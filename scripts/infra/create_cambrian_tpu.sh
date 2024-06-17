#!/bin/bash

# Default arguments
DEFAULT_TPU_NAME="ellis-v4-mllm-1"
DEFAULT_TPU_TYPE="v4-64"
DEFAULT_PD_NAME="ellis-pd-4t"
DEFAULT_BRANCH="ssl_eval"
DEFAULT_ZONE="us-central2-b"
PROJECT="nyu-vision-lab"
RUNTIME_VERSION="tpu-ubuntu2204-base"
SSH_KEY="nyu_key"
WANDB_KEY="${WANDB_API_KEY}"  # Ensure this is set in your environment

# Function to print logs with timestamp and color the time
log() {
    printf "\033[34m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

# Function to display usage
usage() {
    log "Usage: $0 --tpu_name <TPU_NAME> --tpu_type <TPU_TYPE> [options]"
    log "Options:"
    log "  --pd_name <PD_NAME>        (default: $DEFAULT_PD_NAME)"
    log "  --branch <BRANCH>          (default: $DEFAULT_BRANCH)"
    log "  --zone <ZONE>              (default: $DEFAULT_ZONE)"
    log "  --script <SCRIPT>          (optional)"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --tpu_name)
            TPU_NAME="$2"
            shift; shift
            ;;
        --tpu_type)
            TPU_TYPE="$2"
            shift; shift
            ;;
        --pd_name)
            PD_NAME="$2"
            shift; shift
            ;;
        --branch)
            BRANCH="$2"
            shift; shift
            ;;
        --zone)
            ZONE="$2"
            shift; shift
            ;;
        --script)
            SCRIPT="$2"
            shift; shift
            ;;
        --help)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

# Set defaults if not provided
TPU_NAME="${TPU_NAME:-$DEFAULT_TPU_NAME}"
TPU_TYPE="${TPU_TYPE:-$DEFAULT_TPU_TYPE}"
PD_NAME="${PD_NAME:-$DEFAULT_PD_NAME}"
BRANCH="${BRANCH:-$DEFAULT_BRANCH}"
ZONE="${ZONE:-$DEFAULT_ZONE}"

# Function to check TPU state
check_tpu_state() {
    log "Checking TPU state..."
    while true; do
        TPU_STATE=$(gcloud alpha compute tpus queued-resources describe $TPU_NAME --zone $ZONE --project $PROJECT --format="value(state)")
        # strip leading and trailing whitespace
        TPU_STATE=$(echo $TPU_STATE | xargs)
        # strip "state=" prefix
        TPU_STATE=${TPU_STATE#state=}
        log "Current TPU state: $TPU_STATE."
        if [[ "$TPU_STATE" == "ACTIVE" ]]; then
            log "TPU is active and ready."
            break
        elif [[ "$TPU_STATE" == "FAILED" || "$TPU_STATE" == "STOPPED" ]]; then
            log "Error: TPU state is $TPU_STATE. Exiting."
            exit 1
        elif [ -z "$TPU_STATE" ]; then
            log "Error: Unable to find TPU resource. Exiting."
            exit 1
        else
            log "Waiting for TPU to be ACTIVE... (checking again in 15 seconds)"
            sleep 15  # Check every 15 seconds
        fi
    done
}

# Log starting parameters
log "Starting TPU setup with the following parameters:"
log "TPU Name: $TPU_NAME"
log "TPU Type: $TPU_TYPE"
log "Persistent Disk Name: $PD_NAME"
log "Branch: $BRANCH"
log "Zone: $ZONE"

# check if the TPU already exists
TPU_EXISTS=$(gcloud alpha compute tpus queued-resources describe $TPU_NAME --zone $ZONE --project $PROJECT --format="value(name)")

if [ -n "$TPU_EXISTS" ]; then
    log "TPU pod $TPU_NAME already exists."
else
    # 1. Request the TPU pod
    log "TPU pod $TPU_NAME does not exist. Requesting the TPU pod..."
    gcloud alpha compute tpus queued-resources create $TPU_NAME \
        --node-id $TPU_NAME \
        --project $PROJECT \
        --zone $ZONE \
        --accelerator-type $TPU_TYPE \
        --runtime-version $RUNTIME_VERSION \
        --best-effort
    if [ $? -ne 0 ]; then
        log "Error: Failed to request the TPU pod."
        exit 1
    fi
    log "Pod requested successfully."
fi

# 2. Wait for the TPU to be created
log "Waiting for the TPU pod to be created..."
check_tpu_state

# Install dependencies and attach the persistent disk in parallel
install_dependencies() {
    log "Adding SSH key to pods..."
    gcloud compute tpus tpu-vm scp --zone "$ZONE" --project "$PROJECT" --worker=all ~/.ssh/$SSH_KEY $TPU_NAME:~/.ssh/$SSH_KEY
    if [ $? -ne 0 ]; then
        log "Error: Failed to copy SSH key to pods."
        exit 1
    fi
    gcloud compute tpus tpu-vm ssh --zone "$ZONE" $TPU_NAME --project "$PROJECT" --worker=all \
        --command="chmod 600 ~/.ssh/$SSH_KEY && ssh-add ~/.ssh/$SSH_KEY && ssh -o StrictHostKeyChecking=no git@github.com"
    # the above command is expected to error, do not check the return code
    log "SSH key permissions set."

    log "Cloning the repository..."
    gcloud compute tpus tpu-vm ssh --zone "$ZONE" $TPU_NAME --project "$PROJECT" --worker=all \
        --command="git clone git@github.com:cambrian-mllm/cambrian.git"
    if [ $? -ne 0 ]; then
        log "Error: Failed to clone the repository."
        exit 1
    fi
    log "Repository cloned."

    log "Installing the repository and dependencies..."
    gcloud compute tpus tpu-vm ssh --zone "$ZONE" $TPU_NAME --project "$PROJECT" --worker=all \
        --command="cd ~/cambrian && git fetch --all && git checkout $BRANCH && git pull && pip install --upgrade pip setuptools && pip install -e . && pip install -e .[tpu] && pip install torch==2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html && sudo snap refresh google-cloud-cli"
    if [ $? -ne 0 ]; then
        log "Error: Failed to install dependencies."
        exit 1
    fi
    log "Repository and dependencies installed successfully."
}

attach_and_mount_disk() {
    log "Attaching and mounting the Persistent Disk..."
    gcloud alpha compute tpus tpu-vm attach-disk $TPU_NAME \
        --zone $ZONE \
        --disk $PD_NAME \
        --mode read-only
    if [ $? -ne 0 ]; then
        log "Error: Failed to attach the persistent disk."
        exit 1
    fi
    gcloud compute tpus tpu-vm ssh --zone "$ZONE" $TPU_NAME --project "$PROJECT" --worker=all \
        --command="sudo mkdir -p /mnt/disks/storage && sudo mount -o ro,noload /dev/sdb /mnt/disks/storage"
    if [ $? -ne 0 ]; then
        log "Error: Failed to mount the persistent disk."
        exit 1
    fi
    log "Persistent Disk mounted successfully."
}

# Start both processes concurrently
install_dependencies &
attach_and_mount_disk &

# Wait for both processes to finish
wait

# 6. Create a tmux session
log "Creating a new tmux session on all TPU pods..."
gcloud compute tpus tpu-vm ssh --zone "$ZONE" $TPU_NAME --project "$PROJECT" --worker=all \
    --command="tmux new-session -d -s cambrian"
if [ $? -ne 0 ]; then
    log "Error: Failed to create the tmux session."
    exit 1
fi
log "Tmux session created."

# 7. Run the provided script (if available)
if [ -n "$SCRIPT" ]; then
    log "Running the provided script on all TPU pods..."
    gcloud compute tpus tpu-vm ssh --zone "$ZONE" $TPU_NAME --project "$PROJECT" --worker=all \
        --command="cd ~/cambrian && git checkout $BRANCH && git pull && tmux send-keys -t cambrian 'cd ~/cambrian && export WANDB_API_KEY=\"$WANDB_KEY\" && export WANDB_ENTITY=nyu-visionx && export WANDB_PROJECT=cambrian && bash $SCRIPT' C-m"
    if [ $? -ne 0 ]; then
        log "Error: Failed to execute the provided script."
        exit 1
    fi
    log "Script sent to tmux session."
else
    log "No script provided. Initial setup completed."
fi

log "All steps completed successfully."
