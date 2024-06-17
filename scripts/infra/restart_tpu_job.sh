#!/bin/bash

# Default values
DEFAULT_ZONE="us-central2-b"
DEFAULT_PROJECT="nyu-vision-lab"
DEFAULT_SESSION_NAME="cambrian"
DEFAULT_NUM_RETRIES=5
DEFAULT_BRANCH="ssl_eval"
DEFAULT_SCRIPT="scripts/ssl_eval/737k/unfreeze_2_stage.sh --group C --pod_size 256"

WANDB_KEY="${WANDB_API_KEY}"  # Ensure this is set in your environment

# Function to print logs with timestamp and color the time
log() {
    printf "\033[34m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

# Function to display usage
usage() {
    log "Usage: $0 --tpu_name <TPU_NAME> --zone <ZONE> --project <PROJECT> --session_name <SESSION_NAME> --num_retries <NUM_RETRIES> --branch <BRANCH> --script <SCRIPT>"
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
        --zone)
            ZONE="$2"
            shift; shift
            ;;
        --project)
            PROJECT="$2"
            shift; shift
            ;;
        --session_name)
            SESSION_NAME="$2"
            shift; shift
            ;;
        --num_retries)
            NUM_RETRIES="$2"
            shift; shift
            ;;
        --branch)
            BRANCH="$2"
            shift; shift
            ;;
        --script)
            SCRIPT="$2"
            shift; shift
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "$TPU_NAME" ]; then
    log "Error: TPU_NAME is required."
    usage
fi

# Set defaults if not provided
ZONE="${ZONE:-$DEFAULT_ZONE}"
PROJECT="${PROJECT:-$DEFAULT_PROJECT}"
SESSION_NAME="${SESSION_NAME:-$DEFAULT_SESSION_NAME}"
NUM_RETRIES="${NUM_RETRIES:-$DEFAULT_NUM_RETRIES}"
BRANCH="${BRANCH:-$DEFAULT_BRANCH}"
SCRIPT="${SCRIPT:-$DEFAULT_SCRIPT}"

# Function to kill existing processes multiple times
kill_processes() {
    log "Killing existing Python processes and clearing resources on TPU..."
    for ((i = 1; i <= NUM_RETRIES; i++)); do
        gcloud compute tpus tpu-vm ssh --zone "$ZONE" $TPU_NAME --project "$PROJECT" --worker=all \
            --command="sudo pkill -9 python; tmux kill-session -t $SESSION_NAME"
            # --command="sudo pkill -9 python; sudo lsof -w /dev/accel0 | grep .py | awk '{print \"sudo kill -9 \" \$2}' | sh; sudo rm -f /tmp/libtpu_lockfile; tmux kill-session -t $SESSION_NAME"
        log "Attempt $i to kill processes completed."
    done
    log "Finished attempting to kill processes $NUM_RETRIES times."
}

# Function to create a new tmux session
create_tmux_session() {
    log "Creating a new tmux session named $SESSION_NAME..."
    gcloud compute tpus tpu-vm ssh --zone "$ZONE" $TPU_NAME --project "$PROJECT" --worker=all \
        --command="tmux new-session -d -s $SESSION_NAME"
    if [ $? -ne 0 ]; then
        log "Error: Failed to create tmux session $SESSION_NAME."
        exit 1
    fi
    log "Tmux session $SESSION_NAME created."
}

# Function to submit a new script to the tmux session
submit_script() {
    log "Submitting the script to tmux session $SESSION_NAME..."
    gcloud compute tpus tpu-vm ssh --zone "$ZONE" $TPU_NAME --project "$PROJECT" --worker=all \
        --command="cd ~/cambrian && git checkout $BRANCH && git pull && tmux send-keys -t $SESSION_NAME 'cd ~/cambrian && export WANDB_API_KEY=\"$WANDB_KEY\" && export WANDB_ENTITY=nyu-visionx && export WANDB_PROJECT=cambrian && bash $SCRIPT' C-m"
    if [ $? -ne 0 ]; then
        log "Error: Failed to submit the script to tmux session $SESSION_NAME."
        exit 1
    fi
    log "Script submitted to tmux session $SESSION_NAME successfully."
}

# Main execution
kill_processes
create_tmux_session
submit_script

log "All steps completed successfully."
