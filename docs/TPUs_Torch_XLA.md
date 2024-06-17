
# Torch XLA Run Guide

### Install GCP CLI

[https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)



### Helpful TPU Scripts
#### Use parameters:

```bash
PROJECT=nyu-vision-lab
ZONE=us-central2-b
```


- [list_tpu.bash](https://gist.github.com/ellisbrown/daeb16164561b5e30b8aa31f54aebca6) – list the total TPU usage in our project

- [delete_suspended.bash](https://gist.github.com/ellisbrown/67e143d79dc4f114ed0d39b04d0d2328) – delete all suspended TPUs in our project



## Create a TPU

```bash
# set session variables
TPU_NAME="my-tpu-name"
TPU_TYPE=v4-8  # v4-64, v4-128, v4-256 …
PD_NAME=ellis-pd-mllm-1
```

```bash
# create TPU
gcloud alpha compute tpus queued-resources create $TPU_NAME \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --node-id $TPU_NAME \
  --accelerator-type $TPU_TYPE \
  --runtime-version tpu-ubuntu2204-base \
  --best-effort
```

## Connect to a Disk

### Clone a PD (Persistent Disk)

> **note:** only necessary when you need to need to access new data from the `cambrian-data` pd that isn't on your current pd

```bash
gcloud compute disks create $PD_NAME --source-disk cambrian-data-3t --zone us-central2-b
```



#### Delete a PD

```bash
gcloud compute disks delete $PD_NAME --zone=us-central2-b
```



### Attach Disk to TPU

> only needs to be done once at the beginning

```bash
# attach pd
gcloud alpha compute tpus tpu-vm attach-disk $TPU_NAME \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --disk=$PD_NAME \
  --mode=read-only

# mount
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --worker=all \
  --command="sudo mkdir -p /mnt/disks/storage && sudo mount -o ro,noload /dev/sdb /mnt/disks/storage"
```

# Code Setup

> We have been syncing our code to the pods via git. The method I have found that works with private repos is to `scp` your private key to all TPU workers so that you have access to pull from the repo.

#### Add SSH Key to All Workers

```bash
SSH_KEY=my-ssh-key

# copy key
gcloud compute tpus tpu-vm scp ~/.ssh/$SSH_KEY $TPU_NAME:~/.ssh/$SSH_KEY \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --worker=all

# add ssh key & pre-auth with github to avoid phantom error
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --worker=all \
  --command="chmod 600 ~/.ssh/$SSH_KEY && ssh-add ~/.ssh/$SSH_KEY && ssh -o StrictHostKeyChecking=no git@github.com"
```



#### Clone Repo

```bash
# clone
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --worker=all \
  --command="git clone git@github.com:cambrian-mllm/cambrian.git"
```



#### Pip install

```bash
BRANCH=main
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --worker=all \
  --command="cd ~/cambrian && git fetch --all && git checkout $BRANCH && git pull && pip install --upgrade pip setuptools && pip install -e . && pip install -e .[tpu] && pip install torch~=2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html && sudo snap refresh google-cloud-cli"
```




## Running a Script

> Some longer-running jobs may die if the terminal connection is lost or scrollback fills up. To solve this, we create a `tmux` session on each worker and then run the script inside of the session.

#### Create `tmux` session on all workers

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --worker=all \
  --command="tmux new-session -d -s projectx"
```



#### Run Script inside `tmux` session

```bash
SCRIPT=scripts/finetune_fsdp.sh
BRANCH=main
WANDB_KEY=<>
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --worker=all \
  --command="cd ~/cambrian && git checkout $BRANCH && git pull && tmux send-keys -t projectx 'cd ~/cambrian && export WANDB_API_KEY='$WANDB_KEY' && export WANDB_ENTITY=nyu-visionx && export WANDB_PROJECT=cambrian && bash $SCRIPT' C-m"
```



### Debugging

#### SSH into a worker

```bash
# ssh
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --worker=0
```



#### Kill Session

```bash
# killall python & kill session
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --worker=all \
  --command="sudo killall python || true && tmux kill-session -t projectx"
```



#### Clear TPU memory

> We couldn't find any Torch XLA function to clear TPU memory, so we created the following hacky workaround: clear the memory using JAX. The catch is that installing JAX automatically changes the runtime, so we have to re-install PyTorch XLA afterwards...

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=us-central2-b \
  --project=nyu-vision-lab \
  --worker=all \
  --command="pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && python cambrian/clear.py && pip install torch~=2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html"
```

