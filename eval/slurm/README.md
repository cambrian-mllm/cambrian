# MLLM Slurm Eval

## HPC Setup

> [!NOTE]
> This was developed on the <a href="https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene" target="_blank">NYU Greene HPC cluster</a>, and may require modifications to run on other setups.

<details open>
<summary><b>Setup gcloud cli</b> <span style="padding-left: 1em;"\> <i>(only needed if working with GCP Cloud Storage)</i></summary>

```bash
SCRATCH=/scratch/${USER}
cd $SCRATCH

curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-473.0.0-linux-x86_64.tar.gz
tar -xvf google-cloud-sdk-473.0.0-linux-x86_64.tar.gz

./google-cloud-sdk/install.sh

# init --> configure default project, region+zone = us-central2-b
./google-cloud-sdk/bin/gcloud init

# install the alpha components
./google-cloud-sdk/bin/gcloud components install alpha
```

</details>

<details open>
<summary><b>Install miniforge:</b></summary>

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

> IMPORTANT: select /scratch/\<USER\>/miniforge3 as the installation path

</details>

<details open>
<summary><b>Create an environment file:</b> <span style="padding-left: 1em;"\> <i>(important)</i></summary>
Your `~/.bashrc` won't be sourced in the Slurm job, so you need to create an environment file to load the necessary modules in the Slurm script. This file will be sourced in the Slurm scripts to set up the environment.


Create an env file in your scratch directory:
```bash
vim $SCRATCH/env.sh
```

Add the following lines to the `$SCRATCH/env.sh` file:
```bash
#!/bin/bash
source $SCRATCH/miniforge3/etc/profile.d/conda.sh

# activate the conda environment
conda activate eval

# The next line updates PATH for the Google Cloud SDK.
if [ -f $SCRATCH/google-cloud-sdk/path.bash.inc ]; then . $SCRATCH/google-cloud-sdk/path.bash.inc; fi

# ensure you download hf files to /scratch instead of /home
export HF_HOME=$SCRATCH/.cache/huggingface
export HF_HUB_CACHE=$SCRATCH/.cache/huggingface/hub
export HF_DATASETS_CACHE=$SCRATCH/.cache/huggingface/datasets

export EVAL_DIR=$SCRATCH/mllm_eval_hpc # path to this project directory
```

> alternatively to setting the HF vars, you can create a dir in $SCRATCH and symlink it to the default cache dir
> ```bash
> mkdir -p $SCRATCH/.cache
> ln -s $SCRATCH/.cache ~/.cache
> ```


### Singularity

#### Setup

<details open>
<summary><span style="font-size: 1.5em;">[From login node]</span> Setup the overlay and download the singularity container:</summary>

```bash
SCRATCH=/scratch/${USER}

mkdir -p ${SCRATCH}/overlay

scp greene-dtn:/scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz ${SCRATCH}/overlay/cambrian.ext3.gz
gunzip -vvv ${SCRATCH}/overlay/cambrian.ext3.gz
scp -rp greene-dtn:/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif ${SCRATCH}/overlay/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
```

</details>

<details open>
<summary><span style="font-size: 1.5em;">[From GPU compute node]</span> Setup the environment:</summary>

Request a GPU node:

```bash
srun --pty -c 6 --mem=16GB --gres=gpu:rtx8000:1 --time=04:00:00 /bin/bash
```

Load the singularity container in read-write mode:

```bash
SCRATCH=/scratch/${USER}
singularity exec --bind /scratch --nv --overlay $SCRATCH/overlay/cambrian.ext3:rw $SCRATCH/overlay/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
```

Install the environment:

```bash
conda create -n eval python=3.10 -y
conda activate eval

pip install --upgrade pip
pip install -r requirements.txt
```

</details>
</details>


## Running Evals via Slurm

### Download / Consolidate Checkpoint from GCP

from the root of this project directory, run the following command to launch a job to 
1. download a checkpoint from GCP using the `gcloud` CLI
2. consolidate the checkpoint using [`consolidate.py`](scripts/consolidate.py)
3. convert the checkpoint to HF format using [`convert_hf_model.py`](scripts/convert_hf_model.py)
```bash
sbatch slurm/consolidate.slurm <path_to_checkpoint>
```

<details>
<summary>example</summary>

```bash
sbatch slurm/consolidate.slurm gs://us-central2-storage/cambrian/checkpoints/TPU-llava-v1.5-7b-finetune-6993k
```

This will save the consolidated checkpoint to `$SCRATCH/cambrian-TPU-llava-v1.5-7b-finetune-6993k`
> Note: an extra "cambrian-" prefix is added to the checkpoint name if it is not already present to ensure the checkpoint can be loaded properly with the `cambrian` code
</details>

### Run single benchmark for single checkpoint via SBATCH
You can launch a sbatch job to evaluate a model checkpoint on a benchmark using [`submit_eval.bash`](slurm/submit_eval.bash)
```bash
Usage: bash slurm/submit_eval.bash --benchmark <benchmark> --ckpt <ckpt> [OPTIONS]

Submit a job to evaluate a model checkpoint on a benchmark.

Required Arguments:
  --benchmark <benchmark>       The benchmark to evaluate on.
  --ckpt <ckpt>                 The path to the model checkpoint.

Optional Arguments:
  --conv_mode <conv_mode>       The conversation mode to use.
                                    (Default: vicuna_v1)
  --gpus <gpus>                 The number of GPUs to request.
                                    (Default: 2)
  --constraint <constraint>     The gres constraint to use.
                                    (Default: a100|h100|rtx8000)
  --cpus <cpus>                 The number of CPUs per task.
                                    (Default: 64)
  --mem <mem>                   The amount of memory to use.
                                    (Default: 128GB)
  --time <time>                 The time limit for the job.
                                    (Default: 03:00:00)
  --help                        Show this message.
```

<details>
<summary>example</summary>

```bash
bash slurm/submit_eval.bash --ckpt $SCRATCH/checkpoints/llava-yi-finetune-6993k/ --conv_mode chatml_direct --constraint "a100|h100" --gpus 2 --benchmark mmmu
```
</details>


<details>
<summary>Under the hood</summary>

The [`submit_eval.bash`](slurm/submit_eval.bash) script does the following:

1. Parses the command-line arguments and validates them.
2. Determines the appropriate Slurm script to use for the evaluation.
3. Constructs the Slurm command to submit the evaluation job.
4. Submits the evaluation job to the Slurm job scheduler.

The Slurm script sets up the environment, loads the necessary modules, and runs the [`run_benchmark.sh`](scripts/run_benchmark.sh) script with the provided arguments. See [`eval_benchmark.slurm`](slurm/eval_benchmark.slurm) for more details.

The [`run_benchmark.sh`](scripts/run_benchmark.sh) script does the following:

1. Parses the command-line arguments.
2. Validates the benchmark directory and required scripts.
3. Handles the distribution of the evaluation workload across multiple GPUs using chunking.
4. Runs the evaluation script for each chunk in parallel.
5. Combines the results from all the chunks into a single answers file.
6. Runs the testing script on the combined answers file to compute the evaluation metrics.

</details>

### Run all benchmarks for single checkpoint in parallel via SBATCH
The [`submit_all_benchmarks_parallel.bash`](slurm/submit_all_benchmarks_parallel.bash) script will call the `submit_eval.bash` script for each benchmark that has been implemented for a given checkpoint.
```bash
Usage: bash slurm/submit_all_benchmarks_parallel.bash --ckpt <ckpt> [OPTIONS]

Submits jobs to evaluate a model checkpoint on each benchmark.

Required Arguments:
  --ckpt <ckpt>                 The path to the model checkpoint.

Optional Arguments:
  --conv_mode <conv_mode>       The conversation mode to use.
                                    (Default: vicuna_v1)
  --gpus <gpus>                 The number of GPUs to request.
                                    (Default: 2)
  --constraint <constraint>     The gres constraint to use.
                                    (Default: a100|h100|rtx8000)
  --cpus <cpus>                 The number of CPUs per task.
                                    (Default: 64)
  --mem <mem>                   The amount of memory to use.
                                    (Default: 128GB)
  --time <time>                 The time limit for the job.
                                    (Default: 03:00:00)
  --help                        Show this message.
```

<details>
<summary>example</summary>

```bash
bash slurm/submit_all_benchmarks_parallel.bash --ckpt $SCRATCH/checkpoints/llava-TPU-llava-v1.5-7b-finetune-6993k
```

or using the [`nyu-visionx/cambrian-8b`](https://huggingface.co/nyu-visionx/cambrian-8b) HF model:

```bash
bash slurm/submit_all_benchmarks_parallel.bash --ckpt nyu-visionx/cambrian-8b
```

</details>

### E2E: Download, Consolidate, Convert, and Evaluate
The [`e2e.bash`](slurm/e2e.bash) script chains together the downloading, consolidation, conversion, and evaluation steps for a given checkpoint stored on GCP.

```bash
Usage: bash slurm/e2e.bash --ckpt <ckpt> [OPTIONS]

End-to-end script to consolidate a GCP checkpoint and submit eval jobs.

Required Arguments:
  --ckpt <ckpt>                 The path to the model checkpoint.

Optional Arguments:
  --conv_mode <conv_mode>       The conversation mode to use.
                                    (Default: vicuna_v1)
  --gpus <gpus>                 The number of GPUs to request.
                                    (Default: 1)
  --constraint <constraint>     The gres constraint to use.
                                    (Default: a100|h100|rtx8000)
  --cpus <cpus>                 The number of CPUs per task.
                                    (Default: 18)
  --mem <mem>                   The amount of memory to use.
                                    (Default: 32GB)
  --time <time>                 The time limit for the job.
                                    (Default: 10:00:00)
  --help                        Show this message.
```

<details>
<summary>example</summary>

```bash
bash slurm/e2e.bash --ckpt gs://us-central2-storage/cambrian/checkpoints/TPU-llava-v1.5-7b-finetune-6993k
```
</details>

<details>
<summary>Under the hood</summary>

The [`e2e.bash`](slurm/e2e.bash) script performs the following steps:

1. **Consolidation Job Submission**:
   - Submits a Slurm job using `slurm/consolidate.slurm` to:
        1. Download the checkpoint from GCP Cloud Storage.
        2. Consolidate the checkpoint using `scripts/consolidate.py`.
        3. Convert the checkpoint to HuggingFace format using `scripts/convert_hf_model.py`.
   - Captures the job ID of the consolidation job.

2. **Checkpoint Path Processing**:
   - Extracts the checkpoint name from the GCP path.
   - Prepends "cambrian-" to the checkpoint name if not already present.
   - Constructs the full local path where the consolidated checkpoint will be saved.

3. **Evaluation Jobs Submission**:
   - Calls `slurm/submit_all_benchmarks_parallel.bash` to submit evaluation jobs for all implemented benchmarks.
   - Passes along all relevant parameters (checkpoint path, conversation mode, Slurm job settings).
   - Sets up a dependency on the consolidation job, ensuring evaluations only start after consolidation is complete.

This end-to-end process automates the entire workflow from downloading a checkpoint from GCP to running all benchmark evaluations, making it easy to evaluate new checkpoints with a single command.

</details>
