#!/bin/bash
set -e

echo "> e2e.bash $@"

################# Parse Arguments #################

# Default values
# conv_mode="vicuna_v1"
conv_mode="llama_3"
# gpus=2
# cpus=24
# mem="80GB"
# to allow RTX8000:
gpus=1
cpus=18
mem="32GB"
constraint="a100|h100|rtx8000"
time="10:00:00"

# add a help message with no args or -h or --help
helpmsg=$(cat <<-EOF
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
EOF
)

if [[ $# -eq 0 || $1 == "-h" || $1 == "--help" ]]; then
    echo "$helpmsg"
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
        ckpt="$2"
        shift 2
        ;;
    --conv_mode)
        conv_mode="$2"
        shift 2
        ;;
    --gpus)
        gpus="$2"
        shift 2
        ;;
    --constraint)
        constraint="$2"
        shift 2
        ;;
    --cpus)
        cpus="$2"
        shift 2
        ;;
    --mem)
        mem="$2"
        shift 2
        ;;
    --time)
        time="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
  esac
done

if [[ -z "$ckpt" ]]; then
    echo "Error: --ckpt argument is required."
    exit 1
fi

# print required and optional arguments with default values
echo "Required Arguments:"
echo "  ckpt: $ckpt"
echo "Optional Arguments:"
echo "  conv_mode: $conv_mode (Default: vicuna_v1)"
echo "Slurm Arguments:"
echo "  gpus: $gpus (Default: 1)"
echo "  constraint: $constraint (Default: a100|h100|rtx8000)"
echo "  cpus: $cpus (Default: 18)"
echo "  mem: $mem (Default: 32GB)"
echo "  time: $time (Default: 10:00:00)"

################# Process Arguments #################

# verify that optional args are in the correct format
if [[ ! $gpus =~ ^[0-9]+$ ]]; then
    echo "Error: gpus must be an integer"
    exit 1
fi
if [[ ! $constraint =~ ^[a-z0-9\|]+$ ]]; then
    echo "Error: constraint must be in the format <node1>|<node2>|...|<nodeN>"
    exit 1
fi
if [[ ! $cpus =~ ^[0-9]+$ ]]; then
    echo "Error: cpus must be an integer"
    exit 1
fi
if [[ ! $mem =~ ^[0-9]+[A-Z]+$ ]]; then
    echo "Error: mem must be in the format <number><unit> where unit is one of {K,M,G,T}"
    exit 1
fi
if [[ ! $time =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "Error: time must be in the format HH:MM:SS"
    exit 1
fi

################# Submit Jobs #################

 # submit consolidate job and retrieve the slurm job id
echo "Submitting consolidate job with GCP path: $ckpt"
slurm_args="-J consolidate"_"$(basename $ckpt)  --output=./logs/consolidate/%x-%j.out --error=./logs/consolidate/%x-%j.err"
sbcmd="sbatch $slurm_args --parsable slurm/consolidate.slurm $ckpt"
echo "> $sbcmd"

# consolidate_job_id=$(sbatch --parsable slurm/consolidate.slurm $ckpt)
consolidate_job_id=$($sbcmd)
echo "Job id: $consolidate_job_id"

# extract the fs checkpoint dir
ckpt_name=$(basename $ckpt)
# prepend with "cambrian-" if not already
if [[ $ckpt_name != cambrian-* ]]; then
    ckpt_name="cambrian-$ckpt_name"
fi
CKPT_DIR=${CHECKPOINT_DIR:-$SCRATCH/checkpoints}
ckpt_path=$CKPT_DIR/$ckpt_name

# now queue up the eval jobs with this job id as the dependency
echo "Submitting eval jobs using checkpoint $ckpt_path with dependency $consolidate_job_id"
bash slurm/submit_all_benchmarks_parallel.bash \
    --ckpt $ckpt_path \
    --conv_mode $conv_mode \
    --gpus $gpus \
    --constraint $constraint \
    --cpus $cpus \
    --mem $mem \
    --time $time \
    --dependency $consolidate_job_id

echo "Submitted eval jobs using checkpoint $ckpt_path with dependency $consolidate_job_id"
