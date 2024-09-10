#!/bin/bash
set -e

echo "> submit_all_benchmarks_single.bash $@"

################# Parse Arguments #################

# Default values
conv_mode="vicuna_v1"
gpus=2
constraint="a100|h100|rtx8000"
mem="80GB"
cpus=24
time="24:00:00"
dependency=""

# add a help message with no args or -h or --help
helpmsg=$(cat <<-EOF
Usage: bash slurm/submit_all_benchmarks_single.bash --ckpt <ckpt> [OPTIONS]

Submit a job to evaluate a model checkpoint on a benchmark.

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
  --dependency <job_id>         The job ID to depend on.
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
    --dependency)
        dependency="$2"
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
echo "  gpus: $gpus (Default: 2)"
echo "  constraint: $constraint (Default: a100|h100|rtx8000)"
echo "  cpus: $cpus (Default: 64)"
echo "  mem: $mem (Default: 128GB)"
echo "  time: $time (Default: 24:00:00)"
echo "  dependency: $dependency"


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

# validate dependency job ID if provided
if [[ -n "$dependency" ]]; then
    if ! squeue --noheader --format "%i" | grep -q "^$dependency$"; then
        echo "Error: Invalid dependency job ID: $dependency"
        exit 1
    fi
fi

################# Submit Job #################
job_name="all_$(basename $ckpt)"
output_file="./logs/all/%x-%j.out"
error_file="./logs/all/%x-%j.err"

sbatch_args=(
    --job-name="$job_name"
    --output="$output_file"
    --error="$error_file"
    --gres="gpu:$gpus"
    --constraint="$constraint"
    --mem="$mem"
    --cpus-per-task="$cpus"
    --time="$time"
    --requeue
    --mail-type="FAIL,END"
)

if [[ -n "$dependency" ]]; then
    sbatch_args+=(--dependency="afterok:$dependency")
fi

sbatch "${sbatch_args[@]}" --wrap="bash run_all_benchmarks.sh $ckpt $conv_mode"

echo "Submitted job $job_name"
