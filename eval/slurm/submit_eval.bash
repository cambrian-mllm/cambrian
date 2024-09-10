#!/bin/bash
set -e

echo "> submit_eval.bash $@"

################# Parse Arguments #################

# Default values
conv_mode="vicuna_v1"
# gpus=2
# cpus=24
# mem="80GB"
# to allow RTX8000:
gpus=1
cpus=18
mem="32GB"
constraint="a100|h100|rtx8000"
time="10:00:00"
dependency=""

## TODO: should set slurm values to none so they take the eval_benchmark.slurm defaults?

# add a help message with no args or -h or --help
helpmsg=$(cat <<-EOF
Usage: bash slurm/submit_eval.bash --benchmark <benchmark> --ckpt <ckpt> [OPTIONS]

Submit a job to evaluate a model checkpoint on a benchmark.

Required Arguments:
  --benchmark <benchmark>       The benchmark to evaluate on.
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
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
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
    --help)
        echo "$helpmsg"
        exit 1
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
  esac
done

# Check if the required arguments benchmark and ckpt are provided
if [[  -z "$benchmark" || -z "$ckpt" ]]; then
  echo "Error: --benchmark and --ckpt arguments are required."
  exit 1
fi


# print required and optional arguments with default values
echo "Required Arguments:"
echo "  benchmark: $benchmark"
echo "  ckpt: $ckpt"
echo "Optional Arguments:"
echo "  conv_mode: $conv_mode (Default: vicuna_v1)"
echo "Slurm Arguments:"
echo "  gpus: $gpus (Default: 1)"
echo "  constraint: $constraint (Default: a100|h100|rtx8000)"
echo "  cpus: $cpus (Default: 18)"
echo "  mem: $mem (Default: 32GB)"
echo "  time: $time (Default: 10:00:00)"
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

# check that the eval/$benchmark directory exists
if [ ! -d "eval/$benchmark" ]; then
    echo "Error: eval/$benchmark directory does not exist. Benchmark $benchmark may not be supported."
    exit 1
fi

echo "Running $benchmark evaluation on model at $ckpt_path"

script="slurm/eval_$benchmark.slurm"
# default to eval_benchmark.slurm if the benchmark specific script does not exist
if [ ! -f $script ]; then
    echo "Warning: Slurm script $script does not exist. Defaulting to eval_benchmark.slurm"
    script="slurm/eval_benchmark.slurm --benchmark $benchmark"
fi

echo "Using slurm script $script"

################# Submit Job #################
slurm_args="-J $benchmark"_"$(basename $ckpt)  --output=./logs/$benchmark/eval-%x-%j.out --error=./logs/$benchmark/eval-%x-%j.err --gres "gpu:$gpus" --constraint $constraint --mem $mem --cpus-per-task $cpus --time $time"

# add dependency if nonempty
if [[ -n "$dependency" ]]; then
    slurm_args="$slurm_args --dependency=afterok:$dependency"
fi

sbcmd="sbatch $slurm_args $script --ckpt $ckpt --conv_mode $conv_mode"

echo "> $sbcmd"

$sbcmd
