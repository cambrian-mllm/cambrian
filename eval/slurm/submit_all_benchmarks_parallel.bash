#!/bin/bash
set -e

echo "> submit_all_benchmarks_parallel.bash $@"

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
dependency=""

# add a help message with no args or -h or --help
helpmsg=$(cat <<-EOF
Usage: bash slurm/submit_all_benchmarks_parallel.bash --ckpt <ckpt> [OPTIONS]

Submits jobs to evaluate a model checkpoint on each benchmark.

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

################# Submit Jobs #################

benchmarks=(
    # vqav2
    gqa
    vizwiz
    scienceqa
    textvqa
    pope
    mme
    mmbench_en
    mmbench_cn
    seed
    # llava_w
    mmvet
    mmmu
    mathvista
    ai2d
    chartqa
    docvqa
    infovqa
    stvqa
    ocrbench
    mmstar
    realworldqa
    mmvp
    vstar
    synthdog
    # vision
    qbench
    blink
    # CV-Bench
    omni
    ade
    coco
)


for benchmark in "${benchmarks[@]}"; do
    # check if $dependency is empty. If it is, don't pass it to the script to avoid errors caused by "shift 2"
    if [[ -z "$dependency" ]]; then
        bash slurm/submit_eval.bash --ckpt $ckpt --benchmark $benchmark --conv_mode $conv_mode --gpus $gpus --constraint $constraint --cpus $cpus --mem $mem --time $time
    else
        bash slurm/submit_eval.bash --ckpt $ckpt --benchmark $benchmark --conv_mode $conv_mode --gpus $gpus --constraint $constraint --cpus $cpus --mem $mem --time $time --dependency $dependency
    fi
done
