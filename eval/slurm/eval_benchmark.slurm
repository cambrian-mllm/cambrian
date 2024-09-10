#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100|h100|rtx8000
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --time=03:00:00

# name, log paths, etc passed via submit_eval.bash
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL

echo "> eval_benchmark.slurm $@"

################# Parse Arguments #################

# take command line args
# Default values
# conv_mode="vicuna_v1"
conv_mode="llama_3"

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

# Print the values
echo "Benchmark: $benchmark"
echo "Checkpoint path: $ckpt"
echo "Conversation mode: $conv_mode"


export SCRATCH=/scratch/${USER}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-$SCRATCH/checkpoints}
export EVAL_DIR=${EVAL_DIR:-$SCRATCH/workspace/mllm_eval_hpc}


################# Run the benchmark #################

singularity exec --bind /scratch --nv --overlay $SCRATCH/overlay/cambrian.ext3:ro $SCRATCH/overlay/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source $SCRATCH/env.sh

cd $EVAL_DIR/scripts
bash run_benchmark.sh --benchmark $benchmark --ckpt $ckpt --conv_mode $conv_mode
"
