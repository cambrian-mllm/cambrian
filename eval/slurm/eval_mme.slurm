#!/bin/bash
#SBATCH --job-name=mme-eval

#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|rtx8000
#SBATCH --cpus-per-task=16
#SBATCH --mem=40GB

#SBATCH --time=02:00:00

#SBATCH --output=./logs/mme/eval-%x-%j.out
#SBATCH --error=./logs/mme/eval-%x-%j.err
#SBATCH --export=ALL

#SBATCH --mail-type=FAIL

echo "> eval_mme.slurm $@"
benchmark="mme"

################# Parse Arguments #################

# take command line args
# Default values
# conv_mode="vicuna_v1"
conv_mode="llama_3"

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
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
  esac
done

# Check if the required argument ckpt is provided
if [[ -z "$ckpt" ]]; then
  echo "Error: --ckpt argument is required."
  exit 1
fi

# Print the values
echo "Benchmark: $benchmark"
echo "Checkpoint path: $ckpt"
echo "Conversation mode: $conv_mode"


export SCRATCH=/scratch/${USER}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-$SCRATCH/checkpoints}
export EVAL_DIR=${EVAL_DIR:-$SCRATCH/workspace/mllm_eval_hpc}


singularity exec --bind /scratch --nv --overlay $SCRATCH/overlay/cambrian.ext3:ro $SCRATCH/overlay/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source $SCRATCH/env.sh

cd $EVAL_DIR/scripts
bash run_benchmark.sh --benchmark $benchmark --ckpt $ckpt --conv_mode $conv_mode
"
