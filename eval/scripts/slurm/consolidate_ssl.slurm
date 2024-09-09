#!/bin/bash
#SBATCH --job-name=consolidate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=01:00:00
#SBATCH --output=./logs/consolidate/%A_%a.log
#SBATCH --error=./logs/consolidate/%A_%a.err
#SBATCH --export=ALL
#SBATCH --array=0-12
#SBATCH --mail-type=FAIL


# Define the array of checkpoints
checkpoints=(
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-clip-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-SigLIP-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-DFN-CLIP-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-EVACLIP-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-dinov2-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-dinov2-res336-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-iJEPA-vit-g-16-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-iJEPA-vit-h-14-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-MAE-vit-l-16-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-MAE-vit-h-14-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-MOCO-v3-vit-v-16-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-supervised-vit-l-16-in21k-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-supervised-vit-h-14-in21k-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-maws-mae-h-14-p16-res224-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-large-midas-p16-res384-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-large-beit-midas-p16-res512-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-sam_vit_b-res1024-interp256-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-sam_vit_b-res1024-interp576-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-sam_vit_l-res1024-interp256-737k-bs512"
#   "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-sam_vit_l-res1024-interp576-737k-bs512"
  "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-sam_vit_h-res1024-interp256-737k-bs512"
#   "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-sam_vit_h-res1024-interp576-737k-bs512"
#   "gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-clip-convnext-res512-interp256-737k-bs512

)

# Get the checkpoint based on the job array index
checkpoint=${checkpoints[$SLURM_ARRAY_TASK_ID]}

echo "Running consolidation with checkpoint: $checkpoint"

export SCRATCH=/scratch/${USER}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-$SCRATCH/checkpoints/ssl_exps/737k}

singularity exec --bind /scratch --nv --overlay $SCRATCH/overlay/cambrian.ext3:ro $SCRATCH/overlay/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c "
source $SCRATCH/env.sh
conda activate eval

cd $SCRATCH/workspace/mllm_eval_hpc/
bash scripts/consolidate_gcp.bash $checkpoint
"