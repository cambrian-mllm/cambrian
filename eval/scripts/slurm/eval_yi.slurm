#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100|h100
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --time=18:00:00

# name, log paths, etc passed via submit_eval.bash
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL

#SBATCH --output=./logs/yi/%A_%a.log
#SBATCH --error=./logs/yi/%A_%a.err
#SBATCH --array=0-14

echo "> eval_benchmark.slurm $@"

################# Parse Arguments #################

benchmarks=(
    # vqav2
    # gqa
    vizwiz
    scienceqa
    textvqa
    pope
    mme
    mmbench_en
    # mmbench_cn
    seed
    # llava_w
    # mm_vet
    # mmmu
    # mathvista
    # ai2d
    chartqa
    docvqa
    infovqa
    stvqa
    ocrbench
    mmstar
    realworldqa
    # mmvp
    # vstar
)


# take command line args
benchmark=${benchmarks[$SLURM_ARRAY_TASK_ID]}
conv_mode=chatml_direct
ckpt=$SCRATCH/checkpoints/llava-yi-finetune-6993k

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
