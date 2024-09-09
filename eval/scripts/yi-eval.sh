#!/bin/bash

echo "> yi_burst.sh $@"

# sbatch scripts/slurm/consolidate_yi.slurm gs://us-central2-storage/cambrian/checkpoints/yi-finetune-6993k-withpretrain

ckpt=/scratch/eb3174/checkpoints/llava-yi-finetune-6993k-withpretrain
conv_mode=cambrian_chatml
constraint="a100|h100"
dependency=""

mem=128GB

# dependency=46128784
# bash scripts/submit_all_benchmarks_single.bash --ckpt $ckpt --conv_mode $conv_mode --constraint $constraint --dependency $dependency --mem $mem
# bash scripts/submit_all_benchmarks_single.bash --ckpt $ckpt --conv_mode $conv_mode --constraint $constraint --mem $mem



# individual jobs
bash scripts/submit_all_benchmarks.bash --ckpt $ckpt --conv_mode $conv_mode --constraint $constraint --mem $mem --time 10:00:00 --cpus 48

