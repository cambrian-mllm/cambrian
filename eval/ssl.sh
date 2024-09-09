
bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/llava-TPU-llava-v1.5-7b-finetune-6993k $benchmark



benchmark=ai2d
# benchmark=chartqa
# bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-clip-737k-bs512/ $benchmark
# bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-DFN-CLIP-737k-bs512/ $benchmark
# bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-dinov2-737k-bs512/ $benchmark
# bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-MAE-vit-h-14-737k-bs512/ $benchmark
# bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-MAE-vit-l-16-737k-bs512/ $benchmark
# bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-supervised-vit-h-14-in21k-737k-bs512/ $benchmark
bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-EVACLIP-737k-bs512/ $benchmark
bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-iJEPA-vit-h-14-737k-bs512/ $benchmark
bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-MOCO-v3-vit-v-16-737k-bs512/ $benchmark
bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-SigLIP-737k-bs512/ $benchmark



bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/llava-TPU-llava-v1.5-7b-finetune-6993k $benchmark

benchmark=textvqa
benchmark=vizwiz
benchmark=gqa
benchmark=mmmu
benchmark=mme
bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-clip-737k-bs512/ $benchmark
bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-DFN-CLIP-737k-bs512/ $benchmark
bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-dinov2-737k-bs512/ $benchmark
bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-MAE-vit-h-14-737k-bs512/ $benchmark
bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-MAE-vit-l-16-737k-bs512/ $benchmark
bash scripts/submit_eval.bash /scratch/eb3174/checkpoints/ssl_exps/737k/llava-vicuna-7b-supervised-vit-h-14-in21k-737k-bs512/ $benchmark