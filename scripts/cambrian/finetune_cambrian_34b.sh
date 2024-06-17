#!/bin/bash

export PJRT_DEVICE=TPU &&
export XLA_USE_BF16=0 &&
export WANDB_RESUME="allow" &&
export CKPT_NAME="cambrian-34b-finetune" &&

export CKPT_DIR="gs://us-central2-storage/cambrian/checkpoints/$CKPT_NAME" &&


python cambrian/train/train_tpu.py \
    --model_name_or_path NousResearch/Nous-Hermes-2-Yi-34B \
    --version chatml_direct \
    --data_path your_path_to_pretrain_jsonl e.g. Cambrian7M_withsystemprompt.jsonl \
    --image_folder your_path_to_image_folder \
    --pretrain_mm_mlp_adapter ./checkpoints/cambrian-34b-pretrain/mm_projector.bin \
    --vision_tower_aux_list '["siglip/CLIP-ViT-SO400M-14-384", "openai/clip-vit-large-patch14-336", "facebook/dinov2-giant-res378", "clip-convnext-XXL-multi-stage"]' \
    --vision_tower_aux_token_len_list '[576, 576, 576, 9216]' \
    --image_token_len 576 \
    --num_query_group 1 \
    --query_num_list '[576]' \
    --connector_depth 3 \
    --image_position 87 \
    --vision_hidden_size 1024 \
    --connector_only False \
    --num_of_vision_sampler_layers 9 \
    --start_of_vision_sampler_layers 0 \
    --stride_of_vision_sampler_layers 7 \
    --mm_projector_type sva \
    --unfreeze_mm_vision_tower False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir $CKPT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $CKPT_NAME \
    --fsdp "full_shard" \
    --fsdp_config fsdp_config.json


CKPT_PATH=checkpoints/$CKPT_NAME
# check if the checkpoint path exists
if [ ! -d "$CKPT_PATH" ]; then
    echo "Checkpoint path does not exist. Exiting..."
    exit 1
fi
echo "Training finished. Syncing checkpoints to GCS..."
gcloud alpha storage rsync $CKPT_PATH gs://us-central2-storage/cambrian/checkpoints/$CKPT_NAME
echo "Syncing finished. Checkpoints are now available at gs://us-central2-storage/cambrian/checkpoints/$CKPT_NAME"
