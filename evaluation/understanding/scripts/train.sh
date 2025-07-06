#!/bin/bash
ROOT_DIR="/path/to/meditok"
WORKSPACE="${ROOT_DIR}/evaluation/understanding"
cd ${WORKSPACE}

VQ_PATH="${ROOT_DIR}/weights/meditok/meditok.pth"
LLAVAMED_PATH="${ROOT_DIR}/weights/llava-med-v1.5-mistral-7b"
CAPTION_DATA_PATH="${ROOT_DIR}/datasets/understanding/pubmedvision_caption.json"
VQA_DATA_PATH="${ROOT_DIR}/datasets/understanding/pubmedvision_vqa.json"
EXP_NAME="llavamed_meditok_pretrain"
OUTPUT_PROJ_DIR="${ROOT_DIR}/weights/understanding/${EXP_NAME}"

ZERO_CONFIG="${WORKSPACE}/scripts/zero2.json"
BATCH_SIZE=32
GRAD_ACCUM=1

# STAGE 1: Pretraining the projector

deepspeed --num_gpus=8 --master_port=25001 llava/train/train.py \
    --deepspeed ${ZERO_CONFIG} \
    --model_name_or_path ${LLAVAMED_PATH} \
    --vision_tower ${VQ_PATH} \
    --version plain \
    --data_path ${CAPTION_DATA_PATH} \
    --image_folder '' \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_PROJ_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --custom_encoder True \
    --quantize True



sleep 30



# STAGE 2: Fine-tuning the LLM-LORA

PROJ_PATH="${OUTPUT_PROJ_DIR}/mm_projector.bin"
EXP_NAME="llavamed_meditok_sft_lora"
OUTPUT_DIR="${ROOT_DIR}/weights/understanding/${EXP_NAME}"

deepspeed --num_gpus=8 llava/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ${ZERO_CONFIG} \
    --model_name_or_path ${LLAVAMED_PATH} \
    --version v1 \
    --data_path ${VQA_DATA_PATH} \
    --image_folder '' \
    --pretrain_mm_mlp_adapter ${PROJ_PATH} \
    --vision_tower ${VQ_PATH} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --custom_encoder True \
    --quantize True

