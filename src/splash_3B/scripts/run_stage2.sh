#!/bin/bash


export GPUS_PER_NODE=4
export MASTER_PORT=$(shuf -n 1 -i 10000-65535)

ROOT_DIR="$(pwd)"

STUDENT_PATH="${ROOT_DIR}/pretrained/Qwen2.5-VL-3B-Instruct"
TEACHER_PATH="${ROOT_DIR}/pretrained/Qwen2.5-VL-7B-Instruct"
MASK_PATH="${ROOT_DIR}/src/masks/60/mask_wanda.pt"
TRAIN_CONFIG="${ROOT_DIR}/src/configs/finetune-data-train-config.yaml"
EVAL_CONFIG="${ROOT_DIR}/src/configs/finetune-data-eval-config.yaml"
OUTPUT_DIR="${ROOT_DIR}/src/distillation/outputs/pipelineA_40_new"
DS_CONFIG="${ROOT_DIR}/src/distillation/configs/ds_config_stage2.json"

export PYTORCH_ALLOC_CONF=expandable_segments:True

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    src/distillation/stage2_main_distillation.py \
    --student_path "$STUDENT_PATH" \
    --teacher_path "$TEACHER_PATH" \
    --mask_path "$MASK_PATH" \
    --train_data_config "$TRAIN_CONFIG" \
    --eval_data_config "$EVAL_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed "$DS_CONFIG" \
    --run_name "pipelineA_40_new" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --num_train_epochs 3 \
    --weight_decay 0.0 \
    --bf16 True \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --eval_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss_task \
    --report_to "wandb" \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    --use_mask True \