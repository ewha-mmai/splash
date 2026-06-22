#!/bin/bash


export PYTHONPATH=$(pwd)
export GPUS_PER_NODE=2
export MASTER_PORT=$(shuf -n 1 -i 10000-65535)
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

export CUDA_VISIBLE_DEVICES=2,3

export WANDB_API_KEY='your_wandb_api_key'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( dirname "$( dirname "$( dirname "$SCRIPT_DIR" )" )" )"

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

SPARSITY=${1:-60}

STUDENT_PATH="${ROOT_DIR}/pretrained/InternVL2_5-1B"
MASK_PATH="${ROOT_DIR}/src/masks/${SPARSITY}_internvl/mask_wanda_${SPARSITY}_det.pt"

TRAIN_CONFIG="${ROOT_DIR}/src/configs/finetune-data-train-config.yaml"
EVAL_CONFIG="${ROOT_DIR}/src/configs/finetune-data-eval-config.yaml"

OUTPUT_DIR="${ROOT_DIR}/src/outputs/1B_mask_${SPARSITY}_det"
DS_CONFIG="${ROOT_DIR}/src/configs/ds_config_stage2.json"

export PYTORCH_ALLOC_CONF=expandable_segments:True


torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    src/splash_1B/stage2_internvl.py \
    --student_path "$STUDENT_PATH" \
    --mask_path "$MASK_PATH" \
    --train_data_config "$TRAIN_CONFIG" \
    --eval_data_config "$EVAL_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed "$DS_CONFIG" \
    --run_name "internvl_mask_${SPARSITY}_det" \
    --seed 42 \
    --data_seed 42 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.0 \
    --num_train_epochs 3 \
    --bf16 True \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --eval_strategy epoch \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end True \
    --greater_is_better False \
    --report_to "wandb" \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --dataloader_prefetch_factor 2 \
    --dataloader_pin_memory True \
    --remove_unused_columns False \
    --use_mask True