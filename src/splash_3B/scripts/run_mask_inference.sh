#!/bin/bash
ROOT_DIR="$(pwd)"

GPU=0
MODE=else
SPARSITY=${1:-60}
CKPT_STEP=${2:-1214}

BASE_MODEL="${ROOT_DIR}/pretrained/Qwen2.5-VL-3B-Instruct"
PRETRAIN_CKPT="${ROOT_DIR}/src/outputs0510/${SPARSITY}/layerwise_shallow_adaptive_t2/checkpoint-${CKPT_STEP}"
DATASET_ROOT="${ROOT_DIR}/dataset/"
OUTPUT="${ROOT_DIR}/src/outputs0510/${SPARSITY}/layerwise_shallow_adaptive_t2/inference_${CKPT_STEP}.csv"

CUDA_VISIBLE_DEVICES=$GPU python src/splash_3B/inference.py \
  --model_mode $MODE \
  --gpu $GPU \
  --base_model $BASE_MODEL \
  --pretrain_ckpt $PRETRAIN_CKPT \
  --dataset_root $DATASET_ROOT \
  --output_csv $OUTPUT
