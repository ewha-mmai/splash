#!/bin/bash
ROOT_DIR="$(pwd)"

GPU=2
SPARSITY=${1:-60}
CKPT_STEP=${2:-2428}

BASE_MODEL="${ROOT_DIR}/pretrained/InternVL2_5-1B"
CKPT="${ROOT_DIR}/src/outputs/1B_mask_${SPARSITY}/checkpoint-${CKPT_STEP}"

DATASET_ROOT="${ROOT_DIR}/dataset/"

OUTPUT="${ROOT_DIR}/src/outputs/1B_mask_${SPARSITY}/inference_${CKPT_STEP}.csv"

CUDA_VISIBLE_DEVICES=$GPU python src/splash_1B/inference.py \
  --ckpt $CKPT \
  --dataset_root $DATASET_ROOT \
  --output_csv $OUTPUT