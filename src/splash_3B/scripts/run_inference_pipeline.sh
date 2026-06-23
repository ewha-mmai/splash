#!/bin/bash
set -e

ROOT_DIR="$(pwd)"
GPU=${GPU:-0}
MODE=${MODE:-else}
SPARSITY=${1:-60}
CKPT_STEP=${2:-1214}

BASE_MODEL="${ROOT_DIR}/checkpoints/Qwen2.5-VL-3B-Instruct"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"${ROOT_DIR}/src/splash_3B/outputs/${SPARSITY}/checkpoint-${CKPT_STEP}"}
DATASET_ROOT="${ROOT_DIR}/dataset/"
OUTPUT_CSV="${ROOT_DIR}/src/splash_3B/outputs/${SPARSITY}/inference_${CKPT_STEP}.csv"
OUTPUT_JSON="${ROOT_DIR}/src/splash_3B/outputs/${SPARSITY}/evaluation_${CKPT_STEP}_gpt-4o.json"

CUDA_VISIBLE_DEVICES=$GPU python src/splash_3B/inference.py \
  --model_mode "$MODE" \
  --gpu "$GPU" \
  --base_model "$BASE_MODEL" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --dataset_root "$DATASET_ROOT" \
  --output_csv "$OUTPUT_CSV"

PYTHONPATH=$ROOT_DIR python src/evaluation.py \
  --csv_path "$OUTPUT_CSV" \
  --output_json "$OUTPUT_JSON"
