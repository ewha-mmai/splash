#!/bin/bash
set -e

ROOT_DIR="$(pwd)"
export CUDA_VISIBLE_DEVICES=""

SPARSITY=${1:-60}
CKPT_STEP=${2:-2428}
TOPK=${3:-5}
USE_CONTRADICTION=${4:-false}

CSV_PATH=${5:-"${ROOT_DIR}/src/splash_1B/outputs/${SPARSITY}/inference_${CKPT_STEP}.csv"}
OUTPUT_JSON=${6:-"${ROOT_DIR}/src/splash_1B/outputs/${SPARSITY}/objective_evaluation_${CKPT_STEP}.json"}

CMD=(
  python src/objective_evaluation.py
  --csv_path "$CSV_PATH"
  --output_json "$OUTPUT_JSON"
  --topk "$TOPK"
)

if [ "$USE_CONTRADICTION" = true ]; then
  CMD+=(--use_contradiction_penalty)
fi

echo "========================================================="
echo "Running SPLASH-1B Objective Evaluation"
echo "========================================================="
echo "SPARSITY            : $SPARSITY"
echo "CKPT_STEP           : $CKPT_STEP"
echo "CSV_PATH            : $CSV_PATH"
echo "OUTPUT_JSON         : $OUTPUT_JSON"
echo "TOPK                : $TOPK"
echo "USE_CONTRADICTION   : $USE_CONTRADICTION"
echo "========================================================="

PYTHONPATH=$ROOT_DIR "${CMD[@]}"
