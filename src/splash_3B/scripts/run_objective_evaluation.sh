#!/bin/bash

ROOT_DIR="$(pwd)"

export CUDA_VISIBLE_DEVICES=""

TOPK=${1:-5}

USE_CONTRADICTION=${2:-false}


CSV_PATH="${ROOT_DIR}/src/tvl_qwen2_5_vl/outputs/outputs0508/baseline1/finetune_ia3_t2/inference_4856.csv"
OUTPUT_JSON="${ROOT_DIR}/src/tvl_qwen2_5_vl/outputs/ia3_outputs/evaluation_objective.json"


CMD="PYTHONPATH=$ROOT_DIR python src/objective_evaluation.py \
    --csv_path $CSV_PATH \
    --output_json $OUTPUT_JSON \
    --topk $TOPK"

if [ "$USE_CONTRADICTION" = true ]; then
    CMD="$CMD --use_contradiction_penalty"
fi

echo "========================================================="
echo "Running Objective Evaluation"
echo "========================================================="
echo "CSV_PATH            : $CSV_PATH"
echo "OUTPUT_JSON         : $OUTPUT_JSON"
echo "TOPK                : $TOPK"
echo "USE_CONTRADICTION   : $USE_CONTRADICTION"
echo "========================================================="

eval $CMD