#!/bin/bash

ROOT_DIR="$(pwd)"

export CUDA_VISIBLE_DEVICES=""

TOPK=${1:-5}

USE_CONTRADICTION=${2:-false}


CSV_PATH=${3:-"${ROOT_DIR}/outputs/inference.csv"}
OUTPUT_JSON=${4:-"${ROOT_DIR}/outputs/evaluation_objective.json"}


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