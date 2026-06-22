#!/bin/bash
ROOT_DIR="$(pwd)"

export CUDA_VISIBLE_DEVICES=0

SPARSITY=${1:-60}
CKPT_STEP=${2:-1214}

JUDGE_TYPE="gpt4"   # llama | vicuna | gpt4 | gpt5
CSV_PATH="${ROOT_DIR}/src/outputs0510/${SPARSITY}/layerwise_shallow_adaptive_t2/inference_${CKPT_STEP}.csv"

OUTPUT_JSON="${ROOT_DIR}/src/outputs0510/${SPARSITY}/layerwise_shallow_adaptive_t2/evaluation_${CKPT_STEP}_${JUDGE_TYPE}.json"

PYTHONPATH=$ROOT_DIR python src/evaluation.py \
    --csv_path $CSV_PATH \
    --judge_type $JUDGE_TYPE \
    --output_json $OUTPUT_JSON