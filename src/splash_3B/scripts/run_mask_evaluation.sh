#!/bin/bash
ROOT_DIR="$(pwd)"

export CUDA_VISIBLE_DEVICES=0

SPARSITY=${1:-60}
CKPT_STEP=${2:-1214}

JUDGE_TYPE="gpt4"   # llama | vicuna | gpt4 | gpt5
CSV_PATH="${ROOT_DIR}/src/outputs/3B_${SPARSITY}/inference_${CKPT_STEP}.csv"

OUTPUT_JSON="${ROOT_DIR}/src/outputs/3B_${SPARSITY}/evaluation_${CKPT_STEP}_${JUDGE_TYPE}.json"

PYTHONPATH=$ROOT_DIR python src/evaluation.py \
    --csv_path $CSV_PATH \
    --judge_type $JUDGE_TYPE \
    --output_json $OUTPUT_JSON