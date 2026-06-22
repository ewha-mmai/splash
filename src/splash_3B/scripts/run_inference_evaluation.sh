#!/bin/bash

ROOT_DIR="$(pwd)"


GPU=1
MODE=else
JUDGE_TYPE="gpt4"

SPARSITIES=(60)
CHECKPOINTS=(1214)

BASE_MODEL="${ROOT_DIR}/pretrained/Qwen2.5-VL-3B-Instruct"
DATASET_ROOT="${ROOT_DIR}/dataset/"

export CUDA_VISIBLE_DEVICES=$GPU


for S in "${SPARSITIES[@]}"; do
  for C in "${CHECKPOINTS[@]}"; do

    echo "=================================================="
    echo "Running Sparsity=$S Checkpoint=$C"
    echo "=================================================="

    PRETRAIN_CKPT="${ROOT_DIR}/src/outputs0510/tvl_frontend_${S}_acc1/checkpoint-${C}"

    OUTPUT_DIR="${ROOT_DIR}/src/outputs0510/tvl_frontend_${S}_acc1"

    mkdir -p "$OUTPUT_DIR"

    OUTPUT_CSV="${OUTPUT_DIR}/inference_${C}.csv"
    OUTPUT_JSON="${OUTPUT_DIR}/evaluation_${C}_${JUDGE_TYPE}.json"


    echo ""
    echo "🚀 Starting Inference..."
    echo ""

    python src/inference.py \
      --model_mode $MODE \
      --gpu $GPU \
      --base_model $BASE_MODEL \
      --pretrain_ckpt $PRETRAIN_CKPT \
      --dataset_root $DATASET_ROOT \
      --output_csv $OUTPUT_CSV


    echo ""
    echo "📊 Starting Evaluation..."
    echo ""

    PYTHONPATH=$ROOT_DIR python src/evaluation.py \
        --csv_path $OUTPUT_CSV \
        --judge_type $JUDGE_TYPE \
        --output_json $OUTPUT_JSON

    echo ""
    echo "✅ Finished Sparsity=$S Checkpoint=$C"
    echo ""

  done
done

echo "🎉 ALL DONE"