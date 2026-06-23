#!/bin/bash
set -e

ROOT_DIR="$(pwd)"
GPU=${GPU:-0}
SPARSITY=${1:-60}
NUM_SAMPLES=${2:-128}

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

CUDA_VISIBLE_DEVICES=$GPU python src/splash_1B/generate_mask.py \
  --sparsity "$SPARSITY" \
  --num_samples "$NUM_SAMPLES"
