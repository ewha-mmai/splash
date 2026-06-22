export PYTHONPATH=$(pwd)/src:$(pwd)
export GPUS_PER_NODE=1
export MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export CUDA_VISIBLE_DEVICES=1

ROOT_DIR="$(pwd)"

DATASET_ROOT="${ROOT_DIR}/dataset"
MODEL_PATH="${ROOT_DIR}/src/outputs/only_mask_60_skip/checkpoint-1214"
BASE_MODEL="${ROOT_DIR}/pretrained/Qwen2.5-VL-3B-Instruct"

python src/splash_3B/latency.py \
  --model_path $MODEL_PATH \
  --base_model $BASE_MODEL \
  --dataset_root $DATASET_ROOT \
  --dataset_name hct \
  --warmup 50 \
  --runs 200 \
  --steady_start 50