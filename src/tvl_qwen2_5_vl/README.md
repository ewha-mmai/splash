# TVL-Qwen Baseline

Requires [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and [TVL](https://github.com/Max-Fu/tvl) pretrained checkpoints.

```bash
python src/tvl_qwen2_5_vl/inference.py \
    --model_mode finetune \
    --base_model checkpoints/Qwen2.5-VL-3B-Instruct \
    --checkpoint_path <tvl_qwen_checkpoint> \
    --dataset_root dataset/ \
    --output_csv outputs/tvl_qwen_inference.csv
```
