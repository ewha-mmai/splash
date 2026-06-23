# TVL-LLaMA Baseline

Requires [TVL](https://github.com/Max-Fu/tvl) pretrained checkpoints.

```bash
python src/tvl_llama/inference.py \
    --model_path <tvl_model_path> \
    --llama_dir <llama_weights_dir> \
    --dataset_root dataset/ \
    --output_csv outputs/tvl_llama_inference.csv
```
