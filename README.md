# SPLASH: Wake up for Touch! Mask-isolated Tactile Alignment Learning in MLLMs

Yoonhyung Park\*, Minji Kim\*, Sungwon Moon, Jiyoung Lee†

\*Equal contribution †Corresponding author

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://ewha-mmai.github.io/splash/) [![Paper](https://img.shields.io/badge/Paper-arXiv-red)](TBD) [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/splash-team)

This repository contains the official implementation of **SPLASH**.

SPLASH integrates tactile perception into vision-language models through a two-stage pipeline: (1) generating dormant masks via weight & activation importance scoring on the LLM backbone, and (2) fine-tuning with mask-guided sparse training to align tactile representations without degrading existing visual-language capabilities.

## Architecture

SPLASH supports two backbone architectures:

| Variant | Base MLLM | Tactile Frontend |
|---------|-----------|------------------|
| SPLASH-1B | InternVL2.5-1B | ViT-Tiny + MLP adapter |
| SPLASH-3B | Qwen2.5-VL-3B | ViT-Tiny + MLP adapter |

## Project Structure

```text
.
├── src/
│   ├── splash_1B/                # InternVL 2.5-1B SPLASH variant
│   │   ├── models/               # Modified InternVL model definitions
│   │   ├── scripts/              # Training, inference, and evaluation scripts
│   │   ├── dataset.py            # Tactile dataset loader
│   │   ├── generate_mask.py      # Dormant mask generation
│   │   ├── train.py              # Mask-guided fine-tuning
│   │   └── inference.py          # Inference pipeline
│   ├── splash_3B/                # Qwen2.5-VL-3B SPLASH variant
│   │   └── ...                   # Same layout as splash_1B, with Qwen-specific models
│   ├── models_tactile_frontend/  # Tactile encoder, processor, projector
│   ├── tvl_qwen2_5_vl/           # TVL-Qwen baseline (pretrain/finetune/inference)
│   ├── tvl_llama/                # TVL-LLaMA baseline (inference)
│   ├── unitouch/                 # UniTouch baseline (inference)
│   ├── configs/                  # DeepSpeed and data configs
│   ├── util/                     # Data loading and evaluation utilities
│   ├── evaluation.py             # GPT-4o LLM-as-judge evaluation
│   └── objective_evaluation.py   # Keyword-based objective evaluation
├── third_party/                  # Third-party baseline code
│   └── tvl/                      # TVL-derived code for baseline reproducibility
├── requirements.txt
└── README.md
```

## Requirements

### Environment

To train or evaluate SPLASH, first set up the environment:

```bash
conda create -n splash python=3.10
conda activate splash
pip install -r requirements.txt
```

We recommend CUDA >= 12.0 and 2+ GPUs for distributed training.

### Dataset

All datasets should be placed under `dataset/` in the project root.

| Dataset | Purpose | Source |
|---------|---------|--------|
| `LLaVA-CC3M-Pretrain-595K/` | Dormant subspace mask generation (annotations) | [liuhaotian/LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) |
| `cc3m/` | Dormant subspace mask generation (images) | [CC3M](https://ai.google.com/research/ConceptualCaptions/) |
| `tvl_dataset/` | Mask-guided tactile alignment & VTL evaluation | [mlfu7/Touch-Vision-Language-Dataset](https://huggingface.co/datasets/mlfu7/Touch-Vision-Language-Dataset) |
| `TacQuad/` | OOD VTL evaluation (DIGIT) | [xxuan01/TacQuad](https://huggingface.co/datasets/xxuan01/TacQuad) |

#### Download

```bash
# Install git-lfs
sudo apt install git-lfs
git lfs install
mkdir -p dataset

# 1. LLaVA-CC3M-Pretrain-595K
git clone git@hf.co:datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K
mv LLaVA-CC3M-Pretrain-595K dataset/LLaVA-CC3M-Pretrain-595K
# CC3M images should be placed separately under dataset/cc3m/

# 2. TVL Dataset
git clone git@hf.co:datasets/mlfu7/Touch-Vision-Language-Dataset
cd Touch-Vision-Language-Dataset
zip -s0 tvl_dataset_sharded.zip --out tvl_dataset.zip
unzip tvl_dataset.zip
cd ..
mv Touch-Vision-Language-Dataset/tvl_dataset dataset/tvl_dataset

# 3. TacQuad Dataset
git clone git@hf.co:datasets/xxuan01/TacQuad
mv TacQuad dataset/TacQuad
```

#### Prepare Train/Val Splits

After downloading TVL, generate train/val splits:

```bash
python src/util/prepare_dataset_splits.py --dataset_root dataset/tvl_dataset/hct
python src/util/prepare_dataset_splits.py --dataset_root dataset/tvl_dataset/ssvtp
```

This generates `train_split.csv`, `val_split.csv`, `finetune_train.json`, and `finetune_val.json` under each subdirectory.

### Model Checkpoints

**Base MLLMs** (required for mask generation and training):

| Model | Source | Local Path |
|---------|---------|---------|
| InternVL2.5-1B | [OpenGVLab/InternVL2_5-1B](https://huggingface.co/OpenGVLab/InternVL2_5-1B) | `checkpoints/InternVL2_5-1B/` |
| Qwen2.5-VL-3B-Instruct | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | `checkpoints/Qwen2.5-VL-3B-Instruct/` |

**Trained SPLASH checkpoints**:

| Model | Source | Local Path |
|---------|---------|---------|
| SPLASH-1B | [splash-team/SPLASH_1B](https://huggingface.co/splash-team/SPLASH_1B) | `checkpoints/SPLASH-1B/` |
| SPLASH-3B | [splash-team/SPLASH_3B](https://huggingface.co/splash-team/SPLASH_3B) | `checkpoints/SPLASH-3B/` |

### Baseline Dependencies

TVL-derived code used by the TVL-LLaMA baseline is vendored under `third_party/tvl/` for reproducibility. SPLASH uses the minimal tactile preprocessing utility in `src/util/tactile_preprocess.py` for its main training and inference paths.

To run the **UniTouch baseline** (`src/unitouch/`), install [ImageBind](https://github.com/facebookresearch/ImageBind) separately:

```bash
git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind
pip install -e .
```

## Locate Dormant Subspace

Before training SPLASH, generate dormant masks for the LLM backbone. We adopt a Wanda-style weight & activation importance scoring scheme on visual-language calibration data to identify dormant parameters.

For SPLASH-1B:

```bash
bash src/splash_1B/scripts/run_generate_mask.sh 60 128
```

For SPLASH-3B:

```bash
bash src/splash_3B/scripts/run_generate_mask.sh 60 128
```

The first argument is the sparsity percentage. For example, `60` marks 60% of the selected LLM weights as dormant trainable parameters. The second argument is the number of calibration samples. Masks are saved as `src/splash_1B/masks/<SPARSITY>.pt` and `src/splash_3B/masks/<SPARSITY>.pt` by default.

## Train

To train SPLASH, use the provided training scripts. The scripts load the generated mask, train the tactile frontend, and update only the dormant LLM parameters.

For SPLASH-1B:

```bash
bash src/splash_1B/scripts/run_train.sh 60
```

For SPLASH-3B:

```bash
bash src/splash_3B/scripts/run_train.sh 60
```

Most training parameters, such as the learning rate, batch size, number of epochs, and output directory, are set directly in each `run_train.sh`. Data paths are configured in `src/configs/finetune-data-train-config.yaml` and `src/configs/finetune-data-eval-config.yaml`.

## Evaluation

Set `OPENAI_API_KEY` in a `.env` file before running GPT-4o evaluation.

For inference + GPT-4o LLM-as-judge evaluation:

```bash
# SPLASH-1B
bash src/splash_1B/scripts/run_inference_pipeline.sh 60 2428
# SPLASH-3B
bash src/splash_3B/scripts/run_inference_pipeline.sh 60 1214
```

To use downloaded SPLASH checkpoints instead of locally trained ones, pass the path via environment variable:

```bash
# SPLASH-1B
CKPT=checkpoints/SPLASH-1B bash src/splash_1B/scripts/run_inference_pipeline.sh 60 2428
# SPLASH-3B
CHECKPOINT_PATH=checkpoints/SPLASH-3B bash src/splash_3B/scripts/run_inference_pipeline.sh 60 1214
```

For keyword-based objective evaluation (reuses the inference CSV from above):

```bash
# SPLASH-1B
bash src/splash_1B/scripts/run_objective_evaluation.sh 60 2428
# SPLASH-3B
bash src/splash_3B/scripts/run_objective_evaluation.sh 60 1214
```

## Citation

```bibtex
@inproceedings{park2026splash,
  title={Wake up for Touch! Mask-isolated Tactile Alignment Learning in MLLMs},
  author={Yoonhyung Park and Minji Kim and Sungwon Moon and Jiyoung Lee},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Acknowledgements

We thank the authors of the following projects for making their code publicly available:

- [TVL (Touch-Vision-Language)](https://github.com/Max-Fu/tvl)
- [Wanda (Pruning by Weights and Activations)](https://github.com/locuslab/wanda)

## License

License information will be updated upon public release.
