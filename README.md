# SPLASH: Wake up for Touch! Mask-isolated Tactile Alignment Learning in MLLMs

This repository contains the official implementation of **SPLASH**.

SPLASH integrates tactile perception into vision-language models through a two-stage pipeline: (1) generating Wanda-based sparsity masks for the LLM backbone, and (2) fine-tuning with mask-guided sparse training to align tactile representations without degrading existing visual-language capabilities.

## Architecture

SPLASH supports two backbone architectures:

| Variant | Backbone | LLM | Tactile Encoder |
|---------|----------|-----|-----------------|
| SPLASH-1B | InternVL 2.5 | 1B | ViT-Tiny |
| SPLASH-3B | Qwen 2.5-VL | 3B | ViT-Tiny |

## Project Structure

```
.
├── src/
│   ├── splash_1B/              # InternVL 2.5-1B variant
│   │   ├── models/             # Modified InternVL model definitions
│   │   ├── scripts/            # Training and inference shell scripts
│   │   ├── dataset.py          # Tactile dataset loader
│   │   ├── inference.py        # Inference pipeline
│   │   ├── stage1_internvl.py  # Wanda mask generation
│   │   └── stage2_internvl.py  # Mask-guided fine-tuning
│   ├── splash_3B/              # Qwen 2.5-VL-3B variant
│   │   ├── models/             # Modified Qwen2.5-VL model definitions
│   │   ├── scripts/            # Training and inference shell scripts
│   │   ├── dataset.py          # Tactile dataset loader
│   │   ├── inference.py        # Inference pipeline
│   │   ├── stage1_generate_masks_skip.py   # Wanda mask generation
│   │   └── stage2_mask_train.py            # Mask-guided fine-tuning
│   ├── models_tactile_frontend/  # Tactile encoder, processor, projector
│   ├── tvl_qwen2_5_vl/          # TVL-Qwen baseline (pretrain/finetune)
│   ├── tvl_llama/                # TVL-LLaMA baseline (inference/latency)
│   ├── unitouch/                 # UniTouch baseline (inference)
│   ├── util/                     # Data loading and evaluation utilities
│   ├── configs/                  # DeepSpeed and data configs
│   ├── evaluation.py             # LLM-as-judge evaluation
│   └── objective_evaluation.py   # Keyword-based objective evaluation
├── tvl/                          # Touch-Vision-Language encoder
│   ├── tvl_enc/                  # TVL encoder (pretraining)
│   └── tvl_llama/                # TVL-LLaMA (fine-tuning & evaluation)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/ewha-mmai/splash.git
cd SPLASH
pip install -r requirements.txt
```

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 12.0
- 2+ GPUs recommended for distributed training

### Baseline Dependencies (Optional)

To run the **UniTouch baseline** (`src/unitouch/`), you need to install [ImageBind](https://github.com/facebookresearch/ImageBind) separately:
```bash
git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind && pip install -e .
```

### Pretrained Models

Please download the following pretrained models from their official repositories and organize them as follows:

```
SPLASH/
└── pretrained/
    ├── InternVL2_5-1B/
    └── Qwen2.5-VL-3B-Instruct/
```

| Model | Source | Local Path |
|---------|---------|---------|
| InternVL2.5-1B | `OpenGVLab/InternVL2_5-1B` | `pretrained/InternVL2_5-1B/` |
| Qwen2.5-VL-3B-Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` | `pretrained/Qwen2.5-VL-3B-Instruct/` |

## Usage

### Stage 1: Generate Sparsity Masks

Generate Wanda-based sparsity masks for the LLM backbone using calibration data.
**Note:** Stage 1 requires both the pretrained checkpoints and the `LLaVA-CC3M-Pretrain-595K` calibration dataset prepared under the project root.

**SPLASH-1B (InternVL):**
```bash
python src/splash_1B/stage1_internvl.py
```

**SPLASH-3B (Qwen):**
```bash
python src/splash_3B/stage1_generate_masks_skip.py
```

The masks are saved under `src/masks/` with the specified pruning ratio (default: 60%).

### Stage 2: Mask-guided Fine-tuning

Fine-tune the model with tactile data using the generated masks.

**SPLASH-1B:**
```bash
bash src/splash_1B/scripts/run_stage2_internvl.sh [SPARSITY]
# Example: bash src/splash_1B/scripts/run_stage2_internvl.sh 60
```

**SPLASH-3B:**
```bash
bash src/splash_3B/scripts/run_mask_train.sh [SPARSITY]
# Example: bash src/splash_3B/scripts/run_mask_train.sh 60
```

### Inference

**SPLASH-1B:**
```bash
bash src/splash_1B/scripts/run_intern_inference.sh [SPARSITY] [CHECKPOINT_STEP]
# Example: bash src/splash_1B/scripts/run_intern_inference.sh 60 2428
```

**SPLASH-3B:**
```bash
bash src/splash_3B/scripts/run_mask_inference.sh [SPARSITY] [CHECKPOINT_STEP]
```

### Evaluation

Run the full inference + evaluation pipeline:

**SPLASH-1B:**
```bash
bash src/splash_1B/scripts/run_inference_pipeline.sh
```

**SPLASH-3B:**
```bash
bash src/splash_3B/scripts/run_inference_pipeline.sh
```

Or evaluate a specific inference result:
```bash
python src/evaluation.py \
    --csv_path <path_to_inference_csv> \
    --judge_type gpt4 \
    --output_json <path_to_output_json>
```

Supported judge types: `gpt4`, `gpt5`, `llama`, `vicuna`

## Data Preparation

Prepare the datasets under the project root as follows:

```
SPLASH/
├── dataset/
│   ├── tvl_dataset/
│   │   ├── ssvtp/
│   │   │   ├── finetune_train.json
│   │   │   └── finetune_val.json
│   │   └── hct/
│   │       ├── data1/
│   │       ├── data2/
│   │       └── data3/
│   └── LLaVA-CC3M-Pretrain-595K/
│       └── chat.json
└── pretrained/
```

- `LLaVA-CC3M-Pretrain-595K` is required for Stage 1 Wanda mask generation.
- `tvl_dataset` is used for Stage 2 fine-tuning and evaluation.

Each annotation JSON follows the conversation format:
```json
[
  {
    "image": "path/to/image.jpg",
    "tactile": "path/to/tactile.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe the tactile feeling."},
      {"from": "gpt", "value": "The surface feels rough and grainy..."}
    ]
  }
]
```

## Configuration

- **DeepSpeed config:** `src/configs/ds_config_stage2.json` (ZeRO Stage 2)
- **Training data:** `src/configs/finetune-data-train-config.yaml`
- **Eval data:** `src/configs/finetune-data-eval-config.yaml`
- **WandB (optional):**
  - Enable logging: `export WANDB_API_KEY=<your_wandb_api_key>`
  - Disable logging: `export WANDB_MODE=disabled`
- **OpenAI (for GPT-judge):** Set via `OPENAI_API_KEY` in a `.env` file

## Citation

```bibtex
@inproceedings{park2026splash,
  title={Wake up for Touch! Mask-isolated Tactile Alignment Learning in MLLMs},
  author={Yoonhyung Park$^*$, Minji Kim$^*$, Sungwon Moon, and Jiyoung Lee$^\dagger$},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

License information will be updated upon public release.