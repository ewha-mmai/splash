import os
import sys
import torch
import random
import wandb
import timm
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils.data import Subset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    AutoTokenizer,
    AutoProcessor,
    HfArgumentParser,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if os.path.join(root_dir, "tvl") not in sys.path:
    sys.path.insert(0, os.path.join(root_dir, "tvl"))   

from tvl.tvl_enc import tacvis
from src.tvl_qwen2_5_vl.models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.splash_3B.dataset import PretrainDataset, DataCollatorForTactileDataset

_original_linspace = torch.linspace
def _safe_linspace(*args, **kwargs):
    if "device" not in kwargs:
        kwargs["device"] = "cpu"
    return _original_linspace(*args, **kwargs)
torch.linspace = _safe_linspace


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=os.path.join(root_dir, "pretrained/Qwen2.5-VL-3B-Instruct")
    )
    tactile_ckpt: Optional[str] = field(
        default=os.path.join(
            root_dir,
            "pretrained/Touch-Vision-Language-Models/ckpt/tvl_enc/tvl_enc_vittiny.pth"
        )
    )
    baseline_mode: str = field(
        default="baseline1",
        metadata={"help": "baseline1 | baseline2"}
    )


@dataclass
class DataArguments:
    train_data_config: str = field(
        default=os.path.join(root_dir, "src/configs/pretrain-data-train-config.yaml")
    )
    eval_data_config: str = field(
        default=os.path.join(root_dir, "src/configs/pretrain-data-eval-config.yaml")
    )
    wandb_entity: str = field(default="your-wandb-entity")
    wandb_project: str = field(default="SPLASH-Baseline")
    wandb_run_name: str = field(default="tvl_qwen_pretrain")


@dataclass
class CustomTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./outputs")

    seed: int = field(default=42)

    num_train_epochs: float = field(default=4.0)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    gradient_checkpointing: bool = field(default=True)

    blr: float = field(default=1e-4)
    encoder_lr_scale: float = field(default=1.0)
    projector_lr_scale: float = field(default=5.0)
    weight_decay: float = field(default=0.01)
    max_grad_norm: float = field(default=1.0)
    warmup_ratio: float = field(default=0.1)

    bf16: bool = field(default=True)
    fp16: bool = field(default=False)

    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=5)
    logging_first_step: bool = field(default=True)
    logging_nan_inf_filter: bool = field(default=True)

    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=200)

    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)

    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=4)
    dataloader_pin_memory: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=False)


class EpochPrintCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) + 1
        print(f"\n [Epoch {current_epoch}/{args.num_train_epochs}]")

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        print(f"\n [Epoch {current_epoch}] Finished")


class LayerWiseTrainer(Trainer):
    def create_optimizer(self):

        model = self.model
        
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        eff_batch = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * world_size
        )

        base_lr = self.args.blr * eff_batch / 256

        encoder_lr = base_lr * self.args.encoder_lr_scale
        projector_lr = base_lr * self.args.projector_lr_scale

        baseline_mode = getattr(self.args, "baseline_mode", "baseline1")

        projector_params = []
        encoder_params = []
        other_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            print(f"DEBUG: Trainable Param Found -> {name}")
            if "tactile_projector" in name:
                projector_params.append(p)
            elif "tactile_encoder" in name:
                encoder_params.append(p)
            else:
                other_params.append(p)

        optimizer_grouped_parameters = []

        if baseline_mode == "baseline1":
            if projector_params:
                optimizer_grouped_parameters.append({
                    "params": projector_params,
                    "lr": projector_lr,
                    "weight_decay": self.args.weight_decay,
                })

        elif baseline_mode == "baseline2":

            if projector_params:
                optimizer_grouped_parameters.append({
                    "params": projector_params,
                    "lr": projector_lr,
                    "weight_decay": self.args.weight_decay,
                })

            if encoder_params:
                optimizer_grouped_parameters.append({
                    "params": encoder_params,
                    "lr": encoder_lr,
                    "weight_decay": self.args.weight_decay,
                })

        else:
            raise ValueError("baseline_mode must be baseline1 or baseline2")

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

class TactileGradDebugCallback(TrainerCallback):
    def __init__(self, print_every=50):
        self.print_every = print_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.print_every != 0:
            return

        model = kwargs["model"]

        if hasattr(model, "module"):
            model = model.module

        backbone = model.model if hasattr(model, "model") else model

        print("\n ===== Gradient Debug =====")

        for name, p in backbone.tactile_projector.named_parameters():
            if p.requires_grad:
                if p.grad is None:
                    print(f" projector.{name} grad=None")
                else:
                    print(f" projector.{name} grad mean={p.grad.abs().mean().item():.6e}")

        if hasattr(backbone, "tactile_encoder"):
            for name, p in backbone.tactile_encoder.named_parameters():
                if p.requires_grad:
                    if p.grad is None:
                        print(f" encoder.{name} grad=None")
                        break
                    else:
                        print(f" encoder.{name} grad mean={p.grad.abs().mean().item():.6e}")
                        break

        print("================================\n")


def debug_decode(input_ids, tokenizer):
    print("=" * 60)
    text = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(text)
    print("=" * 60)

def find_assistant_start(input_ids, tokenizer):
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    
    indices = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
    
    print("All <|im_start|> positions:", indices.tolist())
    
    if len(indices) > 0:
        last_start = indices[-1].item()
        print("Assistant <|im_start|> index:", last_start)
        
        start = max(0, last_start - 5)
        end = last_start + 10
        print("Context around assistant start:")
        print(tokenizer.decode(input_ids[start:end], skip_special_tokens=False))
        
        return last_start
    else:
        print(" No <|im_start|> found")
        return None

    
def analyze_token_distribution(dataset, num_samples=3):
    print(f"\n[Analysis] Checking first {num_samples} samples...")
    
    tokenizer = dataset.tokenizer
    tactile_pad_id = dataset.tactile_pad_id
    
    stats = {"vision": [], "tactile": [], "text": [], "total": []}
    
    for i in range(min(len(dataset), num_samples)):
        item = dataset[i]
        input_ids = item["input_ids"]

        n_tactile = (input_ids == tactile_pad_id).sum().item()

        if "image_grid_thw" in item:
            if item["image_grid_thw"].ndim == 1:
                T, H, W = item["image_grid_thw"].tolist()
            else:
                T, H, W = item["image_grid_thw"][0].tolist()
            
            n_vision_patches = H * W
            n_vision = n_vision_patches // 4
        else:
            n_vision_patches = 0
            n_vision = 0

        n_total = len(input_ids)
        n_text = n_total - n_tactile - n_vision

        stats["vision"].append(n_vision)
        stats["tactile"].append(n_tactile)
        stats["text"].append(n_text)
        stats["total"].append(n_total)

        print(f"  Sample {i}: Total={n_total} | Vision={n_vision} (Patches={n_vision_patches}) | Tactile={n_tactile} | Text={n_text}")

    print("\n" + "="*40)
    print(f"  Average Stats (First {num_samples} samples)")
    print("-" * 40)
    avg_v = sum(stats['vision']) / num_samples
    avg_t = sum(stats['tactile']) / num_samples
    print(f"  Vision  : {avg_v:.1f}")
    print(f"  Tactile : {avg_t:.1f}")
    if avg_t > 0:
        print(f"  Ratio   : 1 Tactile : {avg_v/avg_t:.2f} Vision")
    print("="*40 + "\n")

def train():

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    torch.manual_seed(training_args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args.seed)

    data_seed = (
        training_args.data_seed
        if training_args.data_seed is not None
        else training_args.seed
    )
    random.seed(data_seed)

    if training_args.process_index == 0:
        if wandb.run is not None:
            wandb.finish()
        run = wandb.init(
            entity=data_args.wandb_entity,
            project=data_args.wandb_project,
            name=data_args.wandb_run_name,
            resume="allow",
        )
        print(f" WandB: {run.get_url()}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    num_new = tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|tactile_start|>",
            "<|tactile_pad|>",
            "<|tactile_end|>",
        ]
    })

    if num_new > 0:
        model.resize_token_embeddings(len(tokenizer))

    tactile_pad_token_id = tokenizer.convert_tokens_to_ids("<|tactile_pad|>")
    tactile_start_token_id = tokenizer.convert_tokens_to_ids("<|tactile_start|>")
    tactile_end_token_id = tokenizer.convert_tokens_to_ids("<|tactile_end|>")

    model.config.tactile_pad_token_id = tactile_pad_token_id
    model.config.tactile_start_token_id = tactile_start_token_id
    model.config.tactile_end_token_id = tactile_end_token_id

    model.config.tactile_token_id = tactile_pad_token_id

    if hasattr(model, "model"):
        model.model.tactile_token_id = tactile_pad_token_id

    backbone = model.model if hasattr(model, "model") else model

    if model_args.baseline_mode == "baseline1":
        assert os.path.exists(model_args.tactile_ckpt)
        ckpt = torch.load(model_args.tactile_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt

        cleaned = {}
        for k, v in state.items():
            k = k.replace("tactile_encoder.", "").replace("encoder.", "")
            if "head" in k:
                continue
            cleaned[k] = v

        if hasattr(backbone.tactile_encoder, "encoder"):
            target = backbone.tactile_encoder.encoder
        else:
            target = backbone.tactile_encoder
        target.load_state_dict(cleaned, strict=False)

    elif model_args.baseline_mode == "baseline2":
        timm_model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
        timm_state = timm_model.state_dict()

        if hasattr(backbone.tactile_encoder, "encoder"):
            target = backbone.tactile_encoder.encoder
        else:
            target = backbone.tactile_encoder

        missing, unexpected = target.load_state_dict(timm_state, strict=False)

        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        
        del timm_model

    else:
        raise ValueError("baseline_mode must be baseline1 or baseline2")

    model.requires_grad_(False)

    for p in backbone.tactile_projector.parameters():
        p.requires_grad = True

    if model_args.baseline_mode == "baseline2":
        for p in backbone.tactile_encoder.parameters():
            p.requires_grad = True

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f" Trainable params: {trainable:,}/{total:,}")

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    processor.tokenizer = tokenizer


    train_dataset = PretrainDataset(
        config_path=data_args.train_data_config,
        qwen_path=model_args.model_name_or_path,
        processor=processor,
        tokenizer=tokenizer,
        augment_tactile=True,
    )

    eval_dataset = PretrainDataset(
        config_path=data_args.eval_data_config,
        qwen_path=model_args.model_name_or_path,
        processor=processor,
        tokenizer=tokenizer,
        augment_tactile=False,
    )

    collator = DataCollatorForTactileDataset(
        tokenizer=tokenizer,
    )

    analyze_token_distribution(train_dataset, num_samples=3)

    training_args.baseline_mode = model_args.baseline_mode

    trainer = LayerWiseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[
            EpochPrintCallback(),
            TactileGradDebugCallback(print_every=20), 
        ],
    )

    print(" Starting Training...")
    trainer.train()

    if training_args.process_index == 0:
        trainer.save_model(training_args.output_dir)

        processor.tokenizer = tokenizer

        processor.save_pretrained(training_args.output_dir)

        tokenizer.save_pretrained(training_args.output_dir)

        wandb.finish()


if __name__ == "__main__":
    train()
