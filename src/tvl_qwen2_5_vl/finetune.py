import os
import sys
import torch
from torch.utils.data import Subset
import random
import wandb
import numpy
from dataclasses import dataclass, field
from typing import Optional, List
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoProcessor, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback, get_linear_schedule_with_warmup, HfArgumentParser
from safetensors.torch import load_file

_original_torch_load = torch.load
def _hack_torch_load(*args, **kwargs):
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _hack_torch_load

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
tvl_inner_dir = os.path.join(root_dir, "tvl")
if tvl_inner_dir not in sys.path:
    sys.path.insert(0, tvl_inner_dir)

_original_linspace = torch.linspace
def _safe_linspace(*args, **kwargs):
    if "device" not in kwargs:
        kwargs["device"] = "cpu"
    return _original_linspace(*args, **kwargs)
torch.linspace = _safe_linspace

from src.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.dataset import FinetuneDataset, DataCollatorForTactileDataset

DEFAULT_PRETRAINED_CKPT = os.path.join(root_dir, "src/baseline/outputs/pretrain_bs2/checkpoint-2428")
DEFAULT_CONFIG_PATH = os.path.join(root_dir, "src/configs/finetune-data-config.yaml")
DEFAULT_DS_CONFIG = os.path.join(root_dir, "src/baseline/configs/ds_config_stage2.json")
DEFAULT_OUTPUT_DIR = os.path.join(root_dir, "src/baseline/outputs/finetune_bs2")
DEFAULT_WANDB_PROJECT = "Baseline"
DEFAULT_WANDB_RUN_NAME = "bs2_finetune"


@dataclass
class ModelArguments:
    """    """
    model_name_or_path: str = field(
        default=DEFAULT_PRETRAINED_CKPT,
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    freeze_tactile_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the tactile encoder"}
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "Attention implementation: 'eager', 'sdpa', or 'flash_attention_2'"}
    )

@dataclass
class DataArguments:
    """  WandB  """
    train_data_config: str = field(
        default=os.path.join(root_dir, "src/configs/finetune-data-train-config.yaml"),
        metadata={"help": "Path to TRAIN data config yaml"}
    )
    eval_data_config: str = field(
        default=os.path.join(root_dir, "src/configs/finetune-data-eval-config.yaml"),
        metadata={"help": "Path to EVAL data config yaml"}
    )
    wandb_entity: str = field(
        default="pyoon0820-ewha-womans-university",
        metadata={"help": "WandB entity name"}
    )
    wandb_project: str = field(
        default=DEFAULT_WANDB_PROJECT,
        metadata={"help": "WandB project name"}
    )
    wandb_run_name: str = field(
        default=DEFAULT_WANDB_RUN_NAME,
        metadata={"help": "WandB run name"}
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    """
     TrainingArguments  User   Default .
            .
    """
    output_dir: str = field(default=DEFAULT_OUTPUT_DIR)
    num_train_epochs: float = field(default=4.0)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    gradient_checkpointing: bool = field(default=True)
    blr: float = field(default=5e-6)
    weight_decay: float = field(default=0.01)
    optim: str = field(default="adamw_torch")
    deepspeed: str = field(default=DEFAULT_DS_CONFIG)
    
    logging_steps: int = field(default=5)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=3)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=200)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    warmup_ratio: float = field(default=0.1)
    
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    dataloader_num_workers: int = field(default=8)
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=False)
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    run_name: str = field(default=DEFAULT_WANDB_RUN_NAME)


class EpochPrintCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) + 1
        print(f"\n [Epoch {current_epoch}/{args.num_train_epochs}] Start")

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        print(f"\n [Epoch {current_epoch}] End")

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.local_rank != -1 and training_args.local_rank != 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    print(f"[Finetune] Train config: {data_args.train_data_config}")
    print(f"[Finetune] Eval config:   {data_args.eval_data_config}")

    if training_args.process_index == 0:
        if wandb.run is not None:
            wandb.finish()

        print(" WandB Connecting...")
        run = wandb.init(
            entity=data_args.wandb_entity,
            project=data_args.wandb_project,
            name=data_args.wandb_run_name,
            resume="allow",
        )
        print(f"\n WandB Link: {run.get_url()}\n")

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    eff_batch = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * world_size
    )

    if training_args.learning_rate is None:
        training_args.learning_rate = training_args.blr * eff_batch / 256

    print(f"Effective batch size: {eff_batch}")
    print(f"Computed LR: {training_args.learning_rate:.2e}")

    print(f"[Finetune] Loading Model from: {model_args.model_name_or_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        _fast_init=False,
        low_cpu_mem_usage=False,
        device_map=None,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )

    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    special_tokens = {
        "additional_special_tokens": [
            "<|tactile_start|>",
            "<|tactile_pad|>",
            "<|tactile_end|>",
        ]
    }
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"[Finetune] Added {num_new_tokens} special tokens.")

    tactile_start_token_id = tokenizer.convert_tokens_to_ids("<|tactile_start|>")
    tactile_pad_token_id = tokenizer.convert_tokens_to_ids("<|tactile_pad|>")
    tactile_end_token_id = tokenizer.convert_tokens_to_ids("<|tactile_end|>")

    model.config.tactile_start_token_id = tactile_start_token_id
    model.config.tactile_pad_token_id = tactile_pad_token_id
    model.config.tactile_end_token_id = tactile_end_token_id
    model.config.tactile_token_id = tactile_pad_token_id

    if hasattr(model, "model"):
        model.model.tactile_token_id = tactile_pad_token_id

    backbone = model.model if hasattr(model, "model") else model

    if hasattr(backbone, "tactile_projector"):
        for param in backbone.tactile_projector.parameters():
            param.requires_grad = True
        print(" Tactile Projector is UN-FROZEN (Trainable).")
    else:
        print(" Warning: 'tactile_projector' not fouznd in model!")

    if hasattr(backbone, "tactile_encoder") and model_args.freeze_tactile_encoder:
        for param in backbone.tactile_encoder.parameters():
            param.requires_grad = False
        print("  Tactile Encoder is FROZEN.")

    print("[Finetune] Applying LoRA Config...")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "tactile_projector"],
        bias="none",
    )

    model = get_peft_model(model, peft_config)

    backbone = model.base_model.model
    if hasattr(backbone, "tactile_projector"):
        for param in backbone.tactile_projector.parameters():
            param.requires_grad = True
        print(" Tactile Projector forced to be TRAINABLE after LoRA init.")

    embed = model.get_input_embeddings() 
    
    vocab_size = embed.weight.shape[0]
    grad_mask = torch.zeros(vocab_size, device=embed.weight.device, dtype=embed.weight.dtype)
    tactile_ids = torch.tensor([
        tactile_start_token_id,
        tactile_pad_token_id,
        tactile_end_token_id,
    ], device=embed.weight.device)
    grad_mask[tactile_ids] = 1.0

    def tactile_only_grad_hook(grad):
        mask = grad_mask.to(grad.device)
        return grad * mask[:, None]

    real_weight = embed.weight if hasattr(embed, "weight") else embed.original_module.weight
    if not hasattr(real_weight, "_tactile_hooked"):
        real_weight.register_hook(tactile_only_grad_hook)
        real_weight._tactile_hooked = True

    model.print_trainable_parameters()

            
    def count_trainable_params(model):

        lora_only = 0
        others = 0
        projector_params = 0

        for name, param in model.named_parameters():

            if not param.requires_grad:
                continue

            if "lora_" in name:
                lora_only += param.numel()

            else:
                others += param.numel()

                if "tactile_projector" in name:
                    projector_params += param.numel()


        embed = model.get_input_embeddings()

        hidden_dim = embed.weight.shape[1]

        effective_embed = 3 * hidden_dim

        effective_total = (
            lora_only
            + projector_params
            + effective_embed
        )

        print("\n" + "=" * 60)
        print(" [Parameter Analysis]")

        print(f"1. Pure LoRA Params:          {lora_only:15,}")
        print(f"2. Full Embed/Proj Counted:  {others:15,}")
        print(f"3. Projector Params:         {projector_params:15,}")

        print("-" * 60)

        print(f" Counted Trainable Total:   {lora_only + others:15,}")

        print("-" * 60)

        print(f" Effective Embed Params:   {effective_embed:15,}")
        print(f" Effective Trainable:      {effective_total:15,}")

        print("=" * 60 + "\n")
    
    count_trainable_params(model)

    pretrain_root = model_args.model_name_or_path.split("/checkpoint")[0]
    processor = AutoProcessor.from_pretrained(
        pretrain_root,
        trust_remote_code=True
    )
    processor.tokenizer = tokenizer

    print(f"[Finetune] Loading TRAIN config: {data_args.train_data_config}")
    train_dataset = FinetuneDataset(
        config_path=data_args.train_data_config,
        qwen_path=model_args.model_name_or_path,
        processor=processor,
        tokenizer=tokenizer,
        augment_tactile=True,
    )

    print(f"[Finetune] Loading EVAL config: {data_args.eval_data_config}")
    eval_dataset = FinetuneDataset(
        config_path=data_args.eval_data_config,
        qwen_path=model_args.model_name_or_path,
        processor=processor,
        tokenizer=tokenizer,
        augment_tactile=False,
    )

    print(f"[Data] Train: {len(train_dataset)}")
    print(f"[Data] Eval:   {len(eval_dataset)}")

    collator = DataCollatorForTactileDataset(processor.tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[EpochPrintCallback()],
    )

    print(" Starting Finetuning (LoRA)...")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if last_checkpoint is not None:
        print(f" Resuming Finetuning from: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print(" Starting Finetuning from SCRATCH...")
        trainer.train()

    print(f"[Finetune]  Finished! Model saved to {training_args.output_dir}")

    trainer.save_model(training_args.output_dir)
    if processor is not None:
        processor.save_pretrained(training_args.output_dir)
        print("   -> Processor saved.")
    if tokenizer is not None:
        tokenizer.save_pretrained(training_args.output_dir)
        print("   -> Tokenizer saved.")

    wandb.finish()


if __name__ == "__main__":
    train()
