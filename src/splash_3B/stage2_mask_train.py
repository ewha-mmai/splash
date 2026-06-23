import os
import sys
import torch
import wandb
import timm
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    AutoProcessor
)

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if os.path.join(root_dir, "tvl") not in sys.path:
    sys.path.insert(0, os.path.join(root_dir, "tvl"))

from src.splash_3B.models.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as Custom_QwenVLT
)
from src.splash_3B.dataset import FinetuneDataset, DataCollatorForTactileDataset

@dataclass
class ModelArguments:
    student_path: str = field(metadata={"help": "Student checkpoint (Base Model)"})
    mask_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    train_data_config: str = field(metadata={"help": "Training data config path"})
    eval_data_config: str = field(metadata={"help": "Validation data config path"})


@dataclass
class MaskArguments:
    use_mask: bool = field(default=True, metadata={"help": "Apply mask guided sparsity"})


class TieWeightsCallback(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kwargs):
        if model is not None:
            model.tie_weights()


class SaveProcessorCallback(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        self.processor.save_pretrained(checkpoint_folder)


class TaskLoggingTrainer(Trainer):

    def __init__(self, *args, masks_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_task_losses = []
        self._eval_task_losses = []
        self.masks_dict = masks_dict

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        if model.training:
            self._train_task_losses.append(loss.detach())
        else:
            self._eval_task_losses.append(loss.detach())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):

        if "loss" in logs and len(self._train_task_losses) > 0:
            mean_train = torch.stack(self._train_task_losses).mean()
            logs["loss_task"] = mean_train.item()
            logs.pop("loss", None)
            self._train_task_losses = []

        if "eval_loss" in logs and len(self._eval_task_losses) > 0:
            mean_eval = torch.stack(self._eval_task_losses).mean()
            logs["eval_loss_task"] = mean_eval.item()
            logs.pop("eval_loss", None)
            self._eval_task_losses = []

        super().log(logs, *args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, MaskArguments, TrainingArguments)
    )
    model_args, data_args, mask_args, training_args = parser.parse_args_into_dataclasses()

    seed = training_args.seed

    import random
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_id_file = os.path.join(training_args.output_dir, "wandb_run_id.txt")

    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
        print(f" Resuming W&B run: {run_id}")
    else:
        run_id = wandb.util.generate_id()
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(run_id_file, "w") as f:
            f.write(run_id)
        print(f" New W&B run: {run_id}")

    wandb.init(
        project="SPLASH",
        id=run_id,
        resume="allow",
        name=training_args.run_name,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

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

    student = Custom_QwenVLT.from_pretrained(
        model_args.student_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    
    print(" Loading ImageNet Pretrained Weights for Tactile Encoder (ViT-Tiny)...")
    timm_model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    timm_state_dict = timm_model.state_dict()

    tactile_encoder = (
        student.model.tactile_encoder
        if hasattr(student, "model")
        else student.tactile_encoder
    )

    encoder_base = (
        tactile_encoder.encoder
        if hasattr(tactile_encoder, "encoder")
        else tactile_encoder
    )

    encoder_base.load_state_dict(timm_state_dict, strict=False)
    del timm_model
    torch.cuda.empty_cache()

    student.config.use_cache = False
    if training_args.gradient_checkpointing:
        student.gradient_checkpointing_enable()

    processor = AutoProcessor.from_pretrained(model_args.student_path)
    tokenizer = processor.tokenizer

    special_tokens = {
        "additional_special_tokens": [
            "<|tactile_start|>",
            "<|tactile_pad|>",
            "<|tactile_end|>",
        ]
    }

    num_new = tokenizer.add_special_tokens(special_tokens)
    if num_new > 0:
        student.resize_token_embeddings(len(tokenizer))
        student.tie_weights()

    student.config.tactile_start_token_id = tokenizer.convert_tokens_to_ids("<|tactile_start|>")
    student.config.tactile_pad_token_id   = tokenizer.convert_tokens_to_ids("<|tactile_pad|>")
    student.config.tactile_end_token_id   = tokenizer.convert_tokens_to_ids("<|tactile_end|>")
    student.config.tactile_token_id = student.config.tactile_pad_token_id


    student.requires_grad_(False)

    for name, param in student.named_parameters():
        if "tactile" in name.lower() or "projector" in name.lower():
            param.requires_grad = True

    trainable_total = sum(1 for _, p in student.named_parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_total}")
    
    loaded_masks = {}

    if mask_args.use_mask:

        if model_args.mask_path is None:
            raise ValueError(
                " use_mask=True but mask_path is None. "
                "Please provide a valid mask file."
            )

        if not os.path.exists(model_args.mask_path):
            raise FileNotFoundError(
                f" Mask file not found at: {model_args.mask_path}"
            )

        print(f" Loading mask from: {model_args.mask_path}")
        raw_masks = torch.load(model_args.mask_path, map_location="cpu")

        for k, v in raw_masks.items():

            if k.startswith("model.layers"):
                new_k = k.replace(
                    "model.layers",
                    "model.language_model.layers"
                )
            else:
                new_k = k
                
            loaded_masks[new_k] = v

        print(f" Loaded {len(loaded_masks)} mask tensors.")

    
    for name, param in student.named_parameters():
        if "model.language_model.layers" in name:
            if not mask_args.use_mask:
                param.requires_grad = True
            elif name in loaded_masks:
                param.requires_grad = True
                
                mask_tensor = loaded_masks[name]
                if mask_tensor.shape == param.shape:
                    def make_mask_hook(m):
                        def hook(grad):
                            mask = m.to(device=grad.device, dtype=grad.dtype)
                            return grad * mask
                        return hook

                    param.register_hook(make_mask_hook(mask_tensor))
    
    embed = student.model.language_model.embed_tokens
    embed.weight.requires_grad = True

    vocab_size = embed.weight.shape[0]
    grad_mask = torch.zeros(
        vocab_size,
        dtype=embed.weight.dtype,
    )

    tactile_ids = torch.tensor(
        [
            student.config.tactile_start_token_id,
            student.config.tactile_pad_token_id,
            student.config.tactile_end_token_id,
        ],
        device=embed.weight.device,
    )

    grad_mask[tactile_ids] = 1.0

    def tactile_only_grad_hook(grad):
        mask = grad_mask.to(device=grad.device, dtype=grad.dtype)
        return grad * mask[:, None]

    if not hasattr(embed.weight, "_tactile_hooked"):
        embed.weight.register_hook(tactile_only_grad_hook)
        embed.weight._tactile_hooked = True

    trainable_cnt = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f" Trainable params (Including masked zero-grad params): {trainable_cnt:,}")
    for name, p in student.named_parameters():
        if p.requires_grad:
            print(name)


    initial_projector_path = os.path.join(training_args.output_dir, "initial_projector.pt")

    os.makedirs(training_args.output_dir, exist_ok=True)

    real_model = student.module if hasattr(student, "module") else student

    torch.save(
        real_model.model.tactile_projector.state_dict(),
        initial_projector_path
    )

    print(f" Saved initial projector weights to: {initial_projector_path}")
    print("--------------------------------------------------\n")

    train_dataset = FinetuneDataset(
        config_path=data_args.train_data_config,
        qwen_path=model_args.student_path,
        tokenizer=tokenizer,
        augment_tactile=True,
    )

    eval_dataset = FinetuneDataset(
        config_path=data_args.eval_data_config,
        qwen_path=model_args.student_path,
        tokenizer=tokenizer,
        augment_tactile=True,
    )

    data_collator = DataCollatorForTactileDataset(
        tokenizer=tokenizer, 
    )

    trainer = TaskLoggingTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            TieWeightsCallback(),
            SaveProcessorCallback(processor),
        ],
        masks_dict=loaded_masks,
    )

    checkpoint_dir = training_args.output_dir

    if os.path.isdir(checkpoint_dir) and any("checkpoint" in d for d in os.listdir(checkpoint_dir)):
        print(" Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print(" Starting fresh training...")
        trainer.train()

    if training_args.report_to and "wandb" in training_args.report_to:
        wandb.finish()

    trainer.save_model(training_args.output_dir)


    final_projector_path = os.path.join(training_args.output_dir, "final_projector.pt")

    real_model = student.module if hasattr(student, "module") else student

    torch.save(
        real_model.model.tactile_projector.state_dict(),
        final_projector_path
    )

    print(f" Saved final projector weights to: {final_projector_path}")

    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()