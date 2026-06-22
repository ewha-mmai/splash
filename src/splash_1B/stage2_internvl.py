
import os
import sys
import torch
import wandb
import timm
import inspect
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
)

import transformers.modeling_utils as modeling_utils
def dummy_warmup(*args, **kwargs):
    return

modeling_utils.caching_allocator_warmup = dummy_warmup

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_dir)

from src.splash_1B.models.modeling_internvl_chat import InternVLChatModel
from src.splash_1B.dataset import FinetuneDataset, DataCollatorForTactileDataset

@dataclass
class ModelArguments:
    student_path: str = field(metadata={"help": "InternVL2.5-1B checkpoint"})
    mask_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    train_data_config: str
    eval_data_config: Optional[str] = None

@dataclass
class MaskArguments:
    use_mask: bool = field(default=True)

def init_tactile_weights_safe(model):
    print(" [Defense] Initializing Tactile modules with safe range...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "tactile" in name:
                if torch.isnan(param).any() or torch.isinf(param).any():
                    nn.init.trunc_normal_(param, std=0.01)
                param.data.clamp_(-10.0, 10.0)

class TaskLoggingTrainer(Trainer):

    def __init__(self, *args, masks_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.masks_dict = masks_dict
        self._train_task_losses = []
        self._eval_task_losses = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        inputs.pop("image_grid_thw", None)
        inputs.pop("tactile_grid_thw", None)

        if "image_flags" not in inputs or inputs["image_flags"] is None:
            if "pixel_values" in inputs:
                bsz = inputs["pixel_values"].shape[0]
                device = inputs["pixel_values"].device
                inputs["image_flags"] = torch.ones(bsz, 1, device=device)

        outputs = model(**inputs)
        loss = outputs.loss

        if model.training:
            self._train_task_losses.append(loss.detach())
        else:
            self._eval_task_losses.append(loss.detach())

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):

        loss = super().training_step(model, inputs, num_items_in_batch)

        real_model = model.module if hasattr(model, "module") else model

        if self.masks_dict is not None:
            self._check_masked_llm_grad(real_model, self.masks_dict)

        return loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and hasattr(self, "_last_task_loss"):
            logs["task_loss"] = self._last_task_loss.item()
        super().log(logs, *args, **kwargs)
        
    def _check_masked_llm_grad(self, model, masks_dict):

        if not self.state.is_world_process_zero:
            return

        real_model = model.module if hasattr(model, "module") else model

        print("\n --- Masked LLM Grad Check ---")

        for name, param in real_model.named_parameters():

            if name in masks_dict and param.grad is not None:

                mask = masks_dict[name].to(param.grad.device)
                grad = param.grad.detach()

                masked_grad = grad[mask == 0]
                active_grad = grad[mask == 1]

                zero_region_norm = masked_grad.norm().item() if masked_grad.numel() > 0 else 0
                active_region_norm = active_grad.norm().item() if active_grad.numel() > 0 else 0

                print(f"{name}")
                print(f"  mask=0 grad norm : {zero_region_norm:.4e}")
                print(f"  mask=1 grad norm : {active_region_norm:.4e}")
                print("-----------------------------------")

        print("----------------------------------------\n")
        

def make_mask_hook(mask):
    def hook(grad):
        return grad * mask.to(grad.device)
    return hook

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, MaskArguments, TrainingArguments)
    )
    model_args, data_args, mask_args, training_args = parser.parse_args_into_dataclasses()

    run_id_file = os.path.join(training_args.output_dir, "wandb_run_id.txt")
    if os.path.exists(run_id_file):
        with open(run_id_file) as f: run_id = f.read().strip()
    else:
        run_id = wandb.util.generate_id()
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(run_id_file, "w") as f: f.write(run_id)

    wandb.init(
        project="VLT-Distillation",
        id=run_id,
        resume="allow",
        name=training_args.run_name,
    )


    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1

    print(" Using mask:", model_args.mask_path)


    model = InternVLChatModel.from_pretrained(
        model_args.student_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )


    for name, param in model.named_parameters():
        if "tactile_encoder" in name:

            def make_hook(n):
                def hook(grad):
                    return grad
                return hook

            param.register_hook(make_hook(name))

    model.config.hidden_size = model.language_model.config.hidden_size

    training_args.gradient_checkpointing = True
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    init_tactile_weights_safe(model)
    print(" Loading ImageNet weights for Tactile ViT...")
    timm_model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    model.tactile_encoder.encoder.load_state_dict(timm_model.state_dict(), strict=False)
    del timm_model
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_args.student_path, trust_remote_code=True, use_fast=False)
    
    tactile_tokens = ["<tac>", "</tac>", "<TAC_CONTEXT>"]
    num_added = tokenizer.add_tokens(tactile_tokens, special_tokens=True)

    if num_added > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            model.language_model.get_input_embeddings().weight[-num_added:].normal_(0.0, 0.02)

    embed = model.language_model.get_input_embeddings()

    tac_start_id = tokenizer.convert_tokens_to_ids("<tac>")
    tac_end_id   = tokenizer.convert_tokens_to_ids("</tac>")
    tac_ctx_id   = tokenizer.convert_tokens_to_ids("<TAC_CONTEXT>")

    tac_id = tac_ctx_id 

    img_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    tactile_ids = torch.tensor(
        [tac_start_id, tac_end_id, tac_ctx_id],
        dtype=torch.long
    )

    
    print(f" Extracted IDs - Image: {img_id}, Tactile: {tac_id}")

    model.config.img_context_token_id = img_id
    model.config.tactile_token_id = tac_id 

    model.img_context_token_id = img_id
    model.tactile_token_id = tac_id
    
    if hasattr(model, 'language_model'):
        model.language_model.config.tactile_token_id = tac_id
    
    print(f" [Final Check] ID Injection Complete")
    print(f"   -> Image Token ID: {model.img_context_token_id}")
    print(f"   -> Tactile Token ID: {model.tactile_token_id}")
    
    print(f" Config Check - Image ID: {model.config.img_context_token_id}, Tactile ID: {model.config.tactile_token_id}")
    

    model.requires_grad_(False)

    for name, param in model.named_parameters():
        if "tactile" in name.lower() or "projector" in name.lower():
            param.requires_grad = True
            param.register_hook(make_hook(name))


    loaded_masks = {}

    if mask_args.use_mask and model_args.mask_path and os.path.exists(model_args.mask_path):

        print(f" Applying masks from: {model_args.mask_path}")

        raw_masks = torch.load(model_args.mask_path, map_location="cpu")

        for k, v in raw_masks.items():

            if k.startswith("language_model.model.layers"):
                loaded_masks[k] = v

            elif k.startswith("model.layers"):
                new_k = k.replace(
                    "model.layers",
                    "language_model.model.layers"
                )
                loaded_masks[new_k] = v

        print(f" Loaded {len(loaded_masks)} mask tensors.")

    
    for name, param in model.named_parameters():

        if name in loaded_masks:

            param.requires_grad = True

            mask_tensor = loaded_masks[name].to(
                device=param.device,
                dtype=param.dtype
            )

            if mask_tensor.shape == param.shape:
                param.register_hook(make_mask_hook(mask_tensor))


    total = 0
    ones = 0

    for v in loaded_masks.values():
        total += v.numel()
        ones += v.sum().item()

    print("Mask density:", ones / total)

    embed = model.language_model.get_input_embeddings()
    embed.weight.requires_grad = True

    vocab_size = embed.weight.shape[0]


    def tactile_embed_hook(grad):
        mask = torch.zeros_like(grad)
        mask[tactile_ids.to(grad.device)] = 1
        return grad * mask

    if not hasattr(embed.weight, "_tactile_hooked"):
        embed.weight.register_hook(tactile_embed_hook)
        embed.weight._tactile_hooked = True


    trainable_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Trainable params: {trainable_cnt:,}")


    print("\n Sample LLM param names:")
    for name, _ in model.named_parameters():
        if "language_model" in name and "layers.0" in name:
            print(name)
            break

    print("\n Sample mask keys:")
    for k in list(loaded_masks.keys())[:5]:
        print(k)

    print("\n --- Real Trainable Parameter Count (Mask=1  | InternVL Safe) ---")

    real_total = 0
    llm_total = 0
    llm_active = 0
    tactile_total = 0
    embed_total = 0

    llm_module = model.language_model
    embed_module = llm_module.get_input_embeddings()

    for name, param in model.named_parameters():

        if name in loaded_masks:

            numel = param.numel()
            llm_total += numel

            active = loaded_masks[name].sum().item()
            llm_active += active
            real_total += active

        elif (
            mask_args.use_mask is False
            and "language_model" in name
            and param.requires_grad
        ):
            numel = param.numel()
            llm_total += numel
            llm_active += numel
            real_total += numel

        elif param.requires_grad and (
            "tactile" in name.lower()
            or "projector" in name.lower()
        ):
            numel = param.numel()
            tactile_total += numel
            real_total += numel

        elif param is embed_module.weight:

            hidden_dim = param.shape[1]
            active_embed = len(tactile_ids) * hidden_dim

            embed_total += active_embed
            real_total += active_embed


    print(f"LLM   (mask ): {llm_total:,}")
    print(f"LLM   (mask=1): {llm_active:,.0f}")

    if llm_total > 0:
        print(f"LLM   : {(llm_active / llm_total) * 100:.2f}%")

    print(f"Tactile  : {tactile_total:,}")
    print(f"Embedding  : {embed_total:,}")
    print(f"     (Mask ): {real_total:,}")
    print("----------------------------------------------------------\n")
    
    
    print([n for n, _ in model.named_modules() if "tactile_projector" in n])


    initial_projector_path = os.path.join(training_args.output_dir, "initial_projector.pt")

    os.makedirs(training_args.output_dir, exist_ok=True)

    real_model = model.module if hasattr(model, "module") else model

    torch.save(
        real_model.tactile_projector.state_dict(),
        initial_projector_path
    )

    print(f" Saved initial projector weights to: {initial_projector_path}")
    print("--------------------------------------------------\n")


    train_dataset = FinetuneDataset(
        config_path=data_args.train_data_config,
        tokenizer=tokenizer,
        qwen_path=model_args.student_path,
        augment_tactile=True,
    )

    eval_dataset = FinetuneDataset(
        config_path=data_args.eval_data_config,
        tokenizer=tokenizer,
        qwen_path=model_args.student_path, 
        augment_tactile=True,
    )


    trainer = TaskLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTactileDataset(tokenizer=tokenizer),
    )

    tactile_params = []
    other_params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            if "tactile" in name.lower():
                tactile_params.append(p)
            else:
                other_params.append(p)

    optimizer_grouped_parameters = [
        {"params": tactile_params, "weight_decay": 0.01},
        {"params": other_params, "weight_decay": 0.0}
    ]

    trainer.optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate
    )


    item = train_dataset[0]

    print("pixel_values_tactile:", item["pixel_values_tactile"].shape if item["pixel_values_tactile"] is not None else "None")
    print("tactile_grid_thw:", item["tactile_grid_thw"])

    tg_list = item["tactile_grid_thw"].tolist() if isinstance(item["tactile_grid_thw"], torch.Tensor) else item["tactile_grid_thw"]
    if len(tg_list) >= 2:
        num_tac = int(tg_list[-2] * tg_list[-1])
    else:
        num_tac = 0
    print("num_tac:", num_tac)
    print(" Starting Optimized Stage2 Training...")
    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    main()