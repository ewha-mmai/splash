import os
import argparse
import sys
import glob
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

MODEL_ID = os.path.join(root_dir, "checkpoints/Qwen2.5-VL-3B-Instruct")
CC3M_TEXT_PATH = os.path.join(root_dir, "dataset/LLaVA-CC3M-Pretrain-595K/chat.json")
CC3M_IMAGE_PATH = os.path.join(root_dir, "dataset/cc3m")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Wanda-style dormant masks for SPLASH-3B.")
    parser.add_argument("--sparsity", type=int, default=60, help="Dormant weight percentage to mark trainable, e.g. 60 for 60%.")
    parser.add_argument("--num_samples", type=int, default=128, help="Number of calibration samples.")
    parser.add_argument("--batch_size", type=int, default=2, help="Calibration batch size.")
    parser.add_argument("--model_path", type=str, default=MODEL_ID)
    parser.add_argument("--calib_text_path", type=str, default=CC3M_TEXT_PATH)
    parser.add_argument("--calib_image_path", type=str, default=CC3M_IMAGE_PATH)
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for the generated mask. Defaults to src/splash_3B/masks/.")
    return parser.parse_args()


def get_llm_layers(model):
    """
       LLM Layer   .
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers

    if hasattr(model, "layers"):
        return model.layers

    raise AttributeError(f"Could not find 'layers' attribute in model {type(model)}")


def get_skip_layers(layers):
    """
    [Sandwich Rule]
      (0)  (N-1)
        (/Pruning) .
    """
    num_layers = len(layers)
    return {0, num_layers - 1}


def load_cc3m_pairs(text_path, image_folder, num_samples):
    """
    TSV JSON  ( )     (Pair) .
    """
    pairs = []
    
    if os.path.isfile(text_path):
        path = text_path
    else:
        files = glob.glob(os.path.join(text_path, "*.tsv")) or \
                glob.glob(os.path.join(text_path, "*.json"))
        if not files:
            print(f"     ! ( : {text_path})")
            return pairs
        path = files[0]
    
    print(f" Reading metadata from: {path}")
    
    try:
        if path.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", header=None, on_bad_lines="skip").dropna()
            text_col = 0 if df[0].str.len().mean() >= df[1].str.len().mean() else 1
            img_col = 1 - text_col 

            df = df.sort_values(by=[img_col, text_col]).reset_index(drop=True)
            
            for _, row in df.iterrows():
                img_path = os.path.join(image_folder, os.path.basename(str(row[img_col]).strip()))
                if os.path.exists(img_path):
                    pairs.append({"image": img_path, "text": str(row[text_col]).strip()})
                if len(pairs) >= num_samples: break

        else:
            with open(path) as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                for img_name in sorted(data.keys()):
                    caption = data[img_name]
                    img_path = os.path.join(image_folder, os.path.basename(img_name))
                    if os.path.exists(img_path):
                        pairs.append({"image": img_path, "text": str(caption)})
                    if len(pairs) >= num_samples: break
                        
            elif isinstance(data, list):
                data = sorted(data, key=lambda x: x.get("image", ""))
                for item in data:
                    if "image" in item and "conversations" in item:
                        img_name = item["image"]
                        
                        caption = ""
                        for conv in item["conversations"]:
                            if conv.get("from") == "gpt":
                                caption = conv.get("value", "")
                                break
                                
                        img_path = os.path.join(image_folder, os.path.basename(img_name))
                        if os.path.exists(img_path) and caption:
                            pairs.append({"image": img_path, "text": str(caption)})
                            
                    if len(pairs) >= num_samples: break
                        
    except Exception as e:
        print(f"     : {e}")

    if len(pairs) < num_samples:
        print(f" :    (Pair) {len(pairs)}. (: {num_samples})")

    pairs = sorted(pairs, key=lambda x: x["image"])
    return pairs[:num_samples]


def generate_weight_wanda_mask(model, processor, num_samples, batch_size, pruning_ratio, save_dir):
    print(f"\n[Mode: Wanda] Generating Mask (Weight Magnitude * Input Activation Norm)...")
    print(f"   - Target Sparsity: {pruning_ratio:.0%} (Dormant(1): {pruning_ratio*100}%, Important(0): {100 - pruning_ratio*100}%)")

    print("   1. Preparing Calibration Data...")
    pairs = load_cc3m_pairs(CC3M_TEXT_PATH, CC3M_IMAGE_PATH, num_samples)
    
    if not pairs:
        raise ValueError(" (Pair)  .   .")

    scaler_row = {}
    handles = []

    layers = get_llm_layers(model)
    skip_indices = get_skip_layers(layers)

    def get_wanda_hook(name):
        def hook(module, input, output):
            inp = input[0].detach().view(-1, input[0].shape[-1])
            comming_sq = inp.pow(2).sum(dim=0).float()
            
            if name in scaler_row:
                scaler_row[name] += comming_sq
            else:
                scaler_row[name] = comming_sq
        return hook

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    for i, layer in enumerate(layers):

        if i in skip_indices:
            continue
            
        for mod_name in target_modules:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, mod_name):
                subset = getattr(layer.self_attn, mod_name)
                full_name = f"model.layers.{i}.self_attn.{mod_name}"
            elif hasattr(layer, "mlp") and hasattr(layer.mlp, mod_name):
                subset = getattr(layer.mlp, mod_name)
                full_name = f"model.layers.{i}.mlp.{mod_name}"
            else:
                continue
                
            handles.append(subset.register_forward_hook(get_wanda_hook(full_name)))

    print(f"   2. Running Inference on {num_samples} samples to collect stats...")
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size)):
            batch_pairs = pairs[i : i + batch_size]
            if not batch_pairs:
                continue

            messages = []
            for pair in batch_pairs:
                img_path = pair["image"]
                cap = pair["text"]
                
                messages.append([
                    {"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": cap}]}
                ])

            text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=text_inputs, images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(model.device)

            model(**inputs)

    for h in handles:
        h.remove()

    print("   3. Calculating Wanda Metric & Pruning...")
    masks = {}

    for i, layer in enumerate(tqdm(layers, desc="Pruning Layers")):

        if i in skip_indices:
            for mod_name in target_modules:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, mod_name):
                    module = getattr(layer.self_attn, mod_name)
                    key = f"model.layers.{i}.self_attn.{mod_name}.weight"
                    masks[key] = torch.zeros_like(module.weight).cpu()

                if hasattr(layer, "mlp") and hasattr(layer.mlp, mod_name):
                    module = getattr(layer.mlp, mod_name)
                    key = f"model.layers.{i}.mlp.{mod_name}.weight"
                    masks[key] = torch.zeros_like(module.weight).cpu()

            continue

        for mod_name in target_modules:

            is_attn = hasattr(layer, "self_attn") and hasattr(layer.self_attn, mod_name)
            is_mlp  = hasattr(layer, "mlp") and hasattr(layer.mlp, mod_name)

            if not (is_attn or is_mlp):
                continue

            block = "self_attn" if is_attn else "mlp"
            full_name = f"model.layers.{i}.{block}.{mod_name}"
            key = f"{full_name}.weight"

            module = getattr(getattr(layer, block), mod_name)
            W = module.weight.detach().float()
            rows, cols = W.shape

            if full_name in scaler_row:
                X_norm = torch.sqrt(scaler_row[full_name]).to(W.device)
                X_norm += 1e-6
            else:
                X_norm = torch.ones(cols, device=W.device)

            W_metric = torch.abs(W) * X_norm.reshape(1, -1)

            k = int(cols * pruning_ratio)

            if k == 0:
                mask = torch.zeros_like(W_metric)
            else:
                _, sorted_idx = torch.sort(W_metric, dim=1)
                pruned_idx = sorted_idx[:, :k]

                mask = torch.zeros_like(W_metric)
                mask.scatter_(1, pruned_idx, 1.0)

            masks[key] = mask.cpu()

    total = 0
    pruned = 0
    for m in masks.values():
        total += m.numel()
        pruned += m.sum().item()

    print(f"Actual Sparsity: {pruned/total:.4f}")

    save_path = os.path.join(save_dir, f"{int(pruning_ratio * 100)}.pt")
    torch.save(masks, save_path)
    print(f" Saved Wanda Mask: {save_path}")


if __name__ == "__main__":
    args = parse_args()

    if args.sparsity < 0 or args.sparsity > 100:
        raise ValueError("--sparsity must be between 0 and 100")

    MODEL_ID = args.model_path
    CC3M_TEXT_PATH = args.calib_text_path
    CC3M_IMAGE_PATH = args.calib_image_path
    num_samples = args.num_samples
    batch_size = args.batch_size
    pruning_ratio = args.sparsity / 100.0
    save_dir = args.output_dir or os.path.join(root_dir, "src/splash_3B/masks")
    os.makedirs(save_dir, exist_ok=True)

    print(f" Loading Model: {MODEL_ID}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE, local_files_only=True, _fast_init=False
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)


    generate_weight_wanda_mask(model, processor, num_samples, batch_size, pruning_ratio, save_dir)

    print("\n All masks generated successfully!")