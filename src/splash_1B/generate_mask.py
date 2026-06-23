import os
import argparse
import sys
import glob
import json
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer
import random
import numpy as np

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
from src.splash_1B.models.modeling_internvl_chat import InternVLChatModel
from src.splash_1B.models.conversation import get_conv_template

MODEL_ID = os.path.join(root_dir, "checkpoints/InternVL2_5-1B")
CC3M_TEXT_PATH = os.path.join(root_dir, "dataset/LLaVA-CC3M-Pretrain-595K/chat.json")
CC3M_IMAGE_PATH = os.path.join(root_dir, "dataset/cc3m")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Wanda-style dormant masks for SPLASH-1B.")
    parser.add_argument("--sparsity", type=int, default=60, help="Dormant weight percentage to mark trainable, e.g. 60 for 60%.")
    parser.add_argument("--num_samples", type=int, default=128, help="Number of calibration samples.")
    parser.add_argument("--model_path", type=str, default=MODEL_ID)
    parser.add_argument("--calib_text_path", type=str, default=CC3M_TEXT_PATH)
    parser.add_argument("--calib_image_path", type=str, default=CC3M_IMAGE_PATH)
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for the generated mask. Defaults to src/splash_1B/masks/.")
    return parser.parse_args()


def build_transform(input_size=448):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def get_llm_layers(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.layers
    raise AttributeError("Could not find LLM layers in InternVLChatModel")

def get_skip_layers(layers):
    num_layers = len(layers)
    return {0, num_layers - 1}


def load_cc3m_pairs(text_path, image_folder, num_samples):
    pairs = []
    if os.path.isfile(text_path):
        path = text_path
    else:
        files = glob.glob(os.path.join(text_path, "*.tsv")) or glob.glob(os.path.join(text_path, "*.json"))
        if not files: return pairs
        path = files[0]
        
    try:
        with open(path) as f:
            data = json.load(f)
            
        if isinstance(data, list):
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
        print(f"error: {e}")
        pairs = sorted(pairs, key=lambda x: x["image"])
    return pairs[:num_samples]


def generate_weight_wanda_mask(model, tokenizer, num_samples, pruning_ratio, save_dir):
    print(f"\n[Mode: Wanda] Generating Mask (Weight Magnitude * Input Activation Norm)...")
    print(f"   - Target Sparsity: {pruning_ratio:.0%}")

    print("   1. Preparing Calibration Data...")
    pairs = load_cc3m_pairs(CC3M_TEXT_PATH, CC3M_IMAGE_PATH, num_samples)
    
    if not pairs:
        raise ValueError("Failed to load calibration data pairs.")

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
        for mod_name in target_modules:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, mod_name):
                subset = getattr(layer.self_attn, mod_name)
                full_name = f"language_model.model.layers.{i}.self_attn.{mod_name}"
            elif hasattr(layer, "mlp") and hasattr(layer.mlp, mod_name):
                subset = getattr(layer.mlp, mod_name)
                full_name = f"language_model.model.layers.{i}.mlp.{mod_name}"
            else:
                continue
            handles.append(subset.register_forward_hook(get_wanda_hook(full_name)))

    print(f"   2. Running Inference on {num_samples} samples to collect stats...")
    model.eval()
    transform = build_transform(448)
    
    img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    if img_context_token_id == tokenizer.unk_token_id or img_context_token_id is None:
        print("Tokenizer does not recognize <IMG_CONTEXT>, adding manually.")
        tokenizer.add_tokens(['<IMG_CONTEXT>', '<img>', '</img>'], special_tokens=True)
        img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    model.img_context_token_id = img_context_token_id

    with torch.no_grad():
        for i in tqdm(range(len(pairs))):
            pair = pairs[i]
            
            img = Image.open(pair["image"]).convert('RGB')
            pixel_values = transform(img).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
            image_flags = torch.ones((1, 1), dtype=torch.long, device=model.device)
            
            question = "<image>\n" + pair["text"]
            template = get_conv_template(model.template)
            template.system_message = model.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()
            
            parts = query.split("<image>", 1)

            ids_0 = tokenizer(parts[0], return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
            ids_1 = tokenizer(parts[1], return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
                
            img_ctx_id = model.img_context_token_id
            img_start_id = tokenizer.convert_tokens_to_ids('<img>')
            img_end_id = tokenizer.convert_tokens_to_ids('</img>')
            
            img_start_list = [img_start_id] if img_start_id not in [tokenizer.unk_token_id, None] else []
            img_end_list = [img_end_id] if img_end_id not in [tokenizer.unk_token_id, None] else []
            
            img_ids = torch.tensor([img_start_list + [img_ctx_id] * model.num_image_token + img_end_list], device=model.device)
            
            input_ids = torch.cat([ids_0, img_ids, ids_1], dim=1)
            attention_mask = torch.ones_like(input_ids)

            model(
                pixel_values=pixel_values,
                image_flags=image_flags,
                input_ids=input_ids,
                attention_mask=attention_mask
            )

    for h in handles:
        h.remove()

    print("   3. Calculating Wanda Metric & Pruning...")
    masks = {}

    for i, layer in enumerate(tqdm(layers, desc="Pruning Layers")):

        if i in skip_indices:
            for mod_name in target_modules:

                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, mod_name):
                    module = getattr(layer.self_attn, mod_name)
                    key = f"language_model.model.layers.{i}.self_attn.{mod_name}.weight"
                    masks[key] = torch.zeros_like(module.weight).cpu()

                if hasattr(layer, "mlp") and hasattr(layer.mlp, mod_name):
                    module = getattr(layer.mlp, mod_name)
                    key = f"language_model.model.layers.{i}.mlp.{mod_name}.weight"
                    masks[key] = torch.zeros_like(module.weight).cpu()

            continue

        for mod_name in target_modules:

            is_attn = hasattr(layer, "self_attn") and hasattr(layer.self_attn, mod_name)
            is_mlp = hasattr(layer, "mlp") and hasattr(layer.mlp, mod_name)

            if not (is_attn or is_mlp):
                continue

            module_block = "self_attn" if is_attn else "mlp"
            full_name = f"language_model.model.layers.{i}.{module_block}.{mod_name}"
            key = f"{full_name}.weight"

            target_module = getattr(getattr(layer, module_block), mod_name)
            W = target_module.weight.detach().float()
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

    save_path = os.path.join(save_dir, f"{int(pruning_ratio * 100)}.pt")
    torch.save(masks, save_path)
    print(f" Saved InternVL Wanda Mask: {save_path}")


if __name__ == "__main__":
    args = parse_args()

    if args.sparsity < 0 or args.sparsity > 100:
        raise ValueError("--sparsity must be between 0 and 100")

    MODEL_ID = args.model_path
    CC3M_TEXT_PATH = args.calib_text_path
    CC3M_IMAGE_PATH = args.calib_image_path
    num_samples = args.num_samples
    pruning_ratio = args.sparsity / 100.0
    save_dir = args.output_dir or os.path.join(root_dir, "src/splash_1B/masks")
    os.makedirs(save_dir, exist_ok=True)

    print(f" Loading InternVL2.5-1B Model: {MODEL_ID}")
    
    torch.set_default_device("cpu") 

    model = InternVLChatModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False
    )

    model = model.to(DEVICE).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)

    generate_weight_wanda_mask(model, tokenizer, num_samples, pruning_ratio, save_dir)

    print("\n All InternVL masks generated successfully!")