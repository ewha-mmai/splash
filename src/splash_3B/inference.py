import os
import sys
import csv
import glob
import argparse
import warnings
from typing import List, Dict, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))

for path in [root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.splash_3B.models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.util.data_utils import load_vision_image, load_tactile_data, inject_tactile_tokens

warnings.filterwarnings("ignore", category=UserWarning)

_original_linspace = torch.linspace
def _safe_linspace(*args, **kwargs):
    if "device" not in kwargs:
        kwargs["device"] = "cpu"
    return _original_linspace(*args, **kwargs)
torch.linspace = _safe_linspace


def parse_args():
    parser = argparse.ArgumentParser(description="Tactile-Vision LLM Inference")
    
    parser.add_argument("--model_mode", type=str, required=True, choices=["pretrain", "finetune", "else"])
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process per dataset")
    parser.add_argument("--gpu", type=str, default="0", help="GPU Device ID")
    
    parser.add_argument("--base_model", type=str, required=True, help="Base HF model path")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--finetune_ckpt", type=str, required=False, help="Path to LoRA adapter")
    
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of datasets")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")

    return parser.parse_args()


def load_model_pipeline(args):
    print(f"[Init] Loading Model Checkpoint from: {args.checkpoint_path}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.checkpoint_path,
        device_map=None,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        local_files_only=True,
    )

    model.to(DEVICE) 

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    processor.tokenizer = tokenizer

    processor.image_processor.min_pixels = 224 * 224
    processor.image_processor.max_pixels = 224 * 224

    if args.model_mode == "finetune" and args.finetune_ckpt:
        print(f"[Init] Loading Adapter: {args.finetune_ckpt}")
        model = PeftModel.from_pretrained(model, args.finetune_ckpt)

    model.eval()

    return model, processor, tokenizer


def load_dataset_samples(root_dir: str, ds_name: str, num_samples: Optional[int] = None) -> List[Dict]:
    samples = []
    
    if ds_name == "ssvtp":
        base_dir = os.path.join(root_dir, "tvl_dataset", "ssvtp")
        csv_path = os.path.join(base_dir, "test.csv")

        if not os.path.exists(csv_path):
            print(f"[Skip] SSVTP csv not found: {csv_path}")
            return []

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            
            for i, row in enumerate(reader):
                if num_samples is not None and i >= num_samples:
                    break
                
                vis_full = os.path.join(base_dir, row[0])
                tac_full = vis_full.replace("rgb", "tac")
                label = row[1] if len(row) > 1 else "N/A"

                samples.append({
                    "vision": vis_full,
                    "tactile": tac_full,
                    "label": label,
                    "type": "ssvtp"
                })

    elif ds_name == "hct":
        target_subdirs = ["data1", "data2", "data3"]
        count = 0
        
        for sub in target_subdirs:
            if num_samples is not None and count >= num_samples:
                break
                
            csv_path = os.path.join(root_dir, "tvl_dataset", "hct", sub, "test.csv")
            folder_path = os.path.dirname(csv_path)

            if not os.path.exists(csv_path):
                print(f"[Skip] HCT csv not found: {csv_path}")
                continue

            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                next(reader)
                
                for row in reader:
                    if num_samples is not None and count >= num_samples:
                        break
                    
                    vis_full = os.path.join(folder_path, row[0])
                    tac_full = os.path.join(folder_path, row[1])
                    label = row[3] if len(row) > 3 else "N/A"

                    samples.append({
                        "vision": vis_full,
                        "tactile": tac_full,
                        "label": label,
                        "type": "hct"
                    })
                    count += 1
                    
    return samples


def load_tacquad_digit_samples(root_dir: str, num_samples: Optional[int] = None):
    samples = []

    indoor_csv = os.path.join(root_dir, "TacQuad", "tacquad", "contact_indoor.csv")
    outdoor_csv = os.path.join(root_dir, "TacQuad", "tacquad", "contact_outdoor.csv")

    count = 0

    def process_csv(csv_path, base_dir):
        nonlocal samples, count

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)

            for row in reader:
                if not row or len(row) < 7:
                    continue

                if num_samples is not None and count >= num_samples:
                    return

                item_name = row[0].strip()

                try:
                    digit_start = int(row[3])
                    digit_end = int(row[4])
                except:
                    continue

                mid_frame = (digit_start + digit_end) // 2

                tactile_path = os.path.join(
                    base_dir,
                    item_name,
                    "digit",
                    f"{mid_frame}.png"
                )

                vision_path = os.path.join(
                    base_dir,
                    item_name,
                    "img_digit",
                    f"{mid_frame}.png"
                )

                if not os.path.exists(tactile_path):
                    continue

                if not os.path.exists(vision_path):
                    continue

                gt_text = row[-1].strip()

                relative_path = os.path.join(
                    os.path.basename(base_dir),
                    item_name,
                    "img_digit",
                    f"{mid_frame}.png"
                )

                samples.append({
                    "vision": vision_path,
                    "tactile": tactile_path,
                    "label": gt_text,
                    "relative_path": relative_path,
                    "type": "tacquad_digit"
                })

                count += 1

    process_csv(indoor_csv, os.path.join(root_dir, "TacQuad", "tacquad", "data_indoor"))
    process_csv(outdoor_csv, os.path.join(root_dir, "TacQuad", "tacquad", "data_outdoor"))

    return samples


def main():
    args = parse_args()

    global DEVICE, DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
    model, processor, tokenizer = load_model_pipeline(args)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results = []

    print(f"\n Inference started. Saving to: {args.output_csv}")

    with open(args.output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Dataset", "File_Name", "GT_Label", "Model_Output"])

        for ds_name in ["hct", "ssvtp", "tacquad_digit"]:

            prompt_text = (
                "List 2–3 tactile attributes of the object shown in the image.\n"
                "Use single-word descriptors only.\n"
                "Output a comma-separated list with no additional text."
            )

            print(f" Processing Dataset: {ds_name.upper()}")
            
            if ds_name == "tacquad_digit":
                samples = load_tacquad_digit_samples(
                    args.dataset_root,
                    num_samples=args.num_samples
                )
            else:
                samples = load_dataset_samples(
                    args.dataset_root,
                    ds_name,
                    num_samples=args.num_samples
                )

            if not samples:
                print(f"   -> No samples found for {ds_name}.")
                continue
            
            print(f"   -> Found {len(samples)} samples.")

            for idx, item in enumerate(samples):
                vis_path = item["vision"]
                tac_path = item["tactile"]
                label = item["label"]

                if "relative_path" in item:
                    file_name = item["relative_path"]
                else:
                    file_name = os.path.basename(vis_path)
                                
                vision_image = load_vision_image(vis_path)
                
                tactile_vals, tactile_grid = load_tactile_data(tac_path, augment=False)

                if tactile_vals is None:
                    print(f"   [Skip] Tactile data missing: {os.path.basename(tac_path)}")
                    continue

                tactile_tensor = tactile_vals.unsqueeze(0).to(DEVICE, dtype=DTYPE)
                tactile_grid = tactile_grid.unsqueeze(0).to(DEVICE)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ] 
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = processor(
                    images=[vision_image], text=[text_input], return_tensors="pt"
                ).to(DEVICE)

                input_ids_1d = inputs.input_ids[0]
                attn_mask_1d = inputs.attention_mask[0]

                final_ids_1d, final_mask_1d = inject_tactile_tokens(
                    input_ids_1d, attn_mask_1d, tokenizer
                )

                final_input_ids = final_ids_1d.unsqueeze(0)
                final_attn_mask = final_mask_1d.unsqueeze(0)

                pixel_values = inputs.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(DTYPE)

                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=final_input_ids,
                        attention_mask=final_attn_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=inputs.get("image_grid_thw"),
                        pixel_values_tactile=tactile_tensor,
                        tactile_grid_thw=tactile_grid,
                        max_new_tokens=64,
                        do_sample=False
                    )

                gen_ids = generated_ids[0][final_input_ids.shape[1]:]
                output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

                writer.writerow([
                    ds_name,
                    file_name,
                    label,
                    output_text.strip()
                ])
                csvfile.flush()
                
                if (idx + 1) % 10 == 0:
                     print(f"   [{idx + 1}/{len(samples)}] Processed...")

    print(f"\n All done! CSV saved at: {os.path.abspath(args.output_csv)}")

if __name__ == "__main__":
    main()