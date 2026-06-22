
import os
import sys
import csv
import argparse
import warnings
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_dir)

from src.splash_1B.models.modeling_internvl_chat import InternVLChatModel
from src.util.data_utils import load_vision_image, load_tactile_data

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    return parser.parse_args()


vision_processor = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])


def load_model(ckpt_path, device):

    print(f"[Init] Loading InternVL from {ckpt_path}")

    model = InternVLChatModel.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        adapter_kwargs=None
    ).to(device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True
        )

    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True
        )

    special_tokens = ["<tac>", "</tac>", "<TAC_CONTEXT>", "<IMG_CONTEXT>"]
    num_added = tokenizer.add_tokens(special_tokens, special_tokens=True)

    if num_added > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))

    img_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    tac_id = tokenizer.convert_tokens_to_ids("<TAC_CONTEXT>")

    model.img_context_token_id = img_id
    model.tactile_token_id = tac_id

    model.config.img_context_token_id = img_id
    model.config.tactile_token_id = tac_id

    if hasattr(model, "language_model"):
        model.language_model.config.tactile_token_id = tac_id

    print("IMG ID:", img_id)
    print("TAC ID:", tac_id)

    model.eval()
    return model, tokenizer


def load_dataset_samples(root_dir: str, ds_name: str, num_samples: Optional[int]):
    samples = []

    if ds_name == "ssvtp":
        base_dir = os.path.join(root_dir, "tvl_dataset", "ssvtp")
        csv_path = os.path.join(base_dir, "test.csv")

        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                if num_samples and i >= num_samples:
                    break
                vis = os.path.join(base_dir, row[0])
                tac = vis.replace("rgb", "tac")
                samples.append({
                    "vision": vis,
                    "tactile": tac,
                    "label": row[1],
                })

    elif ds_name == "hct":
        for sub in ["data1", "data2", "data3"]:
            csv_path = os.path.join(root_dir, "tvl_dataset", "hct", sub, "test.csv")
            folder = os.path.dirname(csv_path)

            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if num_samples and len(samples) >= num_samples:
                        break
                    vis = os.path.join(folder, row[0])
                    tac = os.path.join(folder, row[1])
                    samples.append({
                        "vision": vis,
                        "tactile": tac,
                        "label": row[3],
                    })

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

                if num_samples and count >= num_samples:
                    return

                item_name = row[0].strip()
                digit_start = int(row[3])
                digit_end = int(row[4])
                mid_frame = (digit_start + digit_end) // 2

                tactile_path = os.path.join(base_dir, item_name, "digit", f"{mid_frame}.png")
                vision_path = os.path.join(base_dir, item_name, "img_digit", f"{mid_frame}.png")

                if not os.path.exists(tactile_path) or not os.path.exists(vision_path):
                    continue

                samples.append({
                    "vision": vision_path,
                    "tactile": tactile_path,
                    "label": row[-1],
                })

                count += 1

    process_csv(indoor_csv, os.path.join(root_dir, "TacQuad", "tacquad", "data_indoor"))
    process_csv(outdoor_csv, os.path.join(root_dir, "TacQuad", "tacquad", "data_outdoor"))

    return samples

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(args.ckpt, device)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print(f"\n Inference started → {args.output_csv}")

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "File_Name", "GT_Label", "Model_Output"])

        for ds_name in ["hct", "ssvtp", "tacquad_digit"]:

            print(f"\n {ds_name.upper()}")

            if ds_name == "tacquad_digit":
                samples = load_tacquad_digit_samples(
                    args.dataset_root,
                    args.num_samples
                )
            else:
                samples = load_dataset_samples(
                    args.dataset_root,
                    ds_name,
                    args.num_samples
                )

            print(f"   -> {len(samples)} samples")

            if len(samples) == 0:
                continue

            for idx, item in enumerate(samples):

                vis_path = item["vision"]
                tac_path = item["tactile"]
                label = item["label"]

                file_name = os.path.basename(vis_path)

                image = Image.open(vis_path).convert("RGB")
                pixel_values = vision_processor(image).unsqueeze(0).to(
                    device, dtype=torch.bfloat16
                )

                tactile_vals, tactile_grid = load_tactile_data(
                    tac_path,
                    augment=False
                )

                if tactile_vals is None:
                    print(f"   [Skip] Missing tactile: {file_name}")
                    continue

                tactile_tensor = tactile_vals.unsqueeze(0).to(
                    device, dtype=torch.bfloat16
                )
                tactile_grid = tactile_grid.unsqueeze(0).to(device)

                tg = tactile_grid[0].tolist()
                num_tac = int(tg[-2] * tg[-1]) if len(tg) >= 2 else 0
                num_tac = min(num_tac, 1024)

                user_text = (
                    "<image>\n"
                    "<tactile>\n"
                    "List 2–3 tactile attributes of the object shown in the image.\n"
                    "Use single-word descriptors only.\n"
                    "Output a comma-separated list with no additional text."
                )

                num_img_tokens = 256
                if "<image>" in user_text:
                    img_tokens = " ".join(["<IMG_CONTEXT>"] * num_img_tokens)
                    user_text = user_text.replace("<image>", img_tokens, 1)

                tac_start = "<tac>"
                tac_end = "</tac>"
                tac_ctx = "<TAC_CONTEXT>"

                if num_tac > 0:
                    tac_tokens_str = (
                        tac_start + " " +
                        " ".join([tac_ctx] * num_tac) +
                        " " + tac_end
                    )
                    user_text = user_text.replace("<tactile>", tac_tokens_str, 1)
                else:
                    user_text = user_text.replace("<tactile>\n", "").replace("<tactile>", "")

                chat_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(
                    chat_text,
                    return_tensors="pt"
                ).to(device)

                print("num tactile tokens:", (inputs["input_ids"] == tokenizer.convert_tokens_to_ids("<TAC_CONTEXT>")).sum().item())
                print("num image tokens:", (inputs["input_ids"] == tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")).sum().item())


                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        pixel_values=pixel_values,
                        pixel_values_tactile=tactile_tensor,
                        max_new_tokens=64,
                        do_sample=False
                    )

                response = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                ).strip()

                print("outputs:", outputs)
                print("outputs.shape:", outputs.shape if hasattr(outputs, "shape") else "N/A")
                print("inputs['input_ids'].shape:", inputs["input_ids"].shape)

                writer.writerow([
                    ds_name,
                    file_name,
                    label,
                    response
                ])
                                
    print("\n Inference complete.")


if __name__ == "__main__":
    main()