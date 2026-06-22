import os
import sys
import time
import csv
import argparse
import numpy as np
import torch

from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel
from models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from util.data_utils import load_vision_image, load_tactile_data, inject_tactile_tokens


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_mode", required=True,
                        choices=["pretrain", "finetune"])
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--pretrain_ckpt", required=True)
    parser.add_argument("--finetune_ckpt", required=False)
    parser.add_argument("--dataset_root", required=True)

    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steady_start", type=int, default=50)

    return parser.parse_args()


def load_model(args):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrain_ckpt,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True
    ).to(DEVICE)

    if args.model_mode == "finetune" and args.finetune_ckpt:
        model = PeftModel.from_pretrained(model, args.finetune_ckpt)
        model = model.merge_and_unload()

    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrain_ckpt,
        trust_remote_code=True
    )

    processor.tokenizer = tokenizer

    processor.image_processor.min_pixels = 224 * 224
    processor.image_processor.max_pixels = 224 * 224

    return model, processor, tokenizer


def get_sample(dataset_root):

    csv_path = os.path.join(
        dataset_root,
        "tvl_dataset",
        "hct",
        "data1",
        "test.csv"
    )

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        row = next(reader)

    base = os.path.dirname(csv_path)
    vision_path = os.path.join(base, row[0])
    tactile_path = os.path.join(base, row[1])

    return vision_path, tactile_path


if __name__ == "__main__":

    args = parse_args()

    global DEVICE, DTYPE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    torch.backends.cudnn.benchmark = True

    model, processor, tokenizer = load_model(args)

    vision_path, tactile_path = get_sample(args.dataset_root)

    vision_image = load_vision_image(vision_path)
    tactile_vals, tactile_grid = load_tactile_data(tactile_path, augment=False)

    tactile_tensor = tactile_vals.unsqueeze(0).to(DEVICE, dtype=DTYPE)
    tactile_grid = tactile_grid.unsqueeze(0).to(DEVICE)

    prompt = (
        "List 2–3 tactile attributes of the object shown in the image."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]

    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        images=[vision_image],
        text=[text_input],
        return_tensors="pt"
    ).to(DEVICE)

    input_ids = inputs.input_ids[0]
    attn_mask = inputs.attention_mask[0]

    final_ids, final_mask = inject_tactile_tokens(
        input_ids,
        attn_mask,
        tokenizer
    )

    final_ids = final_ids.unsqueeze(0)
    final_mask = final_mask.unsqueeze(0)

    generate_kwargs = dict(
        input_ids=final_ids,
        attention_mask=final_mask,
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        pixel_values_tactile=tactile_tensor,
        tactile_grid_thw=tactile_grid,
        max_new_tokens=6,
        do_sample=False
    )

    print(f" Warmup ({args.warmup} iterations)...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model.generate(**generate_kwargs)

    torch.cuda.synchronize()

    print(f"⏱ Measuring ({args.runs} iterations)...")

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    latencies = []
    
    with torch.inference_mode():
        for _ in range(args.runs):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            
            starter.record()
            _ = model.generate(**generate_kwargs)
            ender.record()
            
            latencies.append((starter, ender))

    torch.cuda.synchronize()

    final_latencies = []
    for s, e in latencies:
        final_latencies.append(s.elapsed_time(e))
    
    latencies = np.array(final_latencies)

    steady_latencies = latencies[args.steady_start:]

    mean_latency = steady_latencies.mean()
    median_latency = np.median(steady_latencies)
    p90_latency = np.percentile(steady_latencies, 90)
    p99_latency = np.percentile(steady_latencies, 99)

    hz = 1000 / mean_latency

    print("\n==========================================")
    print(" Steady-State End-to-End Latency (Batch=1)")
    print("==========================================")
    print(f"Total Runs       : {args.runs}")
    print(f"Discard First    : {args.steady_start}")
    print("------------------------------------------")
    print(f" Mean Latency  : {mean_latency:.2f} ms")
    print(f" Median        : {median_latency:.2f} ms")
    print(f" P90 Latency   : {p90_latency:.2f} ms")
    print(f" P99 Latency   : {p99_latency:.2f} ms")
    print(f" Control Hz    : {hz:.2f} Hz")
    print("==========================================\n")


import matplotlib.pyplot as plt

print("Min :", steady_latencies.min())
print("Max :", steady_latencies.max())
print("Mean :", steady_latencies.mean())
print("Median:", np.median(steady_latencies))
print("Std :", steady_latencies.std())

plt.figure(figsize=(6,4))
plt.hist(steady_latencies, bins=30)
plt.title("Latency Distribution (Steady-State)")
plt.xlabel("Latency (ms)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("latency_hist.png")
print(" Histogram saved to latency_hist.png")