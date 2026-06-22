import os
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoProcessor
from models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from util.data_utils import (
    load_vision_image,
    load_tactile_data,
    inject_tactile_tokens
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dataset_root", required=True)

    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=["hct", "ssvtp", "tacquad_digit"]
    )

    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steady_start", type=int, default=50)

    return parser.parse_args()


def get_first_sample(dataset_root, dataset_name):

    root = dataset_root

    if dataset_name == "hct":

        csv_path = os.path.join(
            root,
            "tvl_dataset",
            "hct",
            "data1",
            "test.csv"
        )

        base = os.path.dirname(csv_path)

        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)

        vis = os.path.join(base, row[0])
        tac = os.path.join(base, row[1])

        return vis, tac

    raise ValueError("Dataset not supported for latency benchmark")


if __name__ == "__main__":

    args = parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    torch.backends.cudnn.benchmark = True


    print(" Loading model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        local_files_only=True
    ).to(DEVICE)

    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    processor.tokenizer = tokenizer

    processor.image_processor.min_pixels = 224 * 224
    processor.image_processor.max_pixels = 224 * 224

    print(" Model loaded.")


    vision_path, tactile_path = get_first_sample(
        args.dataset_root,
        args.dataset_name
    )

    print(f" Vision sample : {vision_path}")
    print(f" Tactile sample: {tactile_path}")


    vision_image = load_vision_image(vision_path)

    tactile_vals, tactile_grid = load_tactile_data(
        tactile_path,
        augment=False
    )

    tactile_tensor = tactile_vals.unsqueeze(0).to(
        DEVICE,
        dtype=DTYPE
    )

    tactile_grid = tactile_grid.unsqueeze(0).to(DEVICE)

    prompt = (
        "List 2–3 tactile attributes of the object shown in the image.\n"
        "Use single-word descriptors only.\n"
        "Output a comma-separated list with no additional text."
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
        min_new_tokens=6,
        do_sample=False
    )


    print(f" Warmup ({args.warmup} iterations)...")

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model.generate(**generate_kwargs)

    torch.cuda.synchronize()


    print(f"⏱ Measuring ({args.runs} iterations)...")

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

    steady = latencies[args.steady_start:]

    mean_latency = steady.mean()
    median_latency = np.median(steady)
    p90_latency = np.percentile(steady, 90)
    p99_latency = np.percentile(steady, 99)
    std_latency = steady.std()

    hz = 1000 / mean_latency


    print("\n==========================================")
    print(" SPLASH-3B End-to-End Latency (Batch=1)")
    print("==========================================")

    print(f"Dataset         : {args.dataset_name}")
    print(f"Warmup          : {args.warmup}")
    print(f"Total Runs      : {args.runs}")
    print(f"Discard First   : {args.steady_start}")

    print("------------------------------------------")

    print(f" Mean Latency : {mean_latency:.2f} ms")
    print(f" Median       : {median_latency:.2f} ms")
    print(f" P90 Latency  : {p90_latency:.2f} ms")
    print(f" P99 Latency  : {p99_latency:.2f} ms")
    print(f" Std Dev      : {std_latency:.2f} ms")
    print(f" Control Hz   : {hz:.2f}")

    print("==========================================\n")


    print("Min   :", steady.min())
    print("Max   :", steady.max())
    print("Mean  :", steady.mean())
    print("Median:", np.median(steady))
    print("Std   :", steady.std())


