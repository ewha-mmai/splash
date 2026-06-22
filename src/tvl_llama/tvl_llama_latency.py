import os
import sys
import argparse
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

tvl_inner_dir = os.path.join(root_dir, "tvl")
if tvl_inner_dir not in sys.path:
    sys.path.insert(0, tvl_inner_dir)

from tvl.tvl_llama.llama.llama_adapter import load
from tvl.tvl_enc import tacvis
from tvl.tvl_llama import llama

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--llama_dir", required=True)

    parser.add_argument("--llama_type", default="llama-2-7b")
    parser.add_argument("--tactile_model", default="vit_tiny_patch16_224")

    parser.add_argument("--dataset_root", required=True)

    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steady_start", type=int, default=50)

    parser.add_argument(
        "--active_modality_names",
        nargs="+",
        default=["vision", "tactile"]
    )

    return parser.parse_args()


def load_model_pipeline(args, device):

    model = load(
        name=args.model_path,
        llama_dir=args.llama_dir,
        device=device,
        phase="finetune",
        args=args
    )

    model.to(device)
    model.eval()

    return model


def load_one_sample(dataset_root, device):

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

    folder = os.path.dirname(csv_path)

    vis_path = os.path.join(folder, row[0])
    tac_path = os.path.join(folder, row[1])

    vision = tacvis.load_vision_data(
        vis_path,
        device=device,
        dataset_version="v2"
    ).unsqueeze(0)

    tactile = tacvis.load_tactile_data(
        tac_path,
        device=device
    ).unsqueeze(0)

    return vision, tactile


def main():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = True

    model = load_model_pipeline(args, device)

    vision, tactile = load_one_sample(
        args.dataset_root,
        device
    )

    prompt_text = (
        "List 2–3 tactile attributes of the object shown in the image.\n"
        "Use single-word descriptors only.\n"
        "Output a comma-separated list with no additional text."
    )

    formatted_prompt = llama.format_prompt(prompt_text)

    inputs = {
        "vision": [vision, 1],
        "tactile": [tactile, 1],
    }

    generate_kwargs = dict(
        max_gen_len=6
    )

    print(f" Warmup ({args.warmup} iterations)...")

    with torch.inference_mode():
        for _ in range(args.warmup):

            _ = model.generate(
                inputs,
                [formatted_prompt],
                **generate_kwargs
            )

    torch.cuda.synchronize()

    print(f"⏱ Measuring ({args.runs} iterations)...")

    latencies = []

    with torch.inference_mode():

        for _ in range(args.runs):

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            starter.record()

            _ = model.generate(
                inputs,
                [formatted_prompt],
                **generate_kwargs
            )

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
    std_latency = steady_latencies.std()

    hz = 1000 / mean_latency

    print("\n==========================================")
    print(" TVL-LLAMA Steady-State Latency (Batch=1)")
    print("==========================================")
    print(f"Total Runs       : {args.runs}")
    print(f"Discard First    : {args.steady_start}")
    print("------------------------------------------")
    print(f" Mean Latency  : {mean_latency:.2f} ms")
    print(f" Median        : {median_latency:.2f} ms")
    print(f" P90 Latency   : {p90_latency:.2f} ms")
    print(f" P99 Latency   : {p99_latency:.2f} ms")
    print(f" Std           : {std_latency:.2f} ms")
    print(f" Control Hz    : {hz:.2f}")
    print("==========================================\n")

    print("Min   :", steady_latencies.min())
    print("Max   :", steady_latencies.max())
    print("Mean  :", steady_latencies.mean())
    print("Median:", np.median(steady_latencies))
    print("Std   :", steady_latencies.std())

    plt.figure(figsize=(6, 4))

    plt.hist(steady_latencies, bins=30)

    plt.title("TVL-LLAMA Latency Distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig("tvl_llama_latency_hist.png")

    print(" Histogram saved to tvl_llama_latency_hist.png")


if __name__ == "__main__":
    main()