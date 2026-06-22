import os
import sys
import csv
import argparse
from typing import Optional
import torch
import ImageBind.data as data
import llama


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--llama_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=None)

    return parser.parse_args()


def load_model(device, llama_dir):

    print("Loading UniTouch model...")

    model = llama.load(
        "7B",
        llama_dir,
        llama_type="7B",
        knn=False
    )

    model.to(device)
    model.eval()

    return model


def load_dataset_samples(root_dir: str, ds_name: str, num_samples: Optional[int] = None):

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
                samples.append((vis, tac, row[1]))

    elif ds_name == "hct":
        subdirs = ["data1", "data2", "data3"]
        count = 0
        for sub in subdirs:
            csv_path = os.path.join(root_dir, "tvl_dataset", "hct", sub, "test.csv")
            folder = os.path.dirname(csv_path)

            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if num_samples and count >= num_samples:
                        return samples

                    vis = os.path.join(folder, row[0])
                    tac = os.path.join(folder, row[1])
                    samples.append((vis, tac, row[3]))
                    count += 1

    return samples


def load_tacquad_digit(root_dir: str, num_samples=None):

    samples = []
    indoor_csv = os.path.join(root_dir, "TacQuad", "tacquad", "contact_indoor.csv")
    outdoor_csv = os.path.join(root_dir, "TacQuad", "tacquad", "contact_outdoor.csv")

    count = 0

    def process(csv_path, base_dir):
        nonlocal count
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if num_samples and count >= num_samples:
                    return

                item = row[0]
                mid = (int(row[3]) + int(row[4])) // 2

                vis = os.path.join(base_dir, item, "img_digit", f"{mid}.png")
                tac = os.path.join(base_dir, item, "digit", f"{mid}.png")

                if os.path.exists(vis) and os.path.exists(tac):
                    samples.append((vis, tac, row[-1]))
                    count += 1

    process(indoor_csv, os.path.join(root_dir, "TacQuad", "tacquad", "data_indoor"))
    process(outdoor_csv, os.path.join(root_dir, "TacQuad", "tacquad", "data_outdoor"))

    return samples


def run_inference(model, samples, device, writer, dataset_name, dataset_root):

    prompt_text = (
        "List 2–3 tactile attributes of the object shown in the image.\n"
        "Use single-word descriptors only.\n"
        "Output a comma-separated list with no additional text."
    )

    formatted_prompt = llama.format_prompt(prompt_text)

    for idx, (vis_path, tac_path, label) in enumerate(samples):

        vision = data.load_and_transform_vision_data(
            [vis_path],
            device=device
        )

        tactile = data.load_and_transform_vision_data(
            [tac_path],
            device=device
        )

        inputs = {
            "Image": [vision, 1],
            "Touch": [tactile, 1],
        }

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                [formatted_prompt],
                max_gen_len=32
            )

        relative_path = os.path.relpath(vis_path, dataset_root)

        writer.writerow([
            dataset_name,
            relative_path,
            label,
            output_text
        ])

        if (idx + 1) % 10 == 0:
            print(f"{dataset_name}: {idx+1}/{len(samples)} done")


def main():

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(device, args.llama_dir)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "File_Name", "GT_Label", "Model_Output"])

        for ds_name in ["hct", "ssvtp", "tacquad_digit"]:

            print(f"\nRunning {ds_name}...")

            if ds_name == "tacquad_digit":
                samples = load_tacquad_digit(args.dataset_root, args.num_samples)
            else:
                samples = load_dataset_samples(args.dataset_root, ds_name, args.num_samples)

            run_inference(model, samples, device, writer, ds_name, args.dataset_root)

    print("All inference done.")


if __name__ == "__main__":
    main()