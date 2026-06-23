import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_path(path: object) -> str:
    if path is None:
        return ""
    return str(path).strip().lstrip("/")


def is_hct_style(path: str) -> bool:
    first_token = normalize_path(path).split("/")[0]
    return "-" in first_token


def extract_object_id(path: str) -> str:
    return normalize_path(path).split("/")[0]


def read_first_data_path(csv_path: str) -> Optional[str]:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        first_row = next(reader, None)
    if not first_row:
        return None
    return first_row[0]


def split_hct_csv(csv_path: str, output_dir: str, train_ratio: float, seed: int) -> Tuple[str, str]:
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    header, data_rows = rows[0], rows[1:]
    object_to_rows = defaultdict(list)

    for row in data_rows:
        if not row:
            continue
        object_to_rows[extract_object_id(row[0])].append(row)

    object_ids = list(object_to_rows.keys())
    rng = random.Random(seed)
    rng.shuffle(object_ids)

    split_idx = int(len(object_ids) * train_ratio)
    train_objects = set(object_ids[:split_idx])
    val_objects = set(object_ids[split_idx:])

    train_rows = [row for obj in train_objects for row in object_to_rows[obj]]
    val_rows = [row for obj in val_objects for row in object_to_rows[obj]]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")

    write_csv(train_path, header, train_rows)
    write_csv(val_path, header, val_rows)

    print(f"[HCT] {csv_path}")
    print(f"  objects: {len(object_ids)}")
    print(f"  train rows: {len(train_rows)} -> {train_path}")
    print(f"  val rows:   {len(val_rows)} -> {val_path}")

    return train_path, val_path


def split_ssvtp_csv(csv_path: str, output_dir: str, val_ratio: float, seed: int) -> Tuple[str, str]:
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_split.csv")
    val_path = os.path.join(output_dir, "val_split.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"[SSVTP] {csv_path}")
    print(f"  total rows: {len(df)}")
    print(f"  train rows: {len(train_df)} -> {train_path}")
    print(f"  val rows:   {len(val_df)} -> {val_path}")

    return train_path, val_path


def write_csv(path: str, header: List[str], rows: Iterable[List[str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def split_csv(csv_path: str, output_dir: str, train_ratio: float, val_ratio: float, seed: int) -> Tuple[str, str]:
    first_path = read_first_data_path(csv_path)
    if first_path is None:
        raise ValueError(f"No data rows found in {csv_path}")

    if is_hct_style(first_path):
        return split_hct_csv(csv_path, output_dir, train_ratio, seed)
    return split_ssvtp_csv(csv_path, output_dir, val_ratio, seed)


def load_split_paths(csv_path: str) -> set:
    df = pd.read_csv(csv_path)
    if "url" not in df.columns:
        raise ValueError(f"'url' column not found in {csv_path}")

    paths = {normalize_path(x) for x in df["url"].tolist()}
    paths.update(os.path.basename(path) for path in paths)
    return paths


def split_finetune_json(finetune_json_path: str, train_csv_path: str, val_csv_path: str) -> Tuple[str, str]:
    train_images = load_split_paths(train_csv_path)
    val_images = load_split_paths(val_csv_path)

    with open(finetune_json_path) as f:
        data = json.load(f)

    train_data = []
    val_data = []
    unmatched = []

    for item in data:
        image_path = normalize_path(item.get("image", ""))
        image_name = os.path.basename(image_path)

        if image_path in train_images or image_name in train_images:
            train_data.append(item)
        elif image_path in val_images or image_name in val_images:
            val_data.append(item)
        else:
            unmatched.append(image_path)

    base_dir = os.path.dirname(finetune_json_path)
    train_out = os.path.join(base_dir, "finetune_train.json")
    val_out = os.path.join(base_dir, "finetune_val.json")

    with open(train_out, "w") as f:
        json.dump(train_data, f, indent=2)

    with open(val_out, "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"[JSON] {finetune_json_path}")
    print(f"  original: {len(data)}")
    print(f"  train:    {len(train_data)} -> {train_out}")
    print(f"  val:      {len(val_data)} -> {val_out}")
    print(f"  unmatched: {len(unmatched)}")
    if unmatched:
        print(f"  unmatched examples: {unmatched[:5]}")

    return train_out, val_out


def candidate_dataset_dirs(dataset_root: str) -> List[str]:
    dirs = []

    for root, _, files in os.walk(dataset_root):
        if "train.csv" in files:
            dirs.append(root)

    return sorted(dirs)


def process_dataset_dir(dataset_dir: str, train_ratio: float, val_ratio: float, seed: int) -> None:
    train_csv = os.path.join(dataset_dir, "train.csv")
    finetune_json = os.path.join(dataset_dir, "finetune.json")

    if not os.path.exists(train_csv):
        print(f"[Skip] train.csv not found: {dataset_dir}")
        return

    train_split, val_split = split_csv(train_csv, dataset_dir, train_ratio, val_ratio, seed)

    if os.path.exists(finetune_json):
        split_finetune_json(finetune_json, train_split, val_split)
    else:
        print(f"[Skip] finetune.json not found: {dataset_dir}")


def process_dataset_root(dataset_root: str, train_ratio: float, val_ratio: float, seed: int) -> None:
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    dataset_dirs = candidate_dataset_dirs(dataset_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No train.csv found under {dataset_root}")

    for dataset_dir in dataset_dirs:
        process_dataset_dir(dataset_dir, train_ratio, val_ratio, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/val CSV splits and finetune JSON files for SPLASH datasets."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Dataset root to scan, e.g. dataset/tvl_dataset/hct or dataset/tvl_dataset/ssvtp.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Object-level train ratio for HCT-style datasets.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Random validation ratio for SSVTP-style datasets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset_root(args.dataset_root, args.train_ratio, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
