import pandas as pd
import argparse
import os
import re
from collections import defaultdict


TACTILE_VOCAB = {

    "smooth",
    "rough",
    "coarse",
    "grainy",
    "bumpy",
    "textured",
    "woven",
    "fibrous",
    "lined",
    "ridged",
    "grooved",
    "wrinkled",
    "porous",
    "abrasive",
    "fuzzy",
    "slippery",
    "slick",
    "waxy",
    "mesh",
    "dimpled",
    "gritty",

    "matte",
    "glossy",
    "reflective",
    "polished",
    "sleek",
    "shiny",

    "hard",
    "soft",
    "firm",
    "rigid",
    "pliable",
    "flexible",
    "elastic",
    "yielding",
    "compressible",
    "unyielding",
    "solid",
    "deformable",
    "cushioned",
    "plush",

    "flat",
    "uneven",
    "round",
    "thread-like",
    "non-porous",
    "thin",
    "thick",

    "cool",
    "warm",
    "cold",

    "dry",
    "moist",
    "flaky",
    "brittle",
    "ticklish",
    "lightweight",

    "metallic",
    "fabric",
}


OPPOSITE_MAP = {

    "soft": "hard",
    "hard": "soft",

    "smooth": "rough",
    "rough": "smooth",

    "rigid": "flexible",
    "flexible": "rigid",

    "glossy": "matte",
    "matte": "glossy",

    "slippery": "rough",
    "rough": "slippery",

    "firm": "yielding",
    "yielding": "firm",
}


def normalize_text(text):

    text = str(text).lower()

    text = text.replace("/", " ")

    text = re.sub(r"[^a-z0-9\-\s,]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_comma_keywords(text):

    text = normalize_text(text)

    keywords = [
        x.strip()
        for x in text.split(",")
    ]

    keywords = [
        k for k in keywords
        if len(k) > 0
    ]

    return list(dict.fromkeys(keywords))


def extract_keywords(text):

    text = normalize_text(text)

    text = re.sub(r"\b\d+\.", " ", text)

    chunks = re.split(r"[,\n]", text)

    keywords = []

    for c in chunks:

        c = c.strip()

        if len(c) == 0:
            continue


        words = c.split()

        for w in words:

            w = w.strip()

            if w in TACTILE_VOCAB:
                keywords.append(w)

    return list(dict.fromkeys(keywords))


def extract_tactile_keywords(text):

    text = normalize_text(text)

    words = text.replace(",", " ").split()

    keywords = []

    for w in words:

        if w in TACTILE_VOCAB:
            keywords.append(w)

    return list(dict.fromkeys(keywords))


def compute_prf(
    gt_keywords,
    pred_keywords,
    use_contradiction_penalty=False,
    contradiction_penalty_weight=0.5,
):

    gt_set = set(gt_keywords)

    pred_set = set(pred_keywords)

    intersection = gt_set & pred_set


    precision = (
        len(intersection) / len(pred_set)
        if len(pred_set) > 0 else 0.0
    )

    recall = (
        len(intersection) / len(gt_set)
        if len(gt_set) > 0 else 0.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)


    contradiction_detected = False

    if use_contradiction_penalty:

        for p in pred_set:

            if p in OPPOSITE_MAP:

                opposite = OPPOSITE_MAP[p]

                if opposite in gt_set:
                    contradiction_detected = True
                    break

        if contradiction_detected:
            f1 *= contradiction_penalty_weight

    return precision, recall, f1, contradiction_detected


def compute_topk_accuracy(
    gt_keywords,
    pred_keywords,
    k=5
):

    pred_topk = pred_keywords[:k]

    hit = any([p in gt_keywords for p in pred_topk])

    return 1 if hit else 0


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=None
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5
    )

    parser.add_argument(
        "--use_contradiction_penalty",
        action="store_true"
    )

    parser.add_argument(
        "--contradiction_penalty_weight",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    args = parser.parse_args()


    if not os.path.exists(args.csv_path):

        print(f" CSV not found: {args.csv_path}")

        return

    df = pd.read_csv(args.csv_path)

    print(f"Loaded {len(df)} samples")


    dataset_stats = defaultdict(list)

    total_precision = []
    total_recall = []
    total_f1 = []
    total_topk = []

    contradiction_count = 0


    for idx, row in df.iterrows():

        if pd.isna(row["GT_Label"]) or pd.isna(row["Model_Output"]):
            continue

        dataset = str(row["Dataset"]).lower().strip()

        gt_text = str(row["GT_Label"])

        pred_text = str(row["Model_Output"])


        if dataset in ["hct", "ssvtp"]:


            gt_keywords = extract_keywords(gt_text)

            pred_keywords = extract_keywords(pred_text)

        else:

            gt_keywords = extract_tactile_keywords(gt_text)

            pred_keywords = extract_keywords(pred_text)


        precision, recall, f1, contradiction_detected = compute_prf(
            gt_keywords,
            pred_keywords,
            use_contradiction_penalty=args.use_contradiction_penalty,
            contradiction_penalty_weight=args.contradiction_penalty_weight,
        )

        topk_acc = compute_topk_accuracy(
            gt_keywords,
            pred_keywords,
            k=args.topk
        )


        if contradiction_detected:
            contradiction_count += 1

        dataset_stats[dataset].append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "topk": topk_acc
        })

        total_precision.append(precision)

        total_recall.append(recall)

        total_f1.append(f1)

        total_topk.append(topk_acc)


        if args.verbose:

            print("\n" + "=" * 80)

            print(f"[{idx}] DATASET: {dataset}")

            print(f"GT TEXT   : {gt_text}")

            print(f"PRED TEXT : {pred_text}")

            print(f"GT KW     : {gt_keywords}")

            print(f"PRED KW   : {pred_keywords}")

            print(
                f"P={precision:.4f} "
                f"R={recall:.4f} "
                f"F1={f1:.4f} "
                f"Top-{args.topk}={topk_acc}"
            )

            if contradiction_detected:
                print(" Contradiction detected")


    print("\n" + "=" * 80)

    print("Dataset-wise Objective Metrics")

    print("=" * 80)

    macro_f1_components = []

    dataset_result_dict = {}

    for dataset, scores in dataset_stats.items():

        avg_precision = (
            sum([x["precision"] for x in scores]) / len(scores)
        )

        avg_recall = (
            sum([x["recall"] for x in scores]) / len(scores)
        )

        avg_f1 = (
            sum([x["f1"] for x in scores]) / len(scores)
        )

        avg_topk = (
            sum([x["topk"] for x in scores]) / len(scores)
        )

        macro_f1_components.append(avg_f1)

        dataset_result_dict[dataset] = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "topk": avg_topk,
            "samples": len(scores),
        }

        print(
            f"{dataset.upper():15s} | "
            f"P: {avg_precision:.4f} | "
            f"R: {avg_recall:.4f} | "
            f"F1: {avg_f1:.4f} | "
            f"Top-{args.topk}: {avg_topk:.4f}"
        )


    micro_precision = sum(total_precision) / len(total_precision)

    micro_recall = sum(total_recall) / len(total_recall)

    micro_f1 = sum(total_f1) / len(total_f1)

    micro_topk = sum(total_topk) / len(total_topk)

    macro_f1 = (
        sum(macro_f1_components) / len(macro_f1_components)
    )

    print("\n" + "=" * 80)

    print("Overall Results")

    print("=" * 80)

    print(f"Micro Precision : {micro_precision:.4f}")

    print(f"Micro Recall    : {micro_recall:.4f}")

    print(f"Micro F1        : {micro_f1:.4f}")

    print(f"Micro Top-{args.topk} Acc : {micro_topk:.4f}")

    print("-" * 80)

    print(f"Macro F1        : {macro_f1:.4f}")

    print("-" * 80)

    print(f"Contradictions  : {contradiction_count}")

    print("=" * 80)


    results = {

        "overall_results": {

            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "micro_topk": micro_topk,
            "macro_f1": macro_f1,
            "contradictions": contradiction_count,
        },

        "dataset_results": dataset_result_dict
    }

    if args.output_json is not None:

        with open(args.output_json, "w") as f:

            import json

            json.dump(results, f, indent=4)

        print(f"\n Saved results to: {args.output_json}")


if __name__ == "__main__":

    main()