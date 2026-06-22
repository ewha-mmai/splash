import pandas as pd
import argparse
import os
import re
import json
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.util.eval_utils import (
    get_gpt_evaluator,
    get_gpt5_evaluator,
    get_llama3_evaluator,
    get_vicuna_evaluator,
)

load_dotenv()

GPT4_EVAL_MODEL = "gpt-4o"
GPT5_EVAL_MODEL = "gpt-5-mini"

GPT_EVAL_PROMPT = """[User Question]: {prompt}\n\n
[Assistant Response]: {assistant_response}\n
[Correct Response]: {correct_response}\n\n
We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. 
The user asks the question on observing an image. The assistant's response is followed by the correct response.
\nPlease evaluate the assistant's response based on how closely it matches the correct response which describes tactile feelings. Please compare only the semantics of the answers. DO NOT consider grammatical errors in scoring the assistant. The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only one value indicating the score for the assistant. \nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.\n\n
"""


LLAMA_EVAL_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"

LLAMA_SYSTEM_PROMPT = """You are an impartial judge evaluating an AI assistant's performance on a "Tactile Description" task.

We would like to request your feedback on the performance of an AI assistant in response to the user question.
The user asks the question on observing an image. The assistant's response is followed by the correct response.

Please evaluate the assistant's response based on how closely it matches the correct response which describes tactile feelings. 
Please compare only the semantics of the answers. DO NOT consider grammatical errors in scoring the assistant.
**Contradictions are fatal errors.** (e.g., If Correct Response says "Uneven/Rough" and Assistant says "Smooth", the score must be 1).
The assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.

- **1-2:** Completely wrong, contradicts GT, or purely visual words (e.g., "Blue, Blurry").
- **3-4:** Mostly visual, but captures one weak tactile trait (e.g., "Metallic" implying hard).
- **5-6:** Mixed performance. Captured correct tactile trait but included significant visual noise.
- **7-8:** Good tactile description, accurate to GT, minimal visual noise.
- **9-10:** Perfect match with GT, purely tactile.

Please first output a single line containing only one value indicating the score for the assistant.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.
"""

LLAMA_USER_PROMPT = """[User Question]: {prompt}

[Assistant Response]: {assistant_response}

[Correct Response]: {correct_response}
"""


VICUNA_EVAL_MODEL = "lmsys/vicuna-33b-v1.3"

VICUNA_EVAL_PROMPT = """### System Description:
You are an impartial and strict AI judge. Your task is to evaluate the quality of an AI assistant's response by comparing it to the Correct Response.

[User Question]: {prompt}

[Correct Response (Ground Truth)]: {correct_response}

[Assistant Response (Generated)]: {assistant_response}

1. **Compare Semantics**: Analyze whether the tactile feelings (adjectives) in the [Assistant Response] match the [Correct Response].
2. **Fact Check**: If the Assistant predicts an adjective that contradicts the Correct Response (e.g., "hard" vs "soft"), mark it as incorrect. Do not hallucinate matches that are not there.
3. **Ignore Grammar**: Focus only on the meaning of the words, not grammatical errors.
4. **Scoring**: Assign a score from 1 to 10.
   - 10: Perfect match in meaning.
   - 1: Completely incorrect or unrelated.

You must output the score in the first line, followed by the explanation in the next lines.

Score: <Single number between 1 and 10>

Explanation: <Your reasoning here>
"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--judge_type", type=str, default="llama",
                        choices=["llama", "vicuna", "gpt4", "gpt5"])
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=8, 
                        help="Number of parallel workers for GPT API")

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f" CSV file not found: {args.csv_path}")
        return

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} samples from {args.csv_path}")

    if args.judge_type == "gpt4":
        judge_model_name = GPT4_EVAL_MODEL
    elif args.judge_type == "gpt5":
        judge_model_name = GPT5_EVAL_MODEL
    elif args.judge_type == "llama":
        judge_model_name = "llama"
    elif args.judge_type == "vicuna":
        judge_model_name = "vicuna"
    else:
        judge_model_name = args.judge_type

    if args.output_json is None:
        base_name = os.path.splitext(args.csv_path)[0]
        safe_model_name = judge_model_name.replace("/", "_")
        args.output_json = f"{base_name}_eval_{safe_model_name}.json"

    print(f"Results will be saved to: {args.output_json}")
    print(f" Judge Model: {judge_model_name}")

    eval_data = []
    processed_files = set()

    if os.path.exists(args.output_json):
        print(f" Resume mode: loading existing results from {args.output_json}")
        with open(args.output_json, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        for d in eval_data:
            if "file_name" in d:
                processed_files.add(d["file_name"])

        print(f"Already evaluated samples: {len(processed_files)}")
    else:
        print(" Fresh run: no existing evaluation file found.")

    pending_items = []
    for idx, row in df.iterrows():
        file_name = row["File_Name"]
        if file_name in processed_files:
            continue
        if pd.isna(row["GT_Label"]) or pd.isna(row["Model_Output"]):
            continue
        pending_items.append((idx, row))

    if not pending_items:
        print(" All samples are already evaluated!")
    else:
        print(f" Found {len(pending_items)} pending samples to evaluate.")
        user_prompt = "This image gives tactile feelings of?"

        if args.judge_type in ["gpt4", "gpt5"]:
            print(f"\n Running GPT API in PARALLEL (Workers: {args.max_workers})...")
            
            if args.judge_type == "gpt4":
                judge_fn = get_gpt_evaluator(GPT4_EVAL_MODEL, GPT_EVAL_PROMPT)
            elif args.judge_type == "gpt5":
                judge_fn = get_gpt5_evaluator(GPT5_EVAL_MODEL, GPT_EVAL_PROMPT)

            def evaluate_single_gpt(item):
                idx, row = item
                file_name = row["File_Name"]
                gt_label = str(row["GT_Label"])
                model_pred = str(row["Model_Output"])
                dataset_name = str(row["Dataset"]).lower().strip()

                try:
                    eval_output = judge_fn(
                        prompt=user_prompt,
                        assistant_response=model_pred,
                        correct_response=gt_label
                    )
                except Exception as e:
                    eval_output = f"Error: {str(e)}"

                return idx, file_name, dataset_name, gt_label, model_pred, eval_output

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(evaluate_single_gpt, item) for item in pending_items]
                
                for future in tqdm(as_completed(futures), total=len(pending_items), desc="GPT Eval"):
                    idx, file_name, dataset_name, gt_label, model_pred, eval_output = future.result()

                    eval_data.append({
                        "dataset": dataset_name,
                        "gt_label": gt_label,
                        "model_pred": model_pred,
                        "evaluation": eval_output,
                        "file_name": file_name,
                    })

                    with open(args.output_json, "w", encoding="utf-8") as f:
                        json.dump(eval_data, f, indent=4, ensure_ascii=False)

        else:
            if args.judge_type == "llama":
                judge_fn = get_llama3_evaluator(LLAMA_EVAL_MODEL, LLAMA_SYSTEM_PROMPT, LLAMA_USER_PROMPT)
            elif args.judge_type == "vicuna":
                judge_fn = get_vicuna_evaluator(VICUNA_EVAL_MODEL, VICUNA_EVAL_PROMPT)

            print(f"\n Running Local Model sequentially...")
            for idx, row in tqdm(pending_items, total=len(pending_items), desc=f"{args.judge_type.capitalize()} Eval"):
                file_name = row["File_Name"]
                gt_label = str(row["GT_Label"])
                model_pred = str(row["Model_Output"])
                dataset_name = str(row["Dataset"]).lower().strip()

                eval_output = judge_fn(
                    prompt=user_prompt, assistant_response=model_pred, correct_response=gt_label
                )

                print("\n" + "-" * 60)
                print(f"Sample [{idx}/{len(df)}] - {dataset_name}")
                print(f" GT  : {gt_label}")
                print(f" Pred: {model_pred}")
                print(f" Eval: {eval_output.strip()[:100]}...")

                eval_data.append({
                    "dataset": dataset_name,
                    "gt_label": gt_label,
                    "model_pred": model_pred,
                    "evaluation": eval_output,
                    "file_name": file_name,
                })

                with open(args.output_json, "w", encoding="utf-8") as f:
                    json.dump(eval_data, f, indent=4, ensure_ascii=False)


    print("\nCalculating Statistics...")

    dataset_scores = {}
    total_scores = []

    parse_fail_cnt = 0

    for d in eval_data:
        evaluation = d["evaluation"]
        dataset = d.get("dataset", "").lower().strip()

        score = None

        m = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)", evaluation, re.IGNORECASE)
        if m:
            score = float(m.group(1))
        else:
            first_line = evaluation.splitlines()[0].strip()
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", first_line):
                score = float(first_line)

        if score is None:
            parse_fail_cnt += 1
            continue

        total_scores.append(score)

        if dataset not in dataset_scores:
            dataset_scores[dataset] = []
        dataset_scores[dataset].append(score)

    print("\n" + "=" * 50)
    print(" Dataset-wise Results")
    print("=" * 50)

    macro_components = []

    for dataset, scores in dataset_scores.items():
        avg = sum(scores) / len(scores)
        macro_components.append(avg)

        print(
            f"{dataset.upper():10s} | "
            f"Samples: {len(scores):4d} | "
            f"Average: {avg:.3f}"
        )

    print("-" * 50)

    if len(total_scores) > 0:
        micro_avg = sum(total_scores) / len(total_scores)
        print(f"Micro Average (sample-weighted): {micro_avg:.3f}")
    else:
        print("Micro Average: No valid scores")

    if len(macro_components) > 0:
        macro_avg = sum(macro_components) / len(macro_components)
        print(f"Macro Average (benchmark-wise): {macro_avg:.3f}")
    else:
        print("Macro Average: No valid datasets")

    print("-" * 50)
    print(f"Total Valid Samples: {len(total_scores)}")
    print(f"Parse Failures: {parse_fail_cnt}")
    print("=" * 50)


if __name__ == "__main__":
    main()
