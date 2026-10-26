#!/usr/bin/env python3
"""
convert_to_parquet.py - MBPP-full + HumanEval parquet files (optional-plan prompt).
Generates:
  data/mbpp_train_optional_plan.parquet
  data/mbpp_test_optional_plan.parquet
  data/mbpp_val_optional_plan.parquet
  data/humaneval_plan.parquet
"""

import json, os, argparse
import pandas as pd
from datasets import load_dataset


# -- Prompt templates --------------------------------
MBPP_PROMPT = """Solve the following Python programming task.

First, decide whether to write a step-by-step plan.
If you do, write your plan inside <plan></plan> tags.
Then write your solution inside <code></code> tags.
Your code MUST follow your plan if you wrote one.
Your function must use the EXACT name shown in the test case.

Example (with plan):
Task: Write a function to check if a number is prime.
Test: assert is_prime(7) == True

Response:
<plan>
1. Handle edge cases: numbers <= 1 are not prime
2. Check divisibility from 2 to sqrt(n)
3. Return True if no divisors found
</plan>

<code>
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
</code>

Now solve this task:
Task: {task_description}

Test: {example_test}

Your response:"""

HE_PROMPT = """Complete the following Python function.

First, decide whether to write a plan.
If you do, write it inside <plan></plan> tags.
Then write your solution inside <code></code> tags.
Your code MUST follow your plan if you wrote one.

Example:
def multiply(a, b):
    # Multiply two numbers

<plan>
1. Take two parameters
2. Return their product
</plan>

<code>
def multiply(a, b):
    return a * b
</code>

Now complete this:
{prompt_text}

Your response:"""


# -- Helpers -----------------------------------------
def make_mbpp_records(ds, split):
    records, meta = [], []
    for item in ds:
        task_desc = item["text"]
        test_list = item.get("test_list", [])
        gt = json.dumps({
            "task_description": task_desc,
            "test_list": test_list,
            "test_code": "\n".join(test_list)
        })
        records.append({
            "prompt": [{"role": "user", "content": MBPP_PROMPT.format(
                task_description=task_desc,
                example_test=test_list[0] if test_list else "")}],
            # verl / trainer metrics use this as the key, e.g. val-aux/mbpp/reward/... (HF config=full, not mbpp_sanitized)
            "data_source": "mbpp",
            "reward_model": {"ground_truth": gt, "style": "rule"},
            "extra_info": {
                "task_description": task_desc,
                "test_list": test_list,
                "test_code": "\n".join(test_list)
            }
        })
        meta.append({
            "task_id": item.get("task_id", 0),
            "task_description": task_desc,
            "test_code": "\n".join(test_list),
            "test_list": test_list
        })
    tag = "val" if split == "validation" else split
    return records, meta, tag


def save_parquet(records, prefix, tag):
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(records)
    path = f"data/{prefix}_{tag}_optional_plan.parquet"
    df.to_parquet(path, index=False)
    print(f"✅ {path} -> {len(df)} tasks")
    return df


def save_metadata(meta, prefix, tag):
    path = f"data/{prefix}_{tag}_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# -- Main conversion ---------------------------------
def convert_mbpp(split="train"):
    """Generate MBPP-full parquet for the given split (or all if 'train')."""
    if split == "train":
        for s in ["train", "test", "validation"]:
            ds = load_dataset("google-research-datasets/mbpp", "full", split=s)
            recs, meta, tag = make_mbpp_records(ds, s)
            save_parquet(recs, "mbpp", tag)
            save_metadata(meta, "mbpp", tag)
    else:
        ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
        recs, meta, tag = make_mbpp_records(ds, split)
        save_parquet(recs, "mbpp", tag)
        save_metadata(meta, "mbpp", tag)


def convert_humaneval():
    """Generate HumanEval parquet."""
    ds = load_dataset("evalplus/humanevalplus", split="test")
    records = []
    for item in ds:
        test_code = f"{item['test']}\ncheck({item['entry_point']})"
        gt = json.dumps({
            "task_id": item["task_id"],
            "test_code": test_code,
            "test_list": [test_code]
        })
        records.append({
            "prompt": [{"role": "user", "content": HE_PROMPT.format(prompt_text=item["prompt"])}],
            "data_source": "humaneval",
            "reward_model": {"ground_truth": gt, "style": "rule"},
            "extra_info": {
                "task_id": item["task_id"],
                "task_description": item["prompt"],
                "test_code": test_code,
                "test_list": [test_code]
            }
        })
    save_parquet(records, "humaneval", "plan")
    print("✅ HumanEval conversion complete!")


# -- CLI ---------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mbpp", action="store_true", default=True, help="Convert MBPP-full (default)")
    p.add_argument("--humaneval", action="store_true", help="Convert HumanEval")
    p.add_argument("--split", default="train", choices=["train", "test", "validation"])
    args = p.parse_args()

    if args.humaneval:
        convert_humaneval()
    else:
        convert_mbpp(args.split)
        # By default, when split is 'train', we also create test/val
