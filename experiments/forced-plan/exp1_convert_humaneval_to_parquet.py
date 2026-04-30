#!/usr/bin/env python3
"""
convert_humaneval_to_parquet.py — Convert HumanEval+ to parquet for exp1 eval scripts.

Usage:
  python convert_humaneval_to_parquet.py

Output:
  data/humaneval_plus_test.parquet
"""
from datasets import load_dataset
import pandas as pd
import json
import os

os.makedirs("data", exist_ok=True)

SYSTEM_PROMPT = """You are an expert Python programmer. Before writing code, you MUST produce a structured plan inside <plan>...</plan> tags. Then write your solution inside <code>...</code> tags.

Your plan MUST include:
1. The exact algorithm or approach you will use
2. How you handle each edge case
3. The expected output for at least one test case

Then implement your solution."""


def build_row(task):
    task_id     = task['task_id']
    prompt      = task['prompt']       # function signature + docstring
    entry_point = task['entry_point']
    test_code   = task['test']         # full test function (check(candidate))

    user_msg = f"""Task: Complete the following Python function.

Function signature and docstring:
{prompt}

Function name: {entry_point}

Respond in this exact format:

<plan>
Your structured plan here...
</plan>

<code>
Your complete Python code here (implement the function '{entry_point}')...
</code>"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    extra_info = {
        "task_id":         task_id,
        "task_prompt_raw": prompt,
        "func_name":       entry_point,
        "test_code":       test_code,      # full check(candidate) function
        "test_list":       json.dumps([]), # empty — eval script uses test_code
    }

    return {
        "prompt":     messages,
        "extra_info": extra_info,
    }


def main():
    print("Loading HumanEval+ from HuggingFace...")
    ds = load_dataset('evalplus/humanevalplus', split='test')
    print(f"Loaded {len(ds)} tasks")

    rows = [build_row(task) for task in ds]
    df   = pd.DataFrame(rows)

    out = "data/humaneval_plus_test.parquet"
    df.to_parquet(out, index=False)
    print(f"\n✓ Saved {len(df)} tasks to {out}")

    # Sanity check
    sample = df.iloc[0]
    ei = sample['extra_info']
    print(f"\nSample task:")
    print(f"  task_id   : {ei['task_id']}")
    print(f"  func_name : {ei['func_name']}")
    print(f"  test_code : {ei['test_code'][:150]}...")


if __name__ == "__main__":
    main()