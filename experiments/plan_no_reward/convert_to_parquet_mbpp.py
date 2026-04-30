"""
Plan-then-Code: Convert MBPP JSON to Parquet for VERL
=======================================================
Includes all fields VERL needs:
  - prompt: chat-formatted messages
  - reward_model: dict with style and ground_truth (test_list)
  - data_source: dataset name
  - extra_info: task_prompt_raw, test_list, task_id

Usage:
    python convert_to_parquet.py
"""

import json
import os
import pandas as pd

DATA_DIR = os.path.join(os.getcwd(), "data")

OPTIONAL_PLAN_SYSTEM = (
    "You are an expert Python programmer. "
    "You may optionally write a plan before writing code if you find it helpful, "
    "but you are not required to do so."
)

OPTIONAL_PLAN_USER = """Solve the following Python programming task.

You may optionally write a plan inside <plan> tags before writing your code.
Planning is not required — only your code will be evaluated.

If you choose to plan, analyze:
- Problem decomposition: break the task into logical steps
- Edge cases: identify tricky inputs or boundary conditions
- Approach: state your algorithm or strategy

Then, write your Python code inside <code> tags.

Task: {task_prompt}

Example test case:
{example_test}

You may respond in either format:

Format 1 (with optional plan):
<plan>
Your structured plan here... (optional)
</plan>

<code>
Your Python code here...
</code>

Format 2 (code only):
<code>
Your Python code here...
</code>"""


def convert_mbpp_split(json_file: str, parquet_file: str, use_forced_plan: bool = True):
    """Convert an MBPP JSON split to VERL-compatible parquet."""
    with open(json_file) as f:
        tasks = json.load(f)

    print(f"  Loading {json_file}: {len(tasks)} tasks")

    rows = []
    for task in tasks:
        task_id = task.get("task_id", "")
        task_prompt = task.get("prompt", task.get("text", ""))
        test_list = task.get("test_list", [])
        test_imports = task.get("test_imports", [])

        # ─── Build chat-formatted prompt ───
        if use_forced_plan:
            # Include first test case so model knows the expected function name
            example_test = test_list[0] if test_list else ""
            prompt = [
                {"role": "system", "content": OPTIONAL_PLAN_SYSTEM},
                {"role": "user", "content": OPTIONAL_PLAN_USER.format(
                    task_prompt=task_prompt,
                    example_test=example_test,
                )},
            ]
        else:
            prompt = [
                {"role": "user", "content": task_prompt},
            ]

        rows.append({
            # Required by VERL
            "prompt": prompt,
            "data_source": "mbpp_full",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(test_list),
            },
            # Extra info passed to reward function
            "extra_info": {
                "task_id": task_id,
                "task_prompt_raw": task_prompt,
                "test_list": json.dumps(test_list),
                "test_imports": json.dumps(test_imports),
            },
        })

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_file, index=False)
    print(f"  Saved {len(df)} tasks to {parquet_file}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("CONVERTING MBPP JSON → PARQUET FOR VERL")
    print("=" * 60)

    splits = {
        "mbpp_train_full.json": "mbpp_train_full.parquet",
        "mbpp_val_full.json": "mbpp_val_full.parquet",
        "mbpp_test_full.json": "mbpp_test_full.parquet",
    }

    for json_name, parquet_name in splits.items():
        json_path = os.path.join(DATA_DIR, json_name)
        parquet_path = os.path.join(DATA_DIR, parquet_name)

        if os.path.exists(json_path):
            convert_mbpp_split(json_path, parquet_path, use_forced_plan=True)
        else:
            print(f"  SKIP: {json_path} not found")

    print("\n" + "=" * 60)
    print("DONE. Parquet files in data/:")
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith(".parquet"):
            size = os.path.getsize(os.path.join(DATA_DIR, f))
            print(f"  {f:40s} {size/1024:.1f} KB")
