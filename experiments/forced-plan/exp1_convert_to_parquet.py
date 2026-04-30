#!/usr/bin/env python3
"""
exp1_convert_to_parquet.py  —  Experiment 1: All 3 test cases + explicit function name
Changes vs baseline:
  - Shows all 3 test cases in prompt (not just 1)
  - Explicitly states function name and signature extracted from test cases
  - Stronger plan specificity instructions in system prompt
  - Train split: 15% random sample (fixed seed=42) for fast experimentation
  - Test split:  20% random sample (fixed seed=42) for fast eval
  - Output files prefixed exp1_* to avoid overwriting baseline data
"""
import os
import re
import json
import random
import pandas as pd
from datasets import load_dataset

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 42
TRAIN_FRAC = 1.0   # 100% of 374 = 374 tasks
TEST_FRAC  = 1.0   # 100% of 500 = 500 tasks

# ── Prompt templates ──────────────────────────────────────────────────────────

FORCED_PLAN_SYSTEM = (
    "You are an expert Python programmer. Before writing code, you MUST produce "
    "a structured plan inside <plan>...</plan> tags. Then write your solution "
    "inside <code>...</code> tags.\n\n"
    "Your plan MUST include:\n"
    "1. The exact algorithm you will use (be specific, not just 'iterate')\n"
    "2. How you handle each edge case visible in the test cases\n"
    "3. The expected output for at least one test case based on your approach\n\n"
    "Your code MUST:\n"
    "- Use EXACTLY the function name shown in the test cases\n"
    "- Implement every step defined in your plan, no deviations\n"
    "- Handle all edge cases identified in your plan"
)

FORCED_PLAN_USER = """\
Task: {task_prompt}

Function name: {func_name}
Signature: {signature}

Test cases (your solution must pass ALL of these):
{all_tests}

Respond in this exact format:

<plan>
Your structured plan here (must cover algorithm, edge cases, and expected outputs)...
</plan>

<code>
Your Python code here (must use function name '{func_name}' and follow the plan above)...
</code>
"""


def extract_func_name_and_sig(test_list):
    """
    Pull the function name and a best-guess signature from the first assert.
    e.g. 'assert remove_Occ("hello","l") == "heo"'
      -> func_name = 'remove_Occ'
      -> signature = 'remove_Occ(s, c)'  (arg names are placeholders)
    """
    func_name = "solution"
    signature = "solution(...)"

    if not test_list:
        return func_name, signature

    first = test_list[0]
    # Match: assert func_name(args) == ...
    m = re.match(r'assert\s+(\w+)\s*\((.*?)\)\s*==', first)
    if not m:
        # Try set(...) == set(...) pattern
        m = re.match(r'assert\s+\w+\((\w+)\s*\(', first)
        if m:
            func_name = m.group(1)
            signature = f"{func_name}(...)"
            return func_name, signature
        return func_name, signature

    func_name = m.group(1)
    raw_args  = m.group(2).strip()

    # Count positional args to generate placeholder names
    # Simple split on commas not inside brackets
    depth, args, cur = 0, [], ""
    for ch in raw_args:
        if ch in "([{": depth += 1
        elif ch in ")]}": depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())

    # Generic arg names: a, b, c, ... or n if single int-looking arg
    arg_names = []
    for i, _ in enumerate(args):
        arg_names.append(chr(ord('a') + i))
    signature = f"{func_name}({', '.join(arg_names)})"
    return func_name, signature


def build_row(item, data_source):
    task_id      = str(item.get("task_id", ""))
    text         = item.get("text", item.get("prompt", ""))
    test_list    = item.get("test_list", [])
    test_imports = item.get("test_imports", [])

    func_name, signature = extract_func_name_and_sig(test_list)
    all_tests = "\n".join(test_list) if test_list else ""

    return {
        "prompt": [
            {"role": "system", "content": FORCED_PLAN_SYSTEM},
            {"role": "user",   "content": FORCED_PLAN_USER.format(
                task_prompt=text,
                func_name=func_name,
                signature=signature,
                all_tests=all_tests)},
        ],
        "data_source": data_source,
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(test_list),
        },
        "extra_info": {
            "task_id":        task_id,
            "task_prompt_raw": text,
            "test_list":      json.dumps(test_list),
            "test_imports":   json.dumps(test_imports),
            "func_name":      func_name,
            "signature":      signature,
        },
    }


def subsample(items, frac, seed):
    rng = random.Random(seed)
    k = max(1, round(len(items) * frac))
    return rng.sample(items, k)


def save(rows, path, name):
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    prompt_lens = [len(str(r["prompt"])) // 4 for r in rows]
    print(f"  ✓ {name:30s} → {path}  ({len(df)} tasks)")
    print(f"    Prompt tokens approx: min={min(prompt_lens)} | "
          f"median={sorted(prompt_lens)[len(prompt_lens)//2]} | max={max(prompt_lens)}")


print("=" * 60)
print("Experiment 1: All-tests + function name prompt")
print("Loading MBPP full from HuggingFace...")
print("=" * 60)

ds = load_dataset("google-research-datasets/mbpp", "full")

# ── Train: 15% subsample ──────────────────────────────────────────────────────
train_items = list(ds["train"])
train_sub   = subsample(train_items, TRAIN_FRAC, SEED)
train_rows  = [build_row(item, "mbpp_full") for item in train_sub]
save(train_rows, os.path.join(DATA_DIR, "exp1_full_train.parquet"), "train (100%)")

# ── Val: full (keep same as baseline for comparability) ──────────────────────
val_rows = [build_row(item, "mbpp_full") for item in ds["validation"]]
save(val_rows, os.path.join(DATA_DIR, "exp1_full_val.parquet"), "validation (100%)")

# ── Test: 20% subsample, same seed ───────────────────────────────────────────
test_items = list(ds["test"])
test_sub   = subsample(test_items, TEST_FRAC, SEED)
test_rows  = [build_row(item, "mbpp_full") for item in test_sub]
save(test_rows, os.path.join(DATA_DIR, "exp1_full_test.parquet"), "test (100%)")

print("\n" + "=" * 60)
print(f"DONE  seed={SEED}  train={len(train_rows)}  val={len(val_rows)}  test={len(test_sub)}")
print("\nParquet files in data/:")
for fname in sorted(os.listdir(DATA_DIR)):
    if fname.startswith("exp1") and fname.endswith(".parquet"):
        size = os.path.getsize(os.path.join(DATA_DIR, fname))
        print(f"  {fname:40s} {size/1024:.1f} KB")

print("\nUsage:")
print("  python exp1_run_training.sh")
print("  python exp1_evaluate_upd2.py   --checkpoint <ckpt> --split test")
print("  python exp1_evaluate_feedback.py --checkpoint <ckpt> --split test")
