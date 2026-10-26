#!/usr/bin/env python3
"""
annotate_plans.py — Build the plan-annotated SFT dataset (SFT-only baseline, no RL).

For each of the 374 MBPP-full train tasks (the same tasks the GRPO runs train on):
  - prompt   = the Optional Plan prompt (experiments/optional-plan/convert_to_parquet.py)
  - response = "<plan>{plan}</plan>\n\n<code>{reference solution}</code>"
The plan is written by GPT-4o-mini from the task + MBPP reference solution; the code is
the MBPP reference solution. Every response is executed against all of the task's tests
and dropped if it fails.

Output: data/sft_train.jsonl  (one {"task_id", "prompt", "response"} per line)

Usage (from this folder):  python annotate_plans.py
"""

import os, re, sys, json
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "optional-plan"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "eval", "optional_plan"))
from convert_to_parquet import MBPP_PROMPT   # same prompt the GRPO model trained with
from runfix import mbpp_run                  # same test runner as the eval

PLAN_PROMPT = """Write a short step-by-step plan (3-6 numbered steps) for solving this Python task.
The plan must describe the approach the reference solution takes, including any edge cases it handles.
Do NOT write code. Output only the numbered steps.

Task: {task}

Reference solution:
{code}"""

client = OpenAI()


def make_plan(task, code):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PLAN_PROMPT.format(task=task, code=code)}],
        max_tokens=300, temperature=0.0,
    )
    plan = resp.choices[0].message.content.strip()
    return re.sub(r"^```\w*\s*|\s*```$", "", plan).strip()


def build(item):
    task, tests = item["text"], item["test_list"]
    code = item["code"].replace("\r\n", "\n").strip()
    ok, _ = mbpp_run(code, tests)
    if not ok:
        return None, "reference solution fails its tests"
    plan = make_plan(task, code)
    response = f"<plan>\n{plan}\n</plan>\n\n<code>\n{code}\n</code>"
    prompt = MBPP_PROMPT.format(task_description=task, example_test=tests[0] if tests else "")
    return {"task_id": item["task_id"], "prompt": prompt, "response": response}, None


def main():
    ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(build, ds))

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    out = os.path.join(HERE, "data", "sft_train.jsonl")
    kept = [r for r, _ in results if r]
    with open(out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    dropped = [(ds[i]["task_id"], why) for i, (r, why) in enumerate(results) if not r]
    print(f"kept {len(kept)}/{len(ds)} -> {out}")
    for tid, why in dropped:
        print(f"  dropped task {tid}: {why}")


if __name__ == "__main__":
    main()
