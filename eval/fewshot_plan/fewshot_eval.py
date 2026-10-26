#!/usr/bin/env python3
"""
fewshot_eval.py — Few-shot plan-then-code baseline (base model, no RL), all 8 evals.

Runs Qwen2.5-Coder-3B-Instruct with the 3-shot plan-then-code prompt from
experiments/fewshot-plan/fewshot_prompts.py on exactly the same tasks, tests and
metrics as the Optional Plan eval: it reuses the eval functions in
eval/optional_plan/vllm_eval.py unchanged, and only swaps the prompt.

Usage (from this folder):
  python fewshot_eval.py --dataset mbpp      --data ../../experiments/optional-plan/data/mbpp_test_optional_plan.parquet      --outdir logs
  python fewshot_eval.py --dataset humaneval --data ../../experiments/optional-plan/data/humaneval_plan_optional_plan.parquet --outdir logs
"""

import os, sys, json, argparse

import pandas as pd
from vllm import LLM
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments", "fewshot-plan"))
sys.path.insert(0, os.path.join(HERE, "..", "optional_plan"))
import vllm_eval as V            # existing Optional Plan eval, used read-only
import fewshot_prompts as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--dataset",    required=True, choices=["mbpp", "humaneval"])
    ap.add_argument("--data",       required=True,
                    help="Optional Plan test parquet (source of tasks and tests)")
    ap.add_argument("--limit",      type=int, default=0)
    ap.add_argument("--outdir",     default="logs")
    ap.add_argument("--n-samples",  type=int, default=16)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--gpu-mem",    type=float, default=0.9)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_parquet(a.data)
    if a.limit: df = df.iloc[:a.limit]
    n = len(df)

    is_mbpp = (a.dataset == "mbpp")
    mod = V.M if is_mbpp else V.H
    maxtok = 1024 if is_mbpp else 2048

    # Same tasks/tests as the Optional Plan eval; prompt replaced by the few-shot one
    # (MBPP: task + first test; HumanEval+: function signature + docstring).
    metas = []
    for i in range(n):
        task_id, task_prompt, tc, tl, _ = mod.row_meta(df.iloc[i], i)
        user = F.mbpp_user(task_prompt, tl[0] if tl else "") if is_mbpp else F.he_user(task_prompt)
        metas.append((task_id, task_prompt, tc, tl, user))

    prefix = F.shot_messages(is_mbpp)
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    def fmt(p):
        return tok.apply_chat_template(prefix + [{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)

    print(f"Loading model {a.checkpoint} (few-shot, seed={a.seed}) ...", flush=True)
    llm = LLM(model=a.checkpoint, dtype="float16",
              gpu_memory_utilization=a.gpu_mem, max_model_len=4096,
              enforce_eager=True, seed=a.seed)

    print(f"\n=== {a.dataset.upper()} few-shot eval (n={n}) ===", flush=True)
    print("\n--- pass@1 ---", flush=True)
    pass1_entries = V.eval_pass1(llm, fmt, metas, df, is_mbpp, a.outdir, maxtok)
    print("\n--- pass@5 ---", flush=True)
    V.eval_pass5(llm, fmt, metas, df, is_mbpp, a.outdir, maxtok, n_samples=a.n_samples)
    print("\n--- +feedback (no-judge) ---", flush=True)
    V.eval_feedback_nojudge(llm, fmt, metas, df, is_mbpp, a.outdir, maxtok, pass1_entries)
    print("\n--- +feedback (judge) ---", flush=True)
    V.eval_feedback_judge(llm, fmt, metas, df, is_mbpp, a.outdir, maxtok, pass1_entries)

    ds = a.dataset
    def _load(name):
        with open(os.path.join(a.outdir, f"{ds}_{name}", f"summary_{ds}_{name}.json")) as f:
            return json.load(f)[0]
    p1, p5 = _load("pass1"), _load("pass5")
    nj, jd = _load("feedback_nojudge"), _load("feedback_judge")
    summary = {"checkpoint": a.checkpoint, "dataset": ds, "prompt_style": "fewshot-3",
               "seed": a.seed, "n": n, "pass@1": p1["pass_at_1"], "pass@5": p5["pass_at_5"],
               "feedback_nojudge_pass@1": nj["pass_at_1_retry"],
               "feedback_judge_pass@1": jd["pass_at_1_retry"],
               "has_plan_rate": p1["has_plan_rate"], "avg_plan_score": jd["avg_plan_score"]}
    with open(os.path.join(a.outdir, f"summary_{ds}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY: " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
