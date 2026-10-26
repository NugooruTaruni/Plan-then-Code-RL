#!/usr/bin/env python3
"""
fence_rescore.py — Fence-aware re-scoring of saved eval logs (no GPU, no model calls).

The eval extracts code between <code> and </code>. When a model opens <code> but closes
its code with a Markdown fence (```) instead of </code>, everything after <code> —
including the prose explanation after the fence — is taken as code, which then fails
with a SyntaxError even if the code itself is correct. The base model under few-shot
prompting does this on ~9% of HumanEval+ samples; the trained models almost never do.

This script re-tests ONLY those answers (<code> present, </code> missing), cutting the
code at the closing fence. Every other answer keeps its original result, so the only
change vs. the strict scores is the rescue of correct-but-misformatted code. The same
rule is applied to every model.

Reads a logs dir in the vllm_eval layout (plain .jsonl or .jsonl.gz) and writes
summary_fence_aware.json next to the existing summaries (strict and fence-aware numbers).

Usage:
  python eval/fence_rescore.py --logs eval/optional_plan/logs
"""

import os, re, sys, json, gzip, math, argparse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "optional_plan"))
from code_cleaner import clean_code      # same cleaner as the eval
from runfix import mbpp_run, he_run      # same test runners as the eval


def read_jsonl(path):
    if os.path.exists(path):
        return [json.loads(l) for l in open(path, encoding="utf-8")]
    with gzip.open(path + ".gz", "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def unclosed(raw):
    return "<code>" in raw and "</code>" not in raw


def fence_code(raw):
    body = raw.split("<code>", 1)[1]
    body = re.sub(r"^\s*```[\w+-]*[ \t]*\n", "", body)   # optional opening fence
    return body.split("```", 1)[0].strip()


def make_tester(ds):
    def test(raw, test_list):
        code = clean_code(fence_code(raw), test_list=test_list)
        ok, _ = mbpp_run(code, test_list) if ds == "mbpp" else he_run(code, test_list[0])
        return bool(ok)
    return test


def rescue(items, tester, workers=40):
    """items: list of (raw_output, test_list, original_pass) -> fence-aware pass list."""
    idx = [i for i, (raw, _, ok) in enumerate(items) if not ok and unclosed(raw)]
    out = [bool(ok) for _, _, ok in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, ok in zip(idx, ex.map(lambda i: tester(items[i][0], items[i][1]), idx)):
            out[i] = ok
    return out, len(idx), sum(out) - sum(bool(ok) for _, _, ok in items)


def passk(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def rescore(logs, ds):
    t = make_tester(ds)
    res = {}

    p1 = read_jsonl(f"{logs}/{ds}_pass1/{ds}_pass1.jsonl")
    ok, n_unc, n_res = rescue([(r["raw_output"], r["test_list"], r["test_pass"]) for r in p1], t)
    res["pass@1"] = {"strict": round(sum(r["test_pass"] for r in p1) / len(p1), 4),
                     "fence_aware": round(sum(ok) / len(p1), 4), "unclosed": n_unc, "rescued": n_res}

    p5 = read_jsonl(f"{logs}/{ds}_pass5/{ds}_pass5.jsonl")
    ok5, n_unc, n_res = rescue([(r["raw_output"], r["test_list"], r["test_pass"]) for r in p5], t)
    by_task, strict_by_task = {}, {}
    for r, o in zip(p5, ok5):
        by_task.setdefault(r["task_id"], []).append(o)
        strict_by_task.setdefault(r["task_id"], []).append(bool(r["test_pass"]))
    pk = lambda d: round(sum(passk(len(v), sum(v), 5) for v in d.values()) / len(d), 4)
    res["pass@5"] = {"strict": pk(strict_by_task), "fence_aware": pk(by_task),
                     "unclosed": n_unc, "rescued": n_res}

    # Feedback: final pass = first attempt (fence-aware) or retry (fence-aware).
    nj = read_jsonl(f"{logs}/{ds}_feedback_nojudge/{ds}_feedback_nojudge.jsonl")
    first, _, _ = rescue([(r["first_attempt"]["raw_output"], r["test_list"],
                           r["first_attempt"]["test_pass"]) for r in nj], t)
    retr_items = [(r["retry"]["raw_output"], r["test_list"], r["retry"]["test_pass"])
                  if r["retry"] else ("", r["test_list"], 0) for r in nj]
    retry, _, _ = rescue(retr_items, t)
    final = [f or (r["retry"] is not None and rr) for f, rr, r in zip(first, retry, nj)]
    res["feedback_nojudge"] = {"strict": round(sum(r["final_pass"] for r in nj) / len(nj), 4),
                               "fence_aware": round(sum(final) / len(nj), 4)}

    jd = read_jsonl(f"{logs}/{ds}_feedback_judge/{ds}_feedback_judge.jsonl")
    first, _, _ = rescue([(r["raw_output"], r["test_list"], r["test_pass"]) for r in jd], t)
    retr_items = [(r["feedback_attempts"][0]["raw_output"], r["test_list"],
                   r["feedback_attempts"][0]["test_pass"]) if r["feedback_attempts"]
                  else ("", r["test_list"], 0) for r in jd]
    retry, _, _ = rescue(retr_items, t)
    final = [f or (bool(r["feedback_attempts"]) and rr) for f, rr, r in zip(first, retry, jd)]
    res["feedback_judge"] = {"strict": round(sum(r["final_test_pass"] for r in jd) / len(jd), 4),
                             "fence_aware": round(sum(final) / len(jd), 4)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="logs dir in the vllm_eval layout")
    a = ap.parse_args()
    out = {ds: rescore(a.logs, ds) for ds in ["mbpp", "humaneval"]}
    with open(os.path.join(a.logs, "summary_fence_aware.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
