#!/usr/bin/env python3
"""
plan_rescore.py — Re-score plan quality in saved judge-feedback logs (no GPU).

The eval's plan scorer (score_plan in eval/optional_plan/vllm_eval.py) asks GPT-4o-mini
for JSON, but GPT-4o-mini wraps its answer in a ```json fence, so json.loads fails and
the scorer falls back to 0.5 for every dimension. The pass@1 / feedback metrics are not
affected (hints come from a separate call), but every plan_score in the logs is 0.5.

This script re-scores every plan with the SAME judge prompt and model (gpt-4o-mini,
temperature 0, same rubric), parsing the JSON robustly, and rewrites in place, in
{ds}_feedback_judge/{ds}_feedback_judge.jsonl(.gz):
  plan_score, plan_dims, reward, feedback_attempts[*].plan_score/plan_dims/reward,
  final_plan_score, final_plan_dims, final_reward
and avg_plan_score in summary_{ds}_feedback_judge.json and summary_{ds}.json.
Reward follows the eval: 0.5 * plan_score + 1.0 if the tests pass, else 0.
A judge call that still fails is stored as null (not 0.5) and counted.

Usage:
  python eval/plan_rescore.py --logs eval/optional_plan/logs
"""

import os, re, json, gzip, argparse
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

DIMS = ["decomposition", "edge_cases", "approach", "clarity"]
EMPTY = {d: 0.0 for d in DIMS}
client = OpenAI()


def judge_prompt(task_desc, plan_text):   # identical to score_plan() in vllm_eval.py
    return ('Rate this programming plan on 4 dimensions (0.0-1.0 each).\n'
            'Respond ONLY with JSON: {"decomposition":0.0,"edge_cases":0.0,"approach":0.0,"clarity":0.0}\n\n'
            f'Task: {task_desc}\nPlan: {plan_text}')


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    dims = json.loads(m.group(0))
    return {d: max(0.0, min(1.0, float(dims[d]))) for d in DIMS}


def score(task_desc, plan_text):
    """Returns (score, dims); (0.0, zeros) for an empty plan; (None, None) on failure."""
    if not plan_text or not plan_text.strip():
        return 0.0, dict(EMPTY)
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini", temperature=0.0, max_tokens=100,
                messages=[{"role": "user", "content": judge_prompt(task_desc, plan_text)}])
            dims = parse(resp.choices[0].message.content)
            return round(sum(dims.values()) / 4, 3), dims
        except Exception:
            continue
    return None, None


def reward(ps, passed):
    if not passed:
        return 0.0
    return None if ps is None else 0.5 * ps + 1.0


def read(path):
    if os.path.exists(path):
        return [json.loads(l) for l in open(path, encoding="utf-8")], False
    with gzip.open(path + ".gz", "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f], True


def write(path, rows, gz):
    data = "".join(json.dumps(r) + "\n" for r in rows)
    if gz:
        with gzip.open(path + ".gz", "wt", encoding="utf-8", compresslevel=9) as f:
            f.write(data)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)


def update_summary(path, avg):
    if not os.path.exists(path):
        return
    s = json.load(open(path))
    if isinstance(s, list):
        s[0]["avg_plan_score"] = avg
    else:
        s["avg_plan_score"] = avg
    json.dump(s, open(path, "w"), indent=2)


def rescore(logs, ds):
    path = f"{logs}/{ds}_feedback_judge/{ds}_feedback_judge.jsonl"
    rows, gz = read(path)

    jobs = {}
    for r in rows:
        jobs[(r["task_prompt"], r["plan"])] = None
        for fa in r["feedback_attempts"]:
            jobs[(r["task_prompt"], fa["plan"])] = None
    keys = list(jobs)
    with ThreadPoolExecutor(max_workers=16) as ex:
        for k, v in zip(keys, ex.map(lambda k: score(*k), keys)):
            jobs[k] = v

    for r in rows:
        ps, dims = jobs[(r["task_prompt"], r["plan"])]
        r["plan_score"], r["plan_dims"] = ps, dims
        r["reward"] = reward(ps, r["test_pass"])
        for fa in r["feedback_attempts"]:
            fps, fdims = jobs[(r["task_prompt"], fa["plan"])]
            fa["plan_score"], fa["plan_dims"], fa["reward"] = fps, fdims, reward(fps, fa["test_pass"])
        if r["test_pass"] or not r["feedback_attempts"]:
            r["final_plan_score"], r["final_plan_dims"] = ps, dims
        else:
            last = r["feedback_attempts"][-1]
            r["final_plan_score"], r["final_plan_dims"] = last["plan_score"], last["plan_dims"]
        r["final_reward"] = reward(r["final_plan_score"], r["final_test_pass"])
    write(path, rows, gz)

    planned = [r["plan_score"] for r in rows if r["plan"] and r["plan_score"] is not None]
    avg = round(sum(planned) / max(len(planned), 1), 4)
    failed = sum(1 for v in jobs.values() if v[0] is None)
    update_summary(f"{logs}/{ds}_feedback_judge/summary_{ds}_feedback_judge.json", avg)
    update_summary(f"{logs}/summary_{ds}.json", avg)
    return {"avg_plan_score": avg, "plans_scored": len(jobs), "judge_failures": failed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="logs dir in the vllm_eval layout")
    a = ap.parse_args()
    print(json.dumps({ds: rescore(a.logs, ds) for ds in ["mbpp", "humaneval"]}))


if __name__ == "__main__":
    main()
