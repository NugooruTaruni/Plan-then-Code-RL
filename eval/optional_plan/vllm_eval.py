#!/usr/bin/env python3
"""
vllm_eval.py — Fast vLLM evaluation for optional-plan model (all 8 evals).

Writes per-task JSONL + summary JSON to match the original run_all_mbpp.py /
run_all_humaneval.py format, so logs can be dropped directly into the repo's
eval/optional_plan/logs/ directories.

Usage:
  python vllm_eval.py --checkpoint /path/to/model --dataset mbpp     --data /path/mbpp_test.parquet   --outdir /workspace/eval_out
  python vllm_eval.py --checkpoint /path/to/model --dataset humaneval --data /path/he_test.parquet     --outdir /workspace/eval_out
"""

import sys, os, json, time, math, argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from code_cleaner import clean_code
import run_all_mbpp  as M
import run_all_humaneval as H
import runfix
# Override test runners with temp-file versions (no arg-too-long bug)
M.run_tests = runfix.mbpp_run
H.run_test  = runfix.he_run

FENCE = "```"
TEST_WORKERS = 40


# ── sampling helpers ──────────────────────────────────────────────────────────

def greedy(llm, prompts, max_tokens):
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=max_tokens,
                        stop=["</code>"], include_stop_str_in_output=True)
    return llm.generate(prompts, sp)


def sample_n(llm, prompts, n, temperature, top_p, max_tokens):
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens,
                        stop=["</code>"], include_stop_str_in_output=True)
    return llm.generate(prompts, sp)


def passk_unbiased(n, c, k):
    """Chen et al. 2021 unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def judge_feedback(task_desc, error_msg, plan_text):
    try:
        from openai import OpenAI
        resp = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                f"The following Python code failed.\n"
                f"Task: {task_desc}\nError: {error_msg}\nPlan: {plan_text or 'None'}\n\n"
                f"Give a short hint (1-2 sentences) to fix it. Do NOT write code."
            )}],
            max_tokens=100, temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Debug the error carefully."


def score_plan(task_desc, plan_text):
    empty = {"decomposition": 0.0, "edge_cases": 0.0, "approach": 0.0, "clarity": 0.0}
    if not plan_text or not plan_text.strip():
        return 0.0, empty
    try:
        from openai import OpenAI
        resp = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                'Rate this programming plan on 4 dimensions (0.0-1.0 each).\n'
                'Respond ONLY with JSON: {"decomposition":0.0,"edge_cases":0.0,"approach":0.0,"clarity":0.0}\n\n'
                f'Task: {task_desc}\nPlan: {plan_text}'
            )}],
            max_tokens=64, temperature=0.0,
        )
        dims = json.loads(resp.choices[0].message.content.strip())
        dims = {k: max(0.0, min(1.0, float(v))) for k, v in dims.items()}
        return round(sum(dims.values()) / 4, 3), dims
    except Exception:
        return 0.5, {"decomposition": 0.5, "edge_cases": 0.5, "approach": 0.5, "clarity": 0.5}


# ── pass@1 ────────────────────────────────────────────────────────────────────

def eval_pass1(llm, fmt, metas, df, is_mbpp, outdir, maxtok):
    ds = "mbpp" if is_mbpp else "humaneval"
    mod = M if is_mbpp else H
    extract_code = M.extract_code if is_mbpp else H.extract_code
    runtests = runfix.mbpp_run if is_mbpp else (lambda c, tl, tc: runfix.he_run(c, tc))

    n = len(df)
    raw_prompts = [m[4] for m in metas]
    prompts = [fmt(p) for p in raw_prompts]

    t0 = time.time()
    outs = greedy(llm, prompts, maxtok)

    passed, n_plan, entries = 0, 0, []
    for i, out in enumerate(outs):
        task_id, task_prompt, tc, tl, prompt = metas[i]
        txt = out.outputs[0].text
        plan = mod.extract_plan(txt)
        code = clean_code(extract_code(txt), test_list=tl)
        ok, err = runtests(code, tl, tc)
        has_plan = bool(plan)
        if ok:       passed += 1
        if has_plan: n_plan += 1
        entries.append({
            "task_id":       task_id,
            "task_prompt":   task_prompt,
            "prompt_text":   prompt,
            "raw_output":    txt,
            "plan":          plan,
            "code":          code,
            "test_list":     tl,
            "test_pass":     1 if ok else 0,
            "error":         "" if ok else err,
            "has_plan":      has_plan,
        })

    elapsed = time.time() - t0
    out_dir = os.path.join(outdir, f"{ds}_pass1")
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, f"{ds}_pass1.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")

    summary = {"split": "test", "num_tasks": n, "passed": passed,
               "pass_at_1": round(passed / n, 4),
               "has_plan_rate": round(n_plan / n, 4),
               "out_path": jsonl}
    with open(os.path.join(out_dir, f"summary_{ds}_pass1.json"), "w") as f:
        json.dump([summary], f, indent=2)

    print(f"[{ds}] pass@1 = {passed}/{n} = {passed/n*100:.1f}%  "
          f"has_plan={n_plan/n*100:.0f}%  [{elapsed:.0f}s]", flush=True)
    return entries


# ── pass@5 (unbiased, n=16, standard sampling) ────────────────────────────────

def eval_pass5(llm, fmt, metas, df, is_mbpp, outdir, maxtok, n_samples=16):
    ds = "mbpp" if is_mbpp else "humaneval"
    mod = M if is_mbpp else H
    extract_code = M.extract_code if is_mbpp else H.extract_code
    runtests = runfix.mbpp_run if is_mbpp else (lambda c, tl, tc: runfix.he_run(c, tc))

    n = len(df)
    raw_prompts = [m[4] for m in metas]
    prompts = [fmt(p) for p in raw_prompts]

    t0 = time.time()
    outs = sample_n(llm, prompts, n=n_samples, temperature=1.0, top_p=0.95, max_tokens=maxtok)

    # Run all n*n_samples test subprocesses in parallel (sequential execution was
    # the bottleneck: sampled programs often hang until the 20 s timeout).
    jobs = []
    for i, out in enumerate(outs):
        tl, tc = metas[i][3], metas[i][2]
        for o in out.outputs:
            jobs.append((clean_code(extract_code(o.text), test_list=tl), tl, tc))
    with ThreadPoolExecutor(max_workers=TEST_WORKERS) as ex:
        results = iter(list(ex.map(lambda j: runtests(*j), jobs)))

    entries = []
    pass5_tasks = 0
    for i, out in enumerate(outs):
        task_id, task_prompt, tc, tl, prompt = metas[i]
        n_correct = 0
        for si, o in enumerate(out.outputs):
            txt = o.text
            plan = mod.extract_plan(txt)
            code = clean_code(extract_code(txt), test_list=tl)
            ok, err = next(results)
            if ok: n_correct += 1
            entries.append({
                "task_id":     task_id,
                "task_prompt": task_prompt,
                "prompt_text": prompt,
                "raw_output":  txt,
                "plan":        plan,
                "code":        code,
                "test_list":   tl,
                "test_pass":   1 if ok else 0,
                "error":       "" if ok else err,
                "has_plan":    bool(plan),
                "sample_idx":  si,
            })
        # Unbiased estimator for pass@5
        p5_task = passk_unbiased(n_samples, n_correct, 5)
        pass5_tasks += p5_task

    elapsed = time.time() - t0
    pass5_mean = pass5_tasks / n

    out_dir = os.path.join(outdir, f"{ds}_pass5")
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, f"{ds}_pass5.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")

    summary = {"split": "test", "num_tasks": n, "n_samples": n_samples,
               "pass_at_5": round(pass5_mean, 4),
               "out_path": jsonl}
    with open(os.path.join(out_dir, f"summary_{ds}_pass5.json"), "w") as f:
        json.dump([summary], f, indent=2)

    print(f"[{ds}] pass@5 = {pass5_mean*100:.1f}%  (n={n_samples}, unbiased)  [{elapsed:.0f}s]", flush=True)


# ── feedback no-judge ─────────────────────────────────────────────────────────

def eval_feedback_nojudge(llm, fmt, metas, df, is_mbpp, outdir, maxtok, pass1_entries):
    ds = "mbpp" if is_mbpp else "humaneval"
    mod = M if is_mbpp else H
    extract_code = M.extract_code if is_mbpp else H.extract_code
    runtests = runfix.mbpp_run if is_mbpp else (lambda c, tl, tc: runfix.he_run(c, tc))

    n = len(df)
    raw_prompts = [m[4] for m in metas]

    # Build retry prompts for failures
    fail_idx = [i for i in range(n) if not pass1_entries[i]["test_pass"]]
    retry_prompts_raw = []
    for i in fail_idx:
        e = pass1_entries[i]
        hint = ("Review your plan and fix accordingly."
                if e["plan"] else "Consider writing a plan to help debug.")
        rp = (raw_prompts[i] + "\n\nYour previous solution had an error:\n"
              + FENCE + "\n" + e["error"] + "\n" + FENCE + "\n"
              + hint + "\nFix the code. Write your solution inside <code></code> tags.")
        retry_prompts_raw.append(rp)
    retry_prompts = [fmt(p) for p in retry_prompts_raw]

    t0 = time.time()
    retry_outs = greedy(llm, retry_prompts, maxtok) if retry_prompts else []

    p1 = sum(1 for e in pass1_entries if e["test_pass"])
    rec_nj = 0
    entries = []
    ri = 0
    for i in range(n):
        task_id, task_prompt, tc, tl, prompt = metas[i]
        e1 = pass1_entries[i]
        fa = {"raw_output": e1["raw_output"], "plan": e1["plan"], "code": e1["code"],
              "test_pass": e1["test_pass"], "error": e1["error"], "has_plan": e1["has_plan"],
              "reward": 1.25 if e1["test_pass"] else 0.0}

        if e1["test_pass"]:
            retry_rec = None
            final_pass = 1
            final_reward = 1.25
        else:
            txt2 = retry_outs[ri].outputs[0].text; ri += 1
            plan2 = mod.extract_plan(txt2)
            code2 = clean_code(extract_code(txt2), test_list=tl)
            ok2, err2 = runtests(code2, tl, tc)
            retry_rec = {"raw_output": txt2, "plan": plan2, "code": code2,
                         "test_pass": 1 if ok2 else 0, "error": "" if ok2 else err2,
                         "has_plan": bool(plan2), "reward": 1.25 if ok2 else 0.0}
            if ok2: rec_nj += 1
            final_pass = 1 if ok2 else 0
            final_reward = 1.25 if ok2 else 0.0

        entries.append({
            "task_id":      task_id,  "task_prompt": task_prompt,
            "prompt_text":  prompt,   "test_list":   tl,
            "first_attempt": fa,      "retry":       retry_rec,
            "final_pass":   final_pass, "final_reward": final_reward,
        })

    elapsed = time.time() - t0
    pf = p1 + rec_nj

    out_dir = os.path.join(outdir, f"{ds}_feedback_nojudge")
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, f"{ds}_feedback_nojudge.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")

    summary = {"split": "test", "num_tasks": n,
               "passed_attempt_1": p1, "pass_at_1": round(p1/n, 4),
               "recovered": rec_nj,   "passed_final": pf,
               "pass_at_1_retry": round(pf/n, 4), "out_path": jsonl}
    with open(os.path.join(out_dir, f"summary_{ds}_feedback_nojudge.json"), "w") as f:
        json.dump([summary], f, indent=2)

    print(f"[{ds}] +feedback(no-judge) = {pf}/{n} = {pf/n*100:.1f}%  (rec {rec_nj})  [{elapsed:.0f}s]", flush=True)


# ── feedback with judge ───────────────────────────────────────────────────────

def eval_feedback_judge(llm, fmt, metas, df, is_mbpp, outdir, maxtok, pass1_entries):
    ds = "mbpp" if is_mbpp else "humaneval"
    mod = M if is_mbpp else H
    extract_code = M.extract_code if is_mbpp else H.extract_code
    runtests = runfix.mbpp_run if is_mbpp else (lambda c, tl, tc: runfix.he_run(c, tc))

    n = len(df)
    raw_prompts = [m[4] for m in metas]

    # Score plans and get GPT hints in parallel for failures
    fail_idx = [i for i in range(n) if not pass1_entries[i]["test_pass"]]

    n_plan = sum(1 for e in pass1_entries if e["plan"])
    print(f"  Scoring {n_plan} plans via GPT-4o-mini ...", flush=True)

    def _score(i):
        e = pass1_entries[i]
        return score_plan(metas[i][1], e["plan"])

    def _hint(i):
        e = pass1_entries[i]
        return judge_feedback(metas[i][1], e["error"], e["plan"])

    with ThreadPoolExecutor(max_workers=16) as ex:
        pscores = list(ex.map(_score, range(n)))

    print(f"  Getting GPT hints for {len(fail_idx)} failures ...", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        hints = list(ex.map(_hint, fail_idx))

    # Build retry prompts with hints
    retry_prompts_raw = []
    for j, i in enumerate(fail_idx):
        e = pass1_entries[i]
        rp = (raw_prompts[i] + "\n\nYour previous solution had an error:\n"
              + FENCE + "\n" + e["error"] + "\n" + FENCE + "\n"
              + "Hint: " + hints[j] + "\nFix the code. Write your solution inside <code></code> tags.")
        retry_prompts_raw.append(rp)
    retry_prompts = [fmt(p) for p in retry_prompts_raw]

    t0 = time.time()
    retry_outs = greedy(llm, retry_prompts, maxtok) if retry_prompts else []

    p1 = sum(1 for e in pass1_entries if e["test_pass"])
    rec_j = 0
    sum_ps = sum(pscores[i][0] for i in range(n) if pass1_entries[i]["plan"])
    avg_ps = sum_ps / max(n_plan, 1)

    entries = []
    ri = 0
    for i in range(n):
        task_id, task_prompt, tc, tl, prompt = metas[i]
        e1 = pass1_entries[i]
        ps1, dims1 = pscores[i]
        r1_val = (0.5 * ps1 + 1.0) if e1["test_pass"] else 0.0

        code1_func = M.extract_func_name(e1["code"], task_prompt) if is_mbpp else H.extract_func_name(e1["code"], task_prompt)

        if e1["test_pass"]:
            fb_attempts = []
            final_pass = 1; final_code = e1["code"]; final_plan = e1["plan"]
            final_ps = ps1; final_dims = dims1; final_reward = r1_val; recovered = 0
        else:
            hint_j = hints[fail_idx.index(i)] if i in fail_idx else ""
            txt2 = retry_outs[ri].outputs[0].text; ri += 1
            plan2 = mod.extract_plan(txt2)
            code2 = clean_code(extract_code(txt2), test_list=tl)
            ok2, err2 = runtests(code2, tl, tc)
            ps2, dims2 = score_plan(task_prompt, plan2) if plan2 else \
                         (0.0, {"decomposition": 0.0, "edge_cases": 0.0, "approach": 0.0, "clarity": 0.0})
            r2_val = (0.5 * ps2 + 1.0) if ok2 else 0.0
            fb_attempts = [{"attempt": 1, "hint": hint_j,
                            "raw_output": txt2, "plan": plan2, "code": code2,
                            "plan_score": ps2, "plan_dims": dims2,
                            "test_pass": 1 if ok2 else 0, "error": "" if ok2 else err2,
                            "reward": r2_val}]
            if ok2: rec_j += 1
            final_pass = 1 if ok2 else 0
            final_code = code2; final_plan = plan2
            final_ps = ps2; final_dims = dims2
            final_reward = r2_val if ok2 else 0.0
            recovered = 1 if ok2 else 0

        entries.append({
            "task_id":               task_id,    "task_prompt":     task_prompt,
            "func_name":             code1_func, "prompt_text":     prompt,
            "raw_output":            e1["raw_output"], "plan":       e1["plan"],
            "code":                  e1["code"], "plan_score":      ps1,
            "plan_dims":             dims1,      "test_pass":       e1["test_pass"],
            "error":                 e1["error"], "reward":          r1_val,
            "feedback_attempts":     fb_attempts,
            "final_test_pass":       final_pass, "final_code":      final_code,
            "final_plan":            final_plan, "final_plan_score": final_ps,
            "final_plan_dims":       final_dims, "final_reward":    final_reward,
            "recovered_by_feedback": recovered,
            "has_plan":              e1["has_plan"],
            "test_list":             tl,
        })

    elapsed = time.time() - t0
    pf = p1 + rec_j

    out_dir = os.path.join(outdir, f"{ds}_feedback_judge")
    os.makedirs(out_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, f"{ds}_feedback_judge.jsonl")
    with open(jsonl, "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")

    summary = {"split": "test", "num_tasks": n,
               "passed_attempt_1": p1, "pass_at_1": round(p1/n, 4),
               "recovered": rec_j,    "passed_final": pf,
               "pass_at_1_retry": round(pf/n, 4),
               "avg_plan_score": round(avg_ps, 4), "out_path": jsonl}
    with open(os.path.join(out_dir, f"summary_{ds}_feedback_judge.json"), "w") as f:
        json.dump([summary], f, indent=2)

    print(f"[{ds}] +feedback(judge) = {pf}/{n} = {pf/n*100:.1f}%  "
          f"(rec {rec_j})  avg_plan_score={avg_ps:.3f}  [{elapsed:.0f}s]", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset",    required=True, choices=["mbpp", "humaneval"])
    ap.add_argument("--data",       required=True)
    ap.add_argument("--limit",      type=int, default=0)
    ap.add_argument("--outdir",     default="/workspace/eval_out")
    ap.add_argument("--n-samples",  type=int, default=16,
                    help="Number of samples for pass@5 (default: 16, unbiased estimator)")
    ap.add_argument("--prompt-style", default="optional", choices=["optional", "forced"],
                    help="optional: prompts from --data; forced: same tasks/tests as --data, "
                         "but system+user prompts taken from the forced-plan parquet --forced-data")
    ap.add_argument("--forced-data", default="")
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--gpu-mem",    type=float, default=0.9,
                    help="vLLM gpu_memory_utilization (lower it to run several evals on one GPU)")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    df = pd.read_parquet(a.data)
    if a.limit: df = df.iloc[:a.limit]
    n = len(df)

    is_mbpp = (a.dataset == "mbpp")
    mod = M if is_mbpp else H
    maxtok = 1024 if is_mbpp else 2048

    metas = [(mod.row_meta(df.iloc[i], i)) for i in range(n)]

    # Forced-plan prompts: tasks and tests stay those of --data (so every model is
    # scored on identical tests); only the prompt text is swapped for the forced-plan
    # one. The forced system prompt is identical for every row, so fmt() prepends it.
    system = None
    if a.prompt_style == "forced":
        fdf = pd.read_parquet(a.forced_data)
        lookup = {}
        for _, r in fdf.iterrows():
            msgs, ei = list(r["prompt"]), r["extra_info"]
            system = msgs[0]["content"]
            if is_mbpp:
                tl = json.loads(ei["test_list"])
                key = (ei["task_prompt_raw"].strip(), tl[0].strip() if tl else "")
            else:
                key = ei["task_id"]
            lookup[key] = msgs[-1]["content"]
        swapped = []
        for task_id, task_prompt, tc, tl, _ in metas:
            key = (task_prompt.strip(), tl[0].strip() if tl else "") if is_mbpp else task_id
            if key not in lookup:
                sys.exit(f"no forced-plan prompt for task {task_id!r}")
            swapped.append((task_id, task_prompt, tc, tl, lookup[key]))
        metas = swapped

    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    def fmt(p):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": p}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # enforce_eager + fixed seed = reproducible greedy decoding (no CUDA-graph FP drift)
    print(f"Loading model {a.checkpoint} (prompt-style={a.prompt_style}, seed={a.seed}) ...", flush=True)
    llm = LLM(model=a.checkpoint, dtype="float16",
              gpu_memory_utilization=a.gpu_mem, max_model_len=4096,
              enforce_eager=True, seed=a.seed)

    print(f"\n=== {a.dataset.upper()} eval (n={n}) ===", flush=True)

    # 1. pass@1 (greedy, saves JSONL, reused by feedback evals)
    print("\n--- pass@1 ---", flush=True)
    pass1_entries = eval_pass1(llm, fmt, metas, df, is_mbpp, a.outdir, maxtok)

    # 2. pass@5 (standard sampling, unbiased estimator)
    print("\n--- pass@5 ---", flush=True)
    eval_pass5(llm, fmt, metas, df, is_mbpp, a.outdir, maxtok, n_samples=a.n_samples)

    # 3. +feedback no-judge (reuses pass@1 outputs)
    print("\n--- +feedback (no-judge) ---", flush=True)
    eval_feedback_nojudge(llm, fmt, metas, df, is_mbpp, a.outdir, maxtok, pass1_entries)

    # 4. +feedback with judge (GPT-4o-mini hints, reuses pass@1 outputs)
    print("\n--- +feedback (judge) ---", flush=True)
    eval_feedback_judge(llm, fmt, metas, df, is_mbpp, a.outdir, maxtok, pass1_entries)

    # Combined one-line summary for this checkpoint/dataset
    ds = a.dataset
    def _load(name):
        with open(os.path.join(a.outdir, f"{ds}_{name}", f"summary_{ds}_{name}.json")) as f:
            return json.load(f)[0]
    p1, p5 = _load("pass1"), _load("pass5")
    nj, jd = _load("feedback_nojudge"), _load("feedback_judge")
    summary = {"checkpoint": a.checkpoint, "dataset": ds, "prompt_style": a.prompt_style,
               "seed": a.seed, "n": n, "pass@1": p1["pass_at_1"], "pass@5": p5["pass_at_5"],
               "feedback_nojudge_pass@1": nj["pass_at_1_retry"],
               "feedback_judge_pass@1": jd["pass_at_1_retry"],
               "has_plan_rate": p1["has_plan_rate"], "avg_plan_score": jd["avg_plan_score"]}
    with open(os.path.join(a.outdir, f"summary_{ds}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY: " + json.dumps(summary), flush=True)

    print(f"\n=== {a.dataset.upper()} DONE — logs in {a.outdir}/{a.dataset}_* ===", flush=True)


if __name__ == "__main__":
    main()
