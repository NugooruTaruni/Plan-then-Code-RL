#!/usr/bin/env python3
"""
run_all_humaneval.py  —  Evaluate optional-plan model on HumanEval test set.

Outputs per eval:
  logs/eval5_he_pass1.log         humaneval_pass1/humaneval_pass1.jsonl          + summary_optional_plan_humaneval_pass1.json
  logs/eval6_he_pass5.log         humaneval_pass5/humaneval_pass5.jsonl          + summary_optional_plan_humaneval_pass5.json
  logs/eval7_he_fb_nojudge.log    humaneval_feedback_nojudge/humaneval_feedback_nojudge.jsonl + summary
  logs/eval8_he_fb_judge.log      humaneval_feedback_judge/humaneval_feedback_judge.jsonl     + summary
"""

import argparse, re, subprocess, json, os, time
import torch, pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from code_cleaner import clean_code

LOGS_DIR = "logs"


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_code(text):
    text = re.sub(r'^\s*\}?\s*Assistant:\s*', '', text, flags=re.DOTALL).strip()
    blocks = re.findall(r'```(?:python)?\s*(.*?)```', text, re.DOTALL)
    for block in blocks:
        if 'def ' in block or 'class ' in block:
            return block.strip()
    if blocks:
        return blocks[-1].strip()
    m = re.search(r'<code>(.*?)</code>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if '```' in text:
        return text.rsplit('```', 1)[-1].strip()
    return text.strip()


def extract_plan(text):
    m = re.search(r'<plan>(.*?)</plan>', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_func_name(code, task_prompt=""):
    m = re.search(r"def (\w+)\s*\(", code)
    if m:
        return m.group(1)
    m = re.search(r"def (\w+)\s*\(", task_prompt)
    return m.group(1) if m else "solution"


def run_test(code, test):
    if not code.strip():
        return False, "Empty code"
    # Run from a temp file (not `python3 -c`) so HumanEval+'s large test suites
    # don't overflow the OS argument-length limit ("Argument list too long").
    import tempfile
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(f'{code}\n\n{test}'); path = fh.name
        r = subprocess.run(['python3', path], capture_output=True, text=True, timeout=20)
        return (r.returncode == 0, r.stderr[:400])
    except subprocess.TimeoutExpired:
        return (False, "Timeout")
    except Exception as e:
        return (False, str(e)[:400])
    finally:
        if path:
            try:
                os.unlink(path)
            except Exception:
                pass


def safe_list(lst):
    if lst is None:
        return []
    if hasattr(lst, 'tolist'):
        return lst.tolist()
    return list(lst) if not isinstance(lst, list) else lst


def ntok(tok, text):
    return len(tok.encode(text, add_special_tokens=False))


def write_log(filename, msg):
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(os.path.join(LOGS_DIR, filename), "w", encoding="utf-8") as f:
        f.write(msg)
    print(msg, end="")


def row_meta(row, i):
    extra = row.get("extra_info", {}) or {}
    task_id     = str(row.get("task_id", extra.get("task_id", f"HumanEval/{i}")))
    test_code   = extra.get("test_code", "")
    test_list   = safe_list(extra.get("test_list", []))
    prompt_text = row["prompt"][0]["content"] if isinstance(row["prompt"], list) else str(row["prompt"])
    task_prompt = extra.get("task_description", prompt_text)
    return task_id, task_prompt, test_code, test_list, prompt_text


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
    """Returns (score, dims_dict)."""
    empty = {"decomposition": 0.0, "edge_cases": 0.0, "approach": 0.0, "clarity": 0.0}
    if not plan_text or not plan_text.strip():
        return 0.0, empty
    try:
        from openai import OpenAI
        resp = OpenAI().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                'Rate this programming plan on 4 dimensions (0.0–1.0 each).\n'
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


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt):
    tok = AutoTokenizer.from_pretrained(ckpt)
    tok.chat_template = None
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16, device_map="cuda")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model


def generate(model, tok, prompt, max_new_tokens=2048, do_sample=False, temperature=1.0):
    inp = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(
        **inp, max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature, pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)[len(prompt):]


# ── eval_pass1 ────────────────────────────────────────────────────────────────
def eval_pass1(ckpt, data):
    df = pd.read_parquet(data)
    tok, model = load_model(ckpt)
    out_dir = "humaneval_pass1"
    os.makedirs(out_dir, exist_ok=True)

    passed, n_plan, entries = 0, 0, []
    for i in tqdm(range(len(df)), desc="HumanEval pass@1"):
        row = df.iloc[i]
        task_id, task_prompt, test_code, test_list, prompt = row_meta(row, i)

        t0 = time.time()
        resp = generate(model, tok, prompt)
        gen_t = round(time.time() - t0, 2)

        plan = extract_plan(resp)
        code = clean_code(extract_code(resp), test_list=test_list)
        ok, err = run_test(code, test_code)
        has_plan = bool(plan)
        if ok:       passed += 1
        if has_plan: n_plan += 1

        entries.append({
            "task_id":         task_id,
            "task_prompt":     task_prompt,
            "prompt_text":     prompt,
            "raw_output":      resp,
            "plan":            plan,
            "code":            code,
            "test_pass":       1 if ok else 0,
            "error":           "" if ok else err,
            "has_plan":        has_plan,
            "prompt_tokens":   ntok(tok, prompt),
            "response_tokens": ntok(tok, resp),
            "gen_time_s":      gen_t,
        })

    with open(os.path.join(out_dir, "humaneval_pass1.jsonl"), "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")

    n = len(df)
    with open(os.path.join(out_dir, "summary_optional_plan_humaneval_pass1.json"), "w") as f:
        json.dump([{"split": "test", "num_tasks": n, "passed": passed,
                    "pass_at_1": round(passed/n, 3), "has_plan_rate": round(n_plan/n, 3),
                    "out_path": "eval_results_optional_plan/humaneval_pass1.jsonl"}], f, indent=2)

    write_log("eval5_he_pass1.log", f"\n✅ pass@1: {passed}/{n} ({passed/n*100:.1f}%)\n")


# ── eval_pass5 ────────────────────────────────────────────────────────────────
def eval_pass5(ckpt, data, n_samples=5):
    df = pd.read_parquet(data)
    tok, model = load_model(ckpt)
    tok.padding_side = "left"
    out_dir = "humaneval_pass5"
    os.makedirs(out_dir, exist_ok=True)

    tasks_passed, n_plan, entries = 0, 0, []
    for i in tqdm(range(len(df)), desc=f"HumanEval pass@5 (n={n_samples})"):
        row = df.iloc[i]
        task_id, task_prompt, test_code, test_list, prompt = row_meta(row, i)

        inputs = tok([prompt] * n_samples, return_tensors="pt",
                     padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=2048, do_sample=True,
                temperature=0.8, pad_token_id=tok.eos_token_id,
            )

        any_pass = False
        for si, out in enumerate(outputs):
            resp     = tok.decode(out, skip_special_tokens=True)[len(prompt):]
            plan     = extract_plan(resp)
            code     = clean_code(extract_code(resp), test_list=test_list)
            ok, err  = run_test(code, test_code)
            has_plan = bool(plan)
            if ok:       any_pass = True
            if has_plan: n_plan += 1
            entries.append({
                "task_id":     task_id,
                "task_prompt": task_prompt,
                "prompt_text": prompt,
                "raw_output":  resp,
                "plan":        plan,
                "code":        code,
                "test_pass":   1 if ok else 0,
                "error":       "" if ok else err,
                "has_plan":    has_plan,
                "sample_idx":  si,
            })
        if any_pass: tasks_passed += 1

    with open(os.path.join(out_dir, "humaneval_pass5.jsonl"), "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")

    n = len(df)
    with open(os.path.join(out_dir, "summary_optional_plan_humaneval_pass5.json"), "w") as f:
        json.dump([{"split": "test", "num_tasks": n, "n": n_samples,
                    "passed": tasks_passed, "pass_at_5": round(tasks_passed/n, 3),
                    "has_plan_rate": round(n_plan/len(entries), 3),
                    "out_path": "eval_results_optional_plan/humaneval_pass5.jsonl"}], f, indent=2)

    write_log("eval6_he_pass5.log", f"\n✅ pass@5: {tasks_passed}/{n} ({tasks_passed/n*100:.1f}%)\n")


# ── eval_feedback_nojudge ─────────────────────────────────────────────────────
def eval_feedback_nojudge(ckpt, data, max_retries=1):
    df = pd.read_parquet(data)
    tok, model = load_model(ckpt)
    out_dir = "humaneval_feedback_nojudge"
    os.makedirs(out_dir, exist_ok=True)

    p1, pf, rec, n_plan, entries = 0, 0, 0, 0, []
    for i in tqdm(range(len(df)), desc="HumanEval feedback (no judge)"):
        row = df.iloc[i]
        task_id, task_prompt, test_code, test_list, prompt = row_meta(row, i)

        r1    = generate(model, tok, prompt)
        plan1 = extract_plan(r1)
        code1 = clean_code(extract_code(r1), test_list=test_list)
        ok1, err1 = run_test(code1, test_code)
        has_plan1 = bool(plan1)
        if has_plan1: n_plan += 1

        fa = {"raw_output": r1, "plan": plan1, "code": code1,
              "test_pass": 1 if ok1 else 0, "error": "" if ok1 else err1,
              "has_plan": has_plan1, "reward": 1.25 if ok1 else 0.0}

        if ok1:
            p1 += 1; pf += 1
            retry = None; final_pass = 1; final_reward = 1.25
        else:
            hint = ("Review your plan and fix accordingly."
                    if plan1 else "Consider writing a plan to help debug.")
            retry_prompt = (
                f"{prompt}\n\nYour previous solution had an error:\n```\n{err1}\n```\n"
                f"{hint}\nFix the code. Write your solution inside <code></code> tags."
            )
            r2    = generate(model, tok, retry_prompt)
            plan2 = extract_plan(r2)
            code2 = clean_code(extract_code(r2), test_list=test_list)
            ok2, err2 = run_test(code2, test_code)
            retry = {"raw_output": r2, "plan": plan2, "code": code2,
                     "test_pass": 1 if ok2 else 0, "error": "" if ok2 else err2,
                     "has_plan": bool(plan2), "reward": 1.25 if ok2 else 0.0}
            if ok2:
                rec += 1; pf += 1
                final_pass = 1; final_reward = 1.25
            else:
                final_pass = 0; final_reward = 0.0

        entries.append({
            "task_id":       task_id,    "task_prompt":  task_prompt,
            "prompt_text":   prompt,
            "first_attempt": fa,         "retry":        retry,
            "final_pass":    final_pass, "final_reward": final_reward,
        })

    with open(os.path.join(out_dir, "humaneval_feedback_nojudge.jsonl"), "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")

    n = len(df)
    avg_r_fa  = sum(e["first_attempt"]["reward"] for e in entries) / n
    avg_r_fin = sum(e["final_reward"] for e in entries) / n
    with open(os.path.join(out_dir, "summary_optional_plan_humaneval_nojudge.json"), "w") as f:
        json.dump([{"split": "test", "num_tasks": n,
                    "passed_attempt_1": p1,  "pass_at_1": round(p1/n, 3),
                    "recovered": rec,        "passed_final": pf,
                    "pass_at_1_retry": round(pf/n, 3),
                    "has_plan_rate": round(n_plan/n, 3),
                    "avg_reward_attempt1": round(avg_r_fa, 3),
                    "avg_reward_final": round(avg_r_fin, 3),
                    "out_path": "eval_results_optional_plan/humaneval_nojudge_test.jsonl"}], f, indent=2)

    write_log("eval7_he_fb_nojudge.log",
              f"\n✅ pass@1 (attempt 1): {p1}/{n} ({p1/n*100:.1f}%)\n"
              f"✅ pass@1 (w/ 1 retry): {pf}/{n} ({pf/n*100:.1f}%)\n"
              f"✅ Recovered: {rec}\n")


# ── eval_feedback_judge ────────────────────────────────────────────────────────
def eval_feedback_judge(ckpt, data, max_retries=1):
    df = pd.read_parquet(data)
    tok, model = load_model(ckpt)
    out_dir = "humaneval_feedback_judge"
    os.makedirs(out_dir, exist_ok=True)

    p1, pf, rec, n_plan = 0, 0, 0, 0
    sum_ps, sum_r1, sum_rf = 0.0, 0.0, 0.0
    entries = []

    for i in tqdm(range(len(df)), desc="HumanEval feedback (with judge)"):
        row = df.iloc[i]
        task_id, task_prompt, test_code, test_list, prompt = row_meta(row, i)

        t0 = time.time()
        r1    = generate(model, tok, prompt)
        gen_t = round(time.time() - t0, 2)
        plan1  = extract_plan(r1)
        code1  = clean_code(extract_code(r1), test_list=test_list)
        ok1, err1 = run_test(code1, test_code)
        has_plan1 = bool(plan1)
        if has_plan1: n_plan += 1

        ps1, dims1 = score_plan(task_prompt, plan1)
        if has_plan1: sum_ps += ps1
        r1_val = 1.0 if ok1 else 0.0   # HumanEval uses binary reward
        sum_r1 += r1_val
        func_name = extract_func_name(code1, task_prompt)

        if ok1:
            p1 += 1; pf += 1; sum_rf += r1_val
            fb_attempts = []
            final_pass = 1;  final_code = code1; final_plan = plan1
            final_ps = ps1;  final_dims = dims1; final_reward = r1_val; recovered = 0
        else:
            hint = judge_feedback(task_prompt, err1, plan1)
            retry_prompt = (
                f"{prompt}\n\nYour previous solution had an error:\n```\n{err1}\n```\n"
                f"Hint: {hint}\nFix the code. Write your solution inside <code></code> tags."
            )
            t1 = time.time()
            r2     = generate(model, tok, retry_prompt)
            gen_t2 = round(time.time() - t1, 2)
            plan2  = extract_plan(r2)
            code2  = clean_code(extract_code(r2), test_list=test_list)
            ok2, err2 = run_test(code2, test_code)
            ps2, dims2 = score_plan(task_prompt, plan2) if plan2 else \
                         (0.0, {"decomposition": 0.0, "edge_cases": 0.0, "approach": 0.0, "clarity": 0.0})
            r2_val = 1.0 if ok2 else 0.0
            sum_rf += r2_val
            fb_attempts = [{"attempt": 1, "hint": hint,
                            "raw_output": r2, "plan": plan2, "code": code2,
                            "plan_score": ps2, "plan_dims": dims2,
                            "test_pass": 1 if ok2 else 0, "error": "" if ok2 else err2,
                            "reward": r2_val,
                            "response_tokens": ntok(tok, r2), "gen_time_s": gen_t2}]
            if ok2:
                rec += 1; pf += 1
                final_pass = 1;  final_code = code2; final_plan = plan2
                final_ps = ps2;  final_dims = dims2; final_reward = r2_val; recovered = 1
            else:
                final_pass = 0;  final_code = code2; final_plan = plan2
                final_ps = ps2;  final_dims = dims2; final_reward = 0.0;  recovered = 0

        entries.append({
            "task_id":               task_id,    "task_prompt":     task_prompt,
            "func_name":             func_name,  "prompt_text":     prompt,
            "raw_output":            r1,          "plan":            plan1,
            "code":                  code1,       "plan_score":      ps1,
            "plan_dims":             dims1,       "test_pass":       1 if ok1 else 0,
            "error":                 "" if ok1 else err1,
            "reward":                r1_val,      "feedback_attempts": fb_attempts,
            "final_test_pass":       final_pass, "final_code":      final_code,
            "final_plan":            final_plan, "final_plan_score": final_ps,
            "final_plan_dims":       final_dims, "final_reward":    final_reward,
            "recovered_by_feedback": recovered,
            "has_plan":              has_plan1,
            "prompt_tokens":         ntok(tok, prompt), "response_tokens": ntok(tok, r1),
            "gen_time_s":            gen_t,
        })

    with open(os.path.join(out_dir, "humaneval_feedback_judge.jsonl"), "w", encoding="utf-8") as f:
        for e in entries: f.write(json.dumps(e) + "\n")

    n = len(df)
    avg_ps = sum_ps / max(n_plan, 1)
    with open(os.path.join(out_dir, "summary_optional_plan_humaneval_judge.json"), "w") as f:
        json.dump([{"split": "test", "num_tasks": n,
                    "passed_attempt_1": p1,  "pass_at_1": round(p1/n, 3),
                    "recovered": rec,        "passed_final": pf,
                    "pass_at_1_retry": round(pf/n, 3),
                    "has_plan_rate": round(n_plan/n, 3),
                    "avg_plan_score": round(avg_ps, 3),
                    "avg_reward_attempt1": round(sum_r1/n, 3),
                    "avg_reward_final": round(sum_rf/n, 3),
                    "out_path": "eval_results_optional_plan/humaneval_judge_test.jsonl"}], f, indent=2)

    write_log("eval8_he_fb_judge.log",
              f"\n✅ pass@1 (attempt 1): {p1}/{n} ({p1/n*100:.1f}%)\n"
              f"✅ pass@1 (w/ 1 retry): {pf}/{n} ({pf/n*100:.1f}%)\n"
              f"✅ Recovered: {rec}\n"
              f"✅ Avg plan score: {avg_ps:.3f}\n"
              f"✅ Avg reward (attempt 1): {sum_r1/n:.3f}\n"
              f"✅ Avg reward (final): {sum_rf/n:.3f}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="merged_optional_125")
    p.add_argument("--data",          required=True)
    p.add_argument("--max_retries",   type=int, default=1)
    p.add_argument("--n",             type=int, default=5)
    p.add_argument("--skip-feedback", action="store_true")
    args = p.parse_args()

    eval_pass1(args.checkpoint, args.data)
    eval_pass5(args.checkpoint, args.data, n_samples=args.n)

    if not args.skip_feedback:
        eval_feedback_nojudge(args.checkpoint, args.data, max_retries=args.max_retries)
        eval_feedback_judge(args.checkpoint, args.data,   max_retries=args.max_retries)

    print("\n✅ All HumanEval evaluations complete!")
