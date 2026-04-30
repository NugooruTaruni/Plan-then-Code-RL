#!/usr/bin/env python3
"""
Experiment 3 evaluation (no feedback, no judge)
  - SPLIT_FILES points to exp3_* parquet files
  - FEEDBACK_USER_TEMPLATE updated to show all 3 test cases (matches training prompt)
  - Everything else identical for fair comparison
"""
import os, re, json, argparse, subprocess, tempfile, time, logging, signal
from datetime import datetime
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_MODEL      = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATA_DIR        = "data"
RESULTS_DIR     = "eval_results_plan_no_reward_upd2"
SANDBOX_TIMEOUT = 10
MAX_NEW_TOKENS  = 2048


logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)


SPLIT_FILES = {
    "val":  "mbpp_val_full.parquet",
    "test": "mbpp_test_full.parquet",
}


def log_generation(raw_fh, task_id, task_prompt, prompt_text,
                   raw_output, plan, code, test_list, passed, error):
    record = {
        "task_id":    task_id,
        "task_prompt": task_prompt,
        "prompt_text": prompt_text,
        "raw_output":  raw_output,
        "plan":        plan,
        "code":        code,
        "test_list":   test_list,
        "test_pass":   1 if passed else 0,
        "error":       "" if passed else error,
    }
    raw_fh.write(json.dumps(record) + "\n")
    raw_fh.flush()


def extract_plan_and_code(raw):
    plan, code = "", ""
    m = re.search(r"<plan>\s*(.*?)\s*</plan>", raw, re.DOTALL)
    if m:
        plan = m.group(1).strip()

    m = re.search(r"<code>\s*(.*?)\s*</code>", raw, re.DOTALL)
    if m:
        code = m.group(1).strip()
        code = re.sub(r'\n```\w*\s*$', '', code).strip()
        code = re.sub(r'^```\w*\s*\n', '', code).strip()
    else:
        m = re.search(r"<code>\s*(.*?)(?:```|$)", raw, re.DOTALL)
        if m:
            code = m.group(1).strip()
            code = re.sub(r'^```\w*\s*\n', '', code).strip()
        else:
            blocks = re.findall(r"```python\s*\n(.*?)```", raw, re.DOTALL)
            if blocks:
                code = blocks[-1].strip()
            else:
                blocks = re.findall(r"```\s*\n(.*?)```", raw, re.DOTALL)
                if blocks:
                    code = blocks[-1].strip()

    code = re.sub(r'\n```\w*\s*$', '', code).strip()

    lines = code.split('\n')
    clean = []
    for line in lines:
        s = line.strip()
        if s.startswith('assert ') or s.startswith('# Test') or s.startswith('# test'):
            break
        if clean and s and not s.startswith('#') \
                and not line.startswith(' ') and not line.startswith('\t') \
                and not any(s.startswith(kw) for kw in
                    ('def ', 'class ', 'import ', 'from ', '@', 'return', 'if ',
                     'else', 'elif ', 'for ', 'while ', 'try:', 'except', 'with ',
                     'raise ', 'pass', 'break', 'continue')):
            break
        clean.append(line)
    code = '\n'.join(clean).strip()
    return plan, code


def run_tests(code, test_list):
    if not code.strip(): return False, "empty code"
    if not test_list: return False, "no tests"
    full = code + "\n\n" + "\n".join(test_list)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full); tmp = f.name
    proc = None
    try:
        proc = subprocess.Popen(
            ["python", tmp],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        stdout, stderr = proc.communicate(timeout=SANDBOX_TIMEOUT)
        return proc.returncode == 0, stderr.decode()[:500]
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.communicate()
        except Exception:
            pass
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        try: os.unlink(tmp)
        except: pass


def load_model(checkpoint_path):
    log.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = None

    for path in [checkpoint_path,
                 os.path.join(checkpoint_path, "actor", "huggingface"),
                 os.path.join(checkpoint_path, "huggingface")]:
        cfg = os.path.join(path, "config.json")
        has_w = any(os.path.exists(os.path.join(path, w))
                    for w in ["model.safetensors", "pytorch_model.bin",
                              "model.safetensors.index.json",
                              "pytorch_model.bin.index.json"])
        if os.path.exists(cfg) and has_w:
            log.info(f"Loading HF model: {path}")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    path, dtype=torch.float16, device_map="auto")
                log.info("  ✓ HF model loaded"); break
            except Exception as e:
                log.warning(f"  HF load failed: {e}")

    if model is None:
        pt = os.path.join(checkpoint_path, "actor", "model_world_size_1_rank_0.pt")
        if os.path.exists(pt):
            log.info(f"Loading base + .pt weights: {pt}")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, dtype=torch.float16, device_map="auto")
            state = torch.load(pt, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            miss, unex = model.load_state_dict(state, strict=False)
            log.info(f"  ✓ .pt loaded | missing={len(miss)} unexpected={len(unex)}")
        else:
            log.warning("No weights found — using base model")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, dtype=torch.float16, device_map="auto")

    model.eval()
    return model, tokenizer


def load_data(split):
    fname = SPLIT_FILES.get(split)
    if not fname: log.error(f"Unknown split: {split}"); return []
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path): log.error(f"Not found: {path}"); return []
    df = pd.read_parquet(path)
    log.info(f"Loaded {len(df)} rows from {path}")
    return df.to_dict("records")


def evaluate_task(model, tokenizer, row, raw_fh):
    extra        = row.get("extra_info", {})
    task_id      = extra.get("task_id", "?")
    task_prompt  = extra.get("task_prompt_raw", "")
    test_list    = json.loads(extra.get("test_list", "[]"))
    messages     = list(row["prompt"])

    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
    gen_time = time.time() - t0

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out_ids[0][prompt_len:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    plan, code = extract_plan_and_code(raw_output)
    passed, error = run_tests(code, test_list)

    log_generation(raw_fh, task_id, task_prompt, chat_input,
                   raw_output, plan, code, test_list, passed, error)

    return {
        "task_id": task_id, "task_prompt": task_prompt,
        "full_output": raw_output, "plan": plan, "code": code,
        "test_list": test_list, "test_pass": 1 if passed else 0,
        "error": "" if passed else error, "has_plan": bool(plan.strip()),
        "prompt_tokens": prompt_len,
        "response_tokens": len(new_tokens), "gen_time_s": round(gen_time, 2),
    }


def evaluate_split(model, tokenizer, split_name, rows, log_file, raw_fh):
    log.info(f"\n{'='*60}\nEVALUATING: {split_name} ({len(rows)} tasks)\n{'='*60}")
    results, n_passed = [], 0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"eval_{split_name}.jsonl")
    with open(out, "w") as eval_fh:
        for i, row in enumerate(rows):
            result = evaluate_task(model, tokenizer, row, raw_fh)
            results.append(result)
            if result["test_pass"]: n_passed += 1

            eval_fh.write(json.dumps(result) + "\n")
            eval_fh.flush()

            status = "PASS ✓" if result["test_pass"] else "FAIL ✗"
            line = (f"[{i+1:3d}/{len(rows)}] {status} | "
                    f"plan={'Y' if result['has_plan'] else 'N'} | "
                    f"{result['response_tokens']} tok | "
                    f"{result['gen_time_s']:.1f}s | "
                    f"{result['task_prompt'][:60]}")
            log.info(line); log_file.write(line + "\n"); log_file.flush()

    n = len(rows)
    pass_rate = n_passed / n if n else 0
    plan_rate = sum(1 for r in results if r["has_plan"]) / n if n else 0
    avg_tok   = sum(r["response_tokens"] for r in results) / n if n else 0
    log.info(f"\nRESULTS: {split_name}")
    log.info(f"  pass@1    : {n_passed}/{n} ({100*pass_rate:.1f}%)")
    log.info(f"  has_plan  : {sum(1 for r in results if r['has_plan'])}/{n} ({100*plan_rate:.1f}%)")
    log.info(f"  avg tokens: {avg_tok:.0f}")
    log.info(f"  Saved: {out}")

    return {"split": split_name, "num_tasks": n, "passed": n_passed,
            "pass_at_1": pass_rate, "has_plan_rate": plan_rate,
            "avg_tokens": avg_tok, "out_path": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "all"])
    ap.add_argument("--start-from", type=int, default=0, help="Skip first N rows (0-based)")
    args = ap.parse_args()

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_name = os.path.basename(args.checkpoint.rstrip("/"))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f"eval_{step_name}_{ts}.log")
    raw_path = os.path.join(RESULTS_DIR, f"raw_{step_name}_{ts}.jsonl")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"[EXP1] checkpoint={args.checkpoint} | split={args.split}")
    log.info(f"raw generations → {raw_path}")
    model, tokenizer = load_model(args.checkpoint)
    log.info(f"device: {next(model.parameters()).device}")

    splits = ["val", "test"] if args.split == "all" else [args.split]
    all_results = []
    with open(log_path, "a") as lf, open(raw_path, "w") as rf:
        for s in splits:
            rows = load_data(s)
            if rows:
                rows = rows[args.start_from:]
                all_results.append(evaluate_split(model, tokenizer, s, rows, lf, rf))

    log.info("\nFINAL SUMMARY")
    log.info(f"  {'Split':<10} {'pass@1':>10} {'has_plan':>10} {'passed':>8} {'total':>8}")
    for r in all_results:
        log.info(f"  {r['split']:<10} {r['pass_at_1']:>9.1%} "
                 f"{r['has_plan_rate']:>9.1%} {r['passed']:>8} {r['num_tasks']:>8}")

    sp = os.path.join(RESULTS_DIR, f"summary_{step_name}_{ts}.json")
    with open(sp, "w") as f: json.dump(all_results, f, indent=2)
    log.info(f"Summary     → {sp}")
    log.info(f"Raw outputs → {raw_path}")


if __name__ == "__main__":
    main()
