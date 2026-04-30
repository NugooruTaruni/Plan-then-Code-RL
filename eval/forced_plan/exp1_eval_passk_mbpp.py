#!/usr/bin/env python3
"""
eval_passk_mbpp.py — pass@k evaluation for MBPP using exp1 parquet + prompt format

Generates n samples per task with temperature sampling, computes pass@k
using the unbiased estimator from Chen et al. (2021).

Usage:
  python eval_passk_mbpp.py --checkpoint <ckpt>
  python eval_passk_mbpp.py --checkpoint <ckpt> --num-samples 5 --k-values 5
  python eval_passk_mbpp.py --checkpoint Qwen/Qwen2.5-Coder-3B-Instruct --num-samples 5 --k-values 5
"""
import os, re, json, argparse, subprocess, tempfile, time, logging, signal, math
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


BASE_MODEL      = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATA_PATH       = "data/exp1_full_test.parquet"
RESULTS_DIR     = "eval_results_exp1_passk"
SANDBOX_TIMEOUT = 10
MAX_NEW_TOKENS  = 2048
TEMPERATURE     = 0.8

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)


# ── Unbiased pass@k estimator (Chen et al. 2021) ─────────────────────────────
def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ── Extraction ────────────────────────────────────────────────────────────────
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
    return plan, '\n'.join(clean).strip()


# ── Sandbox ───────────────────────────────────────────────────────────────────
def run_tests(code, test_list):
    if not code.strip(): return False
    if not test_list:    return False
    full = code + "\n\n" + "\n".join(test_list)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full); tmp = f.name
    proc = None
    try:
        proc = subprocess.Popen(
            ["python", tmp],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        proc.communicate(timeout=SANDBOX_TIMEOUT)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.communicate()
        except Exception:
            pass
        return False
    except Exception:
        return False
    finally:
        try: os.unlink(tmp)
        except: pass


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path):
    log.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None

    for path in [checkpoint_path,
                 os.path.join(checkpoint_path, "actor", "huggingface"),
                 os.path.join(checkpoint_path, "huggingface")]:
        cfg   = os.path.join(path, "config.json")
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
        log.info(f"Loading from HF hub: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, dtype=torch.float16, device_map="auto")
        log.info("  ✓ HF model loaded")

    model.eval()
    return model, tokenizer


# ── Generate n samples for one task ──────────────────────────────────────────
def generate_n_samples(model, tokenizer, messages, n, temperature):
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    samples = []
    for _ in range(n):
        inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out_ids[0][prompt_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        samples.append(raw)
    return samples


# ── Main evaluation ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   required=True)
    ap.add_argument("--num-samples",  type=int,         default=5,
                    help="Number of samples per task n (default: 5)")
    ap.add_argument("--k-values",     type=int, nargs="+", default=[5],
                    help="k values for pass@k (default: 5)")
    ap.add_argument("--temperature",  type=float,       default=0.8,
                    help="Sampling temperature (default: 0.8)")
    args = ap.parse_args()

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_name = os.path.basename(args.checkpoint.rstrip("/"))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f"eval_{step_name}_{ts}.log")
    raw_path = os.path.join(RESULTS_DIR, f"raw_{step_name}_{ts}.jsonl")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"[MBPP pass@k] checkpoint={args.checkpoint}")
    log.info(f"n={args.num_samples} k={args.k_values} temperature={args.temperature}")

    model, tokenizer = load_model(args.checkpoint)
    log.info(f"device: {next(model.parameters()).device}")

    df   = pd.read_parquet(DATA_PATH)
    rows = df.to_dict("records")
    log.info(f"Loaded {len(rows)} tasks from {DATA_PATH}")

    correct_counts = []
    all_records    = []

    with open(raw_path, "w") as rf:
        for i, row in enumerate(tqdm(rows, desc="MBPP pass@k")):
            extra       = row.get("extra_info", {})
            task_id     = extra.get("task_id", "?")
            task_prompt = extra.get("task_prompt_raw", "")
            test_list   = json.loads(extra.get("test_list", "[]"))
            messages    = list(row["prompt"])

            samples  = generate_n_samples(model, tokenizer, messages,
                                          args.num_samples, args.temperature)
            results  = []
            n_correct = 0
            for s_idx, raw in enumerate(samples):
                _, code = extract_plan_and_code(raw)
                passed  = run_tests(code, test_list)
                if passed: n_correct += 1
                results.append({"sample": s_idx, "passed": passed})

            correct_counts.append(n_correct)

            record = {
                "task_id": task_id, "task_prompt": task_prompt,
                "n_correct": n_correct, "n_total": args.num_samples,
                "samples": results,
            }
            all_records.append(record)
            rf.write(json.dumps(record) + "\n")
            rf.flush()

            if (i + 1) % 10 == 0:
                # Running pass@k estimate
                passk = {k: np.mean([pass_at_k(args.num_samples, c, k)
                                     for c in correct_counts]) * 100
                         for k in args.k_values}
                log.info(f"[{i+1}/{len(rows)}] running " +
                         " | ".join(f"pass@{k}={v:.1f}%" for k, v in passk.items()))

    # Final pass@k
    log.info(f"\nFINAL RESULTS: MBPP pass@k (n={args.num_samples})")
    final = {}
    for k in args.k_values:
        scores = [pass_at_k(args.num_samples, c, k) for c in correct_counts]
        val = np.mean(scores) * 100
        final[f"pass@{k}"] = val
        log.info(f"  pass@{k}: {val:.2f}%")

    summary = {
        "checkpoint":   args.checkpoint,
        "dataset":      "mbpp_test",
        "num_tasks":    len(rows),
        "num_samples":  args.num_samples,
        "temperature":  args.temperature,
        "k_values":     args.k_values,
        "results":      final,
        "raw_path":     raw_path,
    }
    sp = os.path.join(RESULTS_DIR, f"summary_{step_name}_{ts}.json")
    with open(sp, "w") as f: json.dump(summary, f, indent=2)
    log.info(f"Summary → {sp}")
    log.info(f"Raw     → {raw_path}")


if __name__ == "__main__":
    main()