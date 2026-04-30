#!/usr/bin/env python3
"""
evaluate_humaneval_feedback.py — HumanEval+ evaluation with feedback loop

Key differences from exp1_evaluate_feedback.py:
  - Uses data/humaneval_plus_test.parquet
  - Tests via check(candidate) function format, not assert list
  - RESULTS_DIR → eval_results_humaneval

CLI flags
  --no-judge      disable GPT plan scoring + GPT error explanation (pure self-repair)
  --no-feedback   disable the feedback loop entirely
  --max-feedback  max retry attempts per task (default: 1)
  --start-from    skip first N rows (0-based), for resuming

Usage:
  # Pure self-repair, no GPT
  python evaluate_humaneval_feedback.py --checkpoint <ckpt> --no-judge

  # GPT-guided feedback
  python evaluate_humaneval_feedback.py --checkpoint <ckpt>
"""
import os, re, json, argparse, subprocess, tempfile, time, logging, signal
from datetime import datetime
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_MODEL      = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATA_PATH       = "data/humaneval_plus_test.parquet"
RESULTS_DIR     = "eval_results_humaneval"
SANDBOX_TIMEOUT = 10
MAX_NEW_TOKENS  = 2048

GPT_MODEL   = "gpt-4o-mini"
MAX_RETRIES = 3
RETRY_DELAY = 2

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)


# ── Judge prompts ─────────────────────────────────────────────────────────────
JUDGE_SYSTEM = (
    "You are a strict but fair programming plan evaluator. "
    "Respond with only a JSON object containing scores."
)

JUDGE_PROMPT = """You are evaluating the quality of a programming plan.

## Task
{task_prompt}

## Required Function Name
{func_name}

## Plan to Evaluate
{plan_text}

Rate this plan on 4 dimensions, each from 0.0 to 1.0:
1. **Decomposition** (0-1): Does the plan break the problem into clear, logical steps?
2. **Edge Cases** (0-1): Does the plan address boundary conditions?
3. **Approach** (0-1): Is the algorithmic approach correct and appropriate?
4. **Clarity** (0-1): Is the plan specific enough to guide implementation?

Respond with ONLY a JSON object, no other text:
{{"decomposition": 0.0, "edge_cases": 0.0, "approach": 0.0, "clarity": 0.0}}"""

ERROR_EXPLAIN_SYSTEM = """\
You are a concise Python debugging assistant.
Given a coding task, the code that failed, and the error/traceback,
explain in ONE short sentence what is wrong and how to fix it.
No code. No markdown. Plain text only."""

ERROR_EXPLAIN_USER = """\
Task: {task_prompt}

Code:
{code}

Error:
{error}"""

FEEDBACK_USER_TEMPLATE = """\
Task: {task_prompt}

Function name: {func_name}

Your previous attempt failed with this error:
{error}

Hint: {hint}

Please fix your solution. Respond in this exact format:

<plan>
Your updated structured plan here...
</plan>

<code>
Your corrected Python code here (implement the function '{func_name}')...
</code>"""


# ── Error hint (no GPT) ───────────────────────────────────────────────────────
def get_error_hint(error, func_name=""):
    e = error.strip()
    if "TIMEOUT" in e:
        return ("The solution has an infinite loop or exponential complexity. "
                "Use a more efficient algorithm or add a termination condition.")
    if "AssertionError" in e:
        return ("The output does not match the expected value. "
                "Re-check your algorithm logic and edge cases.")
    if "NameError" in e:
        name_hint = ""
        m = re.search(r"name '(\w+)' is not defined", e)
        if m:
            name_hint = f" You used '{m.group(1)}' but it is not defined."
        func_hint = (f" Make sure your function is named exactly '{func_name}'."
                     if func_name else "")
        return f"A variable or function name is not defined.{name_hint}{func_hint}"
    if "TypeError" in e:
        return ("There is a type mismatch — check that your function signature "
                "matches the expected inputs exactly.")
    if "ValueError" in e:
        return ("A value has the right type but wrong content — check your logic "
                "handles all edge cases.")
    if "IndexError" in e:
        return ("You are accessing an index out of range. "
                "Add bounds checking before accessing list or string elements.")
    if "IndentationError" in e or "SyntaxError" in e:
        return ("There is a syntax or indentation error. "
                "Check all blocks are properly indented and all brackets are closed.")
    if "AttributeError" in e:
        return ("You are calling a method or attribute that does not exist. "
                "Check the type of your variable and use the correct method.")
    if "UnboundLocalError" in e:
        return ("A local variable is referenced before assignment. "
                "Initialize all variables before using them.")
    if "empty code" in e:
        return (f"No code was extracted from your response. "
                f"Make sure to wrap your solution in <code>...</code> tags "
                f"using function name '{func_name}'.")
    return e[:400]


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


# ── Sandbox — HumanEval check() format ───────────────────────────────────────
def run_humaneval_test(code, test_code, entry_point):
    if not code.strip():
        return False, "empty code"
    full = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
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


# ── GPT judge ─────────────────────────────────────────────────────────────────
_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def score_plan(task_prompt, plan, func_name=""):
    if not plan.strip():
        return 0.0, {}
    for attempt in range(MAX_RETRIES):
        try:
            client = _get_openai_client()
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": JUDGE_PROMPT.format(
                        task_prompt=task_prompt,
                        func_name=func_name or "N/A",
                        plan_text=plan)},
                ],
                temperature=0.0, max_tokens=200,
            )
            text = response.choices[0].message.content
            m = re.search(r'\{[^{}]+\}', text)
            if m:
                scores = json.loads(m.group())
                dims = ["decomposition", "edge_cases", "approach", "clarity"]
                vals = [max(0.0, min(1.0, float(scores.get(d, 0.0)))) for d in dims]
                return sum(vals) / len(dims), {d: v for d, v in zip(dims, vals)}
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt)); continue
            log.warning(f"plan score failed (attempt {attempt+1}): {e}")
    return 0.0, {}


def explain_error(task_prompt, code, error):
    for attempt in range(MAX_RETRIES):
        try:
            client = _get_openai_client()
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": ERROR_EXPLAIN_SYSTEM},
                    {"role": "user",   "content": ERROR_EXPLAIN_USER.format(
                        task_prompt=task_prompt, code=code, error=error[:400])},
                ],
                temperature=0.0, max_tokens=100,
            )
            hint = response.choices[0].message.content.strip()
            hint = re.sub(r'```.*?```', '', hint, flags=re.DOTALL).strip()
            return hint or error[:200]
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt)); continue
            log.warning(f"error explain failed (attempt {attempt+1}): {e}")
    return error[:200]


def compute_reward(plan_score, test_pass, alpha, beta):
    return alpha * plan_score * test_pass + beta * test_pass


def log_generation(raw_fh, record):
    raw_fh.write(json.dumps(record) + "\n")
    raw_fh.flush()


# ── Policy generation ─────────────────────────────────────────────────────────
def generate(model, tokenizer, messages):
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_time   = time.time() - t0
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out_ids[0][prompt_len:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw_output, chat_input, prompt_len, len(new_tokens), round(gen_time, 2)


# ── Core per-task evaluation ──────────────────────────────────────────────────
def evaluate_task(model, tokenizer, row, raw_fh, use_judge,
                  alpha, beta, use_feedback, max_feedback):
    extra       = row.get("extra_info", {})
    task_id     = extra.get("task_id", "?")
    task_prompt = extra.get("task_prompt_raw", "")
    func_name   = extra.get("func_name", "")
    test_code   = extra.get("test_code", "")
    messages    = list(row["prompt"])

    raw_output, chat_input, prompt_len, n_new_tok, gen_time = \
        generate(model, tokenizer, messages)
    plan, code = extract_plan_and_code(raw_output)
    passed, error = run_humaneval_test(code, test_code, func_name)

    plan_score, plan_dims = (score_plan(task_prompt, plan, func_name=func_name)
                             if use_judge else (-1.0, {}))
    reward = compute_reward(plan_score if plan_score >= 0 else 0.0, int(passed), alpha, beta)

    fb_attempts = []
    final_passed, final_error, final_code, final_plan = passed, error, code, plan
    final_plan_score, final_plan_dims, final_reward = plan_score, plan_dims, reward

    if use_feedback and not passed:
        current_error, current_code = error, code

        for attempt_i in range(max_feedback):
            if use_judge:
                hint = ("The solution has an infinite loop." if current_error == "TIMEOUT"
                        else explain_error(task_prompt, current_code, current_error))
            else:
                hint = get_error_hint(current_error, func_name=func_name)

            fb_messages = list(messages)
            fb_messages.append({"role": "assistant",
                                 "content": raw_output if attempt_i == 0
                                 else fb_attempts[-1]["raw_output"]})
            fb_messages.append({"role": "user", "content":
                FEEDBACK_USER_TEMPLATE.format(
                    task_prompt=task_prompt, func_name=func_name,
                    error=current_error[:400], hint=hint)})

            fb_raw, _, _, fb_n_tok, fb_gen_time = generate(model, tokenizer, fb_messages)
            fb_plan, fb_code = extract_plan_and_code(fb_raw)
            fb_passed, fb_error = run_humaneval_test(fb_code, test_code, func_name)

            fb_plan_score, fb_plan_dims = (score_plan(task_prompt, fb_plan, func_name=func_name)
                                           if use_judge else (-1.0, {}))
            fb_reward = compute_reward(fb_plan_score if fb_plan_score >= 0 else 0.0,
                                       int(fb_passed), alpha, beta)

            fb_attempts.append({
                "attempt": attempt_i + 1, "hint": hint,
                "raw_output": fb_raw, "plan": fb_plan, "code": fb_code,
                "plan_score": fb_plan_score, "plan_dims": fb_plan_dims,
                "test_pass": 1 if fb_passed else 0,
                "error": "" if fb_passed else fb_error,
                "reward": fb_reward, "response_tokens": fb_n_tok, "gen_time_s": fb_gen_time,
            })

            if fb_passed:
                final_passed, final_error, final_code = True, "", fb_code
                final_plan, final_plan_score, final_plan_dims = fb_plan, fb_plan_score, fb_plan_dims
                final_reward = fb_reward
                break
            current_error, current_code = fb_error, fb_code

    record = {
        "task_id": task_id, "task_prompt": task_prompt, "func_name": func_name,
        "prompt_text": chat_input, "raw_output": raw_output,
        "plan": plan, "code": code, "plan_score": plan_score, "plan_dims": plan_dims,
        "test_pass": 1 if passed else 0, "error": "" if passed else error,
        "reward": reward, "feedback_attempts": fb_attempts,
        "final_test_pass": 1 if final_passed else 0,
        "final_code": final_code, "final_plan": final_plan,
        "final_plan_score": final_plan_score, "final_plan_dims": final_plan_dims,
        "final_reward": final_reward,
        "recovered_by_feedback": int(not passed and final_passed),
        "has_plan": bool(plan.strip()),
        "prompt_tokens": prompt_len, "response_tokens": n_new_tok, "gen_time_s": gen_time,
    }
    log_generation(raw_fh, record)
    return record


# ── Dataset-level evaluation ──────────────────────────────────────────────────
def evaluate_all(model, tokenizer, rows, log_file, raw_fh,
                 use_judge, alpha, beta, use_feedback, max_feedback, start_from=0):
    total = len(rows) + start_from
    log.info(f"\n{'='*60}\nEVALUATING: HumanEval+ ({len(rows)} tasks)\n{'='*60}")
    results = []

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix   = f"_from{start_from}" if start_from > 0 else ""
    out_path = os.path.join(RESULTS_DIR, f"eval_humaneval{suffix}.jsonl")

    with open(out_path, "w") as eval_fh:
        for i, row in enumerate(rows):
            result = evaluate_task(model, tokenizer, row, raw_fh,
                                   use_judge, alpha, beta, use_feedback, max_feedback)
            results.append(result)
            eval_fh.write(json.dumps(result) + "\n")
            eval_fh.flush()

            status    = "PASS ✓" if result["test_pass"] else "FAIL ✗"
            fb_status = ""
            if result["feedback_attempts"]:
                fb_status = "→ FIXED ✓" if result["recovered_by_feedback"] else "→ still FAIL"
            ps  = f"{result['plan_score']:.2f}" if result["plan_score"] >= 0 else "N/A"
            line = (f"[{start_from+i+1:3d}/{total}] {status} {fb_status:<12}| "
                    f"plan={'Y' if result['has_plan'] else 'N'} "
                    f"score={ps} R={result['reward']:.2f} | "
                    f"{result['response_tokens']} tok | "
                    f"{result['gen_time_s']:.1f}s | "
                    f"{result['func_name']}")
            log.info(line)
            log_file.write(line + "\n")
            log_file.flush()

    n              = len(results)
    n_passed_1     = sum(r["test_pass"]            for r in results)
    n_passed_final = sum(r["final_test_pass"]       for r in results)
    n_recovered    = sum(r["recovered_by_feedback"] for r in results)
    n_had_fb       = sum(1 for r in results if r["feedback_attempts"])
    scored         = [r for r in results if r["plan_score"] >= 0]
    avg_ps         = sum(r["plan_score"] for r in scored) / len(scored) if scored else float("nan")

    log.info(f"\nRESULTS: HumanEval+")
    log.info(f"  pass@1  (attempt 1)     : {n_passed_1}/{n} ({100*n_passed_1/n:.1f}%)")
    log.info(f"  pass@1  (after feedback): {n_passed_final}/{n} ({100*n_passed_final/n:.1f}%)")
    if use_feedback:
        log.info(f"  tasks entering feedback : {n_had_fb}")
        log.info(f"  recovered by feedback   : {n_recovered} "
                 f"({100*n_recovered/n_had_fb:.1f}% of feedback tasks)"
                 if n_had_fb else "  recovered by feedback   : 0")
    log.info(f"  avg plan_score: {avg_ps:.3f}" if scored else "  avg plan_score: N/A")
    log.info(f"  has_plan rate : {sum(1 for r in results if r['has_plan'])/n:.1%}")
    log.info(f"  Saved: {out_path}")

    return {
        "dataset": "humaneval_plus", "num_tasks": n,
        "passed_attempt1": n_passed_1, "passed_final": n_passed_final,
        "recovered": n_recovered,
        "pass_at_1": n_passed_1 / n if n else 0,
        "pass_at_1_final": n_passed_final / n if n else 0,
        "feedback_recovery_rate": n_recovered / n_had_fb if n_had_fb else 0,
        "avg_plan_score": avg_ps,
        "has_plan_rate": sum(1 for r in results if r["has_plan"]) / n if n else 0,
        "judge_mode": "gpt-4o-mini" if use_judge else "rule-based",
        "alpha": alpha, "beta": beta, "out_path": out_path,
    }


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(checkpoint_path):
    log.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
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


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",   required=True)
    ap.add_argument("--alpha",        type=float, default=0.5)
    ap.add_argument("--beta",         type=float, default=1.0)
    ap.add_argument("--no-judge",     action="store_true")
    ap.add_argument("--no-feedback",  action="store_true")
    ap.add_argument("--max-feedback", type=int, default=1)
    ap.add_argument("--start-from",   type=int, default=0)
    args = ap.parse_args()

    use_judge = not args.no_judge
    if use_judge and not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set. Use --no-judge or set the env variable.")
        return

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_name = os.path.basename(args.checkpoint.rstrip("/"))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f"eval_{step_name}_{ts}.log")
    raw_path = os.path.join(RESULTS_DIR, f"raw_{step_name}_{ts}.jsonl")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    mode = "gpt-4o-mini" if use_judge else "rule-based (no GPT)"
    log.info(f"[HumanEval+] checkpoint={args.checkpoint}")
    log.info(f"alpha={args.alpha} beta={args.beta} judge={mode} "
             f"feedback={'disabled' if args.no_feedback else f'on (max={args.max_feedback})'} "
             f"start_from={args.start_from}")
    log.info(f"raw generations → {raw_path}")

    model, tokenizer = load_model(args.checkpoint)
    log.info(f"device: {next(model.parameters()).device}")

    df   = pd.read_parquet(DATA_PATH)
    rows = df.to_dict("records")[args.start_from:]
    log.info(f"Loaded {len(rows)} tasks (start_from={args.start_from})")

    with open(log_path, "a") as lf, open(raw_path, "w") as rf:
        result = evaluate_all(
            model, tokenizer, rows, lf, rf,
            use_judge, args.alpha, args.beta,
            use_feedback=not args.no_feedback,
            max_feedback=args.max_feedback,
            start_from=args.start_from)

    sp = os.path.join(RESULTS_DIR, f"summary_{step_name}_{ts}.json")
    with open(sp, "w") as f: json.dump(result, f, indent=2)
    log.info(f"Summary → {sp}")
    log.info(f"Raw     → {raw_path}")


if __name__ == "__main__":
    main()