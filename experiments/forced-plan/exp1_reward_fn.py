#!/usr/bin/env python3
"""
exp1_reward_fn.py  —  Experiment 1 reward function
Changes vs baseline reward_fn.py:
  - JUDGE_PROMPT updated: judges plan against ALL test cases (not just one),
    and explicitly checks whether the function name matches what's required.
  - Everything else identical to baseline for fair comparison.
"""

import json
import os
import re
import subprocess
import tempfile
import time

import wandb

# ─── Config ────────────────────────────────────────────────
ALPHA = 0.5
BETA  = 1.0

GPT_MODEL      = "gpt-4o-mini"
MAX_RETRIES    = 3
RETRY_DELAY    = 2

SANDBOX_TIMEOUT = 10

LOG_FILE  = "results/exp1_training_responses.jsonl"
DIAG_FILE = "results/exp1_reward_signal_diagnostics.jsonl"

os.makedirs("results", exist_ok=True)

_step_counter      = 0
_wandb_initialized = False
_recent_pass_rates = []


# ═══════════════════════════════════════════════════════════
# EXTRACTION  (identical to baseline)
# ═══════════════════════════════════════════════════════════

def extract_plan_and_code(raw_output: str):
    plan = ""
    code = ""

    plan_match = re.search(r'<plan>\s*(.*?)\s*</plan>', raw_output, re.DOTALL)
    if plan_match:
        plan = plan_match.group(1).strip()

    code_match = re.search(r'<code>\s*(.*?)\s*</code>', raw_output, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        code_match2 = re.search(r'<code>\s*(.*?)\s*```', raw_output, re.DOTALL)
        if code_match2:
            code = code_match2.group(1).strip()
        elif re.search(r'```python\s*\n(.*?)```', raw_output, re.DOTALL):
            py_blocks = re.findall(r'```python\s*\n(.*?)```', raw_output, re.DOTALL)
            code = py_blocks[-1].strip()
        elif re.findall(r'```\s*\n(.*?)```', raw_output, re.DOTALL):
            blocks = re.findall(r'```\s*\n(.*?)```', raw_output, re.DOTALL)
            code = blocks[-1].strip()
        else:
            after_plan = raw_output[plan_match.end():] if plan_match else raw_output
            lines = after_plan.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith(("def ", "class ", "import ", "from ")):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            if code_lines:
                code = "\n".join(code_lines).strip()

    if not code.strip():
        _log_extraction_failure(raw_output)

    return plan, code


def _log_extraction_failure(raw_output: str):
    failure_log = "results/exp1_extraction_failures.jsonl"
    try:
        with open(failure_log, "a") as f:
            f.write(json.dumps({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "step": _step_counter,
                "raw_output_snippet": raw_output[:500],
                "has_plan_tag": "<plan>" in raw_output,
                "has_code_tag": "<code>" in raw_output,
                "has_backticks": "```" in raw_output,
            }) + "\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════
# GPT JUDGE  — updated for Exp 1: judges against all tests
# ═══════════════════════════════════════════════════════════

JUDGE_SYSTEM = (
    "You are a strict but fair programming plan evaluator. "
    "Respond with only a JSON object containing scores."
)

JUDGE_PROMPT = """You are evaluating the quality of a programming plan.

## Task
{task_prompt}

## Required Function Name
{func_name}

## All Test Cases (the solution must pass all of these)
{all_tests}

## Plan to Evaluate
{plan_text}

Rate this plan on 4 dimensions, each from 0.0 to 1.0:
1. **Decomposition** (0-1): Does the plan break the problem into clear, logical steps?
2. **Edge Cases** (0-1): Does the plan address the boundary conditions shown in the test cases?
3. **Approach** (0-1): Is the algorithmic approach correct and appropriate for ALL test cases?
4. **Clarity** (0-1): Is the plan specific enough to guide implementation using the correct function name?

Respond with ONLY a JSON object, no other text:
{{"decomposition": 0.0, "edge_cases": 0.0, "approach": 0.0, "clarity": 0.0}}"""

_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def score_plan(task_prompt: str, plan_text: str,
               func_name: str = "", all_tests: str = "") -> float:
    if not plan_text.strip():
        return 0.0

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
                        all_tests=all_tests or "N/A",
                        plan_text=plan_text,
                    )},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            text = response.choices[0].message.content
            json_match = re.search(r'\{[^{}]+\}', text)
            if json_match:
                scores = json.loads(json_match.group())
                dims = ["decomposition", "edge_cases", "approach", "clarity"]
                vals = [max(0.0, min(1.0, float(scores.get(d, 0.0)))) for d in dims]
                return sum(vals) / len(vals)
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
            print(f"  [GPTJudge] Error (attempt {attempt+1}): {e}")

    return 0.0


# ═══════════════════════════════════════════════════════════
# CODE EXECUTION SANDBOX  (identical to baseline)
# ═══════════════════════════════════════════════════════════

def run_tests(code: str, test_list_json: str) -> float:
    if not code.strip():
        return 0.0
    try:
        test_list = json.loads(test_list_json)
    except (json.JSONDecodeError, TypeError):
        test_list = []
    if not test_list:
        return 0.0

    test_code = "\n".join(test_list)
    full_code  = f"{code}\n\n{test_code}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True, text=True, timeout=SANDBOX_TIMEOUT,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════
# LOGGING  (identical to baseline, exp1 filenames)
# ═══════════════════════════════════════════════════════════

def log_response(step, task_prompt, full_output, plan, code,
                 plan_score, test_pass, total_reward):
    record = {
        "step": step,
        "task_prompt": task_prompt[:200],
        "full_output": full_output[:1000],
        "plan": plan[:500],
        "code": code[:800],
        "plan_score": plan_score,
        "test_pass": test_pass,
        "total_reward": total_reward,
        "extraction_ok": bool(code.strip()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def log_reward_signal(step, pass_rate, plan_score, total_reward, extraction_ok):
    global _recent_pass_rates
    _recent_pass_rates.append(pass_rate)
    if len(_recent_pass_rates) > 200:
        _recent_pass_rates = _recent_pass_rates[-200:]

    all_zero = sum(1 for p in _recent_pass_rates if p == 0.0)
    all_one  = sum(1 for p in _recent_pass_rates if p == 1.0)
    n = len(_recent_pass_rates)
    dead_signal_frac = (all_zero + all_one) / n if n > 0 else 0

    record = {
        "step": step,
        "pass_rate": pass_rate,
        "plan_score": plan_score,
        "total_reward": total_reward,
        "extraction_ok": extraction_ok,
        "rolling_dead_signal_frac": round(dead_signal_frac, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if dead_signal_frac > 0.8 and n >= 20:
        record["WARNING"] = (
            f"GRPO signal may be dead: {100*dead_signal_frac:.0f}% of last {n} "
            f"rollouts are all-0 or all-1."
        )
    try:
        with open(DIAG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

    return dead_signal_frac


# ═══════════════════════════════════════════════════════════
# MAIN REWARD FUNCTION — called by VERL
# ═══════════════════════════════════════════════════════════

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    global _step_counter, _wandb_initialized
    _step_counter += 1

    plan, code    = extract_plan_and_code(solution_str)
    extraction_ok = bool(code.strip())

    task_prompt    = ""
    test_list_json = ground_truth
    func_name      = ""
    all_tests      = ""

    if extra_info:
        task_prompt    = extra_info.get("task_prompt_raw", "")
        func_name      = extra_info.get("func_name", "")
        if "test_list" in extra_info:
            test_list_json = extra_info["test_list"]
            try:
                tests    = json.loads(test_list_json)
                all_tests = "\n".join(tests)
            except Exception:
                pass

    # Plan scoring — pass func_name and all_tests to judge
    plan_score = (score_plan(task_prompt, plan,
                             func_name=func_name, all_tests=all_tests)
                  if plan.strip() else 0.0)

    test_pass      = run_tests(code, test_list_json)
    plan_component = ALPHA * plan_score * test_pass
    code_component = BETA  * test_pass
    total_reward   = plan_component + code_component

    log_response(_step_counter, task_prompt, solution_str,
                 plan, code, plan_score, test_pass, total_reward)

    dead_signal_frac = log_reward_signal(
        _step_counter, test_pass, plan_score, total_reward, extraction_ok)

    try:
        if not _wandb_initialized:
            if wandb.run is not None:
                _wandb_initialized = True
        if _wandb_initialized and wandb.run is not None:
            if _step_counter % 5 == 0:
                wandb.log({
                    "reward/total":          total_reward,
                    "reward/plan_score":     plan_score,
                    "reward/test_pass":      test_pass,
                    "reward/plan_component": plan_component,
                    "reward/code_component": code_component,
                    "planning/has_plan":     int(bool(plan.strip())),
                    "planning/plan_length":  len(plan),
                    "diag/extraction_ok":    int(extraction_ok),
                    "diag/dead_signal_frac": dead_signal_frac,
                })
    except Exception:
        pass

    return total_reward
