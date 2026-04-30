"""
Plan-then-Code: Custom Reward Function for VERL
=================================================
Logs all generated responses + reward components to a JSONL file.
Includes retry logic for GPT rate limits.
"""

import json
import os
import re
import subprocess
import tempfile
import time
from typing import Dict

import wandb


# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════

ALPHA = 0.0     # for plan_no_reward
BETA = 1.0
GPT_MODEL = "gpt-4o-mini"
SANDBOX_TIMEOUT = 10
MAX_RETRIES = 1
RETRY_DELAY = 0.0 

# Logging
LOG_DIR = "results"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "training_responses.jsonl")

_step_counter = 0
_wandb_initialized = False


# ═══════════════════════════════════════════════════
# PLAN / CODE EXTRACTION
# ═══════════════════════════════════════════════════

def extract_plan_and_code(raw_output: str):
    plan = ""
    code = ""

    plan_match = re.search(r'<plan>\s*(.*?)\s*</plan>', raw_output, re.DOTALL)
    if plan_match:
        plan = plan_match.group(1).strip()

    # Strategy 1: <code>...</code>
    code_match = re.search(r'<code>\s*(.*?)\s*</code>', raw_output, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # Strategy 2: <code>...``` (model opens with <code> but closes with ```)
        code_match2 = re.search(r'<code>\s*(.*?)\s*```', raw_output, re.DOTALL)
        if code_match2:
            code = code_match2.group(1).strip()
        elif re.search(r'```python\s*\n(.*?)```', raw_output, re.DOTALL):
            # Strategy 3: ```python...```
            py_blocks = re.findall(r'```python\s*\n(.*?)```', raw_output, re.DOTALL)
            code = py_blocks[-1].strip()
        elif re.findall(r'```\s*\n(.*?)```', raw_output, re.DOTALL):
            # Strategy 4: ```...```
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

    return plan, code


# ═══════════════════════════════════════════════════
# GPT JUDGE WITH RETRY
# ═══════════════════════════════════════════════════

JUDGE_SYSTEM = (
    "You are a strict but fair programming plan evaluator. "
    "Respond with only a JSON object containing scores."
)

JUDGE_PROMPT = """You are evaluating the quality of a programming plan.

## Task
{task_prompt}

## Plan to Evaluate
{plan_text}

Rate this plan on 4 dimensions, each from 0.0 to 1.0:
1. **Decomposition** (0-1): Does the plan break the problem into clear, logical steps?
2. **Edge Cases** (0-1): Does the plan identify boundary conditions and tricky inputs?
3. **Approach** (0-1): Is the algorithmic approach correct and appropriate?
4. **Clarity** (0-1): Is the plan clearly written and specific enough to guide implementation?

Respond with ONLY a JSON object, no other text:
{{"decomposition": 0.0, "edge_cases": 0.0, "approach": 0.0, "clarity": 0.0}}"""

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def score_plan(task_prompt: str, plan_text: str) -> float:
    if not plan_text.strip():
        return 0.0

    for attempt in range(MAX_RETRIES):
        try:
            client = _get_openai_client()
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": JUDGE_PROMPT.format(
                        task_prompt=task_prompt, plan_text=plan_text
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
                wait = RETRY_DELAY * (2 ** attempt)
                time.sleep(wait)
                continue
            print(f"  [GPTJudge] Error (attempt {attempt+1}): {e}")

    return 0.0


# ═══════════════════════════════════════════════════
# CODE EXECUTION SANDBOX
# ═══════════════════════════════════════════════════

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
    full_code = f"{code}\n\n{test_code}"

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


# ═══════════════════════════════════════════════════
# RESPONSE LOGGING
# ═══════════════════════════════════════════════════

def log_response(
    step: int,
    task_prompt: str,
    full_output: str,
    plan: str,
    code: str,
    plan_score: float,
    test_pass: float,
    total_reward: float,
):
    """Append one response record to the JSONL log file."""
    record = {
        "step": step,
        "task_prompt": task_prompt[:200],
        "full_output": full_output,
        "plan": plan,
        "code": code,
        "plan_score": plan_score,
        "test_pass": test_pass,
        "total_reward": total_reward,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════
# MAIN REWARD FUNCTION — called by VERL
# ═══════════════════════════════════════════════════

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Custom reward function for VERL GRPO training.
    Logs every response to results/training_responses.jsonl
    """
    global _step_counter, _wandb_initialized

    _step_counter += 1

    # ─── Extract plan and code ───
    plan, code = extract_plan_and_code(solution_str)

    # ─── Get task prompt and test list ───
    task_prompt = ""
    test_list_json = ground_truth

    if extra_info:
        task_prompt = extra_info.get("task_prompt_raw", "")
        if "test_list" in extra_info:
            test_list_json = extra_info["test_list"]

    # ─── Score plan with GPT judge (with retry) ───
    plan_score = score_plan(task_prompt, plan) if plan.strip() else 0.0

    # ─── Execute code against unit tests ───
    test_pass = run_tests(code, test_list_json)

    # ─── Compute dual reward ───
    plan_component = ALPHA * plan_score * test_pass
    code_component = BETA * test_pass
    total_reward = plan_component + code_component

    # ─── Log response to file ───
    log_response(
        step=_step_counter,
        task_prompt=task_prompt,
        full_output=solution_str,
        plan=plan,
        code=code,
        plan_score=plan_score,
        test_pass=test_pass,
        total_reward=total_reward,
    )

    # ─── W&B logging ───
    try:
        if not _wandb_initialized:
            if wandb.run is not None:
                _wandb_initialized = True

        if _wandb_initialized and wandb.run is not None:
            if _step_counter % 10 == 0:
                wandb.log({
                    "reward/total": total_reward,
                    "reward/plan_score": plan_score,
                    "reward/test_pass": test_pass,
                    "reward/plan_component": plan_component,
                    "reward/code_component": code_component,
                    "planning/has_plan": int(bool(plan.strip())),
                    "planning/plan_length": len(plan),
                })
    except Exception:
        pass

    return total_reward