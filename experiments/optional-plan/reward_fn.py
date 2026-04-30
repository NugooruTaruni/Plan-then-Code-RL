"""
reward_fn.py — Optional-Plan Reward (GPT-4o-mini Judge)
========================================================
R = 0.5 * plan_score * test_ratio + 1.0 * test_ratio
"""

import os, re, ast, json, subprocess, logging

logger = logging.getLogger(__name__)
ALPHA, BETA, MAX_REWARD = 0.5, 1.0, 1.6
CODE_TIMEOUT = 10
METADATA_PATH = os.environ.get("MBPP_METADATA_PATH", "data/mbpp_train_metadata.json")
_metadata_cache = None
_openai_client = None


def _load_metadata():
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache
    for path in [METADATA_PATH, "data/mbpp_train_metadata.json", "data/mbpp_test_metadata.json"]:
        if os.path.exists(path):
            with open(path) as f:
                ml = json.load(f)
            _metadata_cache = {i["task_description"]: {
                "test_code": i.get("test_code", ""),
                "test_list": i.get("test_list", []),
            } for i in ml}
            logger.info(f"Loaded {len(_metadata_cache)} tasks from {path}")
            return _metadata_cache
    _metadata_cache = {}
    return _metadata_cache


def _get_tests(ground_truth, extra_info):
    if ground_truth and isinstance(ground_truth, str) and ground_truth.strip():
        try:
            gt = json.loads(ground_truth)
            tl, tc = gt.get("test_list", []), gt.get("test_code", "")
            if tl or tc:
                return tl, tc
        except:
            pass
    if extra_info and isinstance(extra_info, dict):
        tl, tc = extra_info.get("test_list", []), extra_info.get("test_code", "")
        if tl or tc:
            return tl, tc
        td = extra_info.get("task_description", "")
        if td:
            meta = _load_metadata()
            info = meta.get(td)
            if info:
                return info.get("test_list", []), info.get("test_code", "")
    return [], ""


def _get_task_desc(ground_truth, extra_info):
    if ground_truth and isinstance(ground_truth, str):
        try:
            return json.loads(ground_truth).get("task_description", "")
        except:
            pass
    if extra_info and isinstance(extra_info, dict):
        return extra_info.get("task_description", "")
    return ""


def _get_openai():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        from openai import OpenAI
        _openai_client = OpenAI()
        logger.info("OpenAI client initialized")
        return _openai_client
    except ImportError:
        logger.error("pip install openai")
        return None


def _score_plan(task_desc, plan_text):
    if not plan_text.strip():
        return 0.0
    client = _get_openai()
    if client is None:
        return 0.5
    prompt = (
        "Rate this plan for solving a Python programming task.\n"
        "Respond with ONLY a number between 0.0 and 1.0.\n\n"
        "0.0 = irrelevant  0.3 = right topic, no steps  "
        "0.5 = vague  0.7 = good steps  1.0 = clear correct algorithm\n\n"
        f"Task: {task_desc}\n\nPlan: {plan_text}\n\nScore:"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16, temperature=0.0,
        )
        nums = re.findall(r"(\d+\.?\d*)", resp.choices[0].message.content.strip())
        if nums:
            return max(0.0, min(1.0, float(nums[0])))
    except Exception as e:
        logger.warning(f"GPT-4o-mini error: {e}")
    return 0.5


def _exec(code, test):
    if not code.strip() or not test.strip():
        return False
    try:
        r = subprocess.run(["python3", "-c", f"{code}\n\n{test}"],
                           capture_output=True, text=True, timeout=CODE_TIMEOUT)
        return r.returncode == 0
    except:
        return False


def _test_ratio(code, test_list, test_code):
    if not code.strip():
        return 0.0
    if test_code and _exec(code, test_code):
        return 1.0
    if not test_list:
        return 0.0
    passed = sum(1 for t in test_list if _exec(code, t))
    return passed / len(test_list)


def _extract_plan(text):
    m = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_code(text):
    m = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None, **kwargs):
    tl, tc = _get_tests(ground_truth, extra_info)
    if not tl and not tc:
        return 0.0
    task_desc = _get_task_desc(ground_truth, extra_info)
    plan = _extract_plan(solution_str)
    code = _extract_code(solution_str)
    tr = _test_ratio(code, tl, tc)
    ps = _score_plan(task_desc, plan) if plan and tr > 0 else 0.0
    reward = ALPHA * ps * tr + BETA * tr
    return min(reward, MAX_REWARD)
