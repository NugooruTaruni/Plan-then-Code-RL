"""
fewshot_prompts.py — Few-shot plan-then-code prompts for the base model (no RL).

Baseline: does prompting Qwen2.5-Coder-3B-Instruct to plan first (3-shot) match
what GRPO training gives the Optional Plan model?

Shots are MBPP tasks 2, 3, 4 from the MBPP *prompt* split (the standard few-shot
examples from Austin et al., 2021) — disjoint from the 500 test tasks (11-510).
Each shot is a (user, assistant) chat turn; the assistant answer is a short plan in
<plan></plan> followed by the solution in <code></code>.
"""

SYSTEM = (
    "You are an expert Python programmer. For every task, first write a short "
    "step-by-step plan inside <plan></plan> tags, then write your solution inside "
    "<code></code> tags. Your code must follow your plan."
)

# (task text, first test, plan, code) — MBPP prompt split, task_ids 2, 3, 4
SHOTS = [
    (
        "Write a function to find the similar elements from the given two tuple lists.",
        "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
        "1. Convert both tuples to sets.\n"
        "2. Take the set intersection to get the shared elements.\n"
        "3. Sort the result so the output order is deterministic and return it as a tuple.",
        "def similar_elements(test_tup1, test_tup2):\n"
        "    return tuple(sorted(set(test_tup1) & set(test_tup2)))",
    ),
    (
        "Write a python function to identify non-prime numbers.",
        "assert is_not_prime(2) == False",
        "1. Numbers below 2 are not prime, so return True for them.\n"
        "2. Check divisors from 2 up to sqrt(n).\n"
        "3. If any divisor divides n, n is not prime: return True.\n"
        "4. Otherwise return False.",
        "import math\n"
        "def is_not_prime(n):\n"
        "    if n < 2:\n"
        "        return True\n"
        "    for i in range(2, int(math.sqrt(n)) + 1):\n"
        "        if n % i == 0:\n"
        "            return True\n"
        "    return False",
    ),
    (
        "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
        "1. Use heapq.nlargest to get the n largest numbers.\n"
        "2. nlargest already returns them in descending order, so return the list directly.",
        "import heapq as hq\n"
        "def heap_queue_largest(nums, n):\n"
        "    return hq.nlargest(n, nums)",
    ),
]

# HumanEval-style versions of the same three tasks (signature + docstring)
HE_SHOTS = [
    (
        'def similar_elements(test_tup1: tuple, test_tup2: tuple) -> tuple:\n'
        '    """Return the elements common to both tuples, sorted ascending.\n'
        '    >>> similar_elements((3, 4, 5, 6), (5, 7, 4, 10))\n'
        '    (4, 5)\n'
        '    """\n',
        SHOTS[0][2], SHOTS[0][3],
    ),
    (
        'def is_not_prime(n: int) -> bool:\n'
        '    """Return True if n is not a prime number.\n'
        '    >>> is_not_prime(2)\n'
        '    False\n'
        '    >>> is_not_prime(10)\n'
        '    True\n'
        '    """\n',
        SHOTS[1][2], SHOTS[1][3],
    ),
    (
        'def heap_queue_largest(nums: list, n: int) -> list:\n'
        '    """Return the n largest numbers in nums, in descending order.\n'
        '    >>> heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3)\n'
        '    [85, 75, 65]\n'
        '    """\n',
        SHOTS[2][2], SHOTS[2][3],
    ),
]


def mbpp_user(task, test):
    return f"Task: {task}\nYour code should pass this test:\n{test}"


def he_user(prompt):
    return f"Complete the following Python function:\n{prompt}"


def answer(plan, code):
    return f"<plan>\n{plan}\n</plan>\n\n<code>\n{code}\n</code>"


def shot_messages(is_mbpp):
    """System prompt + the 3 example (user, assistant) turns."""
    msgs = [{"role": "system", "content": SYSTEM}]
    if is_mbpp:
        for task, test, plan, code in SHOTS:
            msgs += [{"role": "user", "content": mbpp_user(task, test)},
                     {"role": "assistant", "content": answer(plan, code)}]
    else:
        for prompt, plan, code in HE_SHOTS:
            msgs += [{"role": "user", "content": he_user(prompt)},
                     {"role": "assistant", "content": answer(plan, code)}]
    return msgs
