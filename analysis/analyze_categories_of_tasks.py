#!/usr/bin/env python3
"""
analyze_categories_5cat.py — MBPP 5-category analysis with weighted regex.

Usage:
  python analyze_categories_5cat.py \
    --step140 eval_step140_mbpp_pass@1.jsonl \
    --base    raw_Qwen2.5-Coder-3B-Instruct.jsonl \
    --out     category_analysis_forced_plan.json
"""
import argparse, os, json, re
from collections import defaultdict

RULES = [
    ("Bitwise",         ["bit", "xor", "bitwise", "hexadecimal", "binary", "octal", "shift"], 1.5),
    ("Algorithms",      ["recursive", "recursion", "subsequence", "subarray", "dynamic programming",
                         "merge sort", "bubble sort", "pancake", "shell sort", "comb sort",
                         "counting sort", "search", "ascending", "descending", "sort"], 1.3),
    ("Math & Geometry", ["prime", "fibonacci", "factorial", "gcd", "lcm", "divisor", "modulo",
                         "power", "square", "cube", "arithmetic", "geometric", "series",
                         "sequence", "digit", "number", "sum", "count", "average", "product",
                         "remainder", "quotient", "minimum", "maximum", "largest", "smallest",
                         "area", "volume", "perimeter", "radius", "polygon", "coordinate",
                         "circle", "triangle", "sphere", "cone", "cylinder", "parabola",
                         "cuboid", "pentagon", "hexagon", "lateral", "diameter"], 1.2),
    ("String & Text",   ["string", "substring", "regex", "palindrome", "camel", "snake",
                         "ascii", "char", "character", "word", "pattern", "uppercase",
                         "lowercase", "vowel", "replace", "match", "letter"], 1.2),
    ("Collections",     ["list", "array", "sublist", "nested", "tuple",
                         "dictionary", "dict", "stack", "queue", "heap", "linked list"], 1.0),
]
FALLBACK = "Applied"


def get_category(prompt):
    p = prompt.lower()
    scores = defaultdict(float)
    for cat, kws, weight in RULES:
        for kw in kws:
            m = len(re.findall(rf"\b{re.escape(kw)}\b", p))
            if m: scores[cat] += m * weight
    return max(scores, key=scores.get) if scores else FALLBACK


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try: rows.append(json.loads(line))
                except: pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step140",  required=True)
    ap.add_argument("--base",     required=True)
    ap.add_argument("--out",      default=None)
    args = ap.parse_args()

    step140_rows = load_jsonl(args.step140)
    base_rows    = load_jsonl(args.base)
    base_map     = {r["task_id"]: r for r in base_rows}

    stats = defaultdict(lambda: {"n":0,"base":0,"s140":0})
    for r in step140_rows:
        tid = r["task_id"]
        if tid not in base_map: continue
        cat = get_category(r.get("task_prompt",""))
        stats[cat]["n"]    += 1
        stats[cat]["base"] += base_map[tid].get("test_pass",0)
        stats[cat]["s140"] += r.get("test_pass",0)

    cat_order = [c for c,_,_ in RULES] + [FALLBACK]
    results = []
    total_n = total_b = total_m = 0

    print(f"\n{'Category':<20} {'N':>5} {'Base':>8} {'Model':>8} {'Delta':>7}")
    print("─"*53)

    for cat in cat_order:
        s = stats.get(cat)
        if not s or s["n"]==0: continue
        n=s["n"]; b=s["base"]/n*100; m=s["s140"]/n*100; d=m-b
        total_n+=n; total_b+=s["base"]; total_m+=s["s140"]
        sign = "+" if d>=0 else ""
        print(f"{cat:<20} {n:>5} {b:>7.1f}% {m:>7.1f}% {sign}{d:.1f}%")
        results.append({"category":cat,"n":n,
                        "base_pass_rate":round(b,2),
                        "s140_pass_rate":round(m,2),
                        "delta":round(d,2)})

    b_ov = total_b/total_n*100; m_ov = total_m/total_n*100
    print(f"{'OVERALL':<20} {total_n:>5} {b_ov:>7.1f}% {m_ov:>7.1f}% {m_ov-b_ov:>+6.1f}%")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out,"w") as f:
            json.dump({"categories":results,
                       "overall":{"n":total_n,"base":round(b_ov,2),
                                  "s140":round(m_ov,2),"delta":round(m_ov-b_ov,2)}}, f, indent=2)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()