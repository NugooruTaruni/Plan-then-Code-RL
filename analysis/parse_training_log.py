#!/usr/bin/env python3
"""
parse_training_log.py — Parse VERL training log and extract metrics to CSV

Usage:
  python parse_training_log.py --log grpo_optional_training_full.log
  python parse_training_log.py --log grpo_optional_training_full.log --out my_output.csv
"""
import re
import csv
import argparse
import os

KEY_METRICS = {
    "val_acc":                r"val-core/mbpp[^/]*/acc/mean@1:(?:np\.float64\()?([\d.]+)\)?",
    "val_reward":             r"val-aux/mbpp[^/]*/reward/mean@1:(?:np\.float64\()?([\d.]+)\)?",
    "critic_score_mean":      r"critic/score/mean:([\d.]+)",
    "critic_rewards_mean":    r"critic/rewards/mean:([\d.]+)",
    "critic_advantages_mean": r"critic/advantages/mean:(?:np\.float64\()?([-\d.e]+)\)?",
    "actor_entropy":          r"actor/entropy:([\d.]+)",
    "actor_grad_norm":        r"actor/grad_norm:(?:np\.float64\()?([\d.e]+)\)?",
    "response_length_mean":   r"response_length/mean:([\d.]+)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Training log file path")
    ap.add_argument("--out", default=None,  help="Output CSV path (default: <log_name>.csv)")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"ERROR: File not found: {args.log}")
        return

    out_path = args.out or args.log.replace(".log", ".csv").replace(".txt", ".csv") + \
               ("" if args.out else "")
    if args.out is None:
        base = os.path.splitext(args.log)[0]
        out_path = base + "_curves.csv"

    print(f"Parsing: {args.log}")

    with open(args.log, errors="replace") as f:
        content = f.read()

    results = []
    seen    = set()

    for line in content.split("\n"):
        m = re.search(r"\bstep:(\d+)\b", line)
        if not m:
            continue
        step = int(m.group(1))
        if step in seen:
            continue
        seen.add(step)

        row = {"training/global_step": step}
        for metric, pattern in KEY_METRICS.items():
            match = re.search(pattern, line)
            if match:
                v = match.group(1).replace("np.float64(", "").replace(")", "")
                try:
                    row[metric] = float(v)
                except ValueError:
                    row[metric] = None
            else:
                row[metric] = None
        results.append(row)

    results.sort(key=lambda x: x["training/global_step"])

    # Summary
    val_rows = [r for r in results if r.get("val_acc") is not None]
    print(f"  Total steps:       {len(results)}")
    print(f"  Step range:        {results[0]['training/global_step']} - {results[-1]['training/global_step']}")
    print(f"  Val checkpoints:   {len(val_rows)}")

    if val_rows:
        best = max(val_rows, key=lambda r: r["val_acc"])
        print(f"  Best val acc:      {best['val_acc']*100:.1f}% at step {best['training/global_step']:.0f}")

    # Save CSV
    fieldnames = ["training/global_step"] + list(KEY_METRICS.keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()