#!/usr/bin/env python3
"""
plot_h1.py — H1 analysis: Dual rewards > code-only RLVR on pass@1

Usage:
  python plot_h1.py --out plots_h1/
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE    = "#2563EB"
ORANGE  = "#EA580C"
GREEN   = "#16A34A"
PURPLE  = "#7C3AED"
GRAY    = "#9CA3AF"
DARK    = "#1F2937"
LIGHT   = "#F3F4F6"
GRID    = "#E5E7EB"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#D1D5DB",
    "axes.labelcolor":   DARK,
    "axes.titlecolor":   DARK,
    "xtick.color":       DARK,
    "ytick.color":       DARK,
    "grid.color":        GRID,
    "grid.linewidth":    0.8,
    "text.color":        DARK,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
})

# ── Data ──────────────────────────────────────────────────────────────────────
MODELS = [
    "Base Model",
    "Plan-No-Reward\n(code-only proxy)",
    "Forced Plan\n(dual reward)",
    "Optional Plan\n(dual reward)",
]

MBPP   = [57.6, 57.4, 60.6, 61.8]
HEPLUS = [72.6, 73.2, 73.8, 75.0]

COLORS = [GRAY, ORANGE, BLUE, GREEN]

# Dual reward threshold lines
MBPP_DUAL_MIN   = 60.6
HEPLUS_DUAL_MIN = 73.8


def add_labels(ax, bars, values, baseline):
    for bar, val in zip(bars, values):
        delta = val - baseline
        sign  = "+" if delta >= 0 else ""
        label = f"{val:.1f}%\n({sign}{delta:.1f}%)" if delta != 0 else f"{val:.1f}%\n(baseline)"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                label, ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color=DARK)


def style_ax(ax, ylabel="Pass@1 (%)", ylim=(50, 80)):
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.grid(True, axis='y', alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=9)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"{path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots_h1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    x = np.arange(len(MODELS))
    w = 0.55

    # ── 1. MBPP pass@1 bar chart ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(x, MBPP, width=w, color=COLORS, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    # Baseline line
    ax.axhline(MBPP[0], color=GRAY, linestyle='--', linewidth=1.2,
               alpha=0.7, label=f'Base model ({MBPP[0]}%)')

    # Dual reward zone
    ax.axhspan(MBPP_DUAL_MIN, max(MBPP)+2, alpha=0.06, color=GREEN,
               label='Dual reward range')

    add_labels(ax, bars, MBPP, MBPP[0])
    ax.set_title("H1: MBPP pass@1 — Dual Reward vs Code-Only RLVR")
    style_ax(ax, ylim=(53, 68))
    ax.legend(fontsize=9, framealpha=0.8)

    # Annotation bracket for dual reward models
    ax.annotate('', xy=(2.28, 62.5), xytext=(3.28, 62.5),
                arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.5))
    ax.text(2.78, 63.0, 'Dual reward\nmodels', ha='center',
            fontsize=8, color=GREEN, fontweight='bold')

    save(fig, args.out, "1_mbpp_pass1.png")

    # ── 2. HumanEval+ pass@1 bar chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(x, HEPLUS, width=w, color=COLORS, alpha=0.85,
                  edgecolor='white', linewidth=1.5)

    ax.axhline(HEPLUS[0], color=GRAY, linestyle='--', linewidth=1.2,
               alpha=0.7, label=f'Base model ({HEPLUS[0]}%)')
    ax.axhspan(HEPLUS_DUAL_MIN, max(HEPLUS)+2, alpha=0.06, color=GREEN,
               label='Dual reward range')

    add_labels(ax, bars, HEPLUS, HEPLUS[0])
    ax.set_title("H1: HumanEval+ pass@1 — Dual Reward vs Code-Only RLVR")
    style_ax(ax, ylim=(68, 80))
    ax.legend(fontsize=9, framealpha=0.8)

    ax.annotate('', xy=(2.28, 76.5), xytext=(3.28, 76.5),
                arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.5))
    ax.text(2.78, 77.0, 'Dual reward\nmodels', ha='center',
            fontsize=8, color=GREEN, fontweight='bold')

    save(fig, args.out, "2_humaneval_pass1.png")

    # ── 3. Side-by-side: MBPP + HumanEval+ ───────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title, ylim, baseline_val in [
        (ax1, MBPP,   "MBPP Test Set",    (53, 68), MBPP[0]),
        (ax2, HEPLUS, "HumanEval+ Test",  (68, 80), HEPLUS[0]),
    ]:
        bars = ax.bar(x, data, width=w, color=COLORS, alpha=0.85,
                      edgecolor='white', linewidth=1.5)
        ax.axhline(baseline_val, color=GRAY, linestyle='--',
                   linewidth=1.2, alpha=0.7)
        add_labels(ax, bars, data, baseline_val)
        ax.set_title(title, fontsize=12, fontweight='bold')
        style_ax(ax, ylim=ylim)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, fontsize=8.5)

    # Shared legend
    patches = [
        mpatches.Patch(color=GRAY,   alpha=0.85, label='Base Model'),
        mpatches.Patch(color=ORANGE, alpha=0.85, label='Plan-No-Reward (code-only proxy)'),
        mpatches.Patch(color=BLUE,   alpha=0.85, label='Forced Plan (dual reward)'),
        mpatches.Patch(color=GREEN,  alpha=0.85, label='Optional Plan (dual reward)'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("H1: Dual Reward vs Code-Only RLVR — pass@1 Comparison",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, args.out, "3_combined_pass1.png")

    # ── 4. Gain over base model ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    mbpp_gains   = [v - MBPP[0]   for v in MBPP]
    heplus_gains = [v - HEPLUS[0] for v in HEPLUS]

    x2 = np.arange(len(MODELS))
    w2 = 0.35
    b1 = ax.bar(x2 - w2/2, mbpp_gains,   w2, label='MBPP',      color=BLUE,
                alpha=0.85, edgecolor='white')
    b2 = ax.bar(x2 + w2/2, heplus_gains, w2, label='HumanEval+', color=GREEN,
                alpha=0.85, edgecolor='white')

    for bar, val in zip(list(b1)+list(b2), mbpp_gains+heplus_gains):
        sign = "+" if val >= 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.05 if val >= 0 else -0.25),
                f'{sign}{val:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.axhline(0, color=DARK, linewidth=1.0)
    ax.set_title("H1: Gain over Base Model (Δ pass@1)")
    ax.set_ylabel("Δ pass@1 (%)", fontsize=10)
    ax.set_xticks(x2)
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, axis='y', alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-1.5, 6)

    save(fig, args.out, "4_gain_over_base.png")

    print(f"\nAll H1 plots saved to: {args.out}/")
    print("\nH1 Summary:")
    print(f"  Plan-no-reward vs base: {MBPP[1]-MBPP[0]:+.1f}% MBPP, {HEPLUS[1]-HEPLUS[0]:+.1f}% HumanEval+")
    print(f"  Forced Plan vs base:    {MBPP[2]-MBPP[0]:+.1f}% MBPP, {HEPLUS[2]-HEPLUS[0]:+.1f}% HumanEval+")
    print(f"  Optional Plan vs base:  {MBPP[3]-MBPP[0]:+.1f}% MBPP, {HEPLUS[3]-HEPLUS[0]:+.1f}% HumanEval+")
    print(f"  Dual reward vs code-only (MBPP):     {min(MBPP[2],MBPP[3])-MBPP[1]:+.1f}% to {max(MBPP[2],MBPP[3])-MBPP[1]:+.1f}%")
    print(f"  Dual reward vs code-only (HumanEval+): {min(HEPLUS[2],HEPLUS[3])-HEPLUS[1]:+.1f}% to {max(HEPLUS[2],HEPLUS[3])-HEPLUS[1]:+.1f}%")


if __name__ == "__main__":
    main()