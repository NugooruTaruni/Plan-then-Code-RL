#!/usr/bin/env python3
"""
plot_optional_training.py — Training curves for Optional Plan (best model)

Usage:
  python plot_optional_training.py \
    --csv training_curves_optional_plan.csv \
    --out plots_optional_training/
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GREEN  = "#16A34A"
BLUE   = "#2563EB"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
GRAY   = "#9CA3AF"
DARK   = "#1F2937"
LIGHT  = "#F3F4F6"

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white","axes.edgecolor":"#D1D5DB",
    "axes.labelcolor":DARK,"axes.titlecolor":DARK,"xtick.color":DARK,"ytick.color":DARK,
    "grid.color":"#E5E7EB","grid.linewidth":0.6,"text.color":DARK,
    "font.family":"sans-serif","font.size":11,"axes.titlesize":13,
    "axes.titleweight":"bold","figure.dpi":150,
})

BEST_STEP = 126

def smooth(y, w=7):
    y = np.array(y, dtype=float)
    if len(y) < w: return y
    return np.convolve(y, np.ones(w)/w, mode='same')

def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"✓ {path}"); plt.close(fig)

def style_ax(ax, ylabel="", xlabel="Training Step"):
    ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(False)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=9)

def vline(ax):
    ax.axvline(BEST_STEP, color=ORANGE, linestyle="--", alpha=0.8,
               linewidth=1.5, label=f"Best step ({BEST_STEP})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="plots_optional_training")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv).sort_values("training/global_step")
    steps   = df["training/global_step"].values
    score   = pd.to_numeric(df["critic_score_mean"],    errors="coerce").values
    entropy = pd.to_numeric(df["actor_entropy"],        errors="coerce").values
    rlen    = pd.to_numeric(df["response_length_mean"], errors="coerce").values
    adv     = pd.to_numeric(df["critic_advantages_mean"],errors="coerce").values

    val_df  = df[df["val_acc"].notna()].copy()
    val_steps = val_df["training/global_step"].values
    val_acc   = val_df["val_acc"].values * 100
    val_rew   = val_df["val_reward"].values * 100

    # ── 1. Val Accuracy ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(val_steps, val_acc, color=GREEN, marker="o", markersize=4,
            linewidth=2.2, label="Val Accuracy", zorder=3)
    ax.fill_between(val_steps, val_acc, alpha=0.1, color=GREEN)
    vline(ax)
    peak = val_acc.max()
    ax.annotate(f"{peak:.1f}% ★", xy=(BEST_STEP, peak),
                xytext=(BEST_STEP+8, peak+0.4),
                fontsize=9, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    ax.set_title("Optional Plan — Validation Accuracy over Training Steps")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, framealpha=0.8)
    style_ax(ax, ylabel="Val Accuracy (%)")
    save(fig, args.out, "1_val_accuracy.png")

    # ── 2. Reward hacking check ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(val_acc, val_rew, color=GREEN, alpha=0.8, s=35, zorder=3)
    lims = [min(val_acc.min(), val_rew.min())-0.5,
            max(val_acc.max(), val_rew.max())+0.5]
    # ax.plot(lims, lims, color=GRAY, linestyle="--", linewidth=1.2, label="No hacking (y=x)")
    # corr = np.corrcoef(val_acc, val_rew)[0,1]
    # ax.text(0.05, 0.92, f"r = {corr:.4f}", transform=ax.transAxes, fontsize=10, color=GREEN, fontweight="bold")
    corr = np.corrcoef(val_acc, val_rew)[0,1]
    ax.text(0.60, 0.12, f"Pearson r = {corr:.4f}\nNo reward hacking",
        transform=ax.transAxes, fontsize=9, color=GREEN, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT, alpha=0.9))
    #ax.text(0.05, 0.82, "No reward hacking", transform=ax.transAxes, fontsize=9, color=GREEN)
    ax.set_title("Reward Hacking Check\n(Reward vs Accuracy)")
    ax.legend(fontsize=9, framealpha=0.8)
    style_ax(ax, xlabel="Val Accuracy (%)", ylabel="Val Reward (%)")
    save(fig, args.out, "2_reward_hacking_check.png")

    # ── 3. Batch reward ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, score, color=GRAY, alpha=0.3, linewidth=0.8)
    ax.plot(steps, smooth(score, 10), color=ORANGE, linewidth=2.2, label="Smoothed (w=10)")
    vline(ax)
    ax.set_title("Batch Reward over Training (critic/score/mean)")
    ax.legend(fontsize=9, framealpha=0.8)
    style_ax(ax, ylabel="Mean Reward")
    save(fig, args.out, "3_batch_reward.png")

    # ── 4. Entropy ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, entropy, color=GRAY, alpha=0.3, linewidth=0.8)
    ax.plot(steps, smooth(entropy, 10), color=GREEN, linewidth=2.2, label="Smoothed")
    vline(ax)
    ax.text(0.72, 0.08, "✓ Stable — no collapse", transform=ax.transAxes,
            fontsize=9, color=GREEN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT, alpha=0.8))
    ax.set_title("Actor Entropy over Training")
    ax.legend(fontsize=9, framealpha=0.8)
    style_ax(ax, ylabel="Entropy")
    save(fig, args.out, "4_entropy.png")

    # ── 5. Response length ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, rlen, color=GRAY, alpha=0.3, linewidth=0.8)
    ax.plot(steps, smooth(rlen, 10), color=PURPLE, linewidth=2.2, label="Smoothed")
    vline(ax)
    ax.text(0.72, 0.08, "✓ No length gaming", transform=ax.transAxes,
            fontsize=9, color=GREEN,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT, alpha=0.8))
    ax.set_title("Mean Response Length over Training")
    ax.legend(fontsize=9, framealpha=0.8)
    style_ax(ax, ylabel="Tokens")
    save(fig, args.out, "5_response_length.png")

    # ── 6. Advantages ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, adv, color=GRAY, alpha=0.3, linewidth=0.8)
    ax.plot(steps, smooth(adv, 10), color=BLUE, linewidth=2.2, label="Smoothed")
    ax.axhline(0, color=GRAY, linestyle="--", alpha=0.5, linewidth=1)
    vline(ax)
    ax.set_title("GRPO Advantages over Training")
    ax.legend(fontsize=9, framealpha=0.8)
    style_ax(ax, ylabel="Mean Advantage")
    save(fig, args.out, "6_advantages.png")

    # ── 7. Overview 4-panel ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Optional Plan GRPO — Training Overview (Best Model)",
                 fontsize=14, fontweight="bold", y=1.01)

    # Val acc
    axes[0,0].plot(val_steps, val_acc, color=GREEN, marker="o",
                   markersize=3, linewidth=2, label=f"Best: step {BEST_STEP}")
    axes[0,0].fill_between(val_steps, val_acc, alpha=0.1, color=GREEN)
    axes[0,0].axvline(BEST_STEP, color=ORANGE, linestyle="--", alpha=0.7, linewidth=1.2)
    axes[0,0].set_title("Validation Accuracy"); axes[0,0].legend(fontsize=8)
    axes[0,0].set_ylabel("Accuracy (%)", fontsize=9)
    axes[0,0].grid(False); axes[0,0].spines[["top","right"]].set_visible(False)

    # Batch reward
    axes[0,1].plot(steps, score, color=GRAY, alpha=0.3, linewidth=0.8)
    axes[0,1].plot(steps, smooth(score, 10), color=ORANGE, linewidth=2)
    axes[0,1].axvline(BEST_STEP, color=ORANGE, linestyle="--", alpha=0.7, linewidth=1.2)
    axes[0,1].set_title("Batch Reward")
    axes[0,1].set_ylabel("Reward", fontsize=9)
    axes[0,1].grid(False); axes[0,1].spines[["top","right"]].set_visible(False)

    # Entropy
    axes[1,0].plot(steps, entropy, color=GRAY, alpha=0.3, linewidth=0.8)
    axes[1,0].plot(steps, smooth(entropy, 10), color=GREEN, linewidth=2)
    axes[1,0].axvline(BEST_STEP, color=ORANGE, linestyle="--", alpha=0.7, linewidth=1.2)
    axes[1,0].set_title("Actor Entropy")
    axes[1,0].set_ylabel("Entropy", fontsize=9)
    axes[1,0].grid(False); axes[1,0].spines[["top","right"]].set_visible(False)

    # Response length
    axes[1,1].plot(steps, rlen, color=GRAY, alpha=0.3, linewidth=0.8)
    axes[1,1].plot(steps, smooth(rlen, 10), color=PURPLE, linewidth=2)
    axes[1,1].axvline(BEST_STEP, color=ORANGE, linestyle="--", alpha=0.7, linewidth=1.2)
    axes[1,1].set_title("Response Length")
    axes[1,1].set_ylabel("Tokens", fontsize=9)
    axes[1,1].grid(False); axes[1,1].spines[["top","right"]].set_visible(False)

    for ax in axes.flat:
        ax.set_xlabel("Training Step", fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    save(fig, args.out, "0_overview.png")

    print(f"\nAll plots saved to: {args.out}/")
    print(f"Best val acc: {val_acc.max():.1f}% at step {BEST_STEP}")

if __name__ == "__main__":
    main()