#!/usr/bin/env python3
"""
plot_h3.py — H3 analysis plots: Plan quality correlates with test outcomes
Updated with 3-bucket scheme: Low (<0.65), Mid (0.65-0.85), High (>=0.85)

Usage:
  python plot_h3.py --jsonl step_140_eval_feedback_w_judge_mbpp.jsonl --out plots_h3/
"""
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

BLUE="#2563EB"; ORANGE="#EA580C"; GREEN="#16A34A"
RED="#DC2626"; PURPLE="#7C3AED"; GRAY="#9CA3AF"
DARK="#1F2937"; LIGHT="#F3F4F6"; GRID="#E5E7EB"

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white","axes.edgecolor":"#D1D5DB",
    "axes.labelcolor":DARK,"axes.titlecolor":DARK,"xtick.color":DARK,"ytick.color":DARK,
    "grid.color":GRID,"grid.linewidth":0.8,"text.color":DARK,"font.family":"sans-serif",
    "font.size":11,"axes.titlesize":13,"axes.titleweight":"bold","lines.linewidth":2.0,
    "figure.dpi":150,
})

DIMS       = ['decomposition','edge_cases','approach','clarity']
DIM_LABELS = ['Decomposition','Edge Cases','Approach','Clarity']
DIM_COLORS = ["#FF8ECC", "#FFAEB0", "#FFCB97", "#FFE791"]


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"{path}"); plt.close(fig)


def style_ax(ax, xlabel="", ylabel=""):
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    #ax.grid(True, alpha=0.5)
    ax.grid(False)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="step_140_eval_feedback_w_judge_mbpp.jsonl")
    ap.add_argument("--out",   default="plots_h3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.jsonl) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    scored     = [r for r in rows if r.get('plan_score',-1) >= 0 and r.get('has_plan')]
    plan_scores= np.array([r['plan_score'] for r in scored])
    test_pass  = np.array([r['test_pass']  for r in scored])
    plan_lens  = np.array([len(r.get('plan','').split()) for r in scored])
    dim_scores = {d: np.array([r['plan_dims'].get(d,0) for r in scored]) for d in DIMS}
    pass_mask  = test_pass == 1
    fail_mask  = test_pass == 0

    # ── 1. Pass rate by plan score bucket (3 buckets) ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    buckets = [
        ('Low\n(< 0.65)',     plan_scores < 0.65),
        ('Mid\n(0.65–0.85)', (plan_scores >= 0.65) & (plan_scores < 0.85)),
        ('High\n(≥ 0.85)',    plan_scores >= 0.85),
    ]
    labels, rates, ns, colors = [], [], [], [GRAY, "#B0E0E6", "#CCCCFF"] #"#E6E6FA"]
    for label, mask in buckets:
        n    = mask.sum()
        rate = test_pass[mask].mean() * 100 if n > 0 else 0
        labels.append(label); rates.append(rate); ns.append(n)

    bars = ax.bar(labels, rates, color=colors, alpha=0.85, width=0.55,
                  edgecolor="white", linewidth=1.5)
    for bar, rate, n in zip(bars, rates, ns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{rate:.1f}%\n(n={n})', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=DARK)

    # Multiplier annotation
    #ax.text(0.02, 0.92, f'4× difference: Low {rates[0]:.1f}% → High {rates[2]:.1f}%',transform=ax.transAxes, fontsize=9, color=DARK,bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT, alpha=0.8)) 

    ax.set_title("Pass Rate by Plan Quality Bucket\n(Higher plan quality → higher pass rate)")
    ax.set_ylim(0, 85)
    style_ax(ax, xlabel="Plan Score Bucket (GPT-4o-mini Judge)", ylabel="Pass@1 Rate (%)")
    save(fig, args.out, "1_pass_rate_by_plan_bucket.png")

    # ── 2. Plan score distribution: pass vs fail ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 25)
    ax.hist(plan_scores[pass_mask], bins=bins, alpha=0.6, color=GREEN,
            label=f'Passed (n={pass_mask.sum()}, mean={plan_scores[pass_mask].mean():.3f})',
            edgecolor='white', linewidth=0.5)
    ax.hist(plan_scores[fail_mask], bins=bins, alpha=0.6, color=RED,
            label=f'Failed (n={fail_mask.sum()}, mean={plan_scores[fail_mask].mean():.3f})',
            edgecolor='white', linewidth=0.5)
    ax.axvline(plan_scores[pass_mask].mean(), color=GREEN, linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axvline(plan_scores[fail_mask].mean(), color=RED,   linestyle='--', linewidth=1.5, alpha=0.8)
    t, p = stats.ttest_ind(plan_scores[pass_mask], plan_scores[fail_mask])
    ax.text(0.03, 0.92, f'p-value = {p:.4f} {"✓ significant" if p < 0.05 else ""}',
            transform=ax.transAxes, fontsize=9, color=DARK,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT, alpha=0.8))
    ax.set_title("H3: Plan Score Distribution — Passed vs Failed Tasks")
    ax.legend(fontsize=9, framealpha=0.8)
    style_ax(ax, xlabel="Plan Score", ylabel="Number of Tasks")
    save(fig, args.out, "2_plan_score_distribution.png")

    # ── 3. Scatter: plan score vs test pass (jittered) ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    jitter = np.random.uniform(-0.03, 0.03, len(test_pass))
    colors_scatter = [GREEN if p else RED for p in test_pass]
    ax.scatter(plan_scores, test_pass+jitter, c=colors_scatter, alpha=0.3, s=15, zorder=2)
    slope, intercept, r, p_val, _ = stats.linregress(plan_scores, test_pass)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, slope*x_line+intercept, color=BLUE, linewidth=2,
            zorder=3, label=f'Linear fit (r={r:.3f})')
    ax.set_yticks([0,1]); ax.set_yticklabels(['Fail','Pass'])
    ax.set_xlim(-0.05, 1.05)
    ax.text(0.03, 0.88, f'Pearson r = {r:.4f}\np = {p_val:.4f}',
            transform=ax.transAxes, fontsize=10, color=BLUE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT, alpha=0.8))
    pass_patch = mpatches.Patch(color=GREEN, alpha=0.6, label='Passed')
    fail_patch = mpatches.Patch(color=RED,   alpha=0.6, label='Failed')
    ax.legend(handles=[pass_patch, fail_patch, ax.get_lines()[0]], fontsize=9, framealpha=0.8)
    ax.set_title("H3: Plan Score vs Test Outcome\n(Jittered for visibility)")
    style_ax(ax, xlabel="Plan Score (GPT-4o-mini Judge)", ylabel="Test Outcome")
    save(fig, args.out, "3_plan_score_vs_outcome_scatter.png")

    # ── 4. Per-dimension correlations ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    corrs = [np.corrcoef(dim_scores[d], test_pass)[0,1] for d in DIMS]
    bars  = ax.barh(DIM_LABELS, corrs, color=DIM_COLORS, alpha=0.85,
                    edgecolor='white', linewidth=1.5)
    for bar, c in zip(bars, corrs):
        ax.text(c+0.005, bar.get_y()+bar.get_height()/2,
                f'r = {c:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_xlim(0, max(corrs)+0.08)
    ax.set_title("Per-Dimension Plan Score\nCorrelation with Test Pass")
    style_ax(ax, xlabel="Pearson r")
    ax.invert_yaxis()
    save(fig, args.out, "4_per_dimension_correlation.png")

    # ── 5. Plan score by dim: pass vs fail ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DIMS)); w = 0.35
    pass_means = [dim_scores[d][pass_mask].mean() for d in DIMS]
    fail_means = [dim_scores[d][fail_mask].mean() for d in DIMS]
    b1 = ax.bar(x-w/2, pass_means, w, label='Passed', color=GREEN, alpha=0.85, edgecolor='white')
    b2 = ax.bar(x+w/2, fail_means, w, label='Failed',  color=RED,   alpha=0.85, edgecolor='white')
    for bar, val in zip(list(b1)+list(b2), pass_means+fail_means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(DIM_LABELS)
    ax.set_ylim(0.7, 1.05)
    ax.set_title("H3: Plan Quality per Dimension — Passed vs Failed Tasks")
    ax.legend(fontsize=10, framealpha=0.8)
    style_ax(ax, ylabel="Mean Score")
    save(fig, args.out, "5_dim_scores_pass_vs_fail.png")

    # ── 6. Plan length vs outcome ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, max(plan_lens), 30)
    ax.hist(plan_lens[pass_mask], bins=bins, alpha=0.6, color=GREEN,
            label=f'Passed (mean={plan_lens[pass_mask].mean():.0f} words)', edgecolor='white')
    ax.hist(plan_lens[fail_mask], bins=bins, alpha=0.6, color=RED,
            label=f'Failed (mean={plan_lens[fail_mask].mean():.0f} words)',  edgecolor='white')
    ax.axvline(plan_lens[pass_mask].mean(), color=GREEN, linestyle='--', linewidth=1.5)
    ax.axvline(plan_lens[fail_mask].mean(), color=RED,   linestyle='--', linewidth=1.5)
    r_len = np.corrcoef(plan_lens, test_pass)[0,1]
    ax.text(0.65, 0.88, f'r = {r_len:.3f}\n(longer plans → harder tasks)',
            transform=ax.transAxes, fontsize=9, color=DARK,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT, alpha=0.8))
    ax.set_title("Plan Length — Passed vs Failed Tasks\n(Failed tasks have longer plans)")
    ax.legend(fontsize=9, framealpha=0.8)
    style_ax(ax, xlabel="Plan Length (words)", ylabel="Number of Tasks")
    save(fig, args.out, "6_plan_length_distribution.png")

    print(f"\nAll H3 plots saved to: {args.out}/")
    print(f"Bucket summary: Low={rates[0]:.1f}% | Mid={rates[1]:.1f}% | High={rates[2]:.1f}%")
    print(f"Overall Pearson r = {np.corrcoef(plan_scores, test_pass)[0,1]:.4f}")


if __name__ == "__main__":
    main()