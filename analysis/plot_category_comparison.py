#!/usr/bin/env python3
"""
plot_category_5cat.py — 3-model category comparison (5 categories)
Base vs Forced Plan vs Optional Plan

Usage:
  python plot_category_5cat.py \
    --forced   category_forced_5cat.json \
    --optional category_optional_5cat.json \
    --out      plots_5cat/
"""
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE   = "#2563EB"; GREEN  = "#16A34A"; GRAY   = "#9CA3AF"
RED    = "#DC2626"; DARK   = "#1F2937"; LIGHT  = "#F3F4F6"
BLACK = "#000000" ; LBLUE = "#93c5fdff"; LPURPLE = "#b4a7d6ff"

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white","axes.edgecolor":"#D1D5DB",
    "axes.labelcolor":DARK,"axes.titlecolor":DARK,"xtick.color":DARK,"ytick.color":DARK,
    "grid.color":"#E5E7EB","grid.linewidth":0.8,"text.color":DARK,
    "font.family":"sans-serif","font.size":11,"axes.titlesize":13,
    "axes.titleweight":"bold","figure.dpi":150,
})

CAT_ORDER = ["Bitwise","Algorithms","Math & Geometry","String & Text","Collections","Applied"]


def load_json(path):
    with open(path) as f: data = json.load(f)
    return {r["category"]: r for r in data["categories"]}, data["overall"]


def calc_se(rate, n):
    p = rate/100
    return np.sqrt(p*(1-p)/n)*100 if n>0 else 0


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"✓ {path}"); plt.close(fig)


def style_ax(ax, ylabel=""):
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", alpha=0.5, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forced",   required=True)
    ap.add_argument("--optional", required=True)
    ap.add_argument("--out",      default="plots_5cat")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    forced_cats,   forced_ov   = load_json(args.forced)
    optional_cats, optional_ov = load_json(args.optional)

    cats = [c for c in CAT_ORDER if c in forced_cats]
    ns           = [forced_cats[c]["n"]              for c in cats]
    base_rates   = [forced_cats[c]["base_pass_rate"] for c in cats]
    forced_rates = [forced_cats[c]["s140_pass_rate"] for c in cats]
    opt_rates    = [optional_cats.get(c,{}).get("s140_pass_rate",0) for c in cats]
    forced_deltas= [forced_cats[c]["delta"]           for c in cats]
    opt_deltas   = [optional_cats.get(c,{}).get("delta",0) for c in cats]

    x = np.arange(len(cats)); w = 0.25

    # ── 1. Grouped bar: 3 models ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x-w,   base_rates,   w, label="Base Model",    color=GRAY,  alpha=0.80, edgecolor="white")
    b2 = ax.bar(x,     forced_rates, w, label="Forced Plan",   color=LBLUE,  alpha=0.85, edgecolor="white")
    b3 = ax.bar(x+w,   opt_rates,    w, label="Optional Plan", color=LPURPLE, alpha=0.85, edgecolor="white")

    for bar, val in zip(list(b1)+list(b2)+list(b3), base_rates+forced_rates+opt_rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    for i, n in enumerate(ns):
        ax.text(i, 1.5, f"n={n}", ha="center", fontsize=7.5, color=BLACK, style="italic")

    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=10)
    ax.set_title("pass@1 by Category: Base vs Forced Plan vs Optional Plan (MBPP)")
    ax.set_ylim(0, 92); ax.legend(fontsize=10, framealpha=0.8)
    style_ax(ax, "Pass@1 (%)")
    save(fig, args.out, "1_grouped_bar_3models.png")

    # ── 2. Delta comparison side by side ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    w2 = 0.35
    forced_colors = [GREEN if d>0.5 else (RED if d<-0.5 else GRAY) for d in forced_deltas]
    opt_colors    = [GREEN if d>0.5 else (RED if d<-0.5 else GRAY) for d in opt_deltas]

    b_f = ax.bar(x-w2/2, forced_deltas, w2, color=forced_colors,
                 alpha=0.85, edgecolor="white", label="Forced Plan Δ")
    b_o = ax.bar(x+w2/2, opt_deltas,    w2, color=opt_colors,
                 alpha=0.70, edgecolor="white", label="Optional Plan Δ", hatch="//")

    for bar, d in zip(list(b_f)+list(b_o), forced_deltas+opt_deltas):
        sign = "+" if d>=0 else ""
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+(0.3 if d>=0 else -0.8),
                f"{sign}{d:.1f}%", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=DARK)

    ax.axhline(0, color=DARK, linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=10)
    ax.set_title("Δ pass@1 vs Base — Forced Plan vs Optional Plan by Category")
    ax.set_ylim(min(forced_deltas+opt_deltas)-3, max(forced_deltas+opt_deltas)+5)
    style_ax(ax, "Δ pass@1 (%)")
    ax.legend(fontsize=10, framealpha=0.8)
    save(fig, args.out, "2_delta_comparison.png")

    # ── 3. Summary table ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, len(cats)*0.65+2.5))
    ax.axis("off")

    col_labels = ["Category","N","Base","Forced Plan","Δ Forced","Optional Plan","Δ Optional"]
    rows_data  = []
    for cat,n,b,f,fd,o,od in zip(cats,ns,base_rates,forced_rates,
                                   forced_deltas,opt_rates,opt_deltas):
        rows_data.append([
            cat, str(n), f"{b:.1f}%", f"{f:.1f}%",
            f"+{fd:.1f}%" if fd>=0 else f"{fd:.1f}%",
            f"{o:.1f}%",
            f"+{od:.1f}%" if od>=0 else f"{od:.1f}%",
        ])
    rows_data.append([
        "OVERALL", "500",
        f"{forced_ov['base']:.1f}%",
        f"{forced_ov['s140']:.1f}%", f"+{forced_ov['delta']:.1f}%",
        f"{optional_ov['s140']:.1f}%", f"+{optional_ov['delta']:.1f}%",
    ])

    table = ax.table(cellText=rows_data, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(10.5); table.scale(1.2, 2.1)

    for j in range(len(col_labels)):
        table[0,j].set_facecolor("#0F1C3F")
        table[0,j].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(rows_data):
        for dc in [4, 6]:
            try:
                v = float(row[dc].replace("%","").replace("+",""))
                table[i+1,dc].set_facecolor(
                    "#D1FAE5" if v>0.5 else ("#FEE2E2" if v<-0.5 else LIGHT))
            except: pass

    for j in range(len(col_labels)):
        table[len(rows_data),j].set_facecolor("#EFF6FF")
        table[len(rows_data),j].set_text_props(fontweight="bold")

    ax.set_title("Category Analysis — Base vs Forced Plan vs Optional Plan",
                 fontsize=13, fontweight="bold", pad=20, color=DARK)
    save(fig, args.out, "3_summary_table.png")

    print(f"\nAll plots saved to: {args.out}/")
    print(f"Overall: Base={forced_ov['base']}% | "
          f"Forced={forced_ov['s140']}% (Δ+{forced_ov['delta']}%) | "
          f"Optional={optional_ov['s140']}% (Δ+{optional_ov['delta']}%)")


if __name__ == "__main__":
    main()