#!/usr/bin/env python3
"""Create the WaC paper's experimental-workflow figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE.parents[1] / "docs"

COLORS = {
    "source": "#E8F1F8",
    "process": "#FFF2CC",
    "main": "#DDEBF7",
    "extra": "#E2F0D9",
    "baseline": "#ECECEC",
    "evaluation": "#FCE4D6",
    "metrics": "#E4DFEC",
    "edge": "#34495E",
    "muted": "#5B6573",
}


def box(ax, x, y, w, h, text, color, *, fontsize=9, weight="normal",
        edge=None, linestyle="-", radius=0.08, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        linewidth=1.15,
        edgecolor=edge or COLORS["edge"],
        facecolor=color,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight=weight, color="#17202A", linespacing=1.25,
        zorder=zorder + 1,
    )
    return patch


def arrow(ax, start, end, *, color=None, connectionstyle="arc3", lw=1.4,
          linestyle="-"):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color or COLORS["edge"],
        linestyle=linestyle, connectionstyle=connectionstyle,
        shrinkA=2, shrinkB=2, zorder=1,
    )
    ax.add_patch(patch)
    return patch


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(14.4, 5.55))
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 5.55)
    ax.axis("off")

    # Column labels
    headings = [
        (1.30, "TRAINING DATA"),
        (4.15, "PREPARATION"),
        (7.35, "SYSTEMS"),
        (10.45, "TARGET EVALUATION"),
        (13.15, "MEASURES"),
    ]
    for x, label in headings:
        ax.text(x, 5.34, label, ha="center", va="center", fontsize=9.2,
                fontweight="bold", color=COLORS["muted"])

    # Training sources
    box(ax, 0.20, 3.90, 2.20, 0.78,
        "La Trobe\nAustralian interaction", COLORS["source"], weight="bold")
    box(ax, 0.20, 2.90, 2.20, 0.78,
        "Santa Barbara\nUS interaction", COLORS["source"], weight="bold")
    box(ax, 0.20, 1.42, 2.20, 0.78,
        "Griffith\nAustralian speech", COLORS["source"], weight="bold")
    ax.text(1.30, 2.56, "LT + SB: 4.47 h training", ha="center", va="center",
            fontsize=8.2, color=COLORS["muted"])
    ax.text(1.30, 1.15, "third-source experiments", ha="center", va="center",
            fontsize=8.2, color=COLORS["muted"])

    # Preparation
    box(ax, 3.03, 2.40, 2.25, 1.82,
        "Normalize transcript notation\n\n"
        "Cut exact acoustic segments\n\n"
        "Filter overlap and\nmixed-speaker targets",
        COLORS["process"], fontsize=8.8, weight="bold")
    arrow(ax, (2.42, 4.29), (3.00, 3.79))
    arrow(ax, (2.42, 3.29), (3.00, 3.32))
    arrow(ax, (2.42, 1.81), (3.00, 2.67))

    # Systems: main and additional variants
    ax.add_patch(FancyBboxPatch(
        (5.95, 2.72), 2.95, 2.02,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0, edgecolor="#5B9BD5", facecolor="#F7FBFE", zorder=0,
    ))
    ax.text(7.425, 4.52, "MAIN TARGET-DOMAIN COMPARISON",
            ha="center", va="center", fontsize=8.2, fontweight="bold",
            color="#2F5597")
    box(ax, 6.15, 3.68, 1.18, 0.60, "Base\nWhisper", COLORS["baseline"],
        fontsize=8.4, weight="bold")
    box(ax, 7.49, 3.68, 1.18, 0.60, "Crisper-\nWhisper2", COLORS["baseline"],
        fontsize=8.4, weight="bold")
    box(ax, 6.15, 2.94, 1.18, 0.60, "Dual-policy\nLoRA", COLORS["main"],
        fontsize=8.4, weight="bold")
    box(ax, 7.49, 2.94, 1.18, 0.60, "Verbatim\nfull FT", COLORS["main"],
        fontsize=8.4, weight="bold")

    ax.add_patch(FancyBboxPatch(
        (5.95, 0.40), 2.95, 1.85,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0, edgecolor="#70AD47", facecolor="#F8FCF5", zorder=0,
    ))
    ax.text(7.425, 2.04, "ADDITIONAL SOURCE-CORPUS TESTS",
            ha="center", va="center", fontsize=8.2, fontweight="bold",
            color="#548235")
    box(ax, 6.12, 1.18, 1.20, 0.68, "Crisper2 LoRA\nLT + SB", COLORS["extra"],
        fontsize=8.3, weight="bold")
    box(ax, 7.52, 1.18, 1.20, 0.68, "3-source full FT\nLT + SB + Griffith", COLORS["extra"],
        fontsize=8.2, weight="bold")
    ax.text(7.425, 0.73, "held-out source-corpus evaluation",
            ha="center", va="center", fontsize=7.8, color=COLORS["muted"])

    arrow(ax, (5.31, 3.31), (5.91, 3.31))
    arrow(ax, (5.31, 2.73), (5.91, 1.44), connectionstyle="arc3,rad=0.12")

    # Evaluation sample
    box(ax, 9.52, 3.45, 1.93, 1.05,
        "100 manually\ntranscribed\n20-s web clips",
        COLORS["evaluation"], fontsize=8.8, weight="bold")
    box(ax, 9.52, 1.86, 1.93, 0.98,
        "Prespecified\nquality rule",
        "#FFF7F2", fontsize=8.8, weight="bold")
    box(ax, 9.52, 0.42, 1.93, 0.92,
        "57-clip primary set\nAU 27  |  NZ 30",
        COLORS["evaluation"], fontsize=8.6, weight="bold")
    arrow(ax, (8.94, 3.57), (9.49, 3.90))
    arrow(ax, (10.485, 3.42), (10.485, 2.87))
    arrow(ax, (10.485, 1.83), (10.485, 1.37))

    # Metrics and uncertainty
    box(ax, 12.02, 2.73, 2.15, 1.55,
        "Verbatim WER\nNormalized WER\nFiller $F_1$\nRepetition $F_1$",
        COLORS["metrics"], fontsize=8.8, weight="bold")
    box(ax, 12.02, 0.87, 2.15, 1.18,
        "Paired bootstrap\nclustered by\nsource video",
        "#F5F2F8", fontsize=8.7, weight="bold")
    arrow(ax, (11.48, 3.94), (11.99, 3.60))
    arrow(ax, (11.48, 0.88), (11.99, 1.44))

    # Legend for visual distinction.
    ax.text(0.20, 0.34,
            "Blue = systems in the main table    Green = supplementary source-corpus tests",
            ha="left", va="center", fontsize=7.8, color=COLORS["muted"])

    fig.subplots_adjust(left=0.012, right=0.992, top=0.982, bottom=0.035)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "workflow.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        OUTPUT_DIR / "workflow.png", dpi=300, bbox_inches="tight", pad_inches=0.03
    )


if __name__ == "__main__":
    main()
