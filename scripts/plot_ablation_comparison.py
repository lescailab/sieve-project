#!/usr/bin/env python3
"""
Visualize ablation comparison results.

Creates a multi-panel publication-quality figure from the outputs of
``compare_ablation_rankings.py`` and ``ablation_compare.py``:

1. Jaccard heatmap — pairwise overlap at a selected top-k
2. Jaccard by top-k — how overlap evolves with increasing k
3. Level-specific variant counts — variants uniquely important per level
4. AUC comparison — model performance across annotation levels (optional)

Usage:
    python scripts/plot_ablation_comparison.py \\
        --jaccard-tsv ablation_jaccard_matrix.tsv \\
        --level-specific-tsv level_specific_variants.tsv \\
        --summary-yaml ablation_summary.yaml \\
        --output ablation_comparison.png

Author: Francesco Lescai
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


LEVEL_ORDER = ["L0", "L1", "L2", "L3"]
LEVEL_DESCRIPTIONS = {
    "L0": "Genotype only",
    "L1": "+ Position",
    "L2": "+ Consequence",
    "L3": "+ SIFT/PolyPhen",
}
LEVEL_COLORS = {
    "L0": "#e74c3c",
    "L1": "#e67e22",
    "L2": "#27ae60",
    "L3": "#3498db",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jaccard_tsv(path: pathlib.Path) -> List[Dict[str, Any]]:
    """
    Load the Jaccard matrix TSV produced by compare_ablation_rankings.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``ablation_jaccard_matrix.tsv``.

    Returns
    -------
    List[Dict[str, Any]]
        Records with keys: top_k, level_a, level_b, jaccard, overlap,
        size_a, size_b, union.
    """
    import csv

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "top_k": int(row["top_k"]),
                    "level_a": row["level_a"],
                    "level_b": row["level_b"],
                    "jaccard": float(row["jaccard"]),
                    "overlap": int(row["overlap"]),
                    "size_a": int(row["size_a"]),
                    "size_b": int(row["size_b"]),
                    "union": int(row["union"]),
                }
            )
    return rows


def load_level_specific_tsv(path: pathlib.Path) -> List[Dict[str, Any]]:
    """
    Load the level-specific variants TSV.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``level_specific_variants.tsv``.

    Returns
    -------
    List[Dict[str, Any]]
        Records from the TSV.
    """
    import csv

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def load_summary_yaml(path: pathlib.Path) -> Dict[str, Any]:
    """
    Load the ablation summary YAML produced by ablation_compare.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``ablation_summary.yaml``.

    Returns
    -------
    Dict[str, Any]
        Parsed YAML contents.
    """
    if not HAS_YAML:
        print(
            "WARNING: pyyaml not installed, cannot load summary YAML",
            file=sys.stderr,
        )
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def _get_present_levels(jaccard_rows: List[Dict[str, Any]]) -> List[str]:
    """Extract and sort the levels present in the Jaccard data."""
    levels = set()
    for row in jaccard_rows:
        levels.add(row["level_a"])
        levels.add(row["level_b"])
    return sorted(levels, key=lambda l: LEVEL_ORDER.index(l) if l in LEVEL_ORDER else 999)


def plot_jaccard_heatmap(
    ax: plt.Axes,
    jaccard_rows: List[Dict[str, Any]],
    target_top_k: int,
) -> None:
    """
    Plot a Jaccard similarity heatmap for a specific top-k.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    jaccard_rows : List[Dict[str, Any]]
        All Jaccard records.
    target_top_k : int
        The top-k value to display.
    """
    levels = _get_present_levels(jaccard_rows)
    n = len(levels)
    matrix = np.ones((n, n))  # diagonal = 1.0

    level_idx = {l: i for i, l in enumerate(levels)}
    filtered = [r for r in jaccard_rows if r["top_k"] == target_top_k]

    for row in filtered:
        i = level_idx.get(row["level_a"])
        j = level_idx.get(row["level_b"])
        if i is not None and j is not None:
            matrix[i, j] = row["jaccard"]
            matrix[j, i] = row["jaccard"]

    if HAS_SEABORN:
        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=levels,
            yticklabels=levels,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd_r",
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Jaccard Index"},
        )
    else:
        im = ax.imshow(matrix, cmap="YlOrRd_r", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(levels)
        ax.set_yticklabels(levels)
        # Annotate cells
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if matrix[i, j] < 0.5 else "black",
                    fontsize=11, fontweight="bold",
                )
        plt.colorbar(im, ax=ax, shrink=0.8, label="Jaccard Index")

    ax.set_title(f"Variant Overlap (Top {target_top_k})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Annotation Level")
    ax.set_ylabel("Annotation Level")


def plot_jaccard_by_topk(
    ax: plt.Axes,
    jaccard_rows: List[Dict[str, Any]],
) -> None:
    """
    Plot Jaccard similarity as a function of top-k for each level pair.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    jaccard_rows : List[Dict[str, Any]]
        All Jaccard records.
    """
    # Group by pair
    pairs: Dict[str, Dict[int, float]] = {}
    for row in jaccard_rows:
        pair_key = f"{row['level_a']} vs {row['level_b']}"
        if pair_key not in pairs:
            pairs[pair_key] = {}
        pairs[pair_key][row["top_k"]] = row["jaccard"]

    markers = ["o", "s", "^", "D", "v", "P"]
    for idx, (pair_key, topk_vals) in enumerate(sorted(pairs.items())):
        ks = sorted(topk_vals.keys())
        jaccards = [topk_vals[k] for k in ks]
        marker = markers[idx % len(markers)]
        ax.plot(
            ks, jaccards,
            marker=marker,
            linewidth=2,
            markersize=6,
            label=pair_key,
        )

    ax.set_xlabel("Top-k Variants", fontsize=11)
    ax.set_ylabel("Jaccard Index", fontsize=11)
    ax.set_title("Ranking Overlap by Top-k", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)


def plot_level_specific_counts(
    ax: plt.Axes,
    level_specific_rows: List[Dict[str, Any]],
) -> None:
    """
    Bar chart of level-specific variant counts.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    level_specific_rows : List[Dict[str, Any]]
        Records from the level-specific variants TSV.
    """
    # Count per level
    counts: Dict[str, int] = {}
    for row in level_specific_rows:
        level = row.get("specific_to_level", "")
        counts[level] = counts.get(level, 0) + 1

    levels = sorted(
        counts.keys(),
        key=lambda l: LEVEL_ORDER.index(l) if l in LEVEL_ORDER else 999,
    )
    values = [counts[l] for l in levels]
    colors = [LEVEL_COLORS.get(l, "#999999") for l in levels]

    bars = ax.bar(range(len(levels)), values, color=colors, alpha=0.8, edgecolor="black", linewidth=1)

    # Value labels
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                str(val),
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
            )

    # Descriptions below level labels
    labels = [f"{l}\n({LEVEL_DESCRIPTIONS.get(l, '')})" for l in levels]
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Number of Unique Variants", fontsize=11)
    ax.set_title("Level-Specific Variant Discoveries", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def plot_auc_comparison(
    ax: plt.Axes,
    summary: Dict[str, Any],
) -> None:
    """
    Bar chart of AUC values across annotation levels.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to draw on.
    summary : Dict[str, Any]
        Parsed ablation_summary.yaml.
    """
    levels_data = summary.get("levels", [])
    if not levels_data:
        ax.text(
            0.5, 0.5, "No performance data available",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="grey",
        )
        ax.set_title("Model Performance", fontsize=13, fontweight="bold")
        return

    levels: List[str] = []
    aucs: List[float] = []
    stds: List[float] = []

    for entry in levels_data:
        level = entry.get("level", "?")
        auc = entry.get("auc")
        std_auc = entry.get("std_auc")
        if auc is not None:
            levels.append(level)
            aucs.append(float(auc))
            stds.append(float(std_auc) if std_auc is not None else 0.0)

    if not levels:
        ax.text(
            0.5, 0.5, "No AUC values found",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="grey",
        )
        ax.set_title("Model Performance", fontsize=13, fontweight="bold")
        return

    colors = [LEVEL_COLORS.get(l, "#999999") for l in levels]
    x = np.arange(len(levels))

    bars = ax.bar(
        x, aucs,
        yerr=stds if any(s > 0 for s in stds) else None,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Value labels
    for i, (bar, auc, std) in enumerate(zip(bars, aucs, stds)):
        label = f"{auc:.3f}"
        if std > 0:
            label += f"\n\u00B1{std:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (std if std > 0 else 0) + 0.005,
            label,
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Random (0.5)")
    ax.set_xticks(x)
    labels = [f"{l}\n({LEVEL_DESCRIPTIONS.get(l, '')})" for l in levels]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title("Model Performance by Annotation Level", fontsize=13, fontweight="bold")

    # Dynamic y-axis
    min_val = min(aucs) - max(stds) - 0.05 if stds else min(aucs) - 0.05
    max_val = max(aucs) + max(stds) + 0.05 if stds else max(aucs) + 0.05
    ax.set_ylim(max(0.4, min_val), min(1.0, max_val))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)

    # Highlight best
    best_level = summary.get("best_level")
    if best_level and best_level in levels:
        best_idx = levels.index(best_level)
        bars[best_idx].set_edgecolor("#FFD700")
        bars[best_idx].set_linewidth(3)


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------


def create_figure(
    jaccard_rows: List[Dict[str, Any]],
    level_specific_rows: List[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    output_path: pathlib.Path,
    heatmap_top_k: int = 100,
) -> None:
    """
    Assemble the multi-panel ablation comparison figure.

    Parameters
    ----------
    jaccard_rows : List[Dict[str, Any]]
        Jaccard similarity data.
    level_specific_rows : List[Dict[str, Any]]
        Level-specific variant data.
    summary : Dict[str, Any] or None
        Model performance summary (optional).
    output_path : pathlib.Path
        Where to save the figure.
    heatmap_top_k : int
        Which top-k to use for the heatmap panel.
    """
    has_performance = summary is not None and summary.get("levels")

    if has_performance:
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
    else:
        fig = plt.figure(figsize=(16, 5.5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = None

    # Panel 1: Jaccard heatmap
    # Pick the closest available top_k to the target
    available_ks = sorted({r["top_k"] for r in jaccard_rows})
    chosen_k = min(available_ks, key=lambda k: abs(k - heatmap_top_k)) if available_ks else heatmap_top_k
    plot_jaccard_heatmap(ax1, jaccard_rows, chosen_k)

    # Panel 2: Jaccard by top-k
    plot_jaccard_by_topk(ax2, jaccard_rows)

    # Panel 3: Level-specific variants
    plot_level_specific_counts(ax3, level_specific_rows)

    # Panel 4: AUC comparison (if available)
    if ax4 is not None and has_performance:
        plot_auc_comparison(ax4, summary)

    # Supertitle
    fig.suptitle(
        "SIEVE Annotation Ablation Comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98 if has_performance else 1.02,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved to {output_path}", file=sys.stderr)

    # Also save PDF if requested
    if output_path.suffix == ".png":
        pdf_path = output_path.with_suffix(".pdf")
        fig2 = plt.figure(figsize=(16, 12) if has_performance else (16, 5.5))
        gs2 = GridSpec(
            2 if has_performance else 1,
            2 if has_performance else 3,
            figure=fig2,
            hspace=0.35,
            wspace=0.3 if has_performance else 0.35,
        )
        axes = [fig2.add_subplot(gs2[i]) for i in range(gs2.get_geometry()[0] * gs2.get_geometry()[1])]

        plot_jaccard_heatmap(axes[0], jaccard_rows, chosen_k)
        plot_jaccard_by_topk(axes[1], jaccard_rows)
        plot_level_specific_counts(axes[2], level_specific_rows)
        if has_performance and len(axes) > 3:
            plot_auc_comparison(axes[3], summary)

        fig2.suptitle(
            "SIEVE Annotation Ablation Comparison",
            fontsize=16, fontweight="bold",
            y=0.98 if has_performance else 1.02,
        )
        fig2.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"PDF version saved to {pdf_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--jaccard-tsv",
        required=True,
        help="Jaccard matrix TSV from compare_ablation_rankings.py",
    )
    parser.add_argument(
        "--level-specific-tsv",
        required=True,
        help="Level-specific variants TSV from compare_ablation_rankings.py",
    )
    parser.add_argument(
        "--summary-yaml",
        default=None,
        help="Ablation summary YAML from ablation_compare.py (optional, adds AUC panel)",
    )
    parser.add_argument(
        "--heatmap-top-k",
        type=int,
        default=100,
        help="Top-k value to show in the heatmap panel (default: 100)",
    )
    parser.add_argument(
        "--output",
        default="ablation_comparison.png",
        help="Output figure path (default: ablation_comparison.png)",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the ablation comparison plotting."""
    args = parse_args()

    # Load Jaccard data
    jaccard_path = pathlib.Path(args.jaccard_tsv)
    if not jaccard_path.exists():
        print(f"ERROR: Jaccard TSV not found: {jaccard_path}", file=sys.stderr)
        return 1
    jaccard_rows = load_jaccard_tsv(jaccard_path)
    print(f"Loaded {len(jaccard_rows)} Jaccard records", file=sys.stderr)

    # Load level-specific variants
    ls_path = pathlib.Path(args.level_specific_tsv)
    if not ls_path.exists():
        print(f"ERROR: Level-specific TSV not found: {ls_path}", file=sys.stderr)
        return 1
    level_specific_rows = load_level_specific_tsv(ls_path)
    print(f"Loaded {len(level_specific_rows)} level-specific variants", file=sys.stderr)

    # Load performance summary (optional)
    summary: Optional[Dict[str, Any]] = None
    if args.summary_yaml:
        summary_path = pathlib.Path(args.summary_yaml)
        if summary_path.exists():
            summary = load_summary_yaml(summary_path)
            print(
                f"Loaded performance summary ({len(summary.get('levels', []))} levels)",
                file=sys.stderr,
            )
        else:
            print(
                f"WARNING: Summary YAML not found: {summary_path}, "
                f"skipping AUC panel",
                file=sys.stderr,
            )

    # Create figure
    output_path = pathlib.Path(args.output)
    create_figure(
        jaccard_rows=jaccard_rows,
        level_specific_rows=level_specific_rows,
        summary=summary,
        output_path=output_path,
        heatmap_top_k=args.heatmap_top_k,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
