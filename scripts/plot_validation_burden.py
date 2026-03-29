#!/usr/bin/env python3
"""
Collect and visualise validation burden results across annotation levels.

Reads the YAML outputs produced by ``test_burden_enrichment.py`` for multiple
annotation levels, consequence types, and top-k thresholds. Produces:

1. A summary TSV table with all results
2. A multi-panel line plot of -log10(empirical p) vs top-k
3. A multi-panel heatmap of logistic-regression z-statistics

Usage:
    python scripts/plot_validation_burden.py \\
        --input-dirs enrichment_stratified_L0 enrichment_stratified_L1 \\
                     enrichment_stratified_L2 enrichment_stratified_L3 \\
        --top-k 100 500 1000 2000 \\
        --consequence-types total missense lof \\
        --output-dir results/validation_burden

Author: Francesco Lescai
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
LEVEL_MARKERS = {
    "L0": "o",
    "L1": "s",
    "L2": "^",
    "L3": "D",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and plot validation burden results across annotation levels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Directories containing enrichment YAML results (one per level)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each input directory (e.g. L0 L1 L2 L3). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[100, 500, 1000, 2000],
        help="Top-k values to collect",
    )
    parser.add_argument(
        "--consequence-types",
        nargs="+",
        default=["total", "missense", "lof"],
        help="Consequence types to collect",
    )
    parser.add_argument(
        "--output-prefix",
        default="validation_burden",
        help="Prefix for output filenames",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "both"],
        default="png",
        help="Output figure format",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def auto_detect_labels(input_dirs: list[Path]) -> list[str]:
    """Extract annotation-level labels from directory names."""
    labels = []
    for d in input_dirs:
        match = re.search(r"(L\d+)", d.name)
        if match:
            labels.append(match.group(1))
        else:
            labels.append(d.name)
    return labels


def collect_results(
    input_dirs: list[Path],
    labels: list[str],
    top_k_values: list[int],
    consequence_types: list[str],
) -> pd.DataFrame:
    """
    Load enrichment YAML files and return a tidy DataFrame.

    Parameters
    ----------
    input_dirs : list[Path]
        One directory per annotation level.
    labels : list[str]
        Human-readable label for each directory.
    top_k_values : list[int]
        Top-k thresholds to look for.
    consequence_types : list[str]
        Consequence types to collect.

    Returns
    -------
    pd.DataFrame
        Columns: level, consequence_type, top_k, logistic_z, logistic_p,
        mannwhitney_p, empirical_p, percentile_rank, mean_cases,
        mean_controls, n_genes_found.
    """
    rows: list[dict[str, Any]] = []

    for d, label in zip(input_dirs, labels):
        for k in top_k_values:
            for csq in consequence_types:
                if csq == "total":
                    fname = f"enrichment_topK{k}.yaml"
                else:
                    fname = f"enrichment_topK{k}_{csq}.yaml"

                path = d / fname
                if not path.exists():
                    continue

                with open(path) as f:
                    data = yaml.safe_load(f)

                obs = data.get("observed", {})
                perm = data.get("permutation", {})

                rows.append({
                    "level": label,
                    "consequence_type": csq,
                    "top_k": k,
                    "logistic_z": obs.get("logistic_z"),
                    "logistic_p": obs.get("logistic_p"),
                    "mannwhitney_p": obs.get("mannwhitney_p"),
                    "empirical_p": perm.get("empirical_p"),
                    "percentile_rank": perm.get("percentile_rank"),
                    "mean_cases": obs.get("mean_cases"),
                    "mean_controls": obs.get("mean_controls"),
                    "n_genes_found": data.get("n_sieve_genes_found"),
                    "n_permutations": perm.get("n_permutations"),
                })

    df = pd.DataFrame(rows)

    # Enforce level ordering
    present_levels = [lv for lv in LEVEL_ORDER if lv in df["level"].values]
    df["level"] = pd.Categorical(df["level"], categories=present_levels, ordered=True)
    df = df.sort_values(["level", "consequence_type", "top_k"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a formatted summary grouped by consequence type."""
    n_total_tests = len(df)
    bonferroni = 0.05 / n_total_tests if n_total_tests > 0 else 0.05

    print(f"\n{'=' * 90}")
    print("Validation Burden Results Summary")
    print(f"{'=' * 90}")
    print(f"Total tests: {n_total_tests}  |  Bonferroni threshold: {bonferroni:.5f}")

    for csq, grp in df.groupby("consequence_type", sort=False):
        print(f"\n--- {csq} burden ---")
        print(f"{'Level':<6} {'top-k':>6} {'z':>8} {'emp. p':>10} {'pctl':>7} "
              f"{'cases':>8} {'ctrls':>8} {'genes':>6} {'sig':>4}")
        print("-" * 72)
        for _, row in grp.iterrows():
            sig = "*" if row["empirical_p"] < bonferroni else ""
            print(
                f"{row['level']:<6} {row['top_k']:>6} "
                f"{row['logistic_z']:>8.3f} {row['empirical_p']:>10.4f} "
                f"{row['percentile_rank']:>6.1f}% "
                f"{row['mean_cases']:>8.2f} {row['mean_controls']:>8.2f} "
                f"{row['n_genes_found']:>6.0f} {sig:>4}"
            )

    print(f"\n* significant after Bonferroni correction (p < {bonferroni:.5f})\n")


def save_summary_table(df: pd.DataFrame, output_path: Path) -> None:
    """Save results as TSV."""
    df.to_csv(output_path, sep="\t", index=False, float_format="%.6g")
    print(f"Table saved to {output_path}")


# ---------------------------------------------------------------------------
# Plot 1: -log10(empirical p) line plot
# ---------------------------------------------------------------------------


def plot_pvalue_lines(
    df: pd.DataFrame,
    output_path: Path,
    fmt: str = "png",
) -> None:
    """
    Multi-panel line plot of -log10(empirical p) vs top-k.

    One panel per consequence type, one line per annotation level.
    """
    consequences = df["consequence_type"].unique()
    n_panels = len(consequences)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.5 * n_panels + 1, 6),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    # Bonferroni across all tests
    n_total_tests = len(df)
    bonferroni = 0.05 / n_total_tests if n_total_tests > 0 else 0.05

    # Cap for -log10 when empirical_p is at its minimum
    max_n_perm = df["n_permutations"].max() if "n_permutations" in df.columns else 10000
    neg_log10_cap = -np.log10(1 / (max_n_perm + 1))

    for ax, csq in zip(axes, consequences):
        sub = df[df["consequence_type"] == csq]
        top_k_values = sorted(sub["top_k"].unique())

        for level in sub["level"].cat.categories:
            lsub = sub[sub["level"] == level]
            if lsub.empty:
                continue

            x_pos = [top_k_values.index(k) for k in lsub["top_k"]]
            neg_log_p = -np.log10(lsub["empirical_p"].clip(lower=10 ** (-neg_log10_cap)))

            ax.plot(
                x_pos, neg_log_p,
                color=LEVEL_COLORS.get(level, "gray"),
                marker=LEVEL_MARKERS.get(level, "o"),
                markersize=8,
                linewidth=2,
                label=f"{level} ({LEVEL_DESCRIPTIONS.get(level, '')})",
            )

        # Significance thresholds
        ax.axhline(
            -np.log10(0.05), color="gray", linestyle="--", linewidth=1, alpha=0.7,
            label="p = 0.05",
        )
        ax.axhline(
            -np.log10(bonferroni), color="red", linestyle=":", linewidth=1.5, alpha=0.7,
            label=f"Bonferroni (p = {bonferroni:.1e})",
        )

        ax.set_xticks(range(len(top_k_values)))
        ax.set_xticklabels([str(k) for k in top_k_values])
        ax.set_xlabel("Top-k genes", fontsize=12)
        ax.set_title(csq.capitalize(), fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

    axes[0].set_ylabel(r"$-\log_{10}$(empirical p)", fontsize=12)

    # Shared legend below
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc="lower center",
        ncol=min(len(legend_labels), 4),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Validation Burden: Empirical P-values Across Annotation Levels",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    _save_figure(fig, output_path, fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: z-statistic heatmap
# ---------------------------------------------------------------------------


def plot_zscore_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    fmt: str = "png",
) -> None:
    """
    Multi-panel heatmap of logistic-regression z-statistics.

    Rows = annotation levels, columns = top-k values.
    One panel per consequence type. Significant cells annotated with *.
    """
    consequences = df["consequence_type"].unique()
    n_panels = len(consequences)
    n_total_tests = len(df)
    bonferroni = 0.05 / n_total_tests if n_total_tests > 0 else 0.05

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4 * n_panels + 2, 5),
        squeeze=False,
    )
    axes = axes[0]

    # Determine shared color limits for comparable panels
    z_abs_max = df["logistic_z"].abs().max()
    vmin, vmax = -z_abs_max, z_abs_max

    for ax, csq in zip(axes, consequences):
        sub = df[df["consequence_type"] == csq]
        levels = [lv for lv in sub["level"].cat.categories if lv in sub["level"].values]
        top_k_values = sorted(sub["top_k"].unique())

        pivot = sub.pivot(index="level", columns="top_k", values="logistic_z")
        pivot = pivot.reindex(index=levels, columns=top_k_values)

        # Build significance mask
        sig_pivot = sub.pivot(index="level", columns="top_k", values="empirical_p")
        sig_pivot = sig_pivot.reindex(index=levels, columns=top_k_values)

        # Annotation strings: z-value with * if significant
        annot = pivot.copy().astype(str)
        for lv in levels:
            for k in top_k_values:
                z_val = pivot.loc[lv, k]
                p_val = sig_pivot.loc[lv, k]
                if pd.isna(z_val):
                    annot.loc[lv, k] = ""
                elif p_val < bonferroni:
                    annot.loc[lv, k] = f"{z_val:.2f}*"
                else:
                    annot.loc[lv, k] = f"{z_val:.2f}"

        if HAS_SEABORN:
            sns.heatmap(
                pivot, ax=ax,
                annot=annot, fmt="",
                cmap="RdBu_r", center=0,
                vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor="white",
                cbar=ax == axes[-1],
                cbar_kws={"label": "Logistic z-statistic"} if ax == axes[-1] else {},
                annot_kws={"fontsize": 10, "fontweight": "bold"},
            )
        else:
            im = ax.imshow(
                pivot.values.astype(float),
                cmap="RdBu_r", vmin=vmin, vmax=vmax,
                aspect="auto",
            )
            ax.set_xticks(range(len(top_k_values)))
            ax.set_xticklabels([str(k) for k in top_k_values])
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels(levels)
            for i, lv in enumerate(levels):
                for j, k in enumerate(top_k_values):
                    text = annot.loc[lv, k]
                    ax.text(j, i, text, ha="center", va="center",
                            fontsize=10, fontweight="bold")
            if ax == axes[-1]:
                fig.colorbar(im, ax=ax, label="Logistic z-statistic")

        ax.set_title(csq.capitalize(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Top-k genes", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Annotation level", fontsize=11)

    fig.suptitle(
        "Validation Burden: Logistic Regression Z-statistics",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    _save_figure(fig, output_path, fmt)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, base_path: Path, fmt: str) -> None:
    """Save figure in the requested format(s)."""
    if fmt in ("png", "both"):
        path = base_path.with_suffix(".png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {path}")
    if fmt in ("pdf", "both"):
        path = base_path.with_suffix(".pdf")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Auto-detect labels if not provided
    labels = args.labels or auto_detect_labels(args.input_dirs)
    if len(labels) != len(args.input_dirs):
        print(
            f"ERROR: {len(labels)} labels provided for {len(args.input_dirs)} input directories."
        )
        return 1

    # Validate input dirs
    for d in args.input_dirs:
        if not d.is_dir():
            print(f"ERROR: {d} is not a directory.")
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect results
    df = collect_results(args.input_dirs, labels, args.top_k, args.consequence_types)
    if df.empty:
        print("ERROR: No results found. Check input directories and parameters.")
        return 1

    print(f"Collected {len(df)} results across {df['level'].nunique()} levels, "
          f"{df['consequence_type'].nunique()} consequence types, "
          f"{df['top_k'].nunique()} top-k values.")

    # Summary table
    print_summary_table(df)
    save_summary_table(df, args.output_dir / f"{args.output_prefix}.tsv")

    # Plots
    plot_pvalue_lines(
        df,
        args.output_dir / f"{args.output_prefix}_pvalues",
        fmt=args.format,
    )
    plot_zscore_heatmap(
        df,
        args.output_dir / f"{args.output_prefix}_heatmap",
        fmt=args.format,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
