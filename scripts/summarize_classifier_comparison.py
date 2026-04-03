#!/usr/bin/env python3
"""
Summarize non-linear classifier validation: LR vs RF comparison plots.

Scans a results directory for YAML outputs from validate_nonlinear_classifier.py,
pairs logistic-regression and random-forest results for each level + top-k
combination, and produces publication-quality comparison figures.

Each figure is saved as a standalone PNG and collated into a single A4-landscape
PDF.

Usage:
    python scripts/summarize_classifier_comparison.py \
        --results-dir /path/to/nonlinear_validation/ \
        --output-dir /path/to/summary_plots/
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEVEL_ORDER = ["L0", "L1", "L2", "L3"]
LEVEL_LABELS = {
    "L0": "L0 – Genotype only",
    "L1": "L1 – + Position",
    "L2": "L2 – + Consequence",
    "L3": "L3 – + Functional scores",
}

# Colour palette – muted, colour-blind-friendly tones
CLR_LR = "#4878A8"       # steel blue  – logistic regression
CLR_LR_LIGHT = "#A8C4E0"
CLR_RF = "#D4553A"       # brick red   – random forest
CLR_RF_LIGHT = "#ECACA0"
CLR_NULL = "#B0B0B0"     # grey        – null distribution
CLR_GAIN = "#2A9D8F"     # teal        – significant gain highlight
CLR_NS = "#999999"       # grey        – non-significant

# Figure dimensions (inches) – individual plots
FIG_W, FIG_H = 10, 5.0

# A4 landscape in inches
A4_W, A4_H = 11.69, 8.27


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def level_sort_key(level: str) -> Tuple[int, str]:
    """Sort key placing L0–L3 first, then anything else alphabetically."""
    if level in LEVEL_ORDER:
        return (0, LEVEL_ORDER.index(level))
    return (1, level)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_null_aucs(npz_path: Path) -> Optional[np.ndarray]:
    """Load the null-distribution AUC array from an npz file."""
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    return data["null_aucs"]


def discover_results(results_dir: Path) -> Dict[Tuple[str, int], Dict[str, Dict[str, Any]]]:
    """
    Scan *results_dir* for YAML result files and pair RF / LR results.

    Returns a dict keyed by (level, top_k) mapping to
    {"rf": {...}, "lr": {...}} where each value holds the parsed YAML plus
    the null-distribution array.
    """
    pattern = re.compile(
        r"nonlinear_validation_(?P<level>\w+)_topK(?P<k>\d+)(?:_(?P<clf>lr))?\.yaml$"
    )

    paired: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}

    for yaml_path in sorted(results_dir.glob("nonlinear_validation_*.yaml")):
        match = pattern.match(yaml_path.name)
        if not match:
            continue

        level = match.group("level")
        top_k = int(match.group("k"))
        clf_tag = match.group("clf") or "rf"  # primary (no suffix) == rf

        payload = load_yaml(yaml_path)

        # Resolve the matching npz file
        npz_name = yaml_path.name.replace(".yaml", "").replace(
            "nonlinear_validation_", "null_aucs_"
        ) + ".npz"
        null_aucs = load_null_aucs(yaml_path.parent / npz_name)
        payload["_null_aucs"] = null_aucs

        key = (level, top_k)
        paired.setdefault(key, {})
        paired[key][clf_tag] = payload

    return paired


def gaussian_kde_safe(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Compute a Gaussian KDE, returning zeros if the data is degenerate."""
    if data.size < 3 or np.std(data) < 1e-12:
        return np.zeros_like(grid)
    kde = stats.gaussian_kde(data, bw_method="scott")
    return kde(grid)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.5,
    })


def plot_comparison(
    level: str,
    top_k: int,
    rf_data: Dict[str, Any],
    lr_data: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create a two-panel publication figure comparing LR and RF for one
    level + top-k combination.

    Left panel:  overlapping density curves of null distributions with
                 observed AUC markers for each classifier.
    Right panel: dot-and-whisker comparison of per-fold AUCs with
                 non-linear gain annotation.
    """
    _setup_style()

    rf_obs = rf_data["observed"]
    lr_obs = lr_data["observed"]

    rf_mean = rf_obs["mean_auc"]
    lr_mean = lr_obs["mean_auc"]

    rf_folds = np.array([v for v in rf_obs["per_fold_aucs"] if v is not None])
    lr_folds = np.array([v for v in lr_obs["per_fold_aucs"] if v is not None])

    rf_null = rf_data.get("_null_aucs")
    lr_null = lr_data.get("_null_aucs")

    rf_p = rf_data["empirical_p"]
    lr_p = lr_data["empirical_p"]

    delta_auc = rf_mean - lr_mean

    # Determine significance thresholds
    # We don't know total test count here, so report nominal p-values;
    # the user's original script already does Bonferroni.
    rf_sig = rf_p < 0.05
    lr_sig = lr_p < 0.05
    gain_notable = delta_auc > 0

    # ── Figure layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.35)

    ax_dens = fig.add_subplot(gs[0])
    ax_comp = fig.add_subplot(gs[1])

    level_label = LEVEL_LABELS.get(level, level)

    # ── Left panel: density curves ────────────────────────────────────
    if rf_null is not None and lr_null is not None:
        all_null = np.concatenate([rf_null, lr_null])
        grid_lo = min(np.min(all_null), min(rf_mean, lr_mean)) - 0.03
        grid_hi = max(np.max(all_null), max(rf_mean, lr_mean)) + 0.03
    elif rf_null is not None:
        grid_lo = min(np.min(rf_null), min(rf_mean, lr_mean)) - 0.03
        grid_hi = max(np.max(rf_null), max(rf_mean, lr_mean)) + 0.03
    else:
        grid_lo, grid_hi = 0.40, 0.70

    grid = np.linspace(grid_lo, grid_hi, 500)

    # Plot null density for RF
    if rf_null is not None:
        rf_kde = gaussian_kde_safe(rf_null, grid)
        ax_dens.fill_between(grid, rf_kde, alpha=0.18, color=CLR_RF, linewidth=0)
        ax_dens.plot(grid, rf_kde, color=CLR_RF, alpha=0.5, linewidth=1,
                     label="Null (RF)")

    # Plot null density for LR
    if lr_null is not None:
        lr_kde = gaussian_kde_safe(lr_null, grid)
        ax_dens.fill_between(grid, lr_kde, alpha=0.18, color=CLR_LR, linewidth=0)
        ax_dens.plot(grid, lr_kde, color=CLR_LR, alpha=0.5, linewidth=1,
                     label="Null (LR)")

    # Observed markers – tall vertical lines
    ymax = ax_dens.get_ylim()[1] if ax_dens.get_ylim()[1] > 0 else 1.0
    ax_dens.axvline(lr_mean, color=CLR_LR, linewidth=2.2, linestyle="--",
                    label=f"LR observed = {lr_mean:.3f}", zorder=5)
    ax_dens.axvline(rf_mean, color=CLR_RF, linewidth=2.2, linestyle="-",
                    label=f"RF observed = {rf_mean:.3f}", zorder=5)

    # Shade the gain region between the two observed lines
    if gain_notable and rf_mean > lr_mean:
        ax_dens.axvspan(lr_mean, rf_mean, alpha=0.12, color=CLR_GAIN, zorder=1)

    ax_dens.set_xlabel("Mean AUC (permutation test)")
    ax_dens.set_ylabel("Density")
    ax_dens.set_title("Null distributions & observed AUC")
    ax_dens.legend(loc="upper left", frameon=True, framealpha=0.9,
                   edgecolor="none", fontsize=8.5)

    # P-value annotations
    p_text_parts = [
        f"$p_{{\\mathrm{{LR}}}}$ = {lr_p:.4f}",
        f"$p_{{\\mathrm{{RF}}}}$ = {rf_p:.4f}",
    ]
    ax_dens.text(
        0.97, 0.97,
        "\n".join(p_text_parts),
        transform=ax_dens.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.9),
    )

    # ── Right panel: fold-level comparison ────────────────────────────
    positions = [0, 1]
    labels = ["Logistic\nRegression", "Random\nForest"]

    # Strip/jitter plot of per-fold AUCs
    jitter_w = 0.06
    for fold_vals, pos, clr, clr_light in [
        (lr_folds, 0, CLR_LR, CLR_LR_LIGHT),
        (rf_folds, 1, CLR_RF, CLR_RF_LIGHT),
    ]:
        if fold_vals.size == 0:
            continue
        jitter = np.random.default_rng(42).uniform(-jitter_w, jitter_w, size=len(fold_vals))
        ax_comp.scatter(
            pos + jitter, fold_vals,
            s=28, alpha=0.55, color=clr_light, edgecolors=clr,
            linewidths=0.6, zorder=3,
        )

    # Mean + CI bars
    for mean_val, std_val, pos, clr in [
        (lr_mean, lr_obs["std_auc"], 0, CLR_LR),
        (rf_mean, rf_obs["std_auc"], 1, CLR_RF),
    ]:
        ax_comp.errorbar(
            pos, mean_val, yerr=std_val,
            fmt="D", markersize=8, color=clr, markeredgecolor="white",
            markeredgewidth=1.2, capsize=5, capthick=1.8, elinewidth=1.8,
            zorder=5,
        )

    # Connecting line between means
    line_color = CLR_GAIN if (gain_notable and rf_sig) else CLR_NS
    ax_comp.plot(
        positions, [lr_mean, rf_mean],
        color=line_color, linewidth=1.2, linestyle=":", alpha=0.7, zorder=2,
    )

    # Delta annotation
    mid_y = (lr_mean + rf_mean) / 2
    delta_sign = "+" if delta_auc >= 0 else ""
    delta_color = CLR_GAIN if (gain_notable and rf_sig) else CLR_NS

    significance_marker = ""
    if rf_sig and gain_notable:
        significance_marker = " *"

    ax_comp.annotate(
        f"$\\Delta$AUC = {delta_sign}{delta_auc:.3f}{significance_marker}",
        xy=(0.5, mid_y),
        fontsize=10,
        fontweight="bold" if (gain_notable and rf_sig) else "normal",
        color=delta_color,
        ha="center", va="bottom",
        xytext=(0.5, mid_y + 0.008),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=delta_color, alpha=0.85, linewidth=1.2),
        zorder=6,
    )

    ax_comp.set_xticks(positions)
    ax_comp.set_xticklabels(labels, fontweight="medium")
    ax_comp.set_ylabel("AUC")
    ax_comp.set_title("Classifier comparison")
    ax_comp.set_xlim(-0.5, 1.5)

    # Chance-level reference line
    ax_comp.axhline(0.5, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=1)
    ax_comp.text(1.45, 0.5, "chance", fontsize=7.5, color="#AAAAAA",
                 ha="right", va="bottom")

    # Y-axis: auto range with some padding
    all_vals = np.concatenate([lr_folds, rf_folds]) if lr_folds.size and rf_folds.size else np.array([lr_mean, rf_mean])
    y_lo = min(0.5, np.min(all_vals) - 0.02)
    y_hi = max(np.max(all_vals) + 0.02, rf_mean + rf_obs["std_auc"] + 0.02)
    ax_comp.set_ylim(y_lo, y_hi)
    ax_comp.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── Suptitle ──────────────────────────────────────────────────────
    fig.suptitle(
        f"SIEVE validation  ·  {level_label}  ·  top-{top_k} genes",
        fontsize=13, fontweight="bold", y=0.98,
    )

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Produce publication-quality LR-vs-RF comparison plots from "
            "non-linear validation results."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing nonlinear_validation_*.yaml files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for output plots.  Defaults to "
            "<results-dir>/classifier_comparison/."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_dir = args.results_dir.resolve()

    if not results_dir.is_dir():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = (args.output_dir or results_dir / "classifier_comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paired = discover_results(results_dir)

    # Keep only combinations that have BOTH classifiers
    complete = {
        key: classifiers
        for key, classifiers in paired.items()
        if "rf" in classifiers and "lr" in classifiers
    }

    if not complete:
        print("No level/top-k combinations with both RF and LR results found.")
        sys.exit(1)

    # Sort: level order, then top-k ascending
    sorted_keys = sorted(complete.keys(), key=lambda k: (level_sort_key(k[0]), k[1]))

    print(f"Found {len(sorted_keys)} level × top-k combinations with paired results.")
    print(f"Output directory: {output_dir}\n")

    _setup_style()

    pdf_path = output_dir / "classifier_comparison_all.pdf"
    png_paths: List[Path] = []

    with PdfPages(pdf_path) as pdf:
        for level, top_k in sorted_keys:
            rf_data = complete[(level, top_k)]["rf"]
            lr_data = complete[(level, top_k)]["lr"]

            png_name = f"comparison_{level}_topK{top_k}.png"
            png_path = output_dir / png_name
            png_paths.append(png_path)

            print(f"  Plotting {level} top-{top_k} ...")
            plot_comparison(level, top_k, rf_data, lr_data, png_path)

            # Re-read the saved figure for the PDF page (A4 landscape)
            # Create a fresh figure at A4 landscape size
            fig_pdf = plt.figure(figsize=(A4_W, A4_H))
            img = plt.imread(str(png_path))
            ax = fig_pdf.add_axes([0.02, 0.02, 0.96, 0.96])
            ax.imshow(img, aspect="equal")
            ax.axis("off")
            pdf.savefig(fig_pdf, orientation="landscape")
            plt.close(fig_pdf)

    print(f"\nIndividual PNGs written to: {output_dir}/")
    print(f"Combined PDF: {pdf_path}")
    print(f"Total plots: {len(png_paths)}")


if __name__ == "__main__":
    main()
