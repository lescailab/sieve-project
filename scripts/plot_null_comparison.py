#!/usr/bin/env python3
"""
Plot real vs null mean_attribution distributions.

Consumes two raw rankings files (sieve_variant_rankings.csv from real and null
explainability runs) and writes a two-panel PNG:

  Panel 1 — overlaid histograms of mean_attribution (real = blue, null = grey)
  Panel 2 — standardised shape comparison with KDE overlay and shape statistics

Usage:
    python scripts/plot_null_comparison.py \
        --real  /path/to/real/sieve_variant_rankings.csv \
        --null  /path/to/null/sieve_variant_rankings.csv \
        --output-png      /path/to/real_vs_null_attribution_comparison.png

Author: Francesco Lescai
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, skew, kurtosis, anderson_ksamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot real vs null mean_attribution distributions',
    )
    parser.add_argument('--real', type=str, required=True,
                        help='Raw real variant rankings CSV (must have mean_attribution)')
    parser.add_argument('--null', type=str, required=True,
                        help='Raw null variant rankings CSV (must have mean_attribution)')
    parser.add_argument('--output-png', type=str, required=True,
                        help='Path for the output PNG file')
    parser.add_argument('--title-suffix', type=str, default='',
                        help='Optional suffix appended to plot titles (e.g. annotation level)')
    return parser.parse_args()


def _require_mean_attribution(df: pd.DataFrame, label: str) -> None:
    if 'mean_attribution' not in df.columns:
        print(f"ERROR: '{label}' file does not have a 'mean_attribution' column.")
        print("Ensure you are passing the raw sieve_variant_rankings.csv file.")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    real_df = pd.read_csv(args.real)
    _require_mean_attribution(real_df, 'real')
    null_df = pd.read_csv(args.null)
    _require_mean_attribution(null_df, 'null')

    real_attr = pd.to_numeric(real_df['mean_attribution'], errors='coerce').dropna().values
    null_attr = pd.to_numeric(null_df['mean_attribution'], errors='coerce').dropna().values

    if real_attr.size == 0:
        print("ERROR: no finite mean_attribution values found in real file.")
        sys.exit(1)
    if null_attr.size == 0:
        print("ERROR: no finite mean_attribution values found in null file.")
        sys.exit(1)

    suffix = f' — {args.title_suffix}' if args.title_suffix else ''

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel 1: overlaid histograms ---
    ax = axes[0]
    bins = np.linspace(
        min(real_attr.min(), null_attr.min()),
        max(real_attr.max(), null_attr.max()),
        80,
    )
    ax.hist(null_attr, bins=bins, alpha=0.55, label='Null', density=True,
            color='#888888', edgecolor='none')
    ax.hist(real_attr, bins=bins, alpha=0.55, label='Real', density=True,
            color='steelblue', edgecolor='none')
    ax.set_xlabel('mean_attribution')
    ax.set_ylabel('Density')
    ax.set_title(f'Real vs Null attribution distributions{suffix}')
    ax.legend(fontsize=8)

    # --- Panel 2: standardised shape comparison ---
    ax = axes[1]
    real_z = (real_attr - real_attr.mean()) / real_attr.std()
    null_z = (null_attr - null_attr.mean()) / null_attr.std()

    grid_lo = min(real_z.min(), null_z.min()) - 0.5
    grid_hi = max(real_z.max(), null_z.max()) + 0.5
    grid = np.linspace(grid_lo, grid_hi, 500)

    real_kde = gaussian_kde(real_z)(grid)
    null_kde = gaussian_kde(null_z)(grid)

    ax.fill_between(grid, null_kde, alpha=0.3, color='#888888')
    ax.plot(grid, null_kde, color='#888888', linewidth=1.2,
            label='Null (standardised)')
    ax.fill_between(grid, real_kde, alpha=0.3, color='steelblue')
    ax.plot(grid, real_kde, color='steelblue', linewidth=1.2,
            label='Real (standardised)')
    ax.plot(grid, norm.pdf(grid), color='black', linestyle='--',
            linewidth=0.9, label='N(0,1) reference')

    ax.set_xlabel('Standardised attribution (z-score)')
    ax.set_ylabel('Density')
    ax.set_title(f'Shape comparison (standardised){suffix}')
    ax.legend(fontsize=8, loc='upper left')

    skew_real = skew(real_z)
    skew_null = skew(null_z)
    kurt_real = kurtosis(real_z)
    kurt_null = kurtosis(null_z)
    frac_real = float(np.mean(np.abs(real_z) > 3))
    frac_null = float(np.mean(np.abs(null_z) > 3))
    ad_result = anderson_ksamp([real_z, null_z])
    ad_stat = ad_result.statistic
    ad_p = getattr(ad_result, 'pvalue', getattr(ad_result, 'significance_level', float('nan')))

    stats_text = (
        "Shape diagnostics\n"
        "─────────────────────────────────\n"
        f"Skewness:   real {skew_real:.2f}  null {skew_null:.2f}\n"
        f"Ex. kurtosis: real {kurt_real:.2f}  null {kurt_null:.2f}\n"
        f"|z| > 3:    real {frac_real:.1%}  null {frac_null:.1%}\n"
        f"Anderson-Darling: stat {ad_stat:.2f}, p {ad_p:.4f}\n"
        "\n"
        "Interpretation:\n"
        "Higher kurtosis + right skew in the real\n"
        "model indicates signal concentrated on a\n"
        "subset of variants, not spread uniformly."
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=7, family='monospace',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    fig.tight_layout()
    out_path = Path(args.output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot to {out_path}')


if __name__ == '__main__':
    main()
