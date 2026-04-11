#!/usr/bin/env python3
"""
Plot real vs null mean_attribution distributions.

Consumes two raw rankings files (sieve_variant_rankings.csv from real and null
explainability runs) and writes a two-panel PNG:

  Panel 1 — overlaid histograms of mean_attribution (real = blue, null = grey)
  Panel 2 — Q-Q plot of real quantiles vs null quantiles

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

    # --- Panel 2: Q-Q plot ---
    ax = axes[1]
    n_points = 500
    quantiles = np.linspace(0, 100, n_points)
    real_q = np.percentile(real_attr, quantiles)
    null_q = np.percentile(null_attr, quantiles)
    ax.scatter(null_q, real_q, s=6, alpha=0.6, color='steelblue')
    lim = (min(null_q.min(), real_q.min()), max(null_q.max(), real_q.max()))
    ax.plot(lim, lim, color='tomato', linestyle='--', linewidth=0.9,
            label='y = x (no difference)')
    ax.set_xlabel('Null quantiles')
    ax.set_ylabel('Real quantiles')
    ax.set_title(f'Q-Q plot: real vs null{suffix}')
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = Path(args.output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot to {out_path}')


if __name__ == '__main__':
    main()
