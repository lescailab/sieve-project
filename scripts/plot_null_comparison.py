#!/usr/bin/env python3
"""
Plot real vs null z_attribution distributions.

Consumes the two corrected rankings files produced by correct_chrx_bias.py
(both must contain a z_attribution column) and writes a two-panel PNG:

  Panel 1 — overlaid histograms of z_attribution (real = blue, null = grey)
  Panel 2 — Q-Q plot of real quantiles vs null quantiles

Usage:
    python scripts/plot_null_comparison.py \
        --corrected-real  /path/to/real/corrected/corrected_variant_rankings.csv \
        --corrected-null  /path/to/null/corrected/corrected_variant_rankings.csv \
        --output-png      /path/to/real_vs_null_z_attribution_comparison.png

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
        description='Plot real vs null z_attribution distributions',
    )
    parser.add_argument('--corrected-real', type=str, required=True,
                        help='Corrected real variant rankings CSV (must have z_attribution)')
    parser.add_argument('--corrected-null', type=str, required=True,
                        help='Corrected null variant rankings CSV (must have z_attribution)')
    parser.add_argument('--output-png', type=str, required=True,
                        help='Path for the output PNG file')
    parser.add_argument('--title-suffix', type=str, default='',
                        help='Optional suffix appended to plot titles (e.g. annotation level)')
    return parser.parse_args()


def _require_z_attribution(df: pd.DataFrame, label: str) -> None:
    if 'z_attribution' not in df.columns:
        print(f"ERROR: '{label}' file does not have a 'z_attribution' column.")
        print("Run correct_chrx_bias.py on this file first.")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    real_df = pd.read_csv(args.corrected_real)
    _require_z_attribution(real_df, 'corrected-real')
    null_df = pd.read_csv(args.corrected_null)
    _require_z_attribution(null_df, 'corrected-null')

    real_z = pd.to_numeric(real_df['z_attribution'], errors='coerce').dropna().values
    null_z = pd.to_numeric(null_df['z_attribution'], errors='coerce').dropna().values

    if real_z.size == 0:
        print("ERROR: no finite z_attribution values found in corrected-real file.")
        sys.exit(1)
    if null_z.size == 0:
        print("ERROR: no finite z_attribution values found in corrected-null file.")
        sys.exit(1)

    suffix = f' — {args.title_suffix}' if args.title_suffix else ''

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: overlaid histograms ---
    ax = axes[0]
    bins = np.linspace(
        min(real_z.min(), null_z.min()),
        max(real_z.max(), null_z.max()),
        80,
    )
    ax.hist(null_z, bins=bins, alpha=0.55, label='Null', density=True,
            color='#888888', edgecolor='none')
    ax.hist(real_z, bins=bins, alpha=0.55, label='Real', density=True,
            color='steelblue', edgecolor='none')
    for xv, ls in [(2.0, '--'), (3.0, '-')]:
        ax.axvline(xv, color='tomato', linestyle=ls, linewidth=0.9,
                   label=f'z = {xv:.0f}')
    ax.set_xlabel('z_attribution (per-chromosome z-score)')
    ax.set_ylabel('Density')
    ax.set_title(f'Real vs Null attribution distributions{suffix}')
    ax.legend(fontsize=8)

    # --- Panel 2: Q-Q plot ---
    ax = axes[1]
    n_points = 500
    quantiles = np.linspace(0, 100, n_points)
    real_q = np.percentile(real_z, quantiles)
    null_q = np.percentile(null_z, quantiles)
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
