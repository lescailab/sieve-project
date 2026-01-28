#!/usr/bin/env python3
"""
Create Manhattan plot for variant rankings.

Similar to GWAS Manhattan plots, shows -log10(attribution scores) across chromosomes
with top variants annotated by gene name.

Usage:
    python scripts/manhattan_plot.py \
        --variant-rankings results/explainability/sieve_variant_rankings.csv \
        --output results/manhattan_plot.png \
        --top-n 10
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Create Manhattan plot for variant rankings')
    parser.add_argument('--variant-rankings', required=True, help='Path to variant rankings CSV')
    parser.add_argument('--output', required=True, help='Output plot path (PNG, PDF, or SVG)')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top variants to label')
    parser.add_argument('--score-column', default='mean_attribution',
                        help='Column to use for y-axis (default: mean_attribution)')
    parser.add_argument('--figsize', type=str, default='14,6', help='Figure size as width,height')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output image')
    return parser.parse_args()


def create_manhattan_plot(df, score_col='mean_attribution', top_n=10, figsize=(14, 6)):
    """Create Manhattan plot from variant rankings."""

    # Ensure required columns exist
    if 'chromosome' not in df.columns:
        raise ValueError("chromosome column required in variant rankings")
    if 'position' not in df.columns:
        raise ValueError("position column required in variant rankings")
    if score_col not in df.columns:
        raise ValueError(f"{score_col} column required in variant rankings")

    # Convert chromosome to numeric (handle X, Y, MT)
    chrom_map = {str(i): i for i in range(1, 23)}
    chrom_map.update({'X': 23, 'Y': 24, 'MT': 25, 'M': 25})

    df = df.copy()
    df['chrom_num'] = df['chromosome'].astype(str).map(chrom_map)

    # Remove unmapped chromosomes
    df = df.dropna(subset=['chrom_num'])
    df['chrom_num'] = df['chrom_num'].astype(int)

    # Sort by chromosome and position
    df = df.sort_values(['chrom_num', 'position'])

    # Create x-axis positions (cumulative position across chromosomes)
    df['x_pos'] = 0
    current_x = 0
    chrom_centers = []

    for chrom in sorted(df['chrom_num'].unique()):
        chrom_df = df[df['chrom_num'] == chrom]
        n_variants = len(chrom_df)

        # Assign x positions
        df.loc[df['chrom_num'] == chrom, 'x_pos'] = np.arange(current_x, current_x + n_variants)

        # Track chromosome center for labeling
        chrom_center = current_x + n_variants / 2
        chrom_centers.append((chrom, chrom_center))

        current_x += n_variants + 1000  # Add gap between chromosomes

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color chromosomes alternately
    colors = ['#1f77b4', '#ff7f0e']
    for i, chrom in enumerate(sorted(df['chrom_num'].unique())):
        chrom_data = df[df['chrom_num'] == chrom]
        ax.scatter(
            chrom_data['x_pos'],
            chrom_data[score_col],
            c=colors[i % 2],
            s=10,
            alpha=0.6,
            label=None
        )

    # Label top N variants
    if 'gene_name' in df.columns:
        top_variants = df.nlargest(top_n, score_col)
        for _, variant in top_variants.iterrows():
            ax.annotate(
                variant['gene_name'],
                xy=(variant['x_pos'], variant[score_col]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=0.5)
            )

    # Set x-axis labels (chromosome names at centers)
    chrom_labels = []
    chrom_positions = []
    for chrom, center in chrom_centers:
        if chrom <= 22:
            chrom_labels.append(str(chrom))
        elif chrom == 23:
            chrom_labels.append('X')
        elif chrom == 24:
            chrom_labels.append('Y')
        elif chrom == 25:
            chrom_labels.append('MT')
        chrom_positions.append(center)

    ax.set_xticks(chrom_positions)
    ax.set_xticklabels(chrom_labels)

    # Labels
    ax.set_xlabel('Chromosome', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{score_col.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_title('SIEVE Variant Rankings - Manhattan Plot', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    return fig


def main():
    args = parse_args()

    # Load data
    print(f"Loading variant rankings from {args.variant_rankings}...")
    df = pd.read_csv(args.variant_rankings)

    print(f"Loaded {len(df)} variants")

    if 'chromosome' not in df.columns:
        print("ERROR: chromosome column missing!")
        print("Available columns:", list(df.columns))
        print("\nRun fix_ranking_outputs.py first to add chromosome and gene_name columns")
        return 1

    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))

    # Create plot
    print(f"Creating Manhattan plot...")
    fig = create_manhattan_plot(
        df,
        score_col=args.score_column,
        top_n=args.top_n,
        figsize=figsize
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Manhattan plot saved to {output_path}")

    print("\nPlot statistics:")
    print(f"  Chromosomes: {df['chromosome'].nunique()}")
    print(f"  Total variants plotted: {len(df)}")
    print(f"  Top {args.top_n} variants labeled")


if __name__ == '__main__':
    exit(main() or 0)
