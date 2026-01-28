#!/usr/bin/env python3
"""
Create gene enrichment visualization plots.

Creates:
1. Bar plot of top genes by attribution score
2. GO enrichment dot plot (if GO enrichment results available)
3. GWAS validation plot (if GWAS validation available)

Usage:
    python scripts/gene_enrichment_plot.py \
        --gene-rankings results/explainability/sieve_gene_rankings.csv \
        --output-dir results/plots/
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Create gene enrichment plots')
    parser.add_argument('--gene-rankings', required=True, help='Path to gene rankings CSV')
    parser.add_argument('--go-enrichment', help='Path to GO enrichment CSV (optional)')
    parser.add_argument('--gwas-validation', help='Path to GWAS validation CSV (optional)')
    parser.add_argument('--output-dir', required=True, help='Output directory for plots')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top genes to show')
    parser.add_argument('--figsize', type=str, default='10,8', help='Figure size as width,height')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for output images')
    return parser.parse_args()


def plot_top_genes(df, top_n=20, score_col='gene_score', figsize=(10, 8)):
    """Create bar plot of top genes."""

    # Get top genes
    top_genes = df.nlargest(top_n, score_col)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bars
    y_pos = np.arange(len(top_genes))
    gene_labels = top_genes['gene_name'] if 'gene_name' in top_genes.columns else [f"Gene {i}" for i in top_genes.index]
    scores = top_genes[score_col]

    bars = ax.barh(y_pos, scores, color='steelblue', alpha=0.8)

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(
            score + score * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{score:.3f}',
            va='center',
            fontsize=8
        )

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gene_labels, fontsize=10)
    ax.invert_yaxis()  # Highest score at top
    ax.set_xlabel('Gene Attribution Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gene', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Genes by Attribution Score', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def plot_go_enrichment(df, top_n=15, figsize=(10, 8)):
    """Create dot plot of GO enrichment."""

    if df.empty:
        return None

    # Get top terms by FDR
    top_terms = df.nsmallest(top_n, 'fdr')

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create dot plot
    y_pos = np.arange(len(top_terms))

    # Dot size = fold enrichment, color = -log10(FDR)
    sizes = top_terms['fold_enrichment'] * 50
    colors = -np.log10(top_terms['fdr'] + 1e-10)  # Add small epsilon to avoid log(0)

    scatter = ax.scatter(
        top_terms['fold_enrichment'],
        y_pos,
        s=sizes,
        c=colors,
        cmap='Reds',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

    # Color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('-log10(FDR)', fontsize=10)

    # GO term labels (truncate if too long)
    term_labels = [term[:50] + '...' if len(term) > 50 else term for term in top_terms['go_term']]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(term_labels, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel('Fold Enrichment', fontsize=12, fontweight='bold')
    ax.set_ylabel('GO Term', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Enriched GO Terms', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add FDR threshold line
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No enrichment')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_gwas_validation(df, top_n=15, figsize=(10, 8)):
    """Create bar plot of GWAS-validated genes."""

    if df.empty:
        return None

    # Filter to only GWAS-validated genes
    gwas_genes = df[df['in_gwas'] == True]

    if len(gwas_genes) == 0:
        return None

    # Get top by gene score
    top_gwas = gwas_genes.nlargest(min(top_n, len(gwas_genes)), 'gene_score')

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bars colored by number of GWAS studies
    y_pos = np.arange(len(top_gwas))
    gene_labels = top_gwas['gene_name'] if 'gene_name' in top_gwas.columns else [f"Gene {i}" for i in top_gwas.index]
    scores = top_gwas['gene_score']
    study_counts = top_gwas['gwas_studies'] if 'gwas_studies' in top_gwas.columns else [0] * len(top_gwas)

    # Normalize study counts for color
    min_count = min(study_counts)
    max_count = max(study_counts)

    # Handle edge case where all counts are the same
    if min_count == max_count:
        # Use a single color for all bars
        colors_map = ['steelblue'] * len(study_counts)
    else:
        norm = plt.Normalize(vmin=min_count, vmax=max_count)
        colors_map = plt.cm.RdYlGn(norm(study_counts))

    bars = ax.barh(y_pos, scores, color=colors_map, alpha=0.8)

    # Add study count labels
    for i, (bar, score, count) in enumerate(zip(bars, scores, study_counts)):
        ax.text(
            score + score * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{count} studies',
            va='center',
            fontsize=7
        )

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gene_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Gene Attribution Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gene', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {len(top_gwas)} GWAS-Validated Genes', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add colorbar (only if there's variation in study counts)
    if min_count != max_count:
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Number of GWAS Studies', fontsize=10)

    plt.tight_layout()
    return fig


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))

    # Load gene rankings
    print(f"Loading gene rankings from {args.gene_rankings}...")
    gene_df = pd.read_csv(args.gene_rankings)
    print(f"Loaded {len(gene_df)} genes")

    # Check for required columns
    if 'gene_name' not in gene_df.columns:
        print("WARNING: gene_name column missing - using gene_id as labels")

    # Plot 1: Top genes
    print(f"\nCreating top genes plot...")
    fig1 = plot_top_genes(gene_df, top_n=args.top_n, figsize=figsize)
    output1 = output_dir / 'top_genes.png'
    fig1.savefig(output1, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved to {output1}")

    # Plot 2: GO enrichment (if available)
    if args.go_enrichment:
        print(f"\nLoading GO enrichment from {args.go_enrichment}...")
        go_df = pd.read_csv(args.go_enrichment)
        print(f"Loaded {len(go_df)} GO terms")

        if not go_df.empty:
            print(f"Creating GO enrichment plot...")
            fig2 = plot_go_enrichment(go_df, top_n=args.top_n, figsize=figsize)
            if fig2:
                output2 = output_dir / 'go_enrichment.png'
                fig2.savefig(output2, dpi=args.dpi, bbox_inches='tight')
                print(f"Saved to {output2}")
        else:
            print("GO enrichment results empty - skipping plot")

    # Plot 3: GWAS validation (if available)
    if args.gwas_validation:
        print(f"\nLoading GWAS validation from {args.gwas_validation}...")
        gwas_df = pd.read_csv(args.gwas_validation)
        print(f"Loaded {len(gwas_df)} genes")

        if not gwas_df.empty and 'in_gwas' in gwas_df.columns:
            n_gwas = gwas_df['in_gwas'].sum()
            print(f"Found {n_gwas} GWAS-validated genes")

            if n_gwas > 0:
                print(f"Creating GWAS validation plot...")
                fig3 = plot_gwas_validation(gwas_df, top_n=args.top_n, figsize=figsize)
                if fig3:
                    output3 = output_dir / 'gwas_validated_genes.png'
                    fig3.savefig(output3, dpi=args.dpi, bbox_inches='tight')
                    print(f"Saved to {output3}")
        else:
            print("No GWAS validation data - skipping plot")

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
