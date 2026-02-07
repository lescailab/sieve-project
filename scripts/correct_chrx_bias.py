#!/usr/bin/env python3
"""
Post-hoc attribution correction for chrX ploidy bias.

Takes existing attribution rankings and produces corrected rankings with
chrX/chrY bias removed via per-chromosome z-score normalisation.

Usage:
    python scripts/correct_chrx_bias.py \
        --rankings /path/to/sieve_variant_rankings.csv \
        --output-dir /path/to/corrected_results \
        --null-rankings /path/to/null/sieve_variant_rankings.csv \
        --exclude-sex-chroms \
        --genome-build GRCh37 \
        --top-k 100

Author: Lescai Lab
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.genome import (
    GenomeBuild,
    get_genome_build,
    is_autosomal,
    is_sex_chrom,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Correct chrX ploidy bias in SIEVE variant attributions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--rankings', type=str, required=True,
                        help='Path to variant rankings CSV')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for corrected files')
    parser.add_argument('--null-rankings', type=str, default=None,
                        help='Path to null model rankings for enrichment recomputation')
    parser.add_argument('--exclude-sex-chroms', dest='exclude_sex_chroms',
                        action='store_true', default=True,
                        help='Exclude chrX and chrY from final rankings (default)')
    parser.add_argument('--include-sex-chroms', dest='exclude_sex_chroms',
                        action='store_false',
                        help='Include sex chromosomes (flagged) in final rankings')
    parser.add_argument('--genome-build', type=str, default='GRCh37',
                        help='Reference genome build')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of top variants to annotate in output')

    return parser.parse_args()


def compute_chromosome_zscores(
    df: pd.DataFrame, build: GenomeBuild,
) -> pd.DataFrame:
    """
    Compute per-chromosome z-scores of mean_attribution.

    Parameters
    ----------
    df : pd.DataFrame
        Variant rankings with 'chromosome' and 'mean_attribution' columns.
    build : GenomeBuild
        Genome build for sex chromosome identification.

    Returns
    -------
    pd.DataFrame
        Copy of df with added z_attribution, corrected_rank, is_sex_chrom,
        chromosome_mean, and chromosome_std columns.
    """
    df = df.copy()

    chrom_stats = df.groupby('chromosome')['mean_attribution'].agg(['mean', 'std'])
    chrom_stats.columns = ['chromosome_mean', 'chromosome_std']
    # Replace zero std with 1 to avoid division by zero
    chrom_stats['chromosome_std'] = chrom_stats['chromosome_std'].replace(0, 1.0)

    df = df.merge(chrom_stats, left_on='chromosome', right_index=True, how='left')

    df['z_attribution'] = (
        (df['mean_attribution'] - df['chromosome_mean']) / df['chromosome_std']
    )

    df['is_sex_chrom'] = df['chromosome'].apply(lambda c: is_sex_chrom(str(c), build))

    df['corrected_rank'] = df['z_attribution'].rank(ascending=False, method='min').astype(int)

    return df


def compute_enrichment_stats(
    real_df: pd.DataFrame, null_df: pd.DataFrame, build: GenomeBuild,
) -> dict:
    """
    Recompute enrichment statistics using autosomal variants only.

    Parameters
    ----------
    real_df : pd.DataFrame
        Corrected real variant rankings.
    null_df : pd.DataFrame
        Corrected null variant rankings.
    build : GenomeBuild
        Genome build.

    Returns
    -------
    dict
        Enrichment statistics dictionary.
    """
    real_auto = real_df[real_df['chromosome'].apply(
        lambda c: is_autosomal(str(c), build)
    )]['z_attribution'].dropna()
    null_auto = null_df[null_df['chromosome'].apply(
        lambda c: is_autosomal(str(c), build)
    )]['z_attribution'].dropna()

    if len(real_auto) == 0 or len(null_auto) == 0:
        return {'error': 'No autosomal variants for enrichment computation'}

    # KS test
    ks_stat, ks_pval = stats.ks_2samp(real_auto, null_auto)

    # Mann-Whitney U test
    u_stat, u_pval = stats.mannwhitneyu(real_auto, null_auto, alternative='greater')

    # Enrichment at various thresholds
    enrichment = {}
    for threshold in [0.10, 0.05, 0.01, 0.001]:
        null_cutoff = np.quantile(null_auto, 1 - threshold) if len(null_auto) > 0 else 0
        real_above = (real_auto > null_cutoff).sum()
        expected = threshold * len(real_auto)
        fold = real_above / expected if expected > 0 else 0
        enrichment[f'p_{threshold}'] = {
            'n_above_threshold': int(real_above),
            'expected': float(expected),
            'fold_enrichment': float(fold),
        }

    return {
        'genome_build': build.name,
        'n_real_autosomal_variants': int(len(real_auto)),
        'n_null_autosomal_variants': int(len(null_auto)),
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'mannwhitney_u_statistic': float(u_stat),
        'mannwhitney_u_pvalue': float(u_pval),
        'enrichment_thresholds': enrichment,
    }


def create_manhattan_plot(
    df: pd.DataFrame, build: GenomeBuild, output_path: Path,
    top_k: int = 10, exclude_sex: bool = True,
) -> None:
    """
    Generate a Manhattan plot using z-score attributions.

    Parameters
    ----------
    df : pd.DataFrame
        Corrected variant rankings with z_attribution.
    build : GenomeBuild
        Genome build for chromosome ordering.
    output_path : Path
        Where to save the plot.
    top_k : int
        Number of top variants to annotate with gene names.
    exclude_sex : bool
        If True, sex chromosomes are excluded; if False, shaded grey.
    """
    plot_df = df.copy()

    # Build chromosome order from GenomeBuild
    chrom_order = list(build.autosomal_chroms) + list(build.sex_chroms)
    chrom_to_idx = {c: i for i, c in enumerate(chrom_order)}

    plot_df['chrom_idx'] = plot_df['chromosome'].apply(
        lambda c: chrom_to_idx.get(str(c), -1)
    )
    plot_df = plot_df[plot_df['chrom_idx'] >= 0].copy()

    if exclude_sex:
        plot_df = plot_df[~plot_df['is_sex_chrom']].copy()

    if len(plot_df) == 0:
        print("WARNING: No variants to plot after filtering.")
        return

    # Sort by chromosome and position
    plot_df = plot_df.sort_values(['chrom_idx', 'position']).reset_index(drop=True)

    # Compute cumulative x positions
    plot_df['x'] = 0
    offset = 0
    chrom_centers = {}
    for chrom_idx in sorted(plot_df['chrom_idx'].unique()):
        mask = plot_df['chrom_idx'] == chrom_idx
        chrom_data = plot_df.loc[mask, 'position']
        plot_df.loc[mask, 'x'] = chrom_data - chrom_data.min() + offset
        center = offset + (chrom_data.max() - chrom_data.min()) / 2
        chrom_name = [c for c, i in chrom_to_idx.items() if i == chrom_idx][0]
        chrom_centers[chrom_name] = center
        offset += (chrom_data.max() - chrom_data.min()) + 1e6

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))

    colors = ['#1f77b4', '#aec7e8']
    for chrom_idx in sorted(plot_df['chrom_idx'].unique()):
        mask = plot_df['chrom_idx'] == chrom_idx
        chrom_name = [c for c, i in chrom_to_idx.items() if i == chrom_idx][0]
        is_sex = is_sex_chrom(chrom_name, build)
        if is_sex and not exclude_sex:
            color = '#cccccc'
        else:
            color = colors[chrom_idx % 2]

        ax.scatter(
            plot_df.loc[mask, 'x'],
            plot_df.loc[mask, 'z_attribution'],
            c=color, s=4, alpha=0.6, edgecolors='none',
        )

    # Significance threshold
    ax.axhline(y=3, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='z = 3')

    # Annotate top variants
    top_variants = plot_df.nlargest(top_k, 'z_attribution')
    for _, row in top_variants.iterrows():
        gene = row.get('gene_name', row.get('gene_id', ''))
        if gene and str(gene) != 'nan':
            ax.annotate(
                str(gene), (row['x'], row['z_attribution']),
                fontsize=6, rotation=45, ha='left', va='bottom',
                alpha=0.8,
            )

    # Axis labels
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('z-score attribution')
    ax.set_title('SIEVE Corrected Manhattan Plot (per-chromosome z-scores)')

    # Chromosome tick labels
    visible_centers = {c: v for c, v in chrom_centers.items()
                       if not (exclude_sex and is_sex_chrom(c, build))}
    ax.set_xticks(list(visible_centers.values()))
    ax.set_xticklabels(list(visible_centers.keys()), fontsize=7, rotation=45)

    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Manhattan plot to {output_path}")


def create_gene_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-aggregate gene rankings from corrected variant scores.

    Parameters
    ----------
    df : pd.DataFrame
        Corrected variant rankings with z_attribution.

    Returns
    -------
    pd.DataFrame
        Gene-level rankings.
    """
    gene_col = 'gene_name' if 'gene_name' in df.columns else 'gene_id'
    gene_agg = df.groupby(gene_col).agg(
        gene_z_score=('z_attribution', 'max'),
        num_variants=('z_attribution', 'count'),
        mean_z_score=('z_attribution', 'mean'),
        top_variant_pos=('position', 'first'),
    ).reset_index()
    gene_agg = gene_agg.sort_values('gene_z_score', ascending=False).reset_index(drop=True)
    gene_agg['gene_rank'] = range(1, len(gene_agg) + 1)
    return gene_agg


def main():
    """Main entry point."""
    args = parse_args()

    build = get_genome_build(args.genome_build)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SIEVE ChrX Ploidy Bias Correction")
    print("=" * 60)
    print(f"Genome build: {build.name}")
    print(f"Exclude sex chromosomes: {args.exclude_sex_chroms}")

    # Load rankings
    print(f"\nLoading rankings from {args.rankings}")
    df = pd.read_csv(args.rankings)
    print(f"  {len(df)} variants loaded")

    # Ensure chromosome column exists
    if 'chromosome' not in df.columns:
        print("ERROR: 'chromosome' column not found in rankings CSV.")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    # Ensure chromosome values are strings
    df['chromosome'] = df['chromosome'].astype(str)

    # Show pre-correction stats
    chrom_means = df.groupby('chromosome')['mean_attribution'].mean()
    print("\nPre-correction per-chromosome mean attributions:")
    for chrom in sorted(chrom_means.index, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 999, x)):
        flag = " <-- sex chrom" if is_sex_chrom(chrom, build) else ""
        print(f"  Chr {chrom}: {chrom_means[chrom]:.6f}{flag}")

    # Compute z-scores
    print("\nComputing per-chromosome z-scores...")
    corrected = compute_chromosome_zscores(df, build)

    # Filter if excluding sex chroms
    if args.exclude_sex_chroms:
        corrected_filtered = corrected[~corrected['is_sex_chrom']].copy()
        corrected_filtered['corrected_rank'] = (
            corrected_filtered['z_attribution'].rank(ascending=False, method='min').astype(int)
        )
        print(f"  Excluded {corrected['is_sex_chrom'].sum()} sex chromosome variants")
        print(f"  {len(corrected_filtered)} autosomal variants retained")
    else:
        corrected_filtered = corrected.copy()

    # Save corrected rankings
    corrected_path = output_dir / 'corrected_variant_rankings.csv'
    corrected.to_csv(corrected_path, index=False)
    print(f"\nSaved full corrected rankings to {corrected_path}")

    # Save top-k
    top_df = corrected_filtered.nsmallest(args.top_k, 'corrected_rank')
    top_path = output_dir / f'corrected_top{args.top_k}_variants.csv'
    top_df.to_csv(top_path, index=False)
    print(f"Saved top-{args.top_k} to {top_path}")

    # Gene rankings
    gene_rankings = create_gene_rankings(corrected_filtered)
    gene_path = output_dir / 'corrected_gene_rankings.csv'
    gene_rankings.to_csv(gene_path, index=False)
    print(f"Saved corrected gene rankings to {gene_path}")

    # Enrichment recomputation
    if args.null_rankings:
        print("\nRecomputing enrichment statistics (autosomes only)...")
        null_df = pd.read_csv(args.null_rankings)
        null_df['chromosome'] = null_df['chromosome'].astype(str)
        null_corrected = compute_chromosome_zscores(null_df, build)

        enrichment_stats = compute_enrichment_stats(corrected, null_corrected, build)
        enrichment_path = output_dir / 'corrected_comparison_summary.yaml'
        with open(enrichment_path, 'w') as f:
            yaml.dump(enrichment_stats, f, default_flow_style=False, sort_keys=False)
        print(f"Saved enrichment stats to {enrichment_path}")

        print(f"  KS statistic: {enrichment_stats.get('ks_statistic', 'N/A'):.4f}")
        print(f"  KS p-value: {enrichment_stats.get('ks_pvalue', 'N/A'):.2e}")

    # Manhattan plot
    print("\nGenerating corrected Manhattan plot...")
    manhattan_path = output_dir / 'corrected_manhattan_plot.png'
    create_manhattan_plot(
        corrected, build, manhattan_path,
        top_k=min(10, args.top_k),
        exclude_sex=args.exclude_sex_chroms,
    )

    # Correction report
    report = {
        'genome_build': build.name,
        'exclude_sex_chroms': args.exclude_sex_chroms,
        'total_variants': int(len(df)),
        'autosomal_variants': int((~corrected['is_sex_chrom']).sum()),
        'sex_chrom_variants': int(corrected['is_sex_chrom'].sum()),
        'per_chromosome': {},
    }

    for chrom in sorted(corrected['chromosome'].unique(),
                        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 999, x)):
        mask = corrected['chromosome'] == chrom
        report['per_chromosome'][chrom] = {
            'count': int(mask.sum()),
            'mean_attribution': float(corrected.loc[mask, 'mean_attribution'].mean()),
            'std_attribution': float(corrected.loc[mask, 'mean_attribution'].std()),
            'mean_z_score': float(corrected.loc[mask, 'z_attribution'].mean()),
            'is_sex_chrom': bool(is_sex_chrom(chrom, build)),
        }

    report_path = output_dir / 'correction_report.yaml'
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    print(f"Saved correction report to {report_path}")

    print("\nCorrection complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
