#!/usr/bin/env python3
"""
Compare real variant attributions against null baseline.

This script compares attribution scores from a model trained on real labels
against scores from a model trained on permuted labels (null baseline).
It establishes significance thresholds and identifies variants with
attributions unlikely to arise by chance.

Usage:
    python scripts/compare_attributions.py \
        --real results/real/sieve_variant_rankings.csv \
        --null results/null/sieve_variant_rankings.csv \
        --output-dir results/comparison

    # With multiple null permutations
    python scripts/compare_attributions.py \
        --real results/real/sieve_variant_rankings.csv \
        --null-dir results/null_permutations \
        --output-dir results/comparison

Author: Lescai Lab
"""

import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
import yaml


def load_null_attributions(
    null_path: Optional[str] = None,
    null_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Load null attribution(s) from single file or directory.

    Parameters
    ----------
    null_path : str, optional
        Path to single null rankings CSV
    null_dir : str, optional
        Directory containing multiple null rankings CSVs

    Returns
    -------
    pd.DataFrame
        Combined null attributions with 'permutation' column if multiple
    """
    if null_path:
        df = pd.read_csv(null_path)
        df['permutation'] = 0
        return df

    if null_dir:
        null_dir = Path(null_dir)
        null_files = sorted(null_dir.glob("*/sieve_variant_rankings.csv"))

        if not null_files:
            # Try direct CSV files
            null_files = sorted(null_dir.glob("*_variant_rankings*.csv"))

        if not null_files:
            raise ValueError(f"No null ranking files found in {null_dir}")

        dfs = []
        for i, f in enumerate(null_files):
            df = pd.read_csv(f)
            df['permutation'] = i
            dfs.append(df)
            print(f"  Loaded permutation {i}: {f.name}")

        return pd.concat(dfs, ignore_index=True)

    raise ValueError("Must provide either --null or --null-dir")


def compute_null_thresholds(null_attr: np.ndarray) -> Dict[str, float]:
    """
    Compute significance thresholds from null distribution.

    Parameters
    ----------
    null_attr : np.ndarray
        Null attribution values

    Returns
    -------
    dict
        Thresholds at various percentiles
    """
    return {
        'p_0.10': np.percentile(null_attr, 90),
        'p_0.05': np.percentile(null_attr, 95),
        'p_0.01': np.percentile(null_attr, 99),
        'p_0.001': np.percentile(null_attr, 99.9),
        'p_0.0001': np.percentile(null_attr, 99.99),
    }


def compare_distributions(
    real_attr: np.ndarray,
    null_attr: np.ndarray
) -> Dict[str, Any]:
    """
    Statistical comparison of real vs null distributions.

    Parameters
    ----------
    real_attr : np.ndarray
        Real attribution values
    null_attr : np.ndarray
        Null attribution values

    Returns
    -------
    dict
        Statistical comparison results
    """
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(real_attr, null_attr)

    # Mann-Whitney U test
    mw_stat, mw_pval = stats.mannwhitneyu(real_attr, null_attr, alternative='greater')

    # Basic statistics
    results = {
        'real_mean': float(np.mean(real_attr)),
        'real_std': float(np.std(real_attr)),
        'real_median': float(np.median(real_attr)),
        'real_max': float(np.max(real_attr)),
        'real_p95': float(np.percentile(real_attr, 95)),
        'real_p99': float(np.percentile(real_attr, 99)),

        'null_mean': float(np.mean(null_attr)),
        'null_std': float(np.std(null_attr)),
        'null_median': float(np.median(null_attr)),
        'null_max': float(np.max(null_attr)),
        'null_p95': float(np.percentile(null_attr, 95)),
        'null_p99': float(np.percentile(null_attr, 99)),

        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'mannwhitney_statistic': float(mw_stat),
        'mannwhitney_pvalue': float(mw_pval),
    }

    return results


def count_significant_variants(
    real_attr: np.ndarray,
    thresholds: Dict[str, float]
) -> Dict[str, Dict[str, Any]]:
    """
    Count variants exceeding null-derived thresholds.

    Parameters
    ----------
    real_attr : np.ndarray
        Real attribution values
    thresholds : dict
        Significance thresholds from null distribution

    Returns
    -------
    dict
        Counts and enrichments at each threshold
    """
    n_total = len(real_attr)
    results = {}

    for name, threshold in thresholds.items():
        # Parse p-value from name (e.g., 'p_0.05' -> 0.05)
        p_val = float(name.split('_')[1])
        expected = n_total * p_val
        observed = (real_attr > threshold).sum()

        enrichment = observed / expected if expected > 0 else np.nan

        results[name] = {
            'threshold': float(threshold),
            'observed': int(observed),
            'expected': float(expected),
            'enrichment': float(enrichment),
            'p_value': p_val,
        }

    return results


def generate_comparison_plots(
    real_df: pd.DataFrame,
    null_df: pd.DataFrame,
    thresholds: Dict[str, float],
    output_dir: Path
):
    """
    Generate comparison visualisations.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real variant rankings
    null_df : pd.DataFrame
        Null variant rankings
    thresholds : dict
        Significance thresholds
    output_dir : Path
        Output directory for plots
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return

    real_attr = real_df['mean_attribution'].values
    null_attr = null_df['mean_attribution'].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution comparison (histogram)
    ax1 = axes[0, 0]
    ax1.hist(null_attr, bins=100, alpha=0.5, label='Null', density=True, color='grey')
    ax1.hist(real_attr, bins=100, alpha=0.5, label='Real', density=True, color='steelblue')
    ax1.axvline(thresholds['p_0.05'], color='orange', linestyle='--',
                label=f"p<0.05: {thresholds['p_0.05']:.3f}")
    ax1.axvline(thresholds['p_0.01'], color='red', linestyle='--',
                label=f"p<0.01: {thresholds['p_0.01']:.3f}")
    ax1.set_xlabel('Mean Attribution')
    ax1.set_ylabel('Density')
    ax1.set_title('Attribution Distributions')
    ax1.legend()

    # 2. Q-Q plot
    ax2 = axes[0, 1]
    n_points = 1000
    real_quantiles = np.percentile(real_attr, np.linspace(0, 100, n_points))
    null_quantiles = np.percentile(null_attr, np.linspace(0, 100, n_points))
    ax2.scatter(null_quantiles, real_quantiles, alpha=0.5, s=5, color='steelblue')
    max_val = max(null_quantiles.max(), real_quantiles.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', label='y=x (no difference)')
    ax2.set_xlabel('Null Quantiles')
    ax2.set_ylabel('Real Quantiles')
    ax2.set_title('Q-Q Plot: Real vs Null')
    ax2.legend()

    # 3. Top variants comparison
    ax3 = axes[1, 0]
    top_n = min(100, len(real_df), len(null_df))
    real_top = real_df.nlargest(top_n, 'mean_attribution')['mean_attribution'].values
    null_top = null_df.groupby('permutation').apply(
        lambda x: x.nlargest(top_n, 'mean_attribution')['mean_attribution'].values
    )

    # Plot null as range
    if len(null_top) > 1:
        null_matrix = np.vstack(null_top.values)
        null_mean = null_matrix.mean(axis=0)
        null_std = null_matrix.std(axis=0)
        ax3.fill_between(range(1, top_n+1), null_mean - null_std, null_mean + null_std,
                        alpha=0.3, color='grey', label='Null ± 1 std')
        ax3.plot(range(1, top_n+1), null_mean, color='grey', label='Null mean')
    else:
        ax3.plot(range(1, top_n+1), null_top.values[0], color='grey',
                label='Null', marker='.', markersize=3)

    ax3.plot(range(1, top_n+1), real_top, color='steelblue',
            label='Real', marker='.', markersize=3)
    ax3.set_xlabel('Rank')
    ax3.set_ylabel('Mean Attribution')
    ax3.set_title(f'Top {top_n} Variants by Attribution')
    ax3.legend()

    # 4. Enrichment bar plot
    ax4 = axes[1, 1]
    p_levels = ['p_0.10', 'p_0.05', 'p_0.01', 'p_0.001']
    enrichments = []
    for p in p_levels:
        threshold = thresholds[p]
        p_val = float(p.split('_')[1])
        observed = (real_attr > threshold).sum()
        expected = len(real_attr) * p_val
        enrichments.append(observed / expected if expected > 0 else 0)

    bars = ax4.bar(range(len(p_levels)), enrichments, color='steelblue')
    ax4.axhline(1.0, color='red', linestyle='--', label='Expected (no enrichment)')
    ax4.set_xticks(range(len(p_levels)))
    ax4.set_xticklabels([p.replace('p_', 'p<') for p in p_levels])
    ax4.set_ylabel('Enrichment Factor')
    ax4.set_title('Enrichment Above Null Thresholds')
    ax4.legend()

    # Add enrichment values on bars
    for bar, enrich in zip(bars, enrichments):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{enrich:.1f}×', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'real_vs_null_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {output_dir / 'real_vs_null_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare real attributions against null baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--real', type=str, required=True,
                       help='Path to real variant rankings CSV')
    parser.add_argument('--null', type=str, default=None,
                       help='Path to single null variant rankings CSV')
    parser.add_argument('--null-dir', type=str, default=None,
                       help='Directory containing multiple null results')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for comparison results')
    parser.add_argument('--top-k', type=int, default=100,
                       help='Number of top variants to output')

    args = parser.parse_args()

    if not args.null and not args.null_dir:
        raise ValueError("Must provide either --null or --null-dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading real attributions...")
    real_df = pd.read_csv(args.real)
    real_attr = real_df['mean_attribution'].values

    print("Loading null attributions...")
    null_df = load_null_attributions(args.null, args.null_dir)
    null_attr = null_df['mean_attribution'].values

    n_permutations = null_df['permutation'].nunique()
    print(f"  Loaded {n_permutations} null permutation(s)")

    # Compute thresholds from null
    print("\nComputing null-derived significance thresholds...")
    thresholds = compute_null_thresholds(null_attr)

    print("\nSignificance Thresholds (from null distribution):")
    print("-" * 50)
    for name, value in thresholds.items():
        print(f"  {name.replace('p_', 'p < ')}: attribution > {value:.4f}")

    # Statistical comparison
    print("\nComparing distributions...")
    dist_comparison = compare_distributions(real_attr, null_attr)

    print("\nDistribution Statistics:")
    print("-" * 50)
    print(f"  {'Metric':<25} {'Real':>12} {'Null':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Mean':<25} {dist_comparison['real_mean']:>12.4f} {dist_comparison['null_mean']:>12.4f}")
    print(f"  {'Std':<25} {dist_comparison['real_std']:>12.4f} {dist_comparison['null_std']:>12.4f}")
    print(f"  {'Median':<25} {dist_comparison['real_median']:>12.4f} {dist_comparison['null_median']:>12.4f}")
    print(f"  {'Max':<25} {dist_comparison['real_max']:>12.4f} {dist_comparison['null_max']:>12.4f}")
    print(f"  {'95th percentile':<25} {dist_comparison['real_p95']:>12.4f} {dist_comparison['null_p95']:>12.4f}")
    print(f"  {'99th percentile':<25} {dist_comparison['real_p99']:>12.4f} {dist_comparison['null_p99']:>12.4f}")

    print(f"\n  Kolmogorov-Smirnov test: statistic={dist_comparison['ks_statistic']:.4f}, "
          f"p-value={dist_comparison['ks_pvalue']:.2e}")
    print(f"  Mann-Whitney U test: statistic={dist_comparison['mannwhitney_statistic']:.4f}, "
          f"p-value={dist_comparison['mannwhitney_pvalue']:.2e}")

    # Count significant variants
    print("\nSignificant Variants (exceeding null thresholds):")
    print("-" * 70)
    significance = count_significant_variants(real_attr, thresholds)

    print(f"  {'Threshold':<15} {'Observed':>10} {'Expected':>10} {'Enrichment':>12} {'Interpretation':<20}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*20}")

    for name, data in significance.items():
        interp = "Strong signal" if data['enrichment'] > 2 else "Weak/no signal"
        print(f"  {name.replace('p_', 'p < '):<15} {data['observed']:>10} "
              f"{data['expected']:>10.1f} {data['enrichment']:>11.1f}× {interp:<20}")

    # Generate plots
    print("\nGenerating comparison plots...")
    generate_comparison_plots(real_df, null_df, thresholds, output_dir)

    # Annotate real variants with significance
    print("\nAnnotating variants with significance levels...")
    real_df['null_p05_threshold'] = thresholds['p_0.05']
    real_df['null_p01_threshold'] = thresholds['p_0.01']
    real_df['null_p001_threshold'] = thresholds['p_0.001']
    real_df['exceeds_null_p05'] = real_df['mean_attribution'] > thresholds['p_0.05']
    real_df['exceeds_null_p01'] = real_df['mean_attribution'] > thresholds['p_0.01']
    real_df['exceeds_null_p001'] = real_df['mean_attribution'] > thresholds['p_0.001']

    # Save annotated rankings
    annotated_path = output_dir / 'variant_rankings_with_significance.csv'
    real_df.to_csv(annotated_path, index=False)
    print(f"Saved annotated rankings to {annotated_path}")

    # Save significant variants only
    sig_variants = real_df[real_df['exceeds_null_p01']].copy()
    sig_path = output_dir / 'significant_variants_p01.csv'
    sig_variants.to_csv(sig_path, index=False)
    print(f"Saved {len(sig_variants)} significant variants (p<0.01) to {sig_path}")

    # Save top-k with significance annotation
    top_k = real_df.nlargest(args.top_k, 'mean_attribution').copy()
    top_k_path = output_dir / f'top{args.top_k}_variants_annotated.csv'
    top_k.to_csv(top_k_path, index=False)
    print(f"Saved top {args.top_k} variants to {top_k_path}")

    # Save summary YAML
    summary = {
        'analysis_parameters': {
            'real_rankings': args.real,
            'null_source': args.null or args.null_dir,
            'n_null_permutations': n_permutations,
            'n_real_variants': len(real_df),
            'n_null_variants': len(null_df),
        },
        'thresholds': {k: float(v) for k, v in thresholds.items()},
        'distribution_comparison': dist_comparison,
        'significance_counts': {
            k: {kk: float(vv) if isinstance(vv, (int, float, np.floating, np.integer)) else vv
                for kk, vv in v.items()}
            for k, v in significance.items()
        },
        'interpretation': {
            'distributions_differ': dist_comparison['ks_pvalue'] < 0.001,
            'real_higher_than_null': dist_comparison['mannwhitney_pvalue'] < 0.001,
            'enrichment_at_p01': significance['p_0.01']['enrichment'],
            'n_significant_p01': significance['p_0.01']['observed'],
            'n_significant_p001': significance['p_0.001']['observed'],
        }
    }

    summary_path = output_dir / 'comparison_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"Saved summary to {summary_path}")

    # Final interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if dist_comparison['ks_pvalue'] < 0.001:
        print("✓ Real and null distributions are significantly different (KS p < 0.001)")
    else:
        print("✗ Real and null distributions are NOT significantly different")

    if significance['p_0.01']['enrichment'] > 2:
        print(f"✓ Strong enrichment at p<0.01: {significance['p_0.01']['enrichment']:.1f}× more variants than expected")
        print(f"  → {significance['p_0.01']['observed']} variants exceed null threshold")
        print("  → These variants are candidates for biological validation")
    elif significance['p_0.01']['enrichment'] > 1.5:
        print(f"~ Moderate enrichment at p<0.01: {significance['p_0.01']['enrichment']:.1f}×")
        print("  → Some signal present but interpret with caution")
    else:
        print(f"✗ Weak or no enrichment at p<0.01: {significance['p_0.01']['enrichment']:.1f}×")
        print("  → Attributions may not be distinguishable from noise")

    print("\nOutput files:")
    print(f"  - {annotated_path}")
    print(f"  - {sig_path}")
    print(f"  - {top_k_path}")
    print(f"  - {output_dir / 'real_vs_null_comparison.png'}")
    print(f"  - {summary_path}")


if __name__ == '__main__':
    main()
