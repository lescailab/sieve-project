#!/usr/bin/env python3
"""
Compare corrected real variant attributions against a corrected null baseline.

This script expects two inputs that have already been processed by
correct_chrx_bias.py — that is, both must contain a ``z_attribution`` column.
It computes per-variant and per-gene empirical p-values using the
``(k + 1) / (N + 1)`` convention (Phipson & Smyth 2010, "Permutation P-values
Should Never Be Zero") and applies Benjamini–Hochberg FDR correction.

The ordering of operations is intentional and must not be changed:
  1. correct_chrx_bias.py on real rankings  →  corrected_variant_rankings.csv
  2. correct_chrx_bias.py on null rankings  →  corrected_variant_rankings.csv
  3. THIS SCRIPT on the two corrected files →  significance columns

Usage:
    python scripts/compare_attributions.py \\
        --corrected-real  /path/to/real/corrected/corrected_variant_rankings.csv \\
        --corrected-null  /path/to/null/corrected/corrected_variant_rankings.csv \\
        --output-dir      /path/to/comparison_output \\
        --genome-build    GRCh37

Author: Francesco Lescai
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.genome import get_genome_build, is_sex_chrom

try:
    from statsmodels.stats.multitest import multipletests
    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            'Compare corrected real and null variant attributions, '
            'producing empirical p-values and BH-FDR correction.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--corrected-real', type=str, required=True,
        help=(
            'Path to real corrected_variant_rankings.csv '
            '(output of correct_chrx_bias.py — must have z_attribution column)'
        ),
    )
    parser.add_argument(
        '--corrected-null', type=str, required=True,
        help=(
            'Path to null corrected_variant_rankings.csv '
            '(output of correct_chrx_bias.py — must have z_attribution column)'
        ),
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for significance-annotated files',
    )
    parser.add_argument(
        '--genome-build', type=str, default='GRCh37',
        help='Reference genome build (GRCh37 or GRCh38)',
    )
    parser.add_argument(
        '--exclude-sex-chroms', action='store_true', default=False,
        help=(
            'Drop sex-chromosome variants from both real and null before '
            'computing empirical p-values'
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_z_attribution(df: pd.DataFrame, label: str) -> None:
    """Raise ValueError if ``z_attribution`` is absent from *df*."""
    if 'z_attribution' not in df.columns:
        raise ValueError(
            f"The {label} rankings file does not have a 'z_attribution' column. "
            "Run correct_chrx_bias.py on this file before calling "
            "compare_attributions.py."
        )


# ---------------------------------------------------------------------------
# Sex-chromosome filtering
# ---------------------------------------------------------------------------

def _filter_sex_chroms(df: pd.DataFrame, genome_build_name: str) -> pd.DataFrame:
    """Remove sex-chromosome variants from *df* and return the filtered copy."""
    if 'chromosome' not in df.columns:
        print("  WARNING: 'chromosome' column not found — cannot filter sex chromosomes")
        return df
    build = get_genome_build(genome_build_name)
    mask = ~df['chromosome'].apply(lambda c: is_sex_chrom(str(c), build))
    n_removed = (~mask).sum()
    print(f"  Removed {n_removed} sex-chromosome variants "
          f"({len(df)} → {mask.sum()} autosomal)")
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Empirical p-value computation
# ---------------------------------------------------------------------------

def compute_empirical_pvalues(
    real_z: np.ndarray,
    null_z: np.ndarray,
) -> np.ndarray:
    """
    Compute per-variant empirical p-values against a null distribution.

    For each real value z_i, the empirical p-value is::

        p_i = (k + 1) / (N + 1)

    where k is the number of null values >= z_i and N is the total number of
    null values.  The ``+1`` in both numerator and denominator follows the
    recommendation in Phipson & Smyth (2010), "Permutation P-values Should
    Never Be Zero", ensuring that p-values are never exactly zero.

    Implementation uses ``np.searchsorted`` on a pre-sorted null array, giving
    O(N log N) for the sort and O(M log N) for the M lookups, where M is the
    number of real variants.

    Parameters
    ----------
    real_z : np.ndarray, shape (M,)
        z_attribution values for the real variants.
    null_z : np.ndarray, shape (N,)
        z_attribution values for the null variants.

    Returns
    -------
    np.ndarray, shape (M,)
        Empirical p-values in [1/(N+1), 1].
    """
    null_sorted = np.sort(null_z)
    N = len(null_sorted)
    # searchsorted(side='left') returns the number of null values < real_z[i]
    n_below = np.searchsorted(null_sorted, real_z, side='left')
    k = N - n_below  # number of null values >= real_z[i]
    return (k + 1) / (N + 1)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini–Hochberg FDR correction and return adjusted p-values."""
    if _HAS_STATSMODELS:
        _, fdr, _, _ = multipletests(p_values, method='fdr_bh')
        return fdr
    # Minimal fallback: manual BH
    n = len(p_values)
    order = np.argsort(p_values)
    fdr = np.empty(n, dtype=float)
    fdr[order] = p_values[order] * n / (np.arange(1, n + 1))
    # Enforce monotonicity from right to left
    fdr = np.minimum.accumulate(fdr[order[::-1]])[::-1]
    fdr = np.minimum(fdr, 1.0)
    result = np.empty(n, dtype=float)
    result[order] = fdr
    return result


# ---------------------------------------------------------------------------
# Gene-level aggregation (mirrors create_gene_rankings in correct_chrx_bias.py)
# ---------------------------------------------------------------------------

def _aggregate_genes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a corrected variant rankings dataframe to gene level.

    Groups by gene symbol (``gene_name`` if present, else ``gene_id``), takes
    the max ``z_attribution`` as ``gene_z_score``, and counts variants.

    Parameters
    ----------
    df : pd.DataFrame
        Corrected variant rankings with ``z_attribution`` column.

    Returns
    -------
    pd.DataFrame
        Gene-level dataframe sorted descending by ``gene_z_score``.
    """
    gene_col = 'gene_name' if 'gene_name' in df.columns else 'gene_id'
    gene_agg = df.groupby(gene_col).agg(
        gene_z_score=('z_attribution', 'max'),
        num_variants=('z_attribution', 'count'),
    ).reset_index()
    gene_agg = gene_agg.sort_values('gene_z_score', ascending=False).reset_index(drop=True)
    return gene_agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('SIEVE Null-Contrast Significance Analysis')
    print('=' * 60)

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    print(f'\nLoading corrected real rankings from {args.corrected_real}')
    real_df = pd.read_csv(args.corrected_real)
    _require_z_attribution(real_df, 'real')
    real_df['chromosome'] = real_df['chromosome'].astype(str)
    print(f'  {len(real_df):,} real variants loaded')

    print(f'Loading corrected null rankings from {args.corrected_null}')
    null_df = pd.read_csv(args.corrected_null)
    _require_z_attribution(null_df, 'null')
    null_df['chromosome'] = null_df['chromosome'].astype(str)
    print(f'  {len(null_df):,} null variants loaded')

    # ------------------------------------------------------------------
    # Optionally filter sex chromosomes
    # ------------------------------------------------------------------
    if args.exclude_sex_chroms:
        print('\nFiltering sex chromosomes...')
        print('  Real:')
        real_df = _filter_sex_chroms(real_df, args.genome_build)
        print('  Null:')
        null_df = _filter_sex_chroms(null_df, args.genome_build)

    # ------------------------------------------------------------------
    # Distributional sanity check (stdout only — not written to any file)
    # ------------------------------------------------------------------
    real_z = real_df['z_attribution'].dropna().values
    null_z = null_df['z_attribution'].dropna().values

    ks_stat, ks_pval = stats.ks_2samp(real_z, null_z)
    mw_stat, mw_pval = stats.mannwhitneyu(real_z, null_z, alternative='greater')
    print(f'\nDistributional sanity check (not written to output):')
    print(f'  KS test:          statistic={ks_stat:.4f}, p={ks_pval:.2e}')
    print(f'  Mann-Whitney U:   statistic={mw_stat:.4f}, p={mw_pval:.2e}')
    print(f'  Real z-scores:    mean={real_z.mean():.4f}, std={real_z.std():.4f}')
    print(f'  Null z-scores:    mean={null_z.mean():.4f}, std={null_z.std():.4f}')

    N = len(null_z)
    min_achievable_p = 1.0 / (N + 1)
    print(f'\n  Null size N = {N:,}')
    print(f'  Minimum achievable empirical p = 1 / (N+1) = {min_achievable_p:.2e}')

    # ------------------------------------------------------------------
    # Variant-level empirical p-values and FDR
    # ------------------------------------------------------------------
    print('\nComputing per-variant empirical p-values...')
    real_df_out = real_df.copy()
    empirical_p_variant = compute_empirical_pvalues(
        real_df_out['z_attribution'].values, null_z,
    )
    real_df_out['empirical_p_variant'] = empirical_p_variant
    fdr_variant = _bh_fdr(empirical_p_variant)
    real_df_out['fdr_variant'] = fdr_variant

    n_sig_05_var = (fdr_variant < 0.05).sum()
    n_sig_01_var = (fdr_variant < 0.01).sum()
    print(f'  Variants with fdr_variant < 0.05: {n_sig_05_var:,}')
    print(f'  Variants with fdr_variant < 0.01: {n_sig_01_var:,}')

    # ------------------------------------------------------------------
    # Gene-level empirical p-values and FDR
    # ------------------------------------------------------------------
    print('\nAggregating to gene level...')
    real_genes = _aggregate_genes(real_df_out)
    null_genes = _aggregate_genes(null_df)

    null_gene_z = null_genes['gene_z_score'].dropna().values
    real_gene_z = real_genes['gene_z_score'].values
    print(f'  {len(real_genes):,} real genes, {len(null_genes):,} null genes')

    empirical_p_gene = compute_empirical_pvalues(real_gene_z, null_gene_z)
    real_genes['empirical_p_gene'] = empirical_p_gene
    fdr_gene = _bh_fdr(empirical_p_gene)
    real_genes['fdr_gene'] = fdr_gene

    n_sig_05_gene = (fdr_gene < 0.05).sum()
    n_sig_01_gene = (fdr_gene < 0.01).sum()
    print(f'  Genes with fdr_gene < 0.05: {n_sig_05_gene:,}')
    print(f'  Genes with fdr_gene < 0.01: {n_sig_01_gene:,}')

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    var_sig_path = output_dir / 'corrected_variant_rankings_with_significance.csv'
    real_df_out.to_csv(var_sig_path, index=False)
    print(f'\nSaved variant significance file to {var_sig_path}')

    gene_sig_path = output_dir / 'corrected_gene_rankings_with_significance.csv'
    real_genes.to_csv(gene_sig_path, index=False)
    print(f'Saved gene significance file to {gene_sig_path}')

    # ------------------------------------------------------------------
    # Significance summary YAML
    # ------------------------------------------------------------------
    def _count_passing(arr: np.ndarray, threshold: float) -> int:
        return int((arr < threshold).sum())

    summary = {
        'genome_build': args.genome_build,
        'exclude_sex_chroms': args.exclude_sex_chroms,
        'n_real_variants_tested': int(len(real_df_out)),
        'n_null_variants': int(N),
        'n_real_genes_tested': int(len(real_genes)),
        'n_null_genes': int(len(null_genes)),
        'min_achievable_empirical_p': float(min_achievable_p),
        'variant_significance': {
            'fdr_0.05': _count_passing(fdr_variant, 0.05),
            'fdr_0.01': _count_passing(fdr_variant, 0.01),
            'fdr_0.001': _count_passing(fdr_variant, 0.001),
        },
        'gene_significance': {
            'fdr_0.05': _count_passing(fdr_gene, 0.05),
            'fdr_0.01': _count_passing(fdr_gene, 0.01),
            'fdr_0.001': _count_passing(fdr_gene, 0.001),
        },
    }

    summary_path = output_dir / 'significance_summary.yaml'
    with open(summary_path, 'w') as fh:
        yaml.dump(summary, fh, default_flow_style=False, sort_keys=False)
    print(f'Saved significance summary to {summary_path}')

    # ------------------------------------------------------------------
    # Final stdout summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  Real variants tested:       {len(real_df_out):,}')
    print(f'  Null variants:              {N:,}')
    print(f'  Min achievable p:           {min_achievable_p:.2e}')
    print(f'  Variants fdr_variant < 0.05: {n_sig_05_var:,}')
    print(f'  Genes fdr_gene < 0.05:       {n_sig_05_gene:,}')
    print('=' * 60)


if __name__ == '__main__':
    main()
