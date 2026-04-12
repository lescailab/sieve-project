"""
Generate standardised gene list from SIEVE variant-level rankings.

Aggregates variant-level attribution scores to gene-level, applies filters,
and outputs the TSV required by the cross-cohort burden validation scripts.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data.genome import get_genome_build, is_sex_chrom, normalise_chrom


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate gene list TSV from SIEVE variant rankings.",
    )
    parser.add_argument(
        "--variant-rankings",
        type=Path,
        required=True,
        help="Corrected variant rankings CSV (output of correct_chrx_bias.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output gene list TSV path",
    )
    parser.add_argument(
        "--score-column",
        default="z_attribution",
        help="Column to use for scoring (default: z_attribution)",
    )
    parser.add_argument(
        "--exclude-sex-chroms",
        action="store_true",
        default=True,
        help="Exclude sex chromosome genes (default: True)",
    )
    parser.add_argument(
        "--include-sex-chroms",
        action="store_true",
        default=False,
        help="Include sex chromosome genes (overrides --exclude-sex-chroms)",
    )
    parser.add_argument(
        "--min-null-threshold",
        choices=["p05", "p01", "p001"],
        default=None,
        help="Only include genes with at least one variant exceeding this null threshold",
    )
    parser.add_argument(
        "--aggregation",
        choices=["max", "mean"],
        default="max",
        help="How to aggregate variant scores to gene level (default: max)",
    )
    parser.add_argument(
        "--ablation-level",
        default=None,
        help="If set, prefix output filename with level label",
    )
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=None,
        help=(
            "Only include genes with FDR below this threshold (e.g. 0.05). "
            "Determines gene set size dynamically from the cutoff. "
            "Requires --gene-significance or corrected gene rankings "
            "containing fdr_gene."
        ),
    )
    parser.add_argument(
        "--gene-significance",
        type=Path,
        default=None,
        help=(
            "Path to gene_rankings_with_significance.csv (or corrected "
            "gene rankings containing fdr_gene) for FDR-based filtering. "
            "Auto-discovered from the --variant-rankings parent directory "
            "if not specified."
        ),
    )
    return parser.parse_args(argv)


def generate_gene_list(
    variant_rankings: pd.DataFrame,
    score_column: str = "z_attribution",
    exclude_sex_chroms: bool = True,
    min_null_threshold: str | None = None,
    aggregation: str = "max",
    fdr_threshold: float | None = None,
    gene_significance_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Aggregate variant-level rankings to gene-level.

    Parameters
    ----------
    variant_rankings : pd.DataFrame
        Variant-level rankings with at least columns:
        gene_name, chromosome, and the specified score_column.
    score_column : str
        Column to aggregate for gene scoring.
    exclude_sex_chroms : bool
        Whether to exclude sex chromosome variants.
    min_null_threshold : str or None
        If set, only retain variants exceeding this null threshold.
        One of 'p05', 'p01', 'p001'.
    aggregation : str
        'max' or 'mean' aggregation of variant scores per gene.
    fdr_threshold : float or None
        If set, only retain genes with ``fdr_gene`` below this value.
        The gene set size is determined dynamically by the threshold.
    gene_significance_df : pd.DataFrame or None
        Gene-level significance file containing ``fdr_gene``. Used for
        FDR-threshold filtering. When *fdr_threshold* is set, this input
        must provide ``fdr_gene``; no fallback to aggregated
        ``fdr_variant`` values is performed.

    Returns
    -------
    pd.DataFrame
        Gene-level rankings with columns:
        gene_name, gene_rank, gene_score, n_variants, chromosome.
    """
    df = variant_rankings.copy()

    # Normalise chromosome column name
    if "chromosome" not in df.columns and "chrom" in df.columns:
        df = df.rename(columns={"chrom": "chromosome"})

    # Filter sex chromosomes using centralised genome build logic
    if exclude_sex_chroms:
        build = get_genome_build("GRCh37")
        before = len(df)
        df = df[
            ~df["chromosome"]
            .astype(str)
            .apply(lambda c: is_sex_chrom(normalise_chrom(c, build), build))
        ]
        n_removed = before - len(df)
        if n_removed > 0:
            print(f"Excluded {n_removed} sex chromosome variants")

    # Filter by null threshold
    if min_null_threshold is not None:
        threshold_col_map = {
            "p05": "exceeds_null_p05",
            "p01": "exceeds_null_p01",
            "p001": "exceeds_null_p001",
        }
        col = threshold_col_map[min_null_threshold]
        if col not in df.columns:
            print(
                f"Warning: threshold column '{col}' not found in rankings. "
                f"Available columns: {list(df.columns)}. Skipping threshold filter."
            )
        else:
            before = len(df)
            df = df[df[col].astype(bool)]
            print(
                f"Filtered to {len(df)} variants exceeding null {min_null_threshold} "
                f"(removed {before - len(df)})"
            )

    if score_column not in df.columns:
        raise ValueError(
            f"Score column '{score_column}' not found. "
            f"Available: {list(df.columns)}"
        )

    # Determine gene name column
    gene_col = "gene_name"
    if gene_col not in df.columns:
        if "gene" in df.columns:
            gene_col = "gene"
        else:
            raise ValueError(
                f"No gene name column found. "
                f"Expected 'gene_name' or 'gene'. Available: {list(df.columns)}"
            )

    # Aggregate to gene level
    agg_func = "max" if aggregation == "max" else "mean"
    gene_df = (
        df.groupby(gene_col)
        .agg(
            gene_score=(score_column, agg_func),
            n_variants=(score_column, "count"),
            chromosome=("chromosome", "first"),
        )
        .reset_index()
        .rename(columns={gene_col: "gene_name"})
    )

    # Rank by score (descending — highest score = rank 1)
    gene_df = gene_df.sort_values("gene_score", ascending=False).reset_index(drop=True)
    gene_df["gene_rank"] = range(1, len(gene_df) + 1)

    # Apply FDR threshold filtering
    if fdr_threshold is not None:
        fdr_col = None
        if gene_significance_df is not None:
            # Merge fdr_gene from the significance file
            sig_gene_col = (
                "gene_name" if "gene_name" in gene_significance_df.columns
                else "gene_id" if "gene_id" in gene_significance_df.columns
                else None
            )
            if sig_gene_col is not None and "fdr_gene" in gene_significance_df.columns:
                sig_subset = gene_significance_df[[sig_gene_col, "fdr_gene"]].copy()
                sig_subset = sig_subset.rename(columns={sig_gene_col: "gene_name"})
                gene_df = gene_df.merge(sig_subset, on="gene_name", how="left")
                fdr_col = "fdr_gene"
        if fdr_col is None and "fdr_gene" in gene_df.columns:
            fdr_col = "fdr_gene"
        if fdr_col is None:
            raise ValueError(
                f"FDR threshold filtering requested (--fdr-threshold {fdr_threshold}) "
                "but no fdr_gene column found. Provide --gene-significance pointing "
                "to gene_rankings_with_significance.csv or corrected_gene_rankings.csv "
                "containing fdr_gene."
            )
        before = len(gene_df)
        gene_df = gene_df[gene_df[fdr_col] < fdr_threshold].copy()
        print(f"FDR threshold < {fdr_threshold}: {len(gene_df)}/{before} genes pass")
        if len(gene_df) == 0:
            print("WARNING: No genes pass the FDR threshold")
        # Re-rank after filtering
        gene_df = gene_df.sort_values("gene_score", ascending=False).reset_index(drop=True)
        gene_df["gene_rank"] = range(1, len(gene_df) + 1)

    # Reorder columns — include fdr_gene if present
    output_cols = ["gene_name", "gene_rank", "gene_score", "n_variants", "chromosome"]
    if "fdr_gene" in gene_df.columns:
        output_cols.append("fdr_gene")
    gene_df = gene_df[output_cols]

    print(f"Generated gene list: {len(gene_df)} genes")
    print(f"  Aggregation: {aggregation}")
    print(f"  Score column: {score_column}")
    print(f"  Top 5 genes: {list(gene_df['gene_name'].head())}")

    return gene_df


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Validate FDR threshold range
    if args.fdr_threshold is not None and not (0 < args.fdr_threshold <= 1):
        print(
            f"ERROR: --fdr-threshold must be in (0, 1], got {args.fdr_threshold}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load variant rankings
    df = pd.read_csv(args.variant_rankings)
    print(f"Loaded {len(df)} variant rankings from {args.variant_rankings}")

    exclude_sex = args.exclude_sex_chroms and not args.include_sex_chroms

    # Load gene significance file for FDR filtering
    gene_sig_df = None
    if args.fdr_threshold is not None:
        if args.gene_significance is not None:
            gene_sig_path = args.gene_significance
        else:
            # Auto-discover in same directory as variant rankings
            gene_sig_path = args.variant_rankings.parent / "gene_rankings_with_significance.csv"
            if not gene_sig_path.exists():
                # Also try corrected gene rankings (may contain fdr_gene after merge)
                gene_sig_path = args.variant_rankings.parent / "corrected_gene_rankings.csv"

        if gene_sig_path.exists():
            gene_sig_df = pd.read_csv(gene_sig_path)
            print(f"Loaded gene significance from {gene_sig_path}")
        else:
            print(f"WARNING: Gene significance file not found at {gene_sig_path}")

    gene_df = generate_gene_list(
        df,
        score_column=args.score_column,
        exclude_sex_chroms=exclude_sex,
        min_null_threshold=args.min_null_threshold,
        aggregation=args.aggregation,
        fdr_threshold=args.fdr_threshold,
        gene_significance_df=gene_sig_df,
    )

    # Determine output path
    output_path = args.output
    if args.ablation_level:
        output_path = output_path.parent / f"{args.ablation_level}_{output_path.name}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gene_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved gene list to {output_path}")


if __name__ == "__main__":
    main()
