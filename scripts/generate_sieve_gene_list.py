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
    return parser.parse_args(argv)


def generate_gene_list(
    variant_rankings: pd.DataFrame,
    score_column: str = "z_attribution",
    exclude_sex_chroms: bool = True,
    min_null_threshold: str | None = None,
    aggregation: str = "max",
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

    # Reorder columns
    gene_df = gene_df[["gene_name", "gene_rank", "gene_score", "n_variants", "chromosome"]]

    print(f"Generated gene list: {len(gene_df)} genes")
    print(f"  Aggregation: {aggregation}")
    print(f"  Score column: {score_column}")
    print(f"  Top 5 genes: {list(gene_df['gene_name'].head())}")

    return gene_df


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load variant rankings
    df = pd.read_csv(args.variant_rankings)
    print(f"Loaded {len(df)} variant rankings from {args.variant_rankings}")

    exclude_sex = args.exclude_sex_chroms and not args.include_sex_chroms

    gene_df = generate_gene_list(
        df,
        score_column=args.score_column,
        exclude_sex_chroms=exclude_sex,
        min_null_threshold=args.min_null_threshold,
        aggregation=args.aggregation,
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
