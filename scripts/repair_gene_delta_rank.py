#!/usr/bin/env python3
"""
Repair utility: patch existing gene-stats CSVs with gene_delta_rank columns.

Patches an existing gene-stats CSV produced by bootstrap_null_calibration.py
so that it carries gene_delta_rank and gene_delta_rank_aggregation columns,
without requiring the bootstrap to be re-run. For internal use only; not
referenced by any pipeline or documentation.

Usage:
    python scripts/repair_gene_delta_rank.py \
        --variant-rankings /path/to/rank_calibrated.csv \
        --gene-stats /path/to/gene_stats.csv \
        --output /path/to/gene_stats_repaired.csv \
        [--aggregation max|mean] \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MISSING_FRACTION_LIMIT = 0.01
INTEGRITY_SAMPLE_SIZE = 50
INTEGRITY_SEED = 20240101


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Patch existing gene-stats CSVs with gene_delta_rank and "
            "gene_delta_rank_aggregation columns derived from the companion "
            "variant-level rank-calibrated CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant-rankings",
        type=Path,
        required=True,
        help=(
            "Path to the variant-level rank-calibrated CSV containing "
            "delta_rank, gene_name (or gene_id), and the other bootstrap outputs."
        ),
    )
    parser.add_argument(
        "--gene-stats",
        type=Path,
        required=True,
        help="Path to the existing gene-stats CSV to patch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Destination path for the patched gene-stats CSV. "
            "Must not be the same path as --gene-stats."
        ),
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=("max", "mean"),
        default="max",
        help=(
            "How to aggregate variant-level delta_rank values to the gene level. "
            "'max' (default) matches the primary bootstrap workflow convention."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Report what would be computed and exit without writing any files.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    """Return the hex SHA256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_gene_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first candidate column found in df, or raise."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"No gene key column found. Expected one of: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def _integrity_check(
    gene_to_delta: dict[str, float],
    variant_df: pd.DataFrame,
    gene_col_variant: str,
    aggregation: str,
) -> None:
    """Spot-check aggregation for a random subset of genes."""
    rng = np.random.default_rng(INTEGRITY_SEED)
    all_genes = list(gene_to_delta.keys())
    sample_genes = rng.choice(
        all_genes, size=min(INTEGRITY_SAMPLE_SIZE, len(all_genes)), replace=False
    )
    for gene in sample_genes:
        variant_subset = variant_df.loc[
            variant_df[gene_col_variant] == gene, "delta_rank"
        ]
        if aggregation == "max":
            expected = float(variant_subset.max())
        else:
            expected = float(variant_subset.mean())
        computed = gene_to_delta[gene]
        if not math.isclose(computed, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError(
                f"Integrity self-check failed for gene {gene}: "
                f"computed={computed}, expected={expected}"
            )


def main(argv: list[str] | None = None) -> int:
    """Run the gene-stats repair."""
    args = parse_args(argv)

    if args.output.resolve() == args.gene_stats.resolve():
        print(
            "ERROR: --output and --gene-stats must be different paths. "
            "The repair script never overwrites the input gene-stats CSV.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading variant rankings from {args.variant_rankings}...")
    variant_df = pd.read_csv(args.variant_rankings)

    if "delta_rank" not in variant_df.columns:
        print(
            "ERROR: variant-rankings CSV does not contain a 'delta_rank' column.",
            file=sys.stderr,
        )
        sys.exit(1)

    gene_col_candidates = ["gene_name", "gene_id"]
    try:
        gene_col_variant = _find_gene_column(variant_df, gene_col_candidates)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading gene-stats from {args.gene_stats}...")
    gene_stats_df = pd.read_csv(args.gene_stats)

    if "gene_delta_rank" in gene_stats_df.columns or "gene_delta_rank_aggregation" in gene_stats_df.columns:
        print(
            "ERROR: gene-stats CSV already contains 'gene_delta_rank' or "
            "'gene_delta_rank_aggregation'. Refusing to overwrite existing values.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        gene_col_stats = _find_gene_column(gene_stats_df, gene_col_candidates)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if gene_col_stats != gene_col_variant:
        print(
            f"WARNING: gene key column in variant-rankings ('{gene_col_variant}') "
            f"differs from gene-stats ('{gene_col_stats}'). Proceeding with "
            f"'{gene_col_variant}' as the join key.",
        )

    print(f"Computing gene_delta_rank using aggregation='{args.aggregation}'...")
    variant_df[gene_col_variant] = variant_df[gene_col_variant].astype(str)
    gene_stats_df[gene_col_stats] = gene_stats_df[gene_col_stats].astype(str)

    if args.aggregation == "max":
        agg_series = variant_df.groupby(gene_col_variant, sort=False)["delta_rank"].max()
    else:
        agg_series = variant_df.groupby(gene_col_variant, sort=False)["delta_rank"].mean()

    gene_to_delta: dict[str, float] = agg_series.to_dict()

    stats_genes = set(gene_stats_df[gene_col_stats].astype(str))
    variant_genes = set(gene_to_delta.keys())
    missing_genes = stats_genes - variant_genes
    n_stats = len(stats_genes)
    n_matched = len(stats_genes & variant_genes)
    n_missing = len(missing_genes)

    if n_missing > 0 and (n_missing / n_stats) > MISSING_FRACTION_LIMIT:
        print(
            f"ERROR: {n_missing}/{n_stats} ({100.0 * n_missing / n_stats:.1f}%) of "
            "genes in gene-stats are absent from the variant-rankings file. "
            "This exceeds the 1% safety threshold and indicates the two files were "
            "not generated from the same bootstrap run. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    _MAX_MISSING_WARNINGS = 20
    for gene in sorted(missing_genes)[:_MAX_MISSING_WARNINGS]:
        print(f"  WARNING: gene '{gene}' in gene-stats has no variants in variant-rankings; delta_rank will be NaN.")
    if n_missing > _MAX_MISSING_WARNINGS:
        print(f"  WARNING: ... and {n_missing - _MAX_MISSING_WARNINGS} more missing genes (suppressed).")

    _integrity_check(gene_to_delta, variant_df, gene_col_variant, args.aggregation)

    gene_delta_col = gene_stats_df[gene_col_stats].map(gene_to_delta)
    gene_delta_rank_values = gene_delta_col.to_numpy(dtype=float)

    if args.dry_run:
        print(
            f"\n[DRY RUN] Would add columns gene_delta_rank and "
            f"gene_delta_rank_aggregation to {args.gene_stats}."
        )
        print(f"  aggregation: {args.aggregation}")
        print(f"  n_genes_in_gene_stats: {n_stats}")
        print(f"  n_genes_matched: {n_matched}")
        print(f"  n_genes_missing_from_variant_file: {n_missing}")
        finite_vals = gene_delta_rank_values[np.isfinite(gene_delta_rank_values)]
        if len(finite_vals):
            print(f"  gene_delta_rank min/median/max: "
                  f"{float(finite_vals.min()):.4f} / "
                  f"{float(np.median(finite_vals)):.4f} / "
                  f"{float(finite_vals.max()):.4f}")
            print(f"  fraction positive: {float((finite_vals > 0).mean()):.4f}")
        print(f"  output would be written to: {args.output}")
        return 0

    patched_df = gene_stats_df.copy()
    patched_df["gene_delta_rank"] = gene_delta_rank_values
    patched_df["gene_delta_rank_aggregation"] = args.aggregation

    args.output.parent.mkdir(parents=True, exist_ok=True)
    patched_df.to_csv(args.output, index=False)
    print(f"Wrote patched gene-stats to {args.output}")

    finite_vals = gene_delta_rank_values[np.isfinite(gene_delta_rank_values)]
    summary_min = float(finite_vals.min()) if len(finite_vals) else float("nan")
    summary_median = float(np.median(finite_vals)) if len(finite_vals) else float("nan")
    summary_max = float(finite_vals.max()) if len(finite_vals) else float("nan")
    fraction_positive = float((finite_vals > 0).mean()) if len(finite_vals) else float("nan")

    output_prefix = args.output.with_suffix("") if args.output.suffix else args.output
    log_path = output_prefix.with_name(output_prefix.stem + ".repair_log.txt")
    sha_variant = _sha256(args.variant_rankings)
    sha_gene_stats = _sha256(args.gene_stats)
    log_lines = [
        f"UTC timestamp: {datetime.datetime.utcnow().isoformat(timespec='seconds')}Z",
        f"aggregation: {args.aggregation}",
        f"n_genes_in_gene_stats: {n_stats}",
        f"n_genes_matched: {n_matched}",
        f"n_genes_missing_from_variant_file: {n_missing}",
        f"gene_delta_rank_min: {summary_min:.6g}",
        f"gene_delta_rank_median: {summary_median:.6g}",
        f"gene_delta_rank_max: {summary_max:.6g}",
        f"gene_delta_rank_fraction_positive: {fraction_positive:.6g}",
        f"sha256_variant_rankings: {sha_variant}",
        f"sha256_gene_stats: {sha_gene_stats}",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Wrote repair log to {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
