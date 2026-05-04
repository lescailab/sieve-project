#!/usr/bin/env python3
"""Aggregate gene-level interaction signals for the SIEVE project."""

import argparse
import logging
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PAIR_COLUMNS = [
    "gene_a",
    "gene_b",
    "n_cooccur",
    "n_cooccur_cases",
    "n_cooccur_controls",
    "freq_gene_a",
    "freq_gene_b",
    "expected_cooccur",
    "obs_exp_ratio",
    "gene_score_a",
    "gene_score_b",
    "gene_rank_a",
    "gene_rank_b",
    "significant_variant_count_a",
    "significant_variant_count_b",
    "interaction_score",
    "same_chrom",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate gene-level interaction signals for SIEVE.",
    )
    parser.add_argument(
        "--preprocessed-data",
        type=Path,
        required=True,
        help="Path to preprocessed .pt file.",
    )
    parser.add_argument(
        "--variant-rankings",
        type=Path,
        required=True,
        help=(
            "Path to variant rankings. Prefer corrected_variant_rankings.csv so "
            "the interaction score is based on chromosome-normalised z-scores."
        ),
    )
    parser.add_argument(
        "--gene-rankings",
        type=Path,
        required=True,
        help=(
            "Path to gene rankings. Prefer corrected_gene_rankings.csv so gene "
            "selection aligns with chromosome-normalised z-scores."
        ),
    )
    parser.add_argument(
        "--null-rankings",
        type=Path,
        default=None,
        help=(
            "Optional null-model variant rankings CSV. When provided, null thresholds "
            "are recomputed in the same score space as --variant-rankings."
        ),
    )
    parser.add_argument(
        "--cooccurrence",
        type=Path,
        default=None,
        help="Path to cooccurrence_per_pair.csv (optional, from co-occurrence audit).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--min-cooccur-samples",
        type=int,
        default=5,
        help="Minimum number of co-occurring samples to keep a gene pair (default: 5).",
    )
    parser.add_argument(
        "--top-k-genes",
        type=int,
        default=50,
        help="Number of top genes by gene_score to consider (default: 50).",
    )
    parser.add_argument(
        "--min-gene-score",
        type=float,
        default=0.0,
        help="Minimum gene score after all filtering (default: 0.0).",
    )
    parser.add_argument(
        "--score-column",
        choices=["z_attribution", "delta_rank"],
        default="z_attribution",
        help=(
            "Variant-level score column used for gene scoring and ranking. "
            "'z_attribution' uses chrX-corrected z-scores (default, preserves "
            "continuity with earlier runs). 'delta_rank' uses the bootstrap-null-"
            "anchored rank difference from bootstrap_null_calibration.py — "
            "requires a rank-calibrated input CSV."
        ),
    )
    parser.add_argument(
        "--significance-threshold",
        type=str,
        default="p_0.01",
        choices=["p_0.05", "p_0.01", "p_0.001"],
        help=(
            "Null-derived significance threshold to enforce when available. "
            "Note: bootstrap empirical p-values are floored at 1/(B+1) (default "
            "B=1000), so this acts as a coarse pass/fail filter rather than a "
            "fine-grained significance gate. For ranking, prefer --score-column "
            "delta_rank with a top-K cutoff via --top-k-genes, combined with "
            "--allow-nonsignificant-genes to disable the floored-p filter."
        ),
    )
    parser.add_argument(
        "--min-significant-variants",
        type=int,
        default=1,
        help=(
            "Minimum number of null-significant variants a gene must have to be kept "
            "when significance information is available."
        ),
    )
    parser.add_argument(
        "--allow-nonsignificant-genes",
        action="store_true",
        help=(
            "Allow genes with no null-significant variants. Recommended when "
            "running with --score-column delta_rank: keep --null-rankings so "
            "the per-variant exceeds_null_* annotations remain in the output as "
            "descriptive metadata, but pass this flag so gene selection and "
            "ranking are driven by delta_rank rather than by the floored "
            "bootstrap p-value gate."
        ),
    )
    return parser.parse_args()


def compute_chromosome_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-chromosome z-scores for a rankings dataframe."""
    required = {"chromosome", "mean_attribution"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Cannot compute z-scores. Missing columns: {missing}")

    corrected = df.copy()
    corrected["chromosome"] = corrected["chromosome"].astype(str)

    chrom_stats = corrected.groupby("chromosome")["mean_attribution"].agg(["mean", "std"])
    chrom_stats.columns = ["chromosome_mean", "chromosome_std"]
    chrom_stats["chromosome_std"] = (
        chrom_stats["chromosome_std"].replace(0, 1.0).fillna(1.0)
    )
    chrom_stats["chromosome_mean"] = chrom_stats["chromosome_mean"].fillna(0.0)

    corrected = corrected.merge(chrom_stats, left_on="chromosome", right_index=True, how="left")
    corrected["z_attribution"] = (
        (corrected["mean_attribution"] - corrected["chromosome_mean"])
        / corrected["chromosome_std"]
    )
    return corrected


def _resolve_gene_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve a gene-symbol column compatible with preprocessed SampleVariants."""
    resolved = df.copy()

    if "gene_symbol" in resolved.columns and not resolved["gene_symbol"].isna().all():
        resolved["gene_symbol"] = resolved["gene_symbol"].astype(str)
        return resolved
    if "gene_name" in resolved.columns and not resolved["gene_name"].isna().all():
        resolved["gene_symbol"] = resolved["gene_name"].astype(str)
        return resolved
    if "gene" in resolved.columns and not resolved["gene"].isna().all():
        resolved["gene_symbol"] = resolved["gene"].astype(str)
        return resolved
    if "gene_id" in resolved.columns:
        resolved["gene_symbol"] = resolved["gene_id"].astype(str)
        return resolved
    raise ValueError("Input rankings must contain one of: gene_symbol, gene_name, gene, gene_id.")


def standardise_variant_rankings(
    variant_df: pd.DataFrame,
    desired_score_column: Optional[str] = None,
) -> pd.DataFrame:
    """Standardise variant ranking columns and derive a comparable score column."""
    df = _resolve_gene_symbol_column(variant_df)

    if "chromosome" not in df.columns or "position" not in df.columns:
        raise ValueError("Variant rankings must contain 'chromosome' and 'position' columns.")
    df["chromosome"] = df["chromosome"].astype(str)

    if desired_score_column == "z_attribution" and "z_attribution" not in df.columns:
        df = compute_chromosome_zscores(df)

    if desired_score_column is None:
        if "z_attribution" in df.columns:
            score_column = "z_attribution"
        elif "mean_attribution" in df.columns:
            score_column = "mean_attribution"
        else:
            raise ValueError("Variant rankings must contain 'z_attribution' or 'mean_attribution'.")
    else:
        if desired_score_column not in df.columns:
            raise ValueError(
                f"Variant rankings are missing requested score column '{desired_score_column}'."
            )
        score_column = desired_score_column

    df["variant_score"] = pd.to_numeric(df[score_column], errors="coerce")
    df["variant_rank"] = np.nan
    if "corrected_rank" in df.columns:
        df["variant_rank"] = pd.to_numeric(df["corrected_rank"], errors="coerce")
    elif "rank" in df.columns:
        df["variant_rank"] = pd.to_numeric(df["rank"], errors="coerce")

    return df


def annotate_with_null_significance(
    real_variant_df: pd.DataFrame,
    null_rankings_path: Optional[Path],
    significance_threshold: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Annotate real variants with null-derived significance in matching score space."""
    threshold_col = f"exceeds_null_{significance_threshold}"
    score_column = "variant_score"
    metadata: dict[str, Any] = {
        "score_basis": "corrected_z_scores" if "z_attribution" in real_variant_df.columns else "mean_attribution",
        "significance_source": "none",
        "threshold_name": significance_threshold,
    }

    if threshold_col in real_variant_df.columns:
        annotated = real_variant_df.copy()
        annotated[threshold_col] = annotated[threshold_col].fillna(False).astype(bool)
        metadata["significance_source"] = "precomputed_columns"
        return annotated, metadata

    if null_rankings_path is None:
        return real_variant_df.copy(), metadata

    null_df = pd.read_csv(null_rankings_path)
    desired_score_column = "z_attribution" if "z_attribution" in real_variant_df.columns else "mean_attribution"
    null_df = standardise_variant_rankings(null_df, desired_score_column=desired_score_column)

    thresholds = {
        "p_0.05": float(np.percentile(null_df[score_column].dropna().values, 95)),
        "p_0.01": float(np.percentile(null_df[score_column].dropna().values, 99)),
        "p_0.001": float(np.percentile(null_df[score_column].dropna().values, 99.9)),
    }

    annotated = real_variant_df.copy()
    for name, threshold in thresholds.items():
        annotated[f"null_{name}_threshold"] = threshold
        annotated[f"exceeds_null_{name}"] = annotated[score_column] > threshold

    metadata["significance_source"] = "null_rankings_recomputed"
    metadata["null_rankings"] = str(null_rankings_path)
    metadata["thresholds"] = thresholds
    return annotated, metadata


def summarise_gene_support(
    variant_df: pd.DataFrame,
    significance_threshold: str,
) -> pd.DataFrame:
    """Summarise per-gene support from corrected variant rankings."""
    threshold_col = f"exceeds_null_{significance_threshold}"
    has_significance = threshold_col in variant_df.columns
    working = variant_df.copy()
    if has_significance:
        working[threshold_col] = working[threshold_col].fillna(False).astype(bool)

    rows = []
    for gene_symbol, gene_variants in working.groupby("gene_symbol", dropna=False):
        if has_significance:
            significant = gene_variants[gene_variants[threshold_col]]
            significant_variant_count = int(len(significant))
            max_significant_variant_score = (
                float(significant["variant_score"].max())
                if not significant.empty
                else np.nan
            )
        else:
            significant_variant_count = 0
            max_significant_variant_score = np.nan

        rows.append(
            {
                "gene_symbol": str(gene_symbol),
                "n_ranked_variants": int(len(gene_variants)),
                "max_variant_score": float(gene_variants["variant_score"].max()),
                "mean_variant_score": float(gene_variants["variant_score"].mean()),
                "significant_variant_count": significant_variant_count,
                "max_significant_variant_score": max_significant_variant_score,
                "has_significant_variant": bool(significant_variant_count > 0),
            }
        )

    return pd.DataFrame(rows)


def load_gene_rankings(
    gene_rankings_path: Path,
    variant_rankings_df: pd.DataFrame,
    top_k: int,
    min_score: float,
    significance_threshold: str,
    min_significant_variants: int,
    allow_nonsignificant_genes: bool,
    score_column: str = "z_attribution",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load gene rankings, align them to corrected/null-aware variant support, and select genes."""
    gene_df = pd.read_csv(gene_rankings_path)
    logger.info("Loaded %d genes from %s", len(gene_df), gene_rankings_path)
    gene_df = _resolve_gene_symbol_column(gene_df)

    if (
        "gene_id" in gene_df.columns
        and "gene_id" in variant_rankings_df.columns
        and "gene_name" in variant_rankings_df.columns
        and (
            "gene_name" not in gene_df.columns
            or gene_df["gene_name"].isna().all()
        )
    ):
        gene_name_map = (
            variant_rankings_df[["gene_id", "gene_name"]]
            .dropna(subset=["gene_name"])
            .drop_duplicates(subset=["gene_id"])
        )
        gene_df = gene_df.merge(gene_name_map, on="gene_id", how="left")
        if "gene_name" in gene_df.columns and not gene_df["gene_name"].isna().all():
            gene_df["gene_symbol"] = gene_df["gene_name"].astype(str)

    if score_column == "delta_rank":
        if "gene_delta_rank" not in gene_df.columns:
            raise ValueError(
                "Requested --score-column delta_rank but gene rankings file "
                f"{gene_rankings_path} has no 'gene_delta_rank' column. "
                "Run bootstrap_null_calibration.py to produce a rank-calibrated "
                "gene-stats CSV before running this script with delta_rank."
            )
        gene_df["gene_score"] = pd.to_numeric(gene_df["gene_delta_rank"], errors="coerce")

    if "gene_score" not in gene_df.columns:
        if "gene_z_score" in gene_df.columns:
            gene_df["gene_score"] = pd.to_numeric(gene_df["gene_z_score"], errors="coerce")
        elif "mean_z_score" in gene_df.columns:
            gene_df["gene_score"] = pd.to_numeric(gene_df["mean_z_score"], errors="coerce")
        else:
            raise ValueError(
                "Gene rankings must contain 'gene_score', 'gene_z_score', or 'mean_z_score'."
            )

    if "gene_rank" not in gene_df.columns:
        gene_df["gene_rank"] = gene_df["gene_score"].rank(
            method="dense",
            ascending=False,
        ).astype(int)

    gene_df["gene_score_raw"] = pd.to_numeric(gene_df["gene_score"], errors="coerce")
    gene_df["gene_rank_raw"] = pd.to_numeric(gene_df["gene_rank"], errors="coerce")

    support_df = summarise_gene_support(variant_rankings_df, significance_threshold)
    gene_df = gene_df.merge(support_df, on="gene_symbol", how="left")
    gene_df["n_ranked_variants"] = gene_df["n_ranked_variants"].fillna(0).astype(int)
    gene_df["significant_variant_count"] = (
        gene_df["significant_variant_count"].fillna(0).astype(int)
    )
    gene_df["has_significant_variant"] = (
        gene_df["has_significant_variant"].fillna(False).astype(bool)
    )

    has_significance = f"exceeds_null_{significance_threshold}" in variant_rankings_df.columns
    if has_significance:
        gene_df["gene_score"] = gene_df["max_significant_variant_score"].where(
            gene_df["significant_variant_count"] > 0,
            np.nan,
        )
        gene_df["gene_score_source"] = "max_significant_corrected_variant_score"
        if allow_nonsignificant_genes:
            gene_df["gene_score"] = gene_df["gene_score"].fillna(gene_df["gene_score_raw"])
            gene_df["gene_score_source"] = np.where(
                gene_df["significant_variant_count"] > 0,
                "max_significant_corrected_variant_score",
                "raw_gene_score_fallback",
            )
        else:
            before = len(gene_df)
            gene_df = gene_df[
                gene_df["significant_variant_count"] >= min_significant_variants
            ].copy()
            logger.info(
                "Filtered genes by null significance: %d -> %d (threshold=%s, min_significant_variants=%d)",
                before,
                len(gene_df),
                significance_threshold,
                min_significant_variants,
            )
    else:
        gene_df["gene_score"] = gene_df["gene_score_raw"]
        gene_df["gene_score_source"] = "raw_gene_rankings"

    gene_df = gene_df[pd.notna(gene_df["gene_score"])].copy()

    if min_score > 0.0:
        before = len(gene_df)
        gene_df = gene_df[gene_df["gene_score"] >= min_score].copy()
        logger.info(
            "After min_gene_score filter (>= %.4f): %d -> %d genes",
            min_score,
            before,
            len(gene_df),
        )

    gene_df = gene_df.sort_values(
        ["gene_score", "significant_variant_count", "gene_score_raw", "gene_symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    gene_df["gene_rank"] = range(1, len(gene_df) + 1)
    gene_df = gene_df.head(top_k).reset_index(drop=True)
    logger.info("Selected top-%d genes by cleaned gene score", len(gene_df))

    metadata = {
        "uses_null_significance": has_significance,
        "significance_threshold": significance_threshold,
        "allow_nonsignificant_genes": allow_nonsignificant_genes,
        "min_significant_variants": min_significant_variants,
        "gene_score_source": (
            "max_significant_corrected_variant_score" if has_significance else "raw_gene_rankings"
        ),
    }
    return gene_df, metadata


def build_carrier_indices(
    all_samples: list[Any],
) -> tuple[dict[str, set[int]], dict[int, int], dict[str, str], int]:
    """Build gene-level carrier indices from preprocessed sample data."""
    gene_to_samples: dict[str, set[int]] = defaultdict(set)
    sample_labels: dict[int, int] = {}
    gene_to_chrom: dict[str, str] = {}

    for idx, sample in enumerate(all_samples):
        sample_labels[idx] = int(sample.label)
        genes_in_sample: set[str] = set()
        for variant in sample.variants:
            gene = str(variant.gene)
            if not gene or gene in genes_in_sample:
                continue
            genes_in_sample.add(gene)
            gene_to_samples[gene].add(idx)
            gene_to_chrom.setdefault(gene, str(variant.chrom))

    total_samples = len(all_samples)
    logger.info(
        "Built carrier index: %d samples, %d unique genes",
        total_samples,
        len(gene_to_samples),
    )
    return gene_to_samples, sample_labels, gene_to_chrom, total_samples


def compute_gene_pair_cooccurrence(
    top_genes: list[str],
    gene_to_samples: dict[str, set[int]],
    sample_labels: dict[int, int],
    total_samples: int,
    gene_to_chrom: dict[str, str],
    gene_rankings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute co-occurrence statistics for all pairs among the top genes."""
    gene_score_map = dict(
        zip(gene_rankings_df["gene_symbol"], gene_rankings_df["gene_score"])
    )
    gene_rank_map = dict(
        zip(gene_rankings_df["gene_symbol"], gene_rankings_df["gene_rank"])
    )
    if "significant_variant_count" in gene_rankings_df.columns:
        significant_count_map = dict(
            zip(gene_rankings_df["gene_symbol"], gene_rankings_df["significant_variant_count"])
        )
    else:
        significant_count_map = {}

    rows: list[dict[str, Any]] = []
    sorted_genes = sorted(set(top_genes))

    for gene_a, gene_b in combinations(sorted_genes, 2):
        samples_a = gene_to_samples.get(gene_a, set())
        samples_b = gene_to_samples.get(gene_b, set())
        cooccur = samples_a & samples_b

        n_cooccur = len(cooccur)
        n_cooccur_cases = sum(1 for sample_idx in cooccur if sample_labels.get(sample_idx) == 1)
        n_cooccur_controls = n_cooccur - n_cooccur_cases

        freq_a = len(samples_a) / total_samples if total_samples > 0 else 0.0
        freq_b = len(samples_b) / total_samples if total_samples > 0 else 0.0
        expected = freq_a * freq_b * total_samples
        obs_exp_ratio = n_cooccur / expected if expected > 0 else float("nan")

        score_a = float(gene_score_map.get(gene_a, 0.0))
        score_b = float(gene_score_map.get(gene_b, 0.0))
        rank_a = int(gene_rank_map.get(gene_a, -1))
        rank_b = int(gene_rank_map.get(gene_b, -1))
        interaction_score = math.sqrt(max(score_a, 0.0) * max(score_b, 0.0)) * math.log1p(
            n_cooccur
        )

        chrom_a = gene_to_chrom.get(gene_a, "")
        chrom_b = gene_to_chrom.get(gene_b, "")

        rows.append(
            {
                "gene_a": gene_a,
                "gene_b": gene_b,
                "n_cooccur": n_cooccur,
                "n_cooccur_cases": n_cooccur_cases,
                "n_cooccur_controls": n_cooccur_controls,
                "freq_gene_a": round(freq_a, 6),
                "freq_gene_b": round(freq_b, 6),
                "expected_cooccur": round(expected, 4),
                "obs_exp_ratio": round(obs_exp_ratio, 4) if math.isfinite(obs_exp_ratio) else float("nan"),
                "gene_score_a": score_a,
                "gene_score_b": score_b,
                "gene_rank_a": rank_a,
                "gene_rank_b": rank_b,
                "significant_variant_count_a": int(significant_count_map.get(gene_a, 0)),
                "significant_variant_count_b": int(significant_count_map.get(gene_b, 0)),
                "interaction_score": round(interaction_score, 6),
                "same_chrom": bool(chrom_a and chrom_a == chrom_b),
            }
        )

    logger.info("Computed co-occurrence for %d gene pairs", len(rows))
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _normalise_gene_pair(
    gene_a: Any,
    gene_b: Any,
) -> tuple[str, str]:
    """Return a deterministic ordering for a gene pair."""
    ga = str(gene_a)
    gb = str(gene_b)
    return (ga, gb) if ga <= gb else (gb, ga)


def enrich_with_variant_cooccurrence(
    pairs_df: pd.DataFrame,
    cooccurrence_path: Path,
) -> pd.DataFrame:
    """Enrich gene-pair data with aggregated variant-level co-occurrence."""
    cooc_df = pd.read_csv(cooccurrence_path)
    logger.info("Loaded %d variant co-occurrence rows from %s", len(cooc_df), cooccurrence_path)

    gene_col_a = "gene_a" if "gene_a" in cooc_df.columns else None
    gene_col_b = "gene_b" if "gene_b" in cooc_df.columns else None
    if gene_col_a is None or gene_col_b is None:
        logger.warning(
            "Could not identify gene columns in cooccurrence file. Skipping variant-level enrichment."
        )
        pairs_df["n_variant_pairs"] = 0
        pairs_df["mean_variant_cooccur"] = 0.0
        return pairs_df

    cooccur_col = "n_cooccur" if "n_cooccur" in cooc_df.columns else None
    if cooccur_col is None:
        for candidate in cooc_df.columns:
            if "cooccur" in candidate.lower() or "count" in candidate.lower():
                cooccur_col = candidate
                break

    normalised = cooc_df.apply(
        lambda row: _normalise_gene_pair(row[gene_col_a], row[gene_col_b]),
        axis=1,
        result_type="expand",
    )
    cooc_df[["gene_a", "gene_b"]] = normalised

    if cooccur_col is None:
        agg = (
            cooc_df.groupby(["gene_a", "gene_b"])
            .size()
            .reset_index(name="n_variant_pairs")
        )
        agg["mean_variant_cooccur"] = 0.0
    else:
        agg = (
            cooc_df.groupby(["gene_a", "gene_b"])
            .agg(
                n_variant_pairs=(cooccur_col, "count"),
                mean_variant_cooccur=(cooccur_col, "mean"),
            )
            .reset_index()
        )

    enriched = pairs_df.merge(
        agg[["gene_a", "gene_b", "n_variant_pairs", "mean_variant_cooccur"]],
        on=["gene_a", "gene_b"],
        how="left",
    )
    enriched["n_variant_pairs"] = enriched["n_variant_pairs"].fillna(0).astype(int)
    enriched["mean_variant_cooccur"] = enriched["mean_variant_cooccur"].fillna(0.0)

    logger.info(
        "Enriched %d pairs with variant-level co-occurrence data",
        int((enriched["n_variant_pairs"] > 0).sum()),
    )
    return enriched


def build_network_outputs(
    pairs_df: pd.DataFrame,
    gene_rankings_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build gene-gene interaction network edge and node tables."""
    edge_columns = ["gene_a", "gene_b", "weight", "n_cooccur"]
    node_columns = [
        "gene",
        "degree",
        "gene_score",
        "gene_rank",
        "n_partners",
        "significant_variant_count",
        "gene_score_source",
    ]

    if pairs_df.empty:
        return pd.DataFrame(columns=edge_columns), pd.DataFrame(columns=node_columns)

    edges_df = pairs_df[["gene_a", "gene_b", "interaction_score", "n_cooccur"]].rename(
        columns={"interaction_score": "weight"}
    )

    degree: dict[str, int] = defaultdict(int)
    for _, row in pairs_df.iterrows():
        degree[str(row["gene_a"])] += 1
        degree[str(row["gene_b"])] += 1

    gene_score_map = dict(
        zip(gene_rankings_df["gene_symbol"], gene_rankings_df["gene_score"])
    )
    gene_rank_map = dict(
        zip(gene_rankings_df["gene_symbol"], gene_rankings_df["gene_rank"])
    )
    significant_count_map = (
        dict(zip(gene_rankings_df["gene_symbol"], gene_rankings_df["significant_variant_count"]))
        if "significant_variant_count" in gene_rankings_df.columns
        else {}
    )
    score_source_map = (
        dict(zip(gene_rankings_df["gene_symbol"], gene_rankings_df["gene_score_source"]))
        if "gene_score_source" in gene_rankings_df.columns
        else {}
    )

    nodes_rows = [
        {
            "gene": gene,
            "degree": deg,
            "gene_score": float(gene_score_map.get(gene, 0.0)),
            "gene_rank": int(gene_rank_map.get(gene, -1)),
            "n_partners": deg,
            "significant_variant_count": int(significant_count_map.get(gene, 0)),
            "gene_score_source": str(score_source_map.get(gene, "unknown")),
        }
        for gene, deg in sorted(degree.items(), key=lambda item: (-item[1], item[0]))
    ]
    return edges_df, pd.DataFrame(nodes_rows, columns=node_columns)


def write_summary(
    pairs_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Write a YAML summary of the gene interaction analysis."""
    top_pairs = [
        {
            "gene_a": row["gene_a"],
            "gene_b": row["gene_b"],
            "interaction_score": float(row["interaction_score"]),
            "n_cooccur": int(row["n_cooccur"]),
            "obs_exp_ratio": float(row["obs_exp_ratio"])
            if pd.notna(row["obs_exp_ratio"])
            else None,
            "significant_variant_count_a": int(row.get("significant_variant_count_a", 0)),
            "significant_variant_count_b": int(row.get("significant_variant_count_b", 0)),
        }
        for _, row in pairs_df.head(10).iterrows()
    ]

    hub_genes = [
        {
            "gene": row["gene"],
            "degree": int(row["degree"]),
            "gene_score": float(row["gene_score"]),
            "significant_variant_count": int(row.get("significant_variant_count", 0)),
        }
        for _, row in nodes_df.head(10).iterrows()
    ]

    summary: dict[str, Any] = {
        "total_gene_pairs": int(len(pairs_df)),
        "total_genes_in_network": int(len(nodes_df)),
        "top_10_pairs": top_pairs,
        "hub_genes": hub_genes,
        "score_basis": metadata,
        "comparison_with_epidetect_style_centrality": (
            "Degree is reported as a simple network-centrality proxy, analogous to the "
            "post-hoc hub analysis used in EpiDetect/EpiCID."
        ),
    }

    with open(output_path, "w") as handle:
        yaml.dump(summary, handle, default_flow_style=False, sort_keys=False)

    logger.info("Summary written to %s", output_path)


def main() -> None:
    """Main entry point for gene-level interaction aggregation."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_variant_df = pd.read_csv(args.variant_rankings)
    real_variant_df = standardise_variant_rankings(
        real_variant_df,
        desired_score_column=args.score_column,
    )
    # annotate_with_null_significance always derives null thresholds from
    # the z_attribution (or mean_attribution) space, regardless of --score-column.
    # When --score-column delta_rank is used, variant_score is delta_rank and
    # exceeds_null_* comparisons cross score spaces. This is intentional: the
    # p-value floor at 1/(B+1) makes these flags a coarse descriptive annotation
    # rather than a calibrated gate. Use --allow-nonsignificant-genes to disable
    # gene filtering by these flags and rely on delta_rank ordering instead.
    real_variant_df, significance_metadata = annotate_with_null_significance(
        real_variant_df,
        null_rankings_path=args.null_rankings,
        significance_threshold=args.significance_threshold,
    )

    gene_rankings_df, gene_metadata = load_gene_rankings(
        gene_rankings_path=args.gene_rankings,
        variant_rankings_df=real_variant_df,
        top_k=args.top_k_genes,
        min_score=args.min_gene_score,
        significance_threshold=args.significance_threshold,
        min_significant_variants=args.min_significant_variants,
        allow_nonsignificant_genes=args.allow_nonsignificant_genes,
        score_column=args.score_column,
    )

    logger.info("Loading preprocessed data from %s ...", args.preprocessed_data)
    preprocessed = torch.load(args.preprocessed_data, weights_only=False)
    all_samples = preprocessed["samples"]
    logger.info("Loaded %d samples", len(all_samples))

    gene_to_samples, sample_labels, gene_to_chrom, total_samples = build_carrier_indices(
        all_samples
    )

    top_genes = [
        gene
        for gene in gene_rankings_df["gene_symbol"].tolist()
        if gene in gene_to_samples
    ]
    logger.info(
        "Top genes retained after intersecting with cohort data: %d",
        len(top_genes),
    )

    pairs_df = compute_gene_pair_cooccurrence(
        top_genes=top_genes,
        gene_to_samples=gene_to_samples,
        sample_labels=sample_labels,
        total_samples=total_samples,
        gene_to_chrom=gene_to_chrom,
        gene_rankings_df=gene_rankings_df,
    )

    if args.cooccurrence is not None and not pairs_df.empty:
        pairs_df = enrich_with_variant_cooccurrence(pairs_df, args.cooccurrence)

    before = len(pairs_df)
    if not pairs_df.empty:
        pairs_df = pairs_df[pairs_df["n_cooccur"] >= args.min_cooccur_samples].copy()
    logger.info(
        "Filtered gene pairs: %d -> %d (min_cooccur_samples=%d)",
        before,
        len(pairs_df),
        args.min_cooccur_samples,
    )

    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(
            ["interaction_score", "n_cooccur"],
            ascending=[False, False],
        ).reset_index(drop=True)

    edges_df, nodes_df = build_network_outputs(pairs_df, gene_rankings_df)

    pairs_path = args.output_dir / "gene_pair_interactions.csv"
    pairs_df.to_csv(pairs_path, index=False)
    logger.info("Wrote %d gene pairs to %s", len(pairs_df), pairs_path)

    edges_path = args.output_dir / "gene_interaction_network_edges.csv"
    edges_df.to_csv(edges_path, index=False)
    logger.info("Wrote %d edges to %s", len(edges_df), edges_path)

    nodes_path = args.output_dir / "gene_interaction_network_nodes.csv"
    nodes_df.to_csv(nodes_path, index=False)
    logger.info("Wrote %d nodes to %s", len(nodes_df), nodes_path)

    summary_metadata = {
        **significance_metadata,
        **gene_metadata,
        "variant_rankings": str(args.variant_rankings),
        "gene_rankings": str(args.gene_rankings),
        "score_column": args.score_column,
    }
    summary_path = args.output_dir / "gene_interaction_summary.yaml"
    write_summary(pairs_df, nodes_df, summary_path, summary_metadata)
    logger.info("Gene interaction aggregation complete.")


if __name__ == "__main__":
    main()
