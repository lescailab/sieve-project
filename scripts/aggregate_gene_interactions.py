#!/usr/bin/env python3
"""Aggregate gene-level interaction signals for the SIEVE project.

The interaction score uses rank-quantile-normalised gene scores so that the
formula behaves consistently across --score-column choices. Under the legacy
formulation `sqrt(gene_score_a * gene_score_b) * log1p(n_cooccur)`, the two
factors were commensurable for `z_attribution`-derived scores (top values
~3-5) but not for `delta_rank`-derived scores (top values ~10^5), where the
importance term swamped the co-occurrence term by ~10^4. Computing rank
quantiles within the analysed top-K set (highest-scoring gene = 1.0,
lowest-scoring = 1/K) restores the intended balance between joint
importance and co-occurrence support, and makes the score scale-invariant
with respect to the input ranking column. Note that this is a heuristic for
candidate ranking, not an estimate of interaction effect; the
`obs_exp_ratio` column and the new `padj` / `reject` columns (see --correction
and --alternative) are the right quantities for departure-from-independence
inference.

Per-pair departure-from-independence is tested with a Fisher exact test
on the 2x2 carrier contingency table. The direction of the test is
controlled by --alternative:

  - 'greater' (default): tests for excess co-occurrence relative to
    independence. This is the appropriate test for classical
    synthetic-lethal-style epistasis hypotheses, where the scientific
    expectation is that carriers of both genes cluster together more
    than would occur under independent assortment.

  - 'less': tests for deficit of co-occurrence. Use when the scientific
    expectation is that the double-carrier state is selected against --
    for example, in early-onset or developmental phenotypes where the
    combination is incompatible with survival to recruitment, or when
    studying compound heterozygous lethality.

  - 'two-sided': tests for departure in either direction without prior
    commitment. Use when no a priori directional hypothesis is justified.
    Note that this halves per-direction power at the same alpha.

The choice of --alternative is a scientific hypothesis specification
and must be made BEFORE looking at the data. Selecting the direction
post-hoc (e.g. running 'greater', seeing no rejections, switching to
'less') inflates Type I error and invalidates the FDR control. If
genuine uncertainty about direction exists, use 'two-sided'.

Multiple-testing correction across all C(K, 2) tested pairs is applied
per --correction (default 'fdr_bh', Benjamini-Hochberg). The 'reject'
column flags pairs with padj < alpha. The metadata block in
gene_interaction_summary.yaml records both --correction and
--alternative so that any downstream re-interpretation has the
necessary audit trail.

Note that for the top SIEVE-prioritised genes in a typical exome
cohort, gene-level carrier indicators may be saturated (freq ~ 1.0
for large genes), which limits power for this gene-collapsed test
regardless of direction; consult the obs_exp_ratio, freq_a, and
freq_b columns when interpreting near-zero rejection counts.

The pair table is sorted by interaction_score descending, with padj
ascending as a tiebreaker so that among equally-scored pairs, more
significant pairs appear first.
"""

import argparse
import logging
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PAIR_COLUMNS = [
    "gene_a",
    "gene_b",
    "n_cooccur",
    "n_only_a",
    "n_only_b",
    "n_neither",
    "n_cooccur_cases",
    "n_cooccur_controls",
    "freq_gene_a",
    "freq_gene_b",
    "expected_cooccur",
    "obs_exp_ratio",
    "gene_score_a",
    "gene_score_b",
    "gene_score_quantile_a",
    "gene_score_quantile_b",
    "gene_rank_a",
    "gene_rank_b",
    "significant_variant_count_a",
    "significant_variant_count_b",
    "interaction_score",
    "same_chrom",
    "pval_fisher",
    "padj",
    "reject",
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
        nargs="+",
        default=[100],
        help=(
            "Top-K gene set sizes for the gene-pair analysis. May be specified "
            "as a single value or a list (e.g. '--top-k-genes 100 2000' to "
            "match the ablation analysis convention). The pair analysis is "
            "quadratic in K (C(K,2) pairs), so K=2000 produces ~2M pairs and "
            "yields a network too dense for hub interpretation; K=100 is the "
            "recommended primary value for interpretable network analysis."
        ),
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
            "anchored rank difference from bootstrap_null_calibration.py -- "
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
    parser.add_argument(
        "--correction",
        choices=["none", "bonferroni", "fdr_bh"],
        default="fdr_bh",
        help=(
            "Multiple-testing correction applied to per-pair departure-from-"
            "independence p-values. 'fdr_bh' (default) controls the false "
            "discovery rate via the Benjamini-Hochberg procedure. 'bonferroni' "
            "controls the family-wise error rate. 'none' reports raw p-values "
            "without correction."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the per-pair correction (default 0.05).",
    )
    parser.add_argument(
        "--alternative",
        choices=["greater", "two-sided", "less"],
        default="greater",
        help=(
            "Alternative hypothesis for the per-pair Fisher exact test on the "
            "2x2 carrier contingency table. 'greater' (default) tests for excess "
            "co-occurrence over independence (the classical synthetic-lethal-"
            "style epistasis hypothesis: carriers of both genes cluster together "
            "more than expected). 'less' tests for deficit of co-occurrence, "
            "where carrying both is under-represented relative to independence "
            "(e.g. survivorship bias against the double-carrier combination, "
            "or selection against double-hit zygotes). 'two-sided' tests for "
            "either direction without prior commitment. The choice is "
            "scientifically meaningful and should be made before looking at "
            "the data: changing direction post-hoc inflates Type I error. See "
            "documentation/detailed-usage.md for guidance."
        ),
    )
    return parser.parse_args()


def _compute_rank_quantiles(scores: dict[str, float]) -> dict[str, float]:
    """
    Map each gene to a rank quantile in (0, 1], computed within the supplied set.

    The gene with the highest finite positive gene_score receives quantile 1.0,
    the gene with the lowest finite positive score receives quantile 1/N. Genes
    with non-finite or non-positive scores receive quantile 0.0 (excluded from
    contributing to interaction_score).

    Ties are broken by competition ranking (method='min'), so tied genes share
    the same quantile.
    """
    finite_items = [(g, s) for g, s in scores.items() if math.isfinite(s) and s > 0.0]
    if not finite_items:
        return dict.fromkeys(scores, 0.0)

    genes_sorted = sorted(finite_items, key=lambda kv: kv[1])
    n = len(genes_sorted)
    quantiles: dict[str, float] = dict.fromkeys(scores, 0.0)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and genes_sorted[j + 1][1] == genes_sorted[i][1]:
            j += 1
        rank_one_indexed = i + 1
        q = rank_one_indexed / n
        for k in range(i, j + 1):
            quantiles[genes_sorted[k][0]] = q
        i = j + 1

    return quantiles


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
    desired_score_column: str | None = None,
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
    null_rankings_path: Path | None,
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
    alternative: str = "greater",
) -> pd.DataFrame:
    """Compute co-occurrence statistics for all pairs among the top genes.

    Rank quantiles are computed within the supplied top-gene set so the
    interaction score is scale-invariant across --score-column choices.
    Per-pair Fisher exact p-values are computed for the carrier contingency table
    using the supplied alternative hypothesis direction.
    """
    gene_score_map = dict(
        zip(gene_rankings_df["gene_symbol"], gene_rankings_df["gene_score"], strict=False)
    )
    gene_rank_map = dict(
        zip(gene_rankings_df["gene_symbol"], gene_rankings_df["gene_rank"], strict=False)
    )
    if "significant_variant_count" in gene_rankings_df.columns:
        significant_count_map = dict(
            zip(
                gene_rankings_df["gene_symbol"],
                gene_rankings_df["significant_variant_count"],
                strict=False,
            )
        )
    else:
        significant_count_map = {}

    sorted_genes = sorted(set(top_genes))
    score_quantiles = _compute_rank_quantiles(
        {g: float(gene_score_map.get(g, 0.0)) for g in sorted_genes}
    )

    rows: list[dict[str, Any]] = []

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

        q_a = score_quantiles.get(gene_a, 0.0)
        q_b = score_quantiles.get(gene_b, 0.0)
        interaction_score = math.sqrt(q_a * q_b) * math.log1p(n_cooccur)

        chrom_a = gene_to_chrom.get(gene_a, "")
        chrom_b = gene_to_chrom.get(gene_b, "")

        n_only_a = max(0, len(samples_a) - n_cooccur)
        n_only_b = max(0, len(samples_b) - n_cooccur)
        n_neither = max(0, total_samples - n_cooccur - n_only_a - n_only_b)

        contingency = [[n_cooccur, n_only_b], [n_only_a, n_neither]]
        _, pval = fisher_exact(contingency, alternative=alternative)

        rows.append(
            {
                "gene_a": gene_a,
                "gene_b": gene_b,
                "n_cooccur": n_cooccur,
                "n_only_a": n_only_a,
                "n_only_b": n_only_b,
                "n_neither": n_neither,
                "n_cooccur_cases": n_cooccur_cases,
                "n_cooccur_controls": n_cooccur_controls,
                "freq_gene_a": round(freq_a, 6),
                "freq_gene_b": round(freq_b, 6),
                "expected_cooccur": round(expected, 4),
                "obs_exp_ratio": round(obs_exp_ratio, 4) if math.isfinite(obs_exp_ratio) else float("nan"),
                "gene_score_a": score_a,
                "gene_score_b": score_b,
                "gene_score_quantile_a": round(q_a, 6),
                "gene_score_quantile_b": round(q_b, 6),
                "gene_rank_a": rank_a,
                "gene_rank_b": rank_b,
                "significant_variant_count_a": int(significant_count_map.get(gene_a, 0)),
                "significant_variant_count_b": int(significant_count_map.get(gene_b, 0)),
                "interaction_score": round(interaction_score, 6),
                "same_chrom": bool(chrom_a and chrom_a == chrom_b),
                "pval_fisher": float(pval),
            }
        )

    logger.info("Computed co-occurrence for %d gene pairs", len(rows))
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def apply_multiple_testing_correction(
    pairs_df: pd.DataFrame,
    correction: str,
    alpha: float,
) -> pd.DataFrame:
    """Apply multiple-testing correction to pval_fisher and add padj and reject columns."""
    df = pairs_df.copy()
    if df.empty:
        df["padj"] = pd.Series(dtype=float)
        df["reject"] = pd.Series(dtype=bool)
        return df

    pvals = df["pval_fisher"].values

    if correction == "none":
        df["padj"] = pvals
    elif correction in ("bonferroni", "fdr_bh"):
        method = "bonferroni" if correction == "bonferroni" else "fdr_bh"
        _, padj, _, _ = multipletests(pvals, alpha=alpha, method=method)
        df["padj"] = padj
    else:
        raise ValueError(f"Unknown correction: {correction}")

    df["reject"] = df["padj"] < alpha
    return df


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
    cooc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich gene-pair data with aggregated variant-level co-occurrence."""
    cooc_df = cooc_df.copy()

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
        zip(gene_rankings_df["gene_symbol"], gene_rankings_df["gene_score"], strict=False)
    )
    gene_rank_map = dict(
        zip(gene_rankings_df["gene_symbol"], gene_rankings_df["gene_rank"], strict=False)
    )
    significant_count_map = (
        dict(
            zip(
                gene_rankings_df["gene_symbol"],
                gene_rankings_df["significant_variant_count"],
                strict=False,
            )
        )
        if "significant_variant_count" in gene_rankings_df.columns
        else {}
    )
    score_source_map = (
        dict(
            zip(
                gene_rankings_df["gene_symbol"],
                gene_rankings_df["gene_score_source"],
                strict=False,
            )
        )
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


def _output_filename(base: str, k: int, multi_k: bool) -> str:
    """Return the output filename, inserting a _topK{k} suffix when multi-K is active."""
    if not multi_k:
        return base
    dot = base.rfind(".")
    if dot == -1:
        return f"{base}_topK{k}"
    return f"{base[:dot]}_topK{k}{base[dot:]}"


def main() -> None:
    """Main entry point for gene-level interaction aggregation."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    multi_k = len(args.top_k_genes) > 1

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

    # Load gene rankings once using the maximum K to cover all requested K values.
    max_k = max(args.top_k_genes)
    gene_rankings_all_df, gene_metadata = load_gene_rankings(
        gene_rankings_path=args.gene_rankings,
        variant_rankings_df=real_variant_df,
        top_k=max_k,
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

    cooccurrence_df = None
    if args.cooccurrence is not None:
        cooccurrence_df = pd.read_csv(args.cooccurrence)
        logger.info(
            "Loaded %d variant co-occurrence rows from %s",
            len(cooccurrence_df),
            args.cooccurrence,
        )

    # Per-K metadata for the optional combined index
    per_k_index: list[dict[str, Any]] = []

    for k in sorted(args.top_k_genes):
        logger.info("--- Running pair analysis for top-K=%d ---", k)

        gene_rankings_df = gene_rankings_all_df.head(k).reset_index(drop=True)

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
            alternative=args.alternative,
        )

        if cooccurrence_df is not None and not pairs_df.empty:
            pairs_df = enrich_with_variant_cooccurrence(pairs_df, cooccurrence_df)

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
            pairs_df = apply_multiple_testing_correction(pairs_df, args.correction, args.alpha)
            pairs_df = pairs_df.sort_values(
                ["interaction_score", "padj", "n_cooccur"],
                ascending=[False, True, False],
            ).reset_index(drop=True)
        else:
            pairs_df["padj"] = pd.Series(dtype=float)
            pairs_df["reject"] = pd.Series(dtype=bool)

        edges_df, nodes_df = build_network_outputs(pairs_df, gene_rankings_df)

        pairs_fname = _output_filename("gene_pair_interactions.csv", k, multi_k)
        edges_fname = _output_filename("gene_interaction_network_edges.csv", k, multi_k)
        nodes_fname = _output_filename("gene_interaction_network_nodes.csv", k, multi_k)
        summary_fname = _output_filename("gene_interaction_summary.yaml", k, multi_k)

        pairs_path = args.output_dir / pairs_fname
        pairs_df.to_csv(pairs_path, index=False)
        logger.info("Wrote %d gene pairs to %s", len(pairs_df), pairs_path)

        edges_path = args.output_dir / edges_fname
        edges_df.to_csv(edges_path, index=False)
        logger.info("Wrote %d edges to %s", len(edges_df), edges_path)

        nodes_path = args.output_dir / nodes_fname
        nodes_df.to_csv(nodes_path, index=False)
        logger.info("Wrote %d nodes to %s", len(nodes_df), nodes_path)

        n_pairs_rejected = int(pairs_df["reject"].sum()) if not pairs_df.empty and "reject" in pairs_df.columns else 0
        padj_vals = pairs_df["padj"].dropna().values if not pairs_df.empty and "padj" in pairs_df.columns else np.array([])

        summary_metadata: dict[str, Any] = {
            **significance_metadata,
            **gene_metadata,
            "variant_rankings": str(args.variant_rankings),
            "gene_rankings": str(args.gene_rankings),
            "score_column": args.score_column,
            "top_k_genes": k,
            "interaction_score_formula": (
                "sqrt(rank_quantile(gene_score_a) * rank_quantile(gene_score_b)) "
                "* log1p(n_cooccur), with rank quantiles computed within the analysed "
                "top-K gene set"
            ),
            "correction_method": args.correction,
            "alpha": args.alpha,
            "fisher_alternative": args.alternative,
            "n_pairs_tested": int(len(pairs_df)),
            "n_pairs_rejected": n_pairs_rejected,
            "min_padj": float(padj_vals.min()) if padj_vals.size > 0 else None,
            "median_padj": float(np.median(padj_vals)) if padj_vals.size > 0 else None,
        }

        summary_path = args.output_dir / summary_fname
        write_summary(pairs_df, nodes_df, summary_path, summary_metadata)

        per_k_index.append({
            "top_k": k,
            "summary_file": str(summary_path),
            "pairs_file": str(pairs_path),
            "edges_file": str(edges_path),
            "nodes_file": str(nodes_path),
            "total_gene_pairs": int(len(pairs_df)),
            "total_genes_in_network": int(len(nodes_df)),
            "n_pairs_rejected": n_pairs_rejected,
            "hub_genes_top5": [
                str(nodes_df.iloc[i]["gene"])
                for i in range(min(5, len(nodes_df)))
            ],
        })

    if multi_k:
        index_data: dict[str, Any] = {
            "top_k_values": sorted(args.top_k_genes),
            "correction_method": args.correction,
            "alpha": args.alpha,
            "fisher_alternative": args.alternative,
            "score_column": args.score_column,
            "runs": per_k_index,
        }
        index_path = args.output_dir / "gene_interaction_summary_index.yaml"
        with open(index_path, "w") as handle:
            yaml.dump(index_data, handle, default_flow_style=False, sort_keys=False)
        logger.info("Combined index written to %s", index_path)

    logger.info("Gene interaction aggregation complete.")


if __name__ == "__main__":
    main()
