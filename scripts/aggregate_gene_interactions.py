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

import pandas as pd
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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
        help="Path to sieve_variant_rankings.csv.",
    )
    parser.add_argument(
        "--gene-rankings",
        type=Path,
        required=True,
        help="Path to sieve_gene_rankings.csv.",
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
        help="Minimum gene_score threshold (default: 0.0).",
    )
    return parser.parse_args()


def _resolve_gene_symbol_column(
    gene_rankings_df: pd.DataFrame,
    variant_rankings_df: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve a gene-symbol column compatible with preprocessed SampleVariants."""
    resolved = gene_rankings_df.copy()

    if "gene_name" not in resolved.columns or resolved["gene_name"].isna().all():
        if {"gene_id", "gene_name"}.issubset(variant_rankings_df.columns):
            gene_name_map = (
                variant_rankings_df[["gene_id", "gene_name"]]
                .dropna(subset=["gene_name"])
                .drop_duplicates(subset=["gene_id"])
            )
            resolved = resolved.merge(gene_name_map, on="gene_id", how="left")

    if "gene_name" in resolved.columns and not resolved["gene_name"].isna().all():
        resolved["gene_symbol"] = resolved["gene_name"].astype(str)
    elif "gene_id" in resolved.columns:
        resolved["gene_symbol"] = resolved["gene_id"].astype(str)
    else:
        raise ValueError("Gene rankings must contain 'gene_name' or 'gene_id'.")

    return resolved


def load_gene_rankings(
    gene_rankings_path: Path,
    variant_rankings_path: Path,
    top_k: int,
    min_score: float,
) -> pd.DataFrame:
    """Load gene rankings, resolve gene symbols, and select top genes."""
    gene_df = pd.read_csv(gene_rankings_path)
    variant_df = pd.read_csv(variant_rankings_path)
    logger.info("Loaded %d genes from %s", len(gene_df), gene_rankings_path)

    gene_df = _resolve_gene_symbol_column(gene_df, variant_df)

    if "gene_rank" not in gene_df.columns:
        gene_df["gene_rank"] = (
            gene_df["gene_score"].rank(method="dense", ascending=False).astype(int)
        )

    if min_score > 0.0:
        gene_df = gene_df[gene_df["gene_score"] >= min_score].copy()
        logger.info(
            "After min_gene_score filter (>= %.4f): %d genes",
            min_score,
            len(gene_df),
        )

    gene_df = (
        gene_df.sort_values(["gene_score", "gene_rank"], ascending=[False, True])
        .head(top_k)
        .reset_index(drop=True)
    )
    logger.info("Selected top-%d genes by gene_score", len(gene_df))
    return gene_df


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
                "interaction_score": round(interaction_score, 6),
                "same_chrom": bool(chrom_a and chrom_a == chrom_b),
            }
        )

    logger.info("Computed co-occurrence for %d gene pairs", len(rows))
    return pd.DataFrame(rows)


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
    node_columns = ["gene", "degree", "gene_score", "gene_rank", "n_partners"]

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

    nodes_rows = [
        {
            "gene": gene,
            "degree": deg,
            "gene_score": float(gene_score_map.get(gene, 0.0)),
            "gene_rank": int(gene_rank_map.get(gene, -1)),
            "n_partners": deg,
        }
        for gene, deg in sorted(degree.items(), key=lambda item: (-item[1], item[0]))
    ]
    return edges_df, pd.DataFrame(nodes_rows, columns=node_columns)


def write_summary(
    pairs_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    output_path: Path,
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
        }
        for _, row in pairs_df.head(10).iterrows()
    ]

    hub_genes = [
        {
            "gene": row["gene"],
            "degree": int(row["degree"]),
            "gene_score": float(row["gene_score"]),
        }
        for _, row in nodes_df.head(10).iterrows()
    ]

    summary: dict[str, Any] = {
        "total_gene_pairs": int(len(pairs_df)),
        "total_genes_in_network": int(len(nodes_df)),
        "top_10_pairs": top_pairs,
        "hub_genes": hub_genes,
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

    gene_rankings_df = load_gene_rankings(
        gene_rankings_path=args.gene_rankings,
        variant_rankings_path=args.variant_rankings,
        top_k=args.top_k_genes,
        min_score=args.min_gene_score,
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

    if args.cooccurrence is not None:
        pairs_df = enrich_with_variant_cooccurrence(pairs_df, args.cooccurrence)

    before = len(pairs_df)
    pairs_df = pairs_df[pairs_df["n_cooccur"] >= args.min_cooccur_samples].copy()
    logger.info(
        "Filtered gene pairs: %d -> %d (min_cooccur_samples=%d)",
        before,
        len(pairs_df),
        args.min_cooccur_samples,
    )

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

    summary_path = args.output_dir / "gene_interaction_summary.yaml"
    write_summary(pairs_df, nodes_df, summary_path)
    logger.info("Gene interaction aggregation complete.")


if __name__ == "__main__":
    main()
