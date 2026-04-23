"""Tests for delta_rank support in validate_nonlinear_classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.validate_nonlinear_classifier as nonlinear
from tests.test_nonlinear_validation import (
    make_gene_rankings,
    prepare_inputs,
    write_labels,
)


def test_score_column_is_ascending_delta_rank_is_false() -> None:
    """delta_rank must be treated as descending (higher-is-better)."""
    assert nonlinear.score_column_is_ascending("delta_rank") is False


def test_resolve_score_column_delta_rank_prefers_gene_delta_rank() -> None:
    """resolve_score_column('delta_rank') resolves to gene_delta_rank when present."""
    df = pd.DataFrame({"gene_name": ["A"], "gene_delta_rank": [1.0], "delta_rank": [2.0]})
    resolved = nonlinear.resolve_score_column(df, "delta_rank")
    assert resolved == "gene_delta_rank"


def test_resolve_score_column_delta_rank_falls_back_to_delta_rank() -> None:
    """resolve_score_column('delta_rank') falls back to delta_rank when gene_delta_rank absent."""
    df = pd.DataFrame({"gene_name": ["A"], "delta_rank": [2.0]})
    resolved = nonlinear.resolve_score_column(df, "delta_rank")
    assert resolved == "delta_rank"


def _make_gene_stats_with_delta_rank(
    genes: list[str],
    signal_genes: list[str],
) -> pd.DataFrame:
    """Build a gene-stats-style DataFrame with both gene_z_score and gene_delta_rank."""
    tail_genes = [g for g in genes if g not in signal_genes]
    ordered = signal_genes + tail_genes
    n = len(ordered)
    return pd.DataFrame(
        {
            "gene_name": ordered,
            "gene_rank": np.arange(1, n + 1),
            "gene_z_score": np.linspace(4.0, -1.0, n),
            "gene_delta_rank": np.linspace(3.5, -0.5, n),
            "gene_delta_rank_aggregation": ["max"] * n,
            "num_variants": np.ones(n, dtype=int),
            "fdr_gene": np.linspace(0.001, 0.50, n),
        }
    )


def test_end_to_end_delta_rank_uses_gene_delta_rank_column(tmp_path: Path) -> None:
    """End-to-end run with --score-column delta_rank selects genes via gene_delta_rank."""
    burden_path, label_path, rankings_root = prepare_inputs(tmp_path, levels=("L0", "L1"))

    burden_df = pd.read_parquet(burden_path)
    all_genes = burden_df.columns.tolist()
    signal_genes = all_genes[:10]

    for level in ("L0", "L1"):
        level_dir = rankings_root / level
        level_dir.mkdir(parents=True, exist_ok=True)
        stats_df = _make_gene_stats_with_delta_rank(all_genes, signal_genes)
        stats_df.to_csv(
            level_dir / "gene_rankings_with_significance.csv",
            index=False,
        )

    output_tsv = tmp_path / "delta_rank_validation.tsv"
    nonlinear.main(
        [
            "--real-rankings-dir", str(rankings_root),
            "--burden-matrix", str(burden_path),
            "--phenotypes", str(label_path),
            "--output-tsv", str(output_tsv),
            "--top-k", "5,10",
            "--classifiers", "rf",
            "--levels", "L0,L1",
            "--n-permutations", "4",
            "--cv-folds", "3",
            "--n-cores", "1",
            "--seed", "42",
            "--score-column", "delta_rank",
        ]
    )

    summary_df = pd.read_csv(output_tsv, sep="\t")
    assert list(summary_df.columns) == nonlinear.SUMMARY_COLUMNS
    assert len(summary_df) == 2 * 1 * 2
