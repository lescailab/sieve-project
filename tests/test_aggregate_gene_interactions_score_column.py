"""Tests for --score-column support in aggregate_gene_interactions.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

import scripts.aggregate_gene_interactions as interactions
from tests._aggregate_gene_interactions_helpers import (
    genes_in_pairs as _genes_in_pairs,
)
from tests._aggregate_gene_interactions_helpers import (
    make_gene_rankings as _make_gene_rankings,
)
from tests._aggregate_gene_interactions_helpers import (
    make_null_rankings as _make_null_rankings,
)
from tests._aggregate_gene_interactions_helpers import (
    make_pt as _make_pt,
)
from tests._aggregate_gene_interactions_helpers import (
    make_variant_rankings as _make_variant_rankings,
)

# ─── tests ────────────────────────────────────────────────────────────────────


def test_score_column_z_attribution_default(tmp_path: Path, monkeypatch) -> None:
    """Default run (no --score-column) records score_column: z_attribution in the
    summary YAML and gene_score values in the pairs CSV match the gene_z_score-
    derived values from the gene rankings file."""
    pt = _make_pt(tmp_path)
    vrankings = _make_variant_rankings(tmp_path)
    grankings = _make_gene_rankings(tmp_path)
    out_dir = tmp_path / "out_z"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_gene_interactions.py",
            "--preprocessed-data", str(pt),
            "--variant-rankings", str(vrankings),
            "--gene-rankings", str(grankings),
            "--output-dir", str(out_dir),
            "--top-k-genes", "3",
            "--min-cooccur-samples", "1",
        ],
    )

    interactions.main()

    summary = yaml.safe_load((out_dir / "gene_interaction_summary.yaml").read_text())
    assert summary["score_basis"]["score_column"] == "z_attribution"

    pairs_df = pd.read_csv(out_dir / "gene_pair_interactions.csv")
    top_genes = _genes_in_pairs(out_dir)
    assert top_genes == {"GENE_A", "GENE_B", "GENE_C"}

    expected_scores = {"GENE_A": 5.0, "GENE_B": 4.0, "GENE_C": 3.0}
    for _, row in pairs_df.iterrows():
        assert row["gene_score_a"] == pytest.approx(expected_scores[row["gene_a"]])
        assert row["gene_score_b"] == pytest.approx(expected_scores[row["gene_b"]])


def test_score_column_delta_rank_selects_different_genes(tmp_path: Path, monkeypatch) -> None:
    """--score-column delta_rank selects a disjoint top-3 gene set from z_attribution,
    and records score_column: delta_rank in the summary YAML."""
    pt = _make_pt(tmp_path)
    vrankings = _make_variant_rankings(tmp_path)
    grankings = _make_gene_rankings(tmp_path)
    out_z = tmp_path / "out_z"
    out_d = tmp_path / "out_delta"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_gene_interactions.py",
            "--preprocessed-data", str(pt),
            "--variant-rankings", str(vrankings),
            "--gene-rankings", str(grankings),
            "--output-dir", str(out_z),
            "--top-k-genes", "3",
            "--min-cooccur-samples", "1",
            "--score-column", "z_attribution",
        ],
    )
    interactions.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_gene_interactions.py",
            "--preprocessed-data", str(pt),
            "--variant-rankings", str(vrankings),
            "--gene-rankings", str(grankings),
            "--output-dir", str(out_d),
            "--top-k-genes", "3",
            "--min-cooccur-samples", "1",
            "--score-column", "delta_rank",
            "--allow-nonsignificant-genes",
        ],
    )
    interactions.main()

    z_genes = _genes_in_pairs(out_z)
    d_genes = _genes_in_pairs(out_d)

    assert z_genes != d_genes
    assert z_genes == {"GENE_A", "GENE_B", "GENE_C"}
    assert d_genes == {"GENE_D", "GENE_E", "GENE_F"}

    summary = yaml.safe_load((out_d / "gene_interaction_summary.yaml").read_text())
    assert summary["score_basis"]["score_column"] == "delta_rank"


def test_score_column_delta_rank_with_null_rankings_and_allow_nonsignificant(
    tmp_path: Path, monkeypatch
) -> None:
    """delta_rank run with --null-rankings keeps per-variant exceeds_null_* annotations
    (reflected as significant_variant_count > 0 in the pairs CSV), but
    --allow-nonsignificant-genes ensures gene selection is driven by delta_rank
    rather than the floored bootstrap p-value gate — so the selected gene set
    matches the no-null-rankings run from the previous test."""
    pt = _make_pt(tmp_path)
    vrankings = _make_variant_rankings(tmp_path)
    grankings = _make_gene_rankings(tmp_path)
    null_rankings = _make_null_rankings(tmp_path)
    out_dir = tmp_path / "out_delta_null"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_gene_interactions.py",
            "--preprocessed-data", str(pt),
            "--variant-rankings", str(vrankings),
            "--gene-rankings", str(grankings),
            "--null-rankings", str(null_rankings),
            "--output-dir", str(out_dir),
            "--top-k-genes", "3",
            "--min-cooccur-samples", "1",
            "--score-column", "delta_rank",
            "--allow-nonsignificant-genes",
        ],
    )
    interactions.main()

    # Gene selection must still be driven by delta_rank, matching the clean run.
    assert _genes_in_pairs(out_dir) == {"GENE_D", "GENE_E", "GENE_F"}

    summary = yaml.safe_load((out_dir / "gene_interaction_summary.yaml").read_text())
    assert summary["score_basis"]["score_column"] == "delta_rank"
    # Null rankings were processed — significance_source must not be 'none'.
    assert summary["score_basis"]["significance_source"] != "none"

    # The exceeds_null_* annotations were propagated: every gene in the pairs
    # CSV should show significant_variant_count > 0 (all real delta_rank values
    # exceed the near-zero null thresholds).
    pairs_df = pd.read_csv(out_dir / "gene_pair_interactions.csv")
    assert (pairs_df["significant_variant_count_a"] > 0).all()
    assert (pairs_df["significant_variant_count_b"] > 0).all()


def test_score_column_delta_rank_missing_column_errors_cleanly(
    tmp_path: Path, monkeypatch
) -> None:
    """--score-column delta_rank raises a clear error when the variant rankings
    file has no 'delta_rank' column, and the error message names the column."""
    pt = _make_pt(tmp_path)
    vrankings = _make_variant_rankings(tmp_path, include_delta_rank=False)
    grankings = _make_gene_rankings(tmp_path)
    out_dir = tmp_path / "out_err1"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_gene_interactions.py",
            "--preprocessed-data", str(pt),
            "--variant-rankings", str(vrankings),
            "--gene-rankings", str(grankings),
            "--output-dir", str(out_dir),
            "--top-k-genes", "3",
            "--min-cooccur-samples", "1",
            "--score-column", "delta_rank",
        ],
    )

    with pytest.raises(ValueError, match="delta_rank"):
        interactions.main()


def test_score_column_delta_rank_missing_gene_column_errors_cleanly(
    tmp_path: Path, monkeypatch
) -> None:
    """--score-column delta_rank raises a clear error when the gene rankings file
    has no 'gene_delta_rank' column; the message must name the missing column and
    point to bootstrap_null_calibration.py."""
    pt = _make_pt(tmp_path)
    vrankings = _make_variant_rankings(tmp_path, include_delta_rank=True)
    grankings = _make_gene_rankings(tmp_path, include_gene_delta_rank=False)
    out_dir = tmp_path / "out_err2"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_gene_interactions.py",
            "--preprocessed-data", str(pt),
            "--variant-rankings", str(vrankings),
            "--gene-rankings", str(grankings),
            "--output-dir", str(out_dir),
            "--top-k-genes", "3",
            "--min-cooccur-samples", "1",
            "--score-column", "delta_rank",
        ],
    )

    with pytest.raises(ValueError, match="gene_delta_rank") as exc_info:
        interactions.main()
    assert "bootstrap_null_calibration" in str(exc_info.value)
