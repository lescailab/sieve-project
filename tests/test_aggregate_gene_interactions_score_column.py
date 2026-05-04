"""Tests for --score-column support in aggregate_gene_interactions.py."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

import scripts.aggregate_gene_interactions as interactions


# ─── minimal sample / variant stubs ──────────────────────────────────────────


@dataclass
class _Variant:
    chrom: str
    pos: int
    gene: str


@dataclass
class _Sample:
    sample_id: str
    label: int
    variants: list[_Variant] = field(default_factory=list)


# Six genes across two chromosomes; scores are crafted so z_attribution and
# delta_rank produce disjoint top-3 gene sets:
#   z_attribution top-3 → GENE_A(5.0), GENE_B(4.0), GENE_C(3.0)
#   delta_rank    top-3 → GENE_D(8.0), GENE_E(7.0), GENE_F(6.0)
_GENES = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E", "GENE_F"]
_CHROM = {
    "GENE_A": "1", "GENE_B": "1", "GENE_C": "1",
    "GENE_D": "2", "GENE_E": "2", "GENE_F": "2",
}
_POS = {g: 100 * (i + 1) for i, g in enumerate(_GENES)}

_VARIANT_ROWS = [
    {"gene_name": "GENE_A", "chromosome": "1", "position": 100, "z_attribution": 5.0, "delta_rank": 1.0},
    {"gene_name": "GENE_B", "chromosome": "1", "position": 200, "z_attribution": 4.0, "delta_rank": 0.5},
    {"gene_name": "GENE_C", "chromosome": "1", "position": 300, "z_attribution": 3.0, "delta_rank": 0.3},
    {"gene_name": "GENE_D", "chromosome": "2", "position": 100, "z_attribution": 2.0, "delta_rank": 8.0},
    {"gene_name": "GENE_E", "chromosome": "2", "position": 200, "z_attribution": 1.0, "delta_rank": 7.0},
    {"gene_name": "GENE_F", "chromosome": "2", "position": 300, "z_attribution": 0.5, "delta_rank": 6.0},
]

_GENE_ROWS = [
    {"gene_name": "GENE_A", "gene_z_score": 5.0, "gene_delta_rank": 1.0, "gene_rank": 1},
    {"gene_name": "GENE_B", "gene_z_score": 4.0, "gene_delta_rank": 0.5, "gene_rank": 2},
    {"gene_name": "GENE_C", "gene_z_score": 3.0, "gene_delta_rank": 0.3, "gene_rank": 3},
    {"gene_name": "GENE_D", "gene_z_score": 2.0, "gene_delta_rank": 8.0, "gene_rank": 4},
    {"gene_name": "GENE_E", "gene_z_score": 1.0, "gene_delta_rank": 7.0, "gene_rank": 5},
    {"gene_name": "GENE_F", "gene_z_score": 0.5, "gene_delta_rank": 6.0, "gene_rank": 6},
]


# ─── fixture helpers ──────────────────────────────────────────────────────────


def _make_pt(tmp_path: Path) -> Path:
    """Save a minimal preprocessed .pt payload; every sample carries all six genes."""
    samples = [
        _Sample(
            sample_id=f"S{i}",
            label=int(i < 5),
            variants=[_Variant(chrom=_CHROM[g], pos=_POS[g], gene=g) for g in _GENES],
        )
        for i in range(10)
    ]
    pt_path = tmp_path / "preprocessed.pt"
    torch.save({"samples": samples}, pt_path)
    return pt_path


def _make_variant_rankings(tmp_path: Path, include_delta_rank: bool = True) -> Path:
    """Write variant rankings CSV with z_attribution and optionally delta_rank."""
    df = pd.DataFrame(_VARIANT_ROWS)
    if not include_delta_rank:
        df = df.drop(columns=["delta_rank"])
    path = tmp_path / "variant_rankings.csv"
    df.to_csv(path, index=False)
    return path


def _make_gene_rankings(tmp_path: Path, include_gene_delta_rank: bool = True) -> Path:
    """Write gene rankings CSV with gene_z_score and optionally gene_delta_rank."""
    df = pd.DataFrame(_GENE_ROWS)
    if not include_gene_delta_rank:
        df = df.drop(columns=["gene_delta_rank"])
    path = tmp_path / "gene_rankings.csv"
    df.to_csv(path, index=False)
    return path


def _make_null_rankings(tmp_path: Path) -> Path:
    """Write a minimal null rankings CSV with uniformly low z_attribution scores.

    annotate_with_null_significance always derives null thresholds from the
    z_attribution (or mean_attribution) column of the null file, regardless of
    which --score-column is active. When --score-column delta_rank is used,
    exceeds_null_* comparisons therefore cross score spaces; the threshold
    functions as a coarse descriptive annotation rather than a calibrated gate.
    The fixture uses very low z_attribution values (-10.0) so that every real
    variant_score — whether z_attribution or delta_rank — exceeds the threshold,
    keeping all genes marked as 'significant' and avoiding null-gate interference
    in tests that focus on delta_rank gene selection.
    """
    rows = [
        {"gene_name": f"NULL_G{i}", "chromosome": "1", "position": i * 100,
         "z_attribution": -10.0}
        for i in range(20)
    ]
    path = tmp_path / "null_rankings.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _genes_in_pairs(out_dir: Path) -> set[str]:
    df = pd.read_csv(out_dir / "gene_pair_interactions.csv")
    return set(df["gene_a"].tolist()) | set(df["gene_b"].tolist())


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
