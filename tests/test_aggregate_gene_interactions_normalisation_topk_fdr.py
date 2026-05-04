"""Tests for rank-quantile normalisation, top-K sweep, and BH-FDR in
aggregate_gene_interactions.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import scripts.aggregate_gene_interactions as interactions
from tests._aggregate_gene_interactions_helpers import (
    _Sample,
    _Variant,
    genes_in_pairs,
    make_gene_rankings,
    make_pt,
    make_variant_rankings,
)

# ─── helpers ─────────────────────────────────────────────────────────────────


def _run_main(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["aggregate_gene_interactions.py"] + argv)
    interactions.main()


def _pairs_csv(out_dir: Path, suffix: str = "") -> pd.DataFrame:
    return pd.read_csv(out_dir / f"gene_pair_interactions{suffix}.csv")


def _summary_yaml(out_dir: Path, suffix: str = "") -> dict:
    return yaml.safe_load((out_dir / f"gene_interaction_summary{suffix}.yaml").read_text())


# ─── Task 1: rank-quantile tests ─────────────────────────────────────────────


def test_rank_quantile_invariance_across_score_columns(tmp_path: Path, monkeypatch) -> None:
    """Same gene importance ordering encoded in two score scales produces identical
    top-10 pair rankings by interaction_score."""
    # Genes G0..G9: z_attribution in [0.1, 5.0], delta_rank in [0.1e5, 5.0e5]
    # Both encode the same relative ordering.
    n_genes = 10
    z_scores = [float(i + 1) * 0.5 for i in range(n_genes)]
    dr_scores = [s * 1e5 for s in z_scores]
    gene_names = [f"GENE_{i}" for i in range(n_genes)]

    var_rows_z = [
        {"gene_name": g, "chromosome": "1", "position": (i + 1) * 100,
         "z_attribution": z_scores[i], "delta_rank": dr_scores[i]}
        for i, g in enumerate(gene_names)
    ]
    gene_rows = [
        {"gene_name": g, "gene_z_score": z_scores[i], "gene_delta_rank": dr_scores[i], "gene_rank": i + 1}
        for i, g in enumerate(gene_names)
    ]

    chrom_map = dict.fromkeys(gene_names, "1")
    pos_map = {g: (i + 1) * 100 for i, g in enumerate(gene_names)}
    samples = [
        _Sample(
            sample_id=f"S{j}",
            label=int(j < 5),
            variants=[_Variant(chrom=chrom_map[g], pos=pos_map[g], gene=g) for g in gene_names],
        )
        for j in range(20)
    ]
    pt_path = tmp_path / "pre.pt"
    torch.save({"samples": samples}, pt_path)

    var_path = tmp_path / "var.csv"
    pd.DataFrame(var_rows_z).to_csv(var_path, index=False)
    gene_path = tmp_path / "gene.csv"
    pd.DataFrame(gene_rows).to_csv(gene_path, index=False)

    out_z = tmp_path / "out_z"
    out_d = tmp_path / "out_d"

    _run_main(monkeypatch, [
        "--preprocessed-data", str(pt_path),
        "--variant-rankings", str(var_path),
        "--gene-rankings", str(gene_path),
        "--output-dir", str(out_z),
        "--top-k-genes", "10",
        "--min-cooccur-samples", "1",
        "--score-column", "z_attribution",
        "--correction", "none",
    ])

    _run_main(monkeypatch, [
        "--preprocessed-data", str(pt_path),
        "--variant-rankings", str(var_path),
        "--gene-rankings", str(gene_path),
        "--output-dir", str(out_d),
        "--top-k-genes", "10",
        "--min-cooccur-samples", "1",
        "--score-column", "delta_rank",
        "--allow-nonsignificant-genes",
        "--correction", "none",
    ])

    pairs_z = _pairs_csv(out_z).head(10)
    pairs_d = _pairs_csv(out_d).head(10)

    top10_z = set(zip(pairs_z["gene_a"], pairs_z["gene_b"], strict=False))
    top10_d = set(zip(pairs_d["gene_a"], pairs_d["gene_b"], strict=False))
    assert top10_z == top10_d, (
        f"Top-10 pairs differ between z_attribution and delta_rank runs.\n"
        f"z: {top10_z}\ndelta_rank: {top10_d}"
    )


def test_rank_quantile_bounds(tmp_path: Path, monkeypatch) -> None:
    """N=10 genes: highest receives quantile 1.0, lowest receives 0.1 = 1/N.
    All gene_score_quantile_a/b values lie in (0, 1]."""
    n_genes = 10
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    scores = list(range(1, n_genes + 1))

    var_rows = [
        {"gene_name": g, "chromosome": "1", "position": (i + 1) * 100,
         "z_attribution": float(scores[i])}
        for i, g in enumerate(gene_names)
    ]
    gene_rows = [
        {"gene_name": g, "gene_z_score": float(scores[i]), "gene_rank": i + 1}
        for i, g in enumerate(gene_names)
    ]

    chrom_map = dict.fromkeys(gene_names, "1")
    pos_map = {g: (i + 1) * 100 for i, g in enumerate(gene_names)}
    samples = [
        _Sample(
            sample_id=f"S{j}",
            label=int(j < 5),
            variants=[_Variant(chrom=chrom_map[g], pos=pos_map[g], gene=g) for g in gene_names],
        )
        for j in range(20)
    ]
    pt_path = tmp_path / "pre.pt"
    torch.save({"samples": samples}, pt_path)

    var_path = tmp_path / "var.csv"
    pd.DataFrame(var_rows).to_csv(var_path, index=False)
    gene_path = tmp_path / "gene.csv"
    pd.DataFrame(gene_rows).to_csv(gene_path, index=False)

    out_dir = tmp_path / "out"
    _run_main(monkeypatch, [
        "--preprocessed-data", str(pt_path),
        "--variant-rankings", str(var_path),
        "--gene-rankings", str(gene_path),
        "--output-dir", str(out_dir),
        "--top-k-genes", "10",
        "--min-cooccur-samples", "1",
        "--correction", "none",
    ])

    pairs = _pairs_csv(out_dir)
    assert not pairs.empty

    # All quantiles must be in (0, 1]
    for col in ("gene_score_quantile_a", "gene_score_quantile_b"):
        vals = pairs[col].values
        assert (vals > 0.0).all(), f"Some {col} values are not > 0"
        assert (vals <= 1.0).all(), f"Some {col} values exceed 1.0"

    # The gene with the highest score should have quantile == 1.0
    max_score_gene = gene_names[-1]  # gene_names[9] has score 10
    for col_gene, col_q in [("gene_a", "gene_score_quantile_a"), ("gene_b", "gene_score_quantile_b")]:
        mask = pairs[col_gene] == max_score_gene
        if mask.any():
            np.testing.assert_allclose(pairs.loc[mask, col_q].to_numpy(), 1.0)

    # The gene with the lowest score should have quantile == 1/N = 0.1
    min_score_gene = gene_names[0]
    for col_gene, col_q in [("gene_a", "gene_score_quantile_a"), ("gene_b", "gene_score_quantile_b")]:
        mask = pairs[col_gene] == min_score_gene
        if mask.any():
            np.testing.assert_allclose(pairs.loc[mask, col_q].to_numpy(), 1 / n_genes)


def test_rank_quantile_handles_negative_and_nonfinite_scores(tmp_path: Path, monkeypatch) -> None:
    """Genes with negative or NaN scores receive quantile 0.0; pairs involving them
    have interaction_score == 0.0; no crash."""
    gene_names = ["G_NEG_A", "G_NEG_B", "G_NAN", "G_POS"]
    var_rows = [
        {"gene_name": "G_NEG_A", "chromosome": "1", "position": 100, "z_attribution": -1.0},
        {"gene_name": "G_NEG_B", "chromosome": "1", "position": 200, "z_attribution": -0.5},
        {"gene_name": "G_NAN",   "chromosome": "1", "position": 300, "z_attribution": float("nan")},
        {"gene_name": "G_POS",   "chromosome": "1", "position": 400, "z_attribution": 3.0},
    ]
    gene_rows = [
        {"gene_name": "G_NEG_A", "gene_z_score": -1.0,          "gene_rank": 4},
        {"gene_name": "G_NEG_B", "gene_z_score": -0.5,          "gene_rank": 3},
        {"gene_name": "G_NAN",   "gene_z_score": float("nan"),  "gene_rank": 2},
        {"gene_name": "G_POS",   "gene_z_score": 3.0,           "gene_rank": 1},
    ]

    samples = [
        _Sample(
            sample_id=f"S{j}",
            label=int(j < 5),
            variants=[
                _Variant(chrom="1", pos=(i + 1) * 100, gene=g)
                for i, g in enumerate(gene_names)
            ],
        )
        for j in range(10)
    ]
    pt_path = tmp_path / "pre.pt"
    torch.save({"samples": samples}, pt_path)

    var_path = tmp_path / "var.csv"
    pd.DataFrame(var_rows).to_csv(var_path, index=False)
    gene_path = tmp_path / "gene.csv"
    pd.DataFrame(gene_rows).to_csv(gene_path, index=False)

    out_dir = tmp_path / "out"
    _run_main(monkeypatch, [
        "--preprocessed-data", str(pt_path),
        "--variant-rankings", str(var_path),
        "--gene-rankings", str(gene_path),
        "--output-dir", str(out_dir),
        "--top-k-genes", "4",
        "--min-cooccur-samples", "1",
        "--min-gene-score", "-999.0",
        "--correction", "none",
    ])

    pairs = _pairs_csv(out_dir)
    # Pairs involving any of the negative/nan genes should have interaction_score == 0.0
    bad_genes = {"G_NEG_A", "G_NEG_B", "G_NAN"}
    for _, row in pairs.iterrows():
        if row["gene_a"] in bad_genes or row["gene_b"] in bad_genes:
            assert row["interaction_score"] == pytest.approx(0.0), (
                f"Expected interaction_score=0.0 for pair ({row['gene_a']}, {row['gene_b']})"
            )
            if row["gene_a"] in bad_genes:
                assert row["gene_score_quantile_a"] == pytest.approx(0.0)
            if row["gene_b"] in bad_genes:
                assert row["gene_score_quantile_b"] == pytest.approx(0.0)


# ─── Task 2: top-K list tests ─────────────────────────────────────────────────


def test_topk_list_produces_per_k_outputs(tmp_path: Path, monkeypatch) -> None:
    """--top-k-genes 5 10 produces per-K files; topK10 gene set is a superset of topK5;
    index file lists both Ks."""
    pt = make_pt(tmp_path)
    vr = make_variant_rankings(tmp_path)
    gr = make_gene_rankings(tmp_path)
    out_dir = tmp_path / "out"

    _run_main(monkeypatch, [
        "--preprocessed-data", str(pt),
        "--variant-rankings", str(vr),
        "--gene-rankings", str(gr),
        "--output-dir", str(out_dir),
        "--top-k-genes", "3", "6",
        "--min-cooccur-samples", "1",
        "--correction", "none",
    ])

    assert (out_dir / "gene_pair_interactions_topK3.csv").exists()
    assert (out_dir / "gene_pair_interactions_topK6.csv").exists()
    assert (out_dir / "gene_interaction_network_edges_topK3.csv").exists()
    assert (out_dir / "gene_interaction_network_edges_topK6.csv").exists()
    assert (out_dir / "gene_interaction_network_nodes_topK3.csv").exists()
    assert (out_dir / "gene_interaction_network_nodes_topK6.csv").exists()
    assert (out_dir / "gene_interaction_summary_topK3.yaml").exists()
    assert (out_dir / "gene_interaction_summary_topK6.yaml").exists()
    assert (out_dir / "gene_interaction_summary_index.yaml").exists()

    genes_k3 = genes_in_pairs(out_dir, "_topK3")
    genes_k6 = genes_in_pairs(out_dir, "_topK6")
    assert genes_k3.issubset(genes_k6), (
        f"topK3 genes ({genes_k3}) not a subset of topK6 genes ({genes_k6})"
    )

    index = yaml.safe_load((out_dir / "gene_interaction_summary_index.yaml").read_text())
    assert set(index["top_k_values"]) == {3, 6}


def test_topk_single_value_preserves_legacy_filenames(tmp_path: Path, monkeypatch) -> None:
    """Single --top-k-genes value: filenames have no suffix; no index file written."""
    pt = make_pt(tmp_path)
    vr = make_variant_rankings(tmp_path)
    gr = make_gene_rankings(tmp_path)
    out_dir = tmp_path / "out"

    _run_main(monkeypatch, [
        "--preprocessed-data", str(pt),
        "--variant-rankings", str(vr),
        "--gene-rankings", str(gr),
        "--output-dir", str(out_dir),
        "--top-k-genes", "6",
        "--min-cooccur-samples", "1",
        "--correction", "none",
    ])

    assert (out_dir / "gene_pair_interactions.csv").exists()
    assert (out_dir / "gene_interaction_network_edges.csv").exists()
    assert (out_dir / "gene_interaction_network_nodes.csv").exists()
    assert (out_dir / "gene_interaction_summary.yaml").exists()

    assert not (out_dir / "gene_pair_interactions_topK6.csv").exists()
    assert not (out_dir / "gene_interaction_summary_index.yaml").exists()


# ─── Task 3: per-pair BH-FDR tests ───────────────────────────────────────────


def _base_argv(pt: Path, vr: Path, gr: Path, out_dir: Path) -> list[str]:
    return [
        "--preprocessed-data", str(pt),
        "--variant-rankings", str(vr),
        "--gene-rankings", str(gr),
        "--output-dir", str(out_dir),
        "--top-k-genes", "6",
        "--min-cooccur-samples", "1",
    ]


def test_padj_columns_present_and_consistent(tmp_path: Path, monkeypatch) -> None:
    """With --correction fdr_bh, pval_fisher, padj, reject columns exist;
    padj >= pval_fisher for all rows; reject == (padj < alpha) for all rows."""
    pt = make_pt(tmp_path)
    vr = make_variant_rankings(tmp_path)
    gr = make_gene_rankings(tmp_path)
    out_dir = tmp_path / "out"

    _run_main(monkeypatch, _base_argv(pt, vr, gr, out_dir) + ["--correction", "fdr_bh", "--alpha", "0.05"])

    pairs = _pairs_csv(out_dir)
    assert not pairs.empty
    for col in ("pval_fisher", "padj", "reject"):
        assert col in pairs.columns, f"Missing column: {col}"

    assert (pairs["padj"] >= pairs["pval_fisher"] - 1e-12).all(), (
        "padj should be >= pval_fisher for all rows"
    )
    expected_reject = pairs["padj"] < 0.05
    assert (pairs["reject"] == expected_reject).all()


def test_correction_none_padj_equals_pval(tmp_path: Path, monkeypatch) -> None:
    """With --correction none, padj equals pval_fisher for all rows."""
    pt = make_pt(tmp_path)
    vr = make_variant_rankings(tmp_path)
    gr = make_gene_rankings(tmp_path)
    out_dir = tmp_path / "out"

    _run_main(monkeypatch, _base_argv(pt, vr, gr, out_dir) + ["--correction", "none"])

    pairs = _pairs_csv(out_dir)
    assert not pairs.empty
    np.testing.assert_allclose(pairs["padj"].values, pairs["pval_fisher"].values, rtol=1e-10)


def test_correction_changes_rejection_count(tmp_path: Path, monkeypatch) -> None:
    """bonferroni rejects <= fdr_bh rejects <= none rejects (using pval < alpha)."""
    n_genes = 12
    gene_names = [f"G{i}" for i in range(n_genes)]

    # Build a dataset where some pairs have very different carrier frequencies to
    # produce a spread of p-values.
    rng = np.random.default_rng(42)
    n_samples = 100
    samples = []
    for j in range(n_samples):
        variants = []
        for i, g in enumerate(gene_names):
            # Different genes have very different carrier frequencies
            freq = 0.1 + 0.07 * i
            if rng.random() < freq:
                variants.append(_Variant(chrom="1", pos=(i + 1) * 100, gene=g))
        samples.append(_Sample(sample_id=f"S{j}", label=int(j < 50), variants=variants))

    pt_path = tmp_path / "pre.pt"
    torch.save({"samples": samples}, pt_path)

    var_rows = [
        {"gene_name": g, "chromosome": "1", "position": (i + 1) * 100, "z_attribution": float(n_genes - i)}
        for i, g in enumerate(gene_names)
    ]
    gene_rows = [
        {"gene_name": g, "gene_z_score": float(n_genes - i), "gene_rank": i + 1}
        for i, g in enumerate(gene_names)
    ]
    var_path = tmp_path / "var.csv"
    pd.DataFrame(var_rows).to_csv(var_path, index=False)
    gene_path = tmp_path / "gene.csv"
    pd.DataFrame(gene_rows).to_csv(gene_path, index=False)

    results = {}
    for correction in ("none", "fdr_bh", "bonferroni"):
        out_dir = tmp_path / f"out_{correction}"
        _run_main(monkeypatch, [
            "--preprocessed-data", str(pt_path),
            "--variant-rankings", str(var_path),
            "--gene-rankings", str(gene_path),
            "--output-dir", str(out_dir),
            "--top-k-genes", str(n_genes),
            "--min-cooccur-samples", "1",
            "--correction", correction,
            "--alpha", "0.05",
        ])
        pairs = _pairs_csv(out_dir)
        results[correction] = int(pairs["reject"].sum())

    assert results["bonferroni"] <= results["fdr_bh"]
    assert results["fdr_bh"] <= results["none"]


def test_alternative_greater_detects_excess_cooccurrence(tmp_path: Path, monkeypatch) -> None:
    """Synthetic data with strong excess co-occurrence for one pair:
    --alternative greater → low pval + reject=True;
    --alternative less    → high pval + reject=False."""
    n_samples = 200
    # GENE_HI: high carrier freq, GENE_LO: low carrier freq but perfectly co-occurring with GENE_HI
    # Construct: 60 samples carry both, only 5 carry each alone.
    samples = []
    for j in range(n_samples):
        variants = []
        if j < 60:
            variants += [
                _Variant(chrom="1", pos=100, gene="GENE_HI"),
                _Variant(chrom="1", pos=200, gene="GENE_CO"),
            ]
        elif j < 65:
            variants += [_Variant(chrom="1", pos=100, gene="GENE_HI")]
        elif j < 70:
            variants += [_Variant(chrom="1", pos=200, gene="GENE_CO")]
        # Other genes to pad the pair analysis
        if j < 100:
            variants += [
                _Variant(chrom="2", pos=100, gene="GENE_X"),
                _Variant(chrom="2", pos=200, gene="GENE_Y"),
            ]
        samples.append(_Sample(sample_id=f"S{j}", label=int(j < 100), variants=variants))

    pt_path = tmp_path / "pre.pt"
    torch.save({"samples": samples}, pt_path)

    gene_names = ["GENE_HI", "GENE_CO", "GENE_X", "GENE_Y"]
    var_rows = [{"gene_name": g, "chromosome": "1" if i < 2 else "2",
                 "position": (i + 1) * 100, "z_attribution": 5.0 - i}
                for i, g in enumerate(gene_names)]
    gene_rows = [{"gene_name": g, "gene_z_score": 5.0 - i, "gene_rank": i + 1}
                 for i, g in enumerate(gene_names)]

    var_path = tmp_path / "var.csv"
    pd.DataFrame(var_rows).to_csv(var_path, index=False)
    gene_path = tmp_path / "gene.csv"
    pd.DataFrame(gene_rows).to_csv(gene_path, index=False)

    for alt, expect_sig in [("greater", True), ("less", False)]:
        out_dir = tmp_path / f"out_{alt}"
        _run_main(monkeypatch, [
            "--preprocessed-data", str(pt_path),
            "--variant-rankings", str(var_path),
            "--gene-rankings", str(gene_path),
            "--output-dir", str(out_dir),
            "--top-k-genes", "4",
            "--min-cooccur-samples", "1",
            "--correction", "none",
            "--alpha", "0.05",
            "--alternative", alt,
        ])
        pairs = _pairs_csv(out_dir)
        target_mask = (
            ((pairs["gene_a"] == "GENE_HI") & (pairs["gene_b"] == "GENE_CO")) |
            ((pairs["gene_a"] == "GENE_CO") & (pairs["gene_b"] == "GENE_HI"))
        )
        assert target_mask.any(), "Expected GENE_HI/GENE_CO pair not found"
        row = pairs[target_mask].iloc[0]
        if expect_sig:
            assert row["pval_fisher"] < 0.01, (
                f"Expected pval_fisher < 0.01 for excess co-occurrence under alternative='greater', "
                f"got {row['pval_fisher']:.4f}"
            )
            assert row["reject"], "Expected reject=True for excess co-occurrence pair under 'greater'"
        else:
            assert row["pval_fisher"] > 0.5, (
                f"Expected pval_fisher > 0.5 for excess pair under alternative='less', "
                f"got {row['pval_fisher']:.4f}"
            )
            assert not row["reject"], "Expected reject=False for excess pair under 'less'"


def test_alternative_less_detects_deficit_cooccurrence(tmp_path: Path, monkeypatch) -> None:
    """Two near-mutually-exclusive genes: --alternative less → pval < 0.01;
    --alternative greater → pval > 0.5."""
    n_samples = 200
    # GENE_MX1 and GENE_MX2 are almost mutually exclusive: 80 carry only MX1, 80 carry only MX2, 0 carry both.
    samples = []
    for j in range(n_samples):
        variants = []
        if j < 80:
            variants += [_Variant(chrom="1", pos=100, gene="GENE_MX1")]
        elif j < 160:
            variants += [_Variant(chrom="1", pos=200, gene="GENE_MX2")]
        if j < 100:
            variants += [
                _Variant(chrom="2", pos=100, gene="GENE_X"),
                _Variant(chrom="2", pos=200, gene="GENE_Y"),
            ]
        samples.append(_Sample(sample_id=f"S{j}", label=int(j < 100), variants=variants))

    pt_path = tmp_path / "pre.pt"
    torch.save({"samples": samples}, pt_path)

    gene_names = ["GENE_MX1", "GENE_MX2", "GENE_X", "GENE_Y"]
    var_rows = [{"gene_name": g, "chromosome": "1" if i < 2 else "2",
                 "position": (i + 1) * 100, "z_attribution": 5.0 - i}
                for i, g in enumerate(gene_names)]
    gene_rows = [{"gene_name": g, "gene_z_score": 5.0 - i, "gene_rank": i + 1}
                 for i, g in enumerate(gene_names)]

    var_path = tmp_path / "var.csv"
    pd.DataFrame(var_rows).to_csv(var_path, index=False)
    gene_path = tmp_path / "gene.csv"
    pd.DataFrame(gene_rows).to_csv(gene_path, index=False)

    for alt, expect_sig in [("less", True), ("greater", False)]:
        out_dir = tmp_path / f"out_{alt}"
        _run_main(monkeypatch, [
            "--preprocessed-data", str(pt_path),
            "--variant-rankings", str(var_path),
            "--gene-rankings", str(gene_path),
            "--output-dir", str(out_dir),
            "--top-k-genes", "4",
            "--min-cooccur-samples", "0",
            "--correction", "none",
            "--alpha", "0.05",
            "--alternative", alt,
        ])
        pairs = _pairs_csv(out_dir)
        target_mask = (
            ((pairs["gene_a"] == "GENE_MX1") & (pairs["gene_b"] == "GENE_MX2")) |
            ((pairs["gene_a"] == "GENE_MX2") & (pairs["gene_b"] == "GENE_MX1"))
        )
        assert target_mask.any(), "GENE_MX1/GENE_MX2 pair not found"
        row = pairs[target_mask].iloc[0]
        if expect_sig:
            assert row["pval_fisher"] < 0.01, (
                f"Expected pval_fisher < 0.01 for deficit pair under alternative='less', "
                f"got {row['pval_fisher']:.4f}"
            )
        else:
            assert row["pval_fisher"] > 0.5, (
                f"Expected pval_fisher > 0.5 for deficit pair under alternative='greater', "
                f"got {row['pval_fisher']:.4f}"
            )


def test_alternative_two_sided_detects_both_directions(tmp_path: Path, monkeypatch) -> None:
    """Dataset with one excess pair and one deficit pair:
    --alternative two-sided → both have pval < 0.01;
    --alternative greater   → only excess pair survives;
    --alternative less      → only deficit pair survives."""
    n_samples = 200
    # GENE_EX and GENE_EC: strong excess co-occurrence (60 carry both, 5 only each)
    # GENE_MX1 and GENE_MX2: near-mutual exclusivity (80 carry only MX1, 80 carry only MX2)
    samples = []
    for j in range(n_samples):
        variants = []
        if j < 60:
            variants += [
                _Variant(chrom="1", pos=100, gene="GENE_EX"),
                _Variant(chrom="1", pos=200, gene="GENE_EC"),
            ]
        elif j < 65:
            variants += [_Variant(chrom="1", pos=100, gene="GENE_EX")]
        elif j < 70:
            variants += [_Variant(chrom="1", pos=200, gene="GENE_EC")]
        if 70 <= j < 150:
            variants += [_Variant(chrom="2", pos=100, gene="GENE_MX1")]
        elif 150 <= j < 200:
            variants += [_Variant(chrom="2", pos=200, gene="GENE_MX2")]
        samples.append(_Sample(sample_id=f"S{j}", label=int(j < 100), variants=variants))

    pt_path = tmp_path / "pre.pt"
    torch.save({"samples": samples}, pt_path)

    gene_names = ["GENE_EX", "GENE_EC", "GENE_MX1", "GENE_MX2"]
    var_rows = [{"gene_name": g, "chromosome": "1" if i < 2 else "2",
                 "position": (i + 1) * 100, "z_attribution": 5.0 - i}
                for i, g in enumerate(gene_names)]
    gene_rows = [{"gene_name": g, "gene_z_score": 5.0 - i, "gene_rank": i + 1}
                 for i, g in enumerate(gene_names)]

    var_path = tmp_path / "var.csv"
    pd.DataFrame(var_rows).to_csv(var_path, index=False)
    gene_path = tmp_path / "gene.csv"
    pd.DataFrame(gene_rows).to_csv(gene_path, index=False)

    def _get_pval(pairs: pd.DataFrame, g1: str, g2: str) -> float:
        mask = (
            ((pairs["gene_a"] == g1) & (pairs["gene_b"] == g2)) |
            ((pairs["gene_a"] == g2) & (pairs["gene_b"] == g1))
        )
        assert mask.any(), f"Pair {g1}/{g2} not found"
        return float(pairs[mask].iloc[0]["pval_fisher"])

    for alt in ("two-sided", "greater", "less"):
        out_dir = tmp_path / f"out_{alt}"
        _run_main(monkeypatch, [
            "--preprocessed-data", str(pt_path),
            "--variant-rankings", str(var_path),
            "--gene-rankings", str(gene_path),
            "--output-dir", str(out_dir),
            "--top-k-genes", "4",
            "--min-cooccur-samples", "0",
            "--correction", "none",
            "--alpha", "0.05",
            "--alternative", alt,
        ])
        pairs = _pairs_csv(out_dir)
        pval_excess = _get_pval(pairs, "GENE_EX", "GENE_EC")
        pval_deficit = _get_pval(pairs, "GENE_MX1", "GENE_MX2")

        if alt == "two-sided":
            assert pval_excess < 0.01, f"two-sided: excess pair pval={pval_excess:.4f} not < 0.01"
            assert pval_deficit < 0.01, f"two-sided: deficit pair pval={pval_deficit:.4f} not < 0.01"
        elif alt == "greater":
            assert pval_excess < 0.01, f"greater: excess pair pval={pval_excess:.4f} not < 0.01"
            assert pval_deficit > 0.5, f"greater: deficit pair pval={pval_deficit:.4f} not > 0.5"
        else:  # less
            assert pval_excess > 0.5, f"less: excess pair pval={pval_excess:.4f} not > 0.5"
            assert pval_deficit < 0.01, f"less: deficit pair pval={pval_deficit:.4f} not < 0.01"


def test_alternative_recorded_in_metadata(tmp_path: Path, monkeypatch) -> None:
    """Each --alternative value is recorded in the summary YAML fisher_alternative field."""
    pt = make_pt(tmp_path)
    vr = make_variant_rankings(tmp_path)
    gr = make_gene_rankings(tmp_path)

    for alt in ("greater", "two-sided", "less"):
        out_dir = tmp_path / f"out_{alt}"
        _run_main(monkeypatch, [
            "--preprocessed-data", str(pt),
            "--variant-rankings", str(vr),
            "--gene-rankings", str(gr),
            "--output-dir", str(out_dir),
            "--top-k-genes", "6",
            "--min-cooccur-samples", "1",
            "--alternative", alt,
        ])
        summary = _summary_yaml(out_dir)
        assert summary["score_basis"]["fisher_alternative"] == alt, (
            f"Expected fisher_alternative={alt!r}, got {summary['score_basis'].get('fisher_alternative')!r}"
        )


def test_interaction_score_formula_recorded_in_metadata(tmp_path: Path, monkeypatch) -> None:
    """interaction_score_formula appears in score_basis and mentions rank_quantile and log1p."""
    pt = make_pt(tmp_path)
    vr = make_variant_rankings(tmp_path)
    gr = make_gene_rankings(tmp_path)
    out_dir = tmp_path / "out"

    _run_main(monkeypatch, [
        "--preprocessed-data", str(pt),
        "--variant-rankings", str(vr),
        "--gene-rankings", str(gr),
        "--output-dir", str(out_dir),
        "--top-k-genes", "6",
        "--min-cooccur-samples", "1",
    ])
    summary = _summary_yaml(out_dir)
    formula = summary["score_basis"].get("interaction_score_formula", "")
    assert "rank_quantile" in formula, f"'rank_quantile' not found in formula: {formula!r}"
    assert "log1p" in formula, f"'log1p' not found in formula: {formula!r}"
