"""Tests for generate_sieve_gene_list.py — FDR-threshold filtering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_sieve_gene_list import generate_gene_list


def _make_variant_rankings(n_genes: int = 20, n_variants_per_gene: int = 3) -> pd.DataFrame:
    """Create synthetic corrected variant rankings."""
    rng = np.random.default_rng(42)
    rows = []
    for g in range(n_genes):
        for v in range(n_variants_per_gene):
            rows.append({
                "variant_id": f"var_{g}_{v}",
                "chromosome": str((g % 22) + 1),
                "position": 1000 + v,
                "gene_name": f"GENE_{g}",
                "z_attribution": rng.uniform(0, 4),
            })
    return pd.DataFrame(rows)


def _make_gene_significance(gene_names: list[str]) -> pd.DataFrame:
    """Create gene significance file with known FDR values."""
    n = len(gene_names)
    fdr_values = np.linspace(0.001, 0.9, n)
    return pd.DataFrame({
        "gene_name": gene_names,
        "gene_score": np.linspace(5, 0.1, n),
        "num_variants": [3] * n,
        "empirical_p_gene": fdr_values / 2,
        "fdr_gene": fdr_values,
    })


class TestFDRThresholdFiltering:

    def test_fdr_threshold_filters_genes(self) -> None:
        """Only genes below FDR threshold should be kept."""
        variants = _make_variant_rankings(n_genes=20)
        gene_names = sorted(variants["gene_name"].unique())
        gene_sig = _make_gene_significance(gene_names)

        # Count how many genes have fdr_gene < 0.1
        expected_passing = (gene_sig["fdr_gene"] < 0.1).sum()
        assert expected_passing > 0 and expected_passing < len(gene_names)

        result = generate_gene_list(
            variants,
            score_column="z_attribution",
            fdr_threshold=0.1,
            gene_significance_df=gene_sig,
        )

        assert len(result) == expected_passing
        assert "fdr_gene" in result.columns
        assert (result["fdr_gene"] < 0.1).all()

    def test_fdr_threshold_no_genes_pass(self) -> None:
        """Should produce empty DataFrame when no genes pass threshold."""
        variants = _make_variant_rankings(n_genes=10)
        gene_names = sorted(variants["gene_name"].unique())
        gene_sig = _make_gene_significance(gene_names)
        # Set all FDR values above threshold
        gene_sig["fdr_gene"] = 0.99

        result = generate_gene_list(
            variants,
            score_column="z_attribution",
            fdr_threshold=0.05,
            gene_significance_df=gene_sig,
        )

        assert len(result) == 0

    def test_fdr_threshold_without_significance_raises(self) -> None:
        """Should raise ValueError when FDR threshold set but no significance."""
        variants = _make_variant_rankings(n_genes=10)

        with pytest.raises(ValueError, match="fdr_gene"):
            generate_gene_list(
                variants,
                score_column="z_attribution",
                fdr_threshold=0.05,
                gene_significance_df=None,
            )

    def test_no_fdr_threshold_returns_all_genes(self) -> None:
        """Without FDR threshold, all genes should be returned."""
        variants = _make_variant_rankings(n_genes=10)
        result = generate_gene_list(
            variants,
            score_column="z_attribution",
        )
        assert len(result) == 10

    def test_re_ranking_after_fdr_filter(self) -> None:
        """Gene ranks should be contiguous after FDR filtering."""
        variants = _make_variant_rankings(n_genes=20)
        gene_names = sorted(variants["gene_name"].unique())
        gene_sig = _make_gene_significance(gene_names)

        result = generate_gene_list(
            variants,
            score_column="z_attribution",
            fdr_threshold=0.5,
            gene_significance_df=gene_sig,
        )

        assert list(result["gene_rank"]) == list(range(1, len(result) + 1))
