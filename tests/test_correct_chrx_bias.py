"""Tests for correct_chrx_bias.py — gene significance merging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.correct_chrx_bias as chrx


def _make_variant_rankings(n: int = 50) -> pd.DataFrame:
    """Create synthetic variant rankings with significance columns."""
    rng = np.random.default_rng(42)
    chroms = [str(c) for c in range(1, 23)] * (n // 22 + 1)
    return pd.DataFrame({
        "variant_id": [f"var_{i}" for i in range(n)],
        "chromosome": chroms[:n],
        "position": rng.integers(1_000, 100_000, size=n),
        "gene_name": [f"GENE_{i % 20}" for i in range(n)],
        "mean_attribution": rng.uniform(0, 1, size=n),
        "empirical_p_variant": rng.uniform(0, 1, size=n),
        "fdr_variant": rng.uniform(0, 1, size=n),
    })


def _make_gene_significance(genes: list[str]) -> pd.DataFrame:
    """Create synthetic gene-level significance file."""
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "gene_name": genes,
        "gene_score": rng.uniform(0, 1, size=len(genes)),
        "num_variants": rng.integers(1, 10, size=len(genes)),
        "empirical_p_gene": rng.uniform(0, 1, size=len(genes)),
        "fdr_gene": rng.uniform(0, 1, size=len(genes)),
    })


class TestCreateGeneRankingsWithSignificance:

    def test_merge_adds_significance_columns(self) -> None:
        """Corrected gene rankings include empirical_p_gene and fdr_gene."""
        variants = _make_variant_rankings()
        unique_genes = sorted(variants["gene_name"].unique())
        gene_sig = _make_gene_significance(unique_genes)

        from src.data.genome import get_genome_build
        build = get_genome_build("GRCh37")
        corrected = chrx.compute_chromosome_zscores(variants, build)
        result = chrx.create_gene_rankings(corrected, gene_significance_df=gene_sig)

        assert "empirical_p_gene" in result.columns
        assert "fdr_gene" in result.columns
        assert "gene_z_score" in result.columns
        assert result["empirical_p_gene"].notna().sum() > 0

    def test_no_significance_file_still_works(self) -> None:
        """Gene rankings work without significance (backward compatible)."""
        variants = _make_variant_rankings()
        from src.data.genome import get_genome_build
        build = get_genome_build("GRCh37")
        corrected = chrx.compute_chromosome_zscores(variants, build)
        result = chrx.create_gene_rankings(corrected, gene_significance_df=None)

        assert "gene_z_score" in result.columns
        assert "gene_rank" in result.columns
        assert "empirical_p_gene" not in result.columns

    def test_partial_gene_match(self) -> None:
        """Genes missing from significance get NaN for significance columns."""
        variants = _make_variant_rankings()
        from src.data.genome import get_genome_build
        build = get_genome_build("GRCh37")
        corrected = chrx.compute_chromosome_zscores(variants, build)

        unique_genes = sorted(variants["gene_name"].unique())
        # Only provide significance for half the genes
        gene_sig = _make_gene_significance(unique_genes[:len(unique_genes) // 2])
        result = chrx.create_gene_rankings(corrected, gene_significance_df=gene_sig)

        assert "fdr_gene" in result.columns
        assert result["fdr_gene"].isna().sum() > 0
        assert result["fdr_gene"].notna().sum() > 0


class TestMainAutoDiscovery:

    def test_auto_discovers_gene_significance(self, tmp_path: Path) -> None:
        """main() auto-discovers gene_rankings_with_significance.csv."""
        variants = _make_variant_rankings()
        unique_genes = sorted(variants["gene_name"].unique())
        gene_sig = _make_gene_significance(unique_genes)

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        rankings_path = input_dir / "variant_rankings_with_significance.csv"
        variants.to_csv(rankings_path, index=False)
        gene_sig.to_csv(
            input_dir / "gene_rankings_with_significance.csv", index=False,
        )

        output_dir = tmp_path / "output"
        chrx.main.__wrapped__ if hasattr(chrx.main, '__wrapped__') else None
        # Call via CLI arguments
        import sys
        old_argv = sys.argv
        sys.argv = [
            "correct_chrx_bias.py",
            "--rankings", str(rankings_path),
            "--output-dir", str(output_dir),
            "--genome-build", "GRCh37",
            "--top-k", "5",
        ]
        try:
            chrx.main()
        finally:
            sys.argv = old_argv

        gene_path = output_dir / "corrected_gene_rankings.csv"
        assert gene_path.exists()
        gene_df = pd.read_csv(gene_path)
        assert "fdr_gene" in gene_df.columns
        assert "empirical_p_gene" in gene_df.columns
