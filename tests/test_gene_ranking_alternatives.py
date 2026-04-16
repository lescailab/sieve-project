"""
Acceptance tests for B4: gene-ranking aggregation alternatives.

Tests cover:
1. 'max' aggregation (existing behaviour) preserves gene_size column.
2. 'mean' aggregation produces a lower gene_score than 'max' when variants differ.
3. 'size_normalised' penalises large genes relative to single-variant genes.
4. 'gene_size' column is present in all aggregation outputs.
5. Unknown aggregation raises ValueError.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explain.variant_ranking import VariantRanker


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_variant_rankings() -> pd.DataFrame:
    """
    Minimal variant-rankings DataFrame with two genes:
    - GENE_0: 1 variant, high attribution (0.9)
    - GENE_1: 3 variants, mixed attributions (0.1, 0.2, 0.3)
    """
    records = [
        # gene 0 — single high-attribution variant
        {'chromosome': '1', 'position': 1000, 'gene_name': 'GENE_0', 'gene_id': 0,
         'mean_attribution': 0.9, 'max_attribution': 0.9,
         'median_attribution': 0.9, 'std_attribution': 0.0,
         'num_samples': 10, 'score': 0.9, 'rank': 1},
        # gene 1 — three low-attribution variants
        {'chromosome': '1', 'position': 2000, 'gene_name': 'GENE_1', 'gene_id': 1,
         'mean_attribution': 0.1, 'max_attribution': 0.3,
         'median_attribution': 0.1, 'std_attribution': 0.1,
         'num_samples': 10, 'score': 0.1, 'rank': 2},
        {'chromosome': '1', 'position': 2100, 'gene_name': 'GENE_1', 'gene_id': 1,
         'mean_attribution': 0.2, 'max_attribution': 0.3,
         'median_attribution': 0.2, 'std_attribution': 0.1,
         'num_samples': 10, 'score': 0.2, 'rank': 3},
        {'chromosome': '1', 'position': 2200, 'gene_name': 'GENE_1', 'gene_id': 1,
         'mean_attribution': 0.3, 'max_attribution': 0.3,
         'median_attribution': 0.3, 'std_attribution': 0.0,
         'num_samples': 10, 'score': 0.3, 'rank': 4},
    ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGeneRankingAggregations:
    """Tests for rank_genes() aggregation alternatives."""

    def test_max_returns_highest_variant_score_per_gene(self):
        """max aggregation: gene_score = highest variant mean_attribution."""
        ranker = VariantRanker()
        vr = _make_variant_rankings()
        gr = ranker.rank_genes(vr, aggregation='max')

        gene0 = gr[gr['gene_id'] == 0].iloc[0]
        gene1 = gr[gr['gene_id'] == 1].iloc[0]

        assert abs(gene0['gene_score'] - 0.9) < 1e-6
        # max mean_attribution across gene1's 3 variants
        assert abs(gene1['gene_score'] - 0.3) < 1e-6

    def test_mean_returns_average_attribution_per_gene(self):
        """mean aggregation: gene_score = mean of variant mean_attributions."""
        ranker = VariantRanker()
        vr = _make_variant_rankings()
        gr = ranker.rank_genes(vr, aggregation='mean')

        gene1 = gr[gr['gene_id'] == 1].iloc[0]
        expected_mean = (0.1 + 0.2 + 0.3) / 3
        assert abs(gene1['gene_score'] - expected_mean) < 1e-6

    def test_mean_score_leq_max_score(self):
        """mean gene_score is always ≤ max gene_score for the same gene."""
        ranker = VariantRanker()
        vr = _make_variant_rankings()
        gr_max = ranker.rank_genes(vr, aggregation='max')
        gr_mean = ranker.rank_genes(vr, aggregation='mean')

        for gene_id in vr['gene_id'].unique():
            s_max = gr_max[gr_max['gene_id'] == gene_id].iloc[0]['gene_score']
            s_mean = gr_mean[gr_mean['gene_id'] == gene_id].iloc[0]['gene_score']
            assert s_mean <= s_max + 1e-9, (
                f"Gene {gene_id}: mean score {s_mean} > max score {s_max}"
            )

    def test_size_normalised_penalises_large_gene(self):
        """
        size_normalised: a single-variant gene with score 0.9 should rank
        above a 3-variant gene with mean 0.9 (because / sqrt(3) < / sqrt(1)).
        """
        # Build variant_rankings where gene1 also has mean_attribution=0.9
        records = [
            {'chromosome': '1', 'position': 1000, 'gene_name': 'GENE_0', 'gene_id': 0,
             'mean_attribution': 0.9, 'max_attribution': 0.9,
             'median_attribution': 0.9, 'std_attribution': 0.0,
             'num_samples': 10, 'score': 0.9, 'rank': 1},
            {'chromosome': '1', 'position': 2000, 'gene_name': 'GENE_1', 'gene_id': 1,
             'mean_attribution': 0.9, 'max_attribution': 0.9,
             'median_attribution': 0.9, 'std_attribution': 0.0,
             'num_samples': 10, 'score': 0.9, 'rank': 2},
            {'chromosome': '1', 'position': 2100, 'gene_name': 'GENE_1', 'gene_id': 1,
             'mean_attribution': 0.9, 'max_attribution': 0.9,
             'median_attribution': 0.9, 'std_attribution': 0.0,
             'num_samples': 10, 'score': 0.9, 'rank': 3},
            {'chromosome': '1', 'position': 2200, 'gene_name': 'GENE_1', 'gene_id': 1,
             'mean_attribution': 0.9, 'max_attribution': 0.9,
             'median_attribution': 0.9, 'std_attribution': 0.0,
             'num_samples': 10, 'score': 0.9, 'rank': 4},
        ]
        vr = pd.DataFrame(records)
        ranker = VariantRanker()
        gr = ranker.rank_genes(vr, aggregation='size_normalised')

        gene0 = gr[gr['gene_id'] == 0].iloc[0]
        gene1 = gr[gr['gene_id'] == 1].iloc[0]

        # gene0: 0.9 / sqrt(1) = 0.9; gene1: 0.9 / sqrt(3) ≈ 0.52
        assert gene0['gene_score'] > gene1['gene_score']
        assert abs(gene0['gene_score'] - 0.9) < 1e-6
        assert abs(gene1['gene_score'] - 0.9 / np.sqrt(3)) < 1e-6

    def test_size_normalised_expected_score_formula(self):
        """size_normalised gene_score = max(mean_attribution) / sqrt(num_variants)."""
        ranker = VariantRanker()
        vr = _make_variant_rankings()
        gr = ranker.rank_genes(vr, aggregation='size_normalised')

        gene1 = gr[gr['gene_id'] == 1].iloc[0]
        expected = 0.3 / np.sqrt(3)
        assert abs(gene1['gene_score'] - expected) < 1e-6


class TestGeneSizeColumn:
    """gene_size column must be present in ALL aggregation outputs."""

    @pytest.mark.parametrize('aggregation', ['max', 'mean', 'sum', 'size_normalised'])
    def test_gene_size_column_present(self, aggregation):
        """gene_size is always present regardless of aggregation method."""
        ranker = VariantRanker()
        vr = _make_variant_rankings()
        gr = ranker.rank_genes(vr, aggregation=aggregation)
        assert 'gene_size' in gr.columns, (
            f"gene_size column missing for aggregation='{aggregation}'"
        )

    @pytest.mark.parametrize('aggregation', ['max', 'mean', 'sum', 'size_normalised'])
    def test_gene_size_equals_num_variants(self, aggregation):
        """gene_size == num_variants for all aggregation methods."""
        ranker = VariantRanker()
        vr = _make_variant_rankings()
        gr = ranker.rank_genes(vr, aggregation=aggregation)
        assert (gr['gene_size'] == gr['num_variants']).all(), (
            f"gene_size != num_variants for aggregation='{aggregation}'"
        )

    def test_unknown_aggregation_raises(self):
        """Unknown aggregation raises ValueError."""
        ranker = VariantRanker()
        vr = _make_variant_rankings()
        with pytest.raises(ValueError, match='Unknown aggregation'):
            ranker.rank_genes(vr, aggregation='invalid')
