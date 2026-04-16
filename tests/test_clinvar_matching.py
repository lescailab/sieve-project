"""
Acceptance tests for B5: hardened ClinVar matching.

Tests cover:
1. Exact chromosome+position match is required and returns correct significance.
2. 'chr'-prefix normalisation ('chr1' vs '1') — both sides.
3. Missing 'chromosome' column in variant_rankings raises ValueError.
4. Missing 'chrom' column in clinvar_df raises ValueError.
5. Position-only match that would previously succeed now correctly MISSES
   when the chromosome differs.
6. Empty ClinVar DataFrame returns top_k rows unchanged.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explain.biological_validation import BiologicalValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_variant_rankings(chroms=None, positions=None, n=5) -> pd.DataFrame:
    chroms = chroms or (['1'] * n)
    positions = positions or list(range(1000, 1000 + n))
    return pd.DataFrame({
        'chromosome': chroms,
        'position': positions,
        'gene_name': [f'GENE_{i}' for i in range(n)],
        'mean_attribution': [float(i) / n for i in range(n, 0, -1)],
        'rank': list(range(1, n + 1)),
    })


def _make_clinvar(chroms=None, positions=None, significance=None, n=3) -> pd.DataFrame:
    chroms = chroms or (['1'] * n)
    positions = positions or list(range(1000, 1000 + n))
    significance = significance or (['Pathogenic'] * n)
    return pd.DataFrame({
        'chrom': chroms,
        'pos': positions,
        'clinical_significance': significance,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClinVarMatchingHardened:

    def test_exact_chrom_pos_match(self):
        """Variants matching ClinVar on chrom+pos are flagged in_clinvar=True."""
        vr = _make_variant_rankings(chroms=['1', '2', '3', '4', '5'],
                                    positions=[1000, 2000, 3000, 4000, 5000])
        cv = _make_clinvar(chroms=['1', '3'],
                           positions=[1000, 3000],
                           significance=['Pathogenic', 'Likely_pathogenic'])

        validator = BiologicalValidator()
        result = validator.validate_variants_against_clinvar(vr, cv, top_k=5)

        assert result.loc[result['position'] == 1000, 'in_clinvar'].iloc[0] is True or \
               result.loc[result['position'] == 1000, 'in_clinvar'].iloc[0] == True
        assert result.loc[result['position'] == 3000, 'in_clinvar'].iloc[0] == True
        assert result.loc[result['position'] == 2000, 'in_clinvar'].iloc[0] == False

    def test_significance_propagated(self):
        """Clinical significance is copied from ClinVar to matched variants."""
        vr = _make_variant_rankings(chroms=['1'], positions=[1000], n=1)
        cv = _make_clinvar(chroms=['1'], positions=[1000], significance=['Pathogenic'], n=1)

        validator = BiologicalValidator()
        result = validator.validate_variants_against_clinvar(vr, cv, top_k=1)

        assert result.iloc[0]['clinvar_significance'] == 'Pathogenic'

    def test_chr_prefix_normalisation_variants_have_chr(self):
        """Variants with 'chr1' match ClinVar with '1'."""
        vr = _make_variant_rankings(chroms=['chr1', 'chr2'], positions=[1000, 2000], n=2)
        cv = _make_clinvar(chroms=['1'], positions=[1000], significance=['Pathogenic'], n=1)

        validator = BiologicalValidator()
        result = validator.validate_variants_against_clinvar(vr, cv, top_k=2)

        assert result.loc[result['position'] == 1000, 'in_clinvar'].iloc[0] == True
        assert result.loc[result['position'] == 2000, 'in_clinvar'].iloc[0] == False

    def test_chr_prefix_normalisation_clinvar_has_chr(self):
        """Variants with '1' match ClinVar with 'chr1'."""
        vr = _make_variant_rankings(chroms=['1', '2'], positions=[1000, 2000], n=2)
        cv = _make_clinvar(chroms=['chr1'], positions=[1000], significance=['Pathogenic'], n=1)

        validator = BiologicalValidator()
        result = validator.validate_variants_against_clinvar(vr, cv, top_k=2)

        assert result.loc[result['position'] == 1000, 'in_clinvar'].iloc[0] == True

    def test_same_position_different_chrom_does_not_match(self):
        """Position 1000 on chr1 must NOT match ClinVar position 1000 on chr2."""
        vr = _make_variant_rankings(chroms=['1'], positions=[1000], n=1)
        cv = _make_clinvar(chroms=['2'], positions=[1000], significance=['Pathogenic'], n=1)

        validator = BiologicalValidator()
        result = validator.validate_variants_against_clinvar(vr, cv, top_k=1)

        assert result.iloc[0]['in_clinvar'] == False

    def test_missing_chromosome_in_variants_raises(self):
        """ValueError when variant_rankings has no 'chromosome' column."""
        vr = pd.DataFrame({'position': [1000], 'mean_attribution': [0.9], 'rank': [1]})
        cv = _make_clinvar()

        validator = BiologicalValidator()
        with pytest.raises(ValueError, match='chromosome'):
            validator.validate_variants_against_clinvar(vr, cv, top_k=1)

    def test_missing_chrom_in_clinvar_raises(self):
        """ValueError when clinvar_df has no 'chrom' column."""
        vr = _make_variant_rankings()
        cv = pd.DataFrame({'pos': [1000], 'clinical_significance': ['Pathogenic']})

        validator = BiologicalValidator()
        with pytest.raises(ValueError, match='chrom'):
            validator.validate_variants_against_clinvar(vr, cv, top_k=5)

    def test_empty_clinvar_returns_top_k_unchanged(self):
        """Empty ClinVar DataFrame returns the top-k variant rows unmodified."""
        vr = _make_variant_rankings(n=10)
        cv = pd.DataFrame()

        validator = BiologicalValidator()
        result = validator.validate_variants_against_clinvar(vr, cv, top_k=5)

        assert len(result) == 5
        assert 'in_clinvar' not in result.columns  # unchanged = no new columns added
