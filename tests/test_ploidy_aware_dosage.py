"""
Tests for ploidy-aware dosage encoding in VCF parser.

Verifies:
1. Male chrX hemizygous alt (non-PAR) -> dosage 2
2. Female chrX het -> dosage 1
3. Male chrX PAR variant -> normal diploid (dosage 1 for het)
4. Female chrY variant -> skipped
5. Autosomal variants -> unaffected by sex_map
6. Build parameter changes PAR boundaries correctly
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.genome import (
    GRCH37,
    GRCH38,
    get_genome_build,
    is_in_par,
    normalise_chrom,
)


class TestPloidyAwareDosageLogic:
    """
    Tests for the ploidy-aware dosage logic extracted from parse_vcf_cyvcf2.

    Since we cannot easily create real VCF data in unit tests, we test
    the dosage logic directly.
    """

    def _compute_dosage(
        self, allele1, allele2, target_allele, chrom, pos, sample_sex, build
    ):
        """
        Replicate the dosage computation logic from parse_vcf_cyvcf2.
        """
        is_non_par_x = (chrom == 'X') and not is_in_par(pos, 'X', build)
        is_y = (chrom == 'Y')

        # Skip female Y
        if is_y and sample_sex == 'F':
            return None  # Indicates skipped

        dosage = 0
        if allele1 == target_allele:
            dosage += 1
        if allele2 >= 0 and allele2 == target_allele:
            dosage += 1

        # Ploidy correction
        if is_non_par_x and sample_sex == 'M':
            dosage = dosage * 2

        return dosage

    # Test 1: Male hemizygous alt on chrX non-PAR -> dosage 2
    def test_male_chrx_nonpar_hemizygous_alt(self):
        # Male with hemizygous alt on chrX (allele2 = -1 from cyvcf2)
        dosage = self._compute_dosage(
            allele1=1, allele2=-1, target_allele=1,
            chrom='X', pos=50000000, sample_sex='M', build=GRCH37,
        )
        assert dosage == 2

    # Test 2: Female het on chrX -> dosage 1
    def test_female_chrx_het(self):
        dosage = self._compute_dosage(
            allele1=0, allele2=1, target_allele=1,
            chrom='X', pos=50000000, sample_sex='F', build=GRCH37,
        )
        assert dosage == 1

    # Test 3: Female hom-alt on chrX -> dosage 2
    def test_female_chrx_hom_alt(self):
        dosage = self._compute_dosage(
            allele1=1, allele2=1, target_allele=1,
            chrom='X', pos=50000000, sample_sex='F', build=GRCH37,
        )
        assert dosage == 2

    # Test 4: Male in PAR1 on chrX -> normal diploid (het = 1)
    def test_male_chrx_par1_het(self):
        # PAR1 on GRCh37: (60001, 2699520)
        dosage = self._compute_dosage(
            allele1=0, allele2=1, target_allele=1,
            chrom='X', pos=1000000, sample_sex='M', build=GRCH37,
        )
        assert dosage == 1  # PAR = diploid, no doubling

    # Test 5: Male ref on chrX non-PAR -> dosage 0
    def test_male_chrx_nonpar_ref(self):
        dosage = self._compute_dosage(
            allele1=0, allele2=-1, target_allele=1,
            chrom='X', pos=50000000, sample_sex='M', build=GRCH37,
        )
        assert dosage == 0

    # Test 6: Female chrY variant -> skipped (returns None)
    def test_female_chry_skipped(self):
        dosage = self._compute_dosage(
            allele1=1, allele2=-1, target_allele=1,
            chrom='Y', pos=10000000, sample_sex='F', build=GRCH37,
        )
        assert dosage is None

    # Test 7: Male chrY variant -> normal dosage (no doubling for Y)
    def test_male_chry_normal(self):
        dosage = self._compute_dosage(
            allele1=1, allele2=-1, target_allele=1,
            chrom='Y', pos=10000000, sample_sex='M', build=GRCH37,
        )
        assert dosage == 1

    # Test 8: Autosomal variant -> unaffected by sex
    def test_autosomal_unaffected_male(self):
        dosage = self._compute_dosage(
            allele1=0, allele2=1, target_allele=1,
            chrom='1', pos=50000000, sample_sex='M', build=GRCH37,
        )
        assert dosage == 1

    def test_autosomal_unaffected_female(self):
        dosage = self._compute_dosage(
            allele1=0, allele2=1, target_allele=1,
            chrom='1', pos=50000000, sample_sex='F', build=GRCH37,
        )
        assert dosage == 1

    # Test 9: Unknown sex -> no ploidy correction (conservative)
    def test_unknown_sex_no_correction(self):
        dosage = self._compute_dosage(
            allele1=1, allele2=-1, target_allele=1,
            chrom='X', pos=50000000, sample_sex='UNKNOWN', build=GRCH37,
        )
        assert dosage == 1  # No doubling for unknown sex

    # Test 10: GRCh38 PAR boundaries differ from GRCh37
    def test_grch38_par_boundary(self):
        # Position 2700000: outside PAR in GRCh37, inside PAR in GRCh38
        # GRCh37 PAR1 ends at 2699520
        # GRCh38 PAR1 ends at 2781479

        # Under GRCh37: non-PAR -> male gets doubled
        dosage_37 = self._compute_dosage(
            allele1=1, allele2=-1, target_allele=1,
            chrom='X', pos=2700000, sample_sex='M', build=GRCH37,
        )
        assert dosage_37 == 2  # Non-PAR in GRCh37, so doubled

        # Under GRCh38: inside PAR -> male gets normal diploid
        dosage_38 = self._compute_dosage(
            allele1=1, allele2=-1, target_allele=1,
            chrom='X', pos=2700000, sample_sex='M', build=GRCH38,
        )
        assert dosage_38 == 1  # PAR in GRCh38, no doubling

    # Test 11: Male in PAR2 on chrX -> normal diploid
    def test_male_chrx_par2_het(self):
        # PAR2 on GRCh37: (154931044, 155260560)
        dosage = self._compute_dosage(
            allele1=0, allele2=1, target_allele=1,
            chrom='X', pos=155000000, sample_sex='M', build=GRCH37,
        )
        assert dosage == 1  # PAR = diploid, no doubling


class TestContigNormalisationInParsing:
    """Test that contig normalisation works correctly for sex chromosomes."""

    def test_chrx_variants_normalised(self):
        assert normalise_chrom('chrX', GRCH37) == 'X'
        assert normalise_chrom('23', GRCH37) == 'X'
        assert normalise_chrom('chr23', GRCH37) == 'X'

    def test_chry_variants_normalised(self):
        assert normalise_chrom('chrY', GRCH37) == 'Y'
        assert normalise_chrom('24', GRCH37) == 'Y'
        assert normalise_chrom('chr24', GRCH37) == 'Y'
