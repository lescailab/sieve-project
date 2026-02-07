"""
Tests for src/data/genome.py — genome build definitions.

Tests cover:
- GenomeBuild retrieval and aliases
- PAR region boundary checks for GRCh37 and GRCh38
- Chromosome classification (autosomal, sex)
- Contig normalisation (chr prefix, numeric aliases)
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.genome import (
    GRCH37,
    GRCH38,
    get_genome_build,
    is_autosomal,
    is_in_par,
    is_sex_chrom,
    normalise_chrom,
)


class TestGetGenomeBuild:
    """Tests for get_genome_build()."""

    def test_canonical_names(self):
        assert get_genome_build('GRCh37').name == 'GRCh37'
        assert get_genome_build('GRCh38').name == 'GRCh38'

    def test_case_insensitive(self):
        assert get_genome_build('grch37').name == 'GRCh37'
        assert get_genome_build('GRCH38').name == 'GRCh38'

    def test_aliases(self):
        assert get_genome_build('hg19').name == 'GRCh37'
        assert get_genome_build('hg38').name == 'GRCh38'
        assert get_genome_build('b37').name == 'GRCh37'
        assert get_genome_build('b38').name == 'GRCh38'

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported genome build"):
            get_genome_build('T2T')

    def test_unsupported_with_info(self):
        with pytest.raises(ValueError, match="hg19"):
            get_genome_build('hg20')


class TestIsInPar:
    """Tests for is_in_par() across both builds."""

    # GRCh37 PAR1 on X: (60001, 2699520)
    def test_grch37_par1_inside(self):
        assert is_in_par(60001, 'X', GRCH37) is True
        assert is_in_par(1000000, 'X', GRCH37) is True
        assert is_in_par(2699520, 'X', GRCH37) is True

    def test_grch37_par1_outside(self):
        assert is_in_par(60000, 'X', GRCH37) is False
        assert is_in_par(2699521, 'X', GRCH37) is False

    # GRCh37 PAR2 on X: (154931044, 155260560)
    def test_grch37_par2_inside(self):
        assert is_in_par(154931044, 'X', GRCH37) is True
        assert is_in_par(155000000, 'X', GRCH37) is True
        assert is_in_par(155260560, 'X', GRCH37) is True

    def test_grch37_par2_outside(self):
        assert is_in_par(154931043, 'X', GRCH37) is False
        assert is_in_par(155260561, 'X', GRCH37) is False

    # GRCh38 PAR1 on X: (10001, 2781479)
    def test_grch38_par1_inside(self):
        assert is_in_par(10001, 'X', GRCH38) is True
        assert is_in_par(2781479, 'X', GRCH38) is True

    def test_grch38_par1_outside(self):
        assert is_in_par(10000, 'X', GRCH38) is False
        assert is_in_par(2781480, 'X', GRCH38) is False

    # Position that differs between builds
    def test_build_specific_boundary(self):
        # GRCh37 PAR1 ends at 2699520, GRCh38 PAR1 ends at 2781479
        # Position 2700000 is outside PAR in GRCh37, inside PAR in GRCh38
        assert is_in_par(2700000, 'X', GRCH37) is False
        assert is_in_par(2700000, 'X', GRCH38) is True

    def test_autosomal_not_in_par(self):
        assert is_in_par(1000000, '1', GRCH37) is False

    def test_y_par(self):
        # GRCh37 Y PAR1: (10001, 2649520)
        assert is_in_par(1000000, 'Y', GRCH37) is True
        assert is_in_par(3000000, 'Y', GRCH37) is False


class TestChromClassification:
    """Tests for is_sex_chrom() and is_autosomal()."""

    def test_sex_chroms(self):
        assert is_sex_chrom('X', GRCH37) is True
        assert is_sex_chrom('Y', GRCH37) is True
        assert is_sex_chrom('1', GRCH37) is False
        assert is_sex_chrom('22', GRCH37) is False

    def test_autosomal(self):
        for c in range(1, 23):
            assert is_autosomal(str(c), GRCH37) is True
        assert is_autosomal('X', GRCH37) is False
        assert is_autosomal('Y', GRCH37) is False
        assert is_autosomal('MT', GRCH37) is False


class TestNormaliseChrom:
    """Tests for normalise_chrom()."""

    def test_strip_chr_prefix(self):
        assert normalise_chrom('chr1', GRCH37) == '1'
        assert normalise_chrom('chrX', GRCH37) == 'X'
        assert normalise_chrom('chrY', GRCH37) == 'Y'

    def test_already_normalised(self):
        assert normalise_chrom('1', GRCH37) == '1'
        assert normalise_chrom('X', GRCH37) == 'X'
        assert normalise_chrom('22', GRCH37) == '22'

    def test_numeric_sex_chrom_aliases(self):
        assert normalise_chrom('23', GRCH37) == 'X'
        assert normalise_chrom('24', GRCH37) == 'Y'
        assert normalise_chrom('chr23', GRCH37) == 'X'
        assert normalise_chrom('chr24', GRCH37) == 'Y'


class TestGenomeBuildImmutability:
    """Verify GenomeBuild is frozen (immutable)."""

    def test_frozen(self):
        with pytest.raises(AttributeError):
            GRCH37.name = 'modified'

    def test_builds_are_separate(self):
        assert GRCH37.name != GRCH38.name
        assert GRCH37.par_regions['X'] != GRCH38.par_regions['X']
