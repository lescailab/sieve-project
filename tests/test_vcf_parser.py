"""
Tests for src/data/vcf_parser.py — load_phenotypes function.

Tests cover:
- Correctly formatted phenotype file loads without error
- File with a header row raises ValueError with a helpful message
- Various known header field names are detected
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.vcf_parser import load_phenotypes


def _write_tsv(path: Path, lines: list[str]) -> None:
    """Write lines (without trailing newline on last) to path."""
    path.write_text('\n'.join(lines) + '\n')


class TestLoadPhenotypesCorrectFormat:
    """load_phenotypes with well-formed input."""

    def test_controls_and_cases(self, tmp_path):
        """Standard 1/2-encoded file without header loads correctly."""
        f = tmp_path / 'pheno.tsv'
        _write_tsv(f, [
            'SAMPLE001\t1',
            'SAMPLE002\t2',
            'SAMPLE003\t1',
        ])
        result = load_phenotypes(f)
        assert result == {'SAMPLE001': 0, 'SAMPLE002': 1, 'SAMPLE003': 0}

    def test_comments_are_skipped(self, tmp_path):
        """Lines beginning with '#' are silently ignored."""
        f = tmp_path / 'pheno.tsv'
        _write_tsv(f, [
            '# This is a comment',
            'SAMPLE001\t1',
            '# Another comment',
            'SAMPLE002\t2',
        ])
        result = load_phenotypes(f)
        assert 'SAMPLE001' in result
        assert 'SAMPLE002' in result
        assert len(result) == 2

    def test_empty_lines_are_skipped(self, tmp_path):
        """Empty lines do not cause errors."""
        f = tmp_path / 'pheno.tsv'
        _write_tsv(f, [
            '',
            'SAMPLE001\t2',
            '',
            'SAMPLE002\t1',
        ])
        result = load_phenotypes(f)
        assert result == {'SAMPLE001': 1, 'SAMPLE002': 0}


class TestLoadPhenotypesHeaderGuard:
    """load_phenotypes rejects files with a header row."""

    @pytest.mark.parametrize('header_field', [
        'sample_id',
        'SAMPLE_ID',
        'Sample_Id',
        'sample',
        'SAMPLE',
        'id',
        'ID',
        'phenotype',
        'PHENOTYPE',
        'status',
        'STATUS',
        'label',
        'LABEL',
        'iid',
        'IID',
        'fid',
        'FID',
        '#iid',
        '#IID',
    ])
    def test_known_header_names_raise(self, tmp_path, header_field):
        """First-row fields that look like column headers raise ValueError."""
        f = tmp_path / 'pheno_with_header.tsv'
        _write_tsv(f, [
            f'{header_field}\tphenotype',
            'SAMPLE001\t1',
            'SAMPLE002\t2',
        ])
        with pytest.raises(ValueError, match='header'):
            load_phenotypes(f)

    def test_error_message_is_helpful(self, tmp_path):
        """The ValueError message names the offending field and explains the fix."""
        f = tmp_path / 'pheno_with_header.tsv'
        _write_tsv(f, [
            'sample_id\tphenotype',
            'SAMPLE001\t1',
        ])
        with pytest.raises(ValueError) as exc_info:
            load_phenotypes(f)

        msg = str(exc_info.value)
        assert 'header' in msg.lower()
        assert 'sample_id' in msg
