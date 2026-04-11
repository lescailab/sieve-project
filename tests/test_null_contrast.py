"""
Unit tests for the null-contrast ranking pipeline:
  - compare_attributions.py  (empirical p-value / FDR logic on raw mean_attribution)
  - correct_chrx_bias.py     (removal of --null-rankings flag; significance column passthrough)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.compare_attributions as compare_mod
import scripts.correct_chrx_bias as chrx_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_df(
    attr_values: np.ndarray,
    chromosomes: list[str] | None = None,
    gene_names: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a minimal raw variant rankings DataFrame."""
    n = len(attr_values)
    rng = np.random.default_rng(seed)
    if chromosomes is None:
        chromosomes = [str(rng.integers(1, 23)) for _ in range(n)]
    if gene_names is None:
        gene_names = [f'GENE_{i}' for i in range(n)]
    return pd.DataFrame({
        'chromosome': chromosomes,
        'position': rng.integers(1_000_000, 250_000_000, size=n),
        'mean_attribution': attr_values,
        'gene_name': gene_names,
    })


# ---------------------------------------------------------------------------
# compute_empirical_pvalues
# ---------------------------------------------------------------------------

class TestComputeEmpiricalPvalues:

    def test_minimum_p_equals_one_over_n_plus_one(self) -> None:
        """
        If the highest real value exceeds all null values the empirical
        p-value should equal 1 / (N + 1), never zero (Phipson & Smyth 2010).
        """
        null_z = np.random.default_rng(0).normal(0, 1, size=999)
        real_z = np.array([null_z.max() + 10.0])
        p = compare_mod.compute_empirical_pvalues(real_z, null_z)
        assert p[0] == pytest.approx(1.0 / (len(null_z) + 1), rel=1e-9)

    def test_p_values_in_valid_range(self) -> None:
        """All empirical p-values must lie in [1/(N+1), 1]."""
        rng = np.random.default_rng(7)
        null_z = rng.normal(0, 1, size=500)
        real_z = rng.normal(0, 1, size=200)
        p = compare_mod.compute_empirical_pvalues(real_z, null_z)
        N = len(null_z)
        assert np.all(p >= 1.0 / (N + 1))
        assert np.all(p <= 1.0)

    def test_strong_signal_gets_low_p(self) -> None:
        """
        Synthetic: 10 signal variants with z=5, 990 noise.
        Signal variants should get p ≈ 1/(N+1), noise should be ~Uniform(0,1).
        """
        rng = np.random.default_rng(42)
        null_z = rng.normal(0, 1, size=1000)
        signal_z = np.full(10, 5.0)
        noise_z = rng.normal(0, 1, size=990)
        real_z = np.concatenate([signal_z, noise_z])

        p = compare_mod.compute_empirical_pvalues(real_z, null_z)
        N = len(null_z)
        min_p = 1.0 / (N + 1)

        # Signal variants should have p at or near the minimum.
        assert np.all(p[:10] <= min_p * 10), \
            "Signal variants should have very small p-values"

        # Noise variants should spread across (0, 1].
        p_noise = p[10:]
        assert p_noise.mean() > 0.3, \
            "Noise p-values should not be uniformly near zero"

    def test_convention_k_plus_1_over_n_plus_1(self) -> None:
        """Direct check of the (k+1)/(N+1) formula."""
        null_z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # N = 5
        real_z = np.array([3.5])  # k = 2 null values >= 3.5 (i.e. 4.0 and 5.0)
        p = compare_mod.compute_empirical_pvalues(real_z, null_z)
        expected = (2 + 1) / (5 + 1)
        assert p[0] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# BH FDR correction
# ---------------------------------------------------------------------------

class TestBhFdr:

    def test_fdr_monotone_nondecreasing_when_sorted_by_p(self) -> None:
        """FDR-adjusted values must be non-decreasing when input is sorted."""
        rng = np.random.default_rng(1)
        p_values = np.sort(rng.uniform(0, 1, size=500))
        fdr = compare_mod._bh_fdr(p_values)
        assert np.all(np.diff(fdr) >= -1e-12), \
            "FDR values must be monotone non-decreasing when p-values are sorted"

    def test_strong_signal_passes_fdr(self) -> None:
        """Variants with very small p-values should pass FDR < 0.01."""
        n = 1000
        p_values = np.ones(n) * 0.5
        p_values[:10] = 1e-6  # 10 very significant variants
        fdr = compare_mod._bh_fdr(p_values)
        assert np.all(fdr[:10] < 0.01), \
            "Variants with p=1e-6 should pass FDR < 0.01"

    def test_fdr_capped_at_one(self) -> None:
        """No FDR value should exceed 1.0."""
        p_values = np.ones(50)  # all p=1, worst case
        fdr = compare_mod._bh_fdr(p_values)
        assert np.all(fdr <= 1.0)


# ---------------------------------------------------------------------------
# End-to-end: main() produces expected output files
# ---------------------------------------------------------------------------

class TestCompareAttributionsMain:

    def _write_raw(self, df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path, index=False)

    def test_outputs_three_files(self, tmp_path: Path) -> None:
        """Running main() should produce the three expected output files."""
        rng = np.random.default_rng(99)
        real_attr = rng.normal(0.5, 1, size=300)
        null_attr = rng.normal(0, 1, size=400)

        real_df = _make_raw_df(real_attr)
        null_df = _make_raw_df(null_attr)

        real_path = tmp_path / 'real_rankings.csv'
        null_path = tmp_path / 'null_rankings.csv'
        self._write_raw(real_df, real_path)
        self._write_raw(null_df, null_path)

        import sys as _sys
        old_argv = _sys.argv[:]
        _sys.argv = [
            'compare_attributions.py',
            '--real', str(real_path),
            '--null', str(null_path),
            '--output-dir', str(tmp_path / 'output'),
        ]
        try:
            compare_mod.main()
        finally:
            _sys.argv = old_argv

        out = tmp_path / 'output'
        assert (out / 'variant_rankings_with_significance.csv').exists()
        assert (out / 'gene_rankings_with_significance.csv').exists()
        assert (out / 'significance_summary.yaml').exists()

    def test_augmented_variant_file_has_significance_columns(
        self, tmp_path: Path,
    ) -> None:
        """Augmented real variant file must have empirical_p_variant and fdr_variant."""
        rng = np.random.default_rng(13)
        real_df = _make_raw_df(rng.normal(0.5, 1, size=200))
        null_df = _make_raw_df(rng.normal(0, 1, size=300))

        real_path = tmp_path / 'r.csv'
        null_path = tmp_path / 'n.csv'
        real_df.to_csv(real_path, index=False)
        null_df.to_csv(null_path, index=False)

        import sys as _sys
        old_argv = _sys.argv[:]
        _sys.argv = [
            'compare_attributions.py',
            '--real', str(real_path),
            '--null', str(null_path),
            '--output-dir', str(tmp_path / 'out'),
        ]
        try:
            compare_mod.main()
        finally:
            _sys.argv = old_argv

        out_df = pd.read_csv(tmp_path / 'out' / 'variant_rankings_with_significance.csv')
        assert 'empirical_p_variant' in out_df.columns
        assert 'fdr_variant' in out_df.columns
        # All columns from the input should still be present.
        for col in real_df.columns:
            assert col in out_df.columns, f"Column '{col}' missing from output"

    def test_gene_file_has_significance_columns(self, tmp_path: Path) -> None:
        """Gene-level output must contain empirical_p_gene and fdr_gene."""
        rng = np.random.default_rng(21)
        real_df = _make_raw_df(rng.normal(0.5, 1, size=200))
        null_df = _make_raw_df(rng.normal(0, 1, size=300))

        real_path = tmp_path / 'r2.csv'
        null_path = tmp_path / 'n2.csv'
        real_df.to_csv(real_path, index=False)
        null_df.to_csv(null_path, index=False)

        import sys as _sys
        old_argv = _sys.argv[:]
        _sys.argv = [
            'compare_attributions.py',
            '--real', str(real_path),
            '--null', str(null_path),
            '--output-dir', str(tmp_path / 'out2'),
        ]
        try:
            compare_mod.main()
        finally:
            _sys.argv = old_argv

        gene_df = pd.read_csv(
            tmp_path / 'out2' / 'gene_rankings_with_significance.csv'
        )
        assert 'empirical_p_gene' in gene_df.columns
        assert 'fdr_gene' in gene_df.columns

    def test_fails_if_mean_attribution_missing_in_real(self, tmp_path: Path) -> None:
        """Script must raise ValueError if real file lacks mean_attribution."""
        rng = np.random.default_rng(5)
        bad_real = pd.DataFrame({
            'chromosome': ['1'] * 10,
            'some_other_column': rng.normal(size=10),
            # deliberately no mean_attribution
        })
        null_df = _make_raw_df(rng.normal(size=100))

        bad_path = tmp_path / 'bad_real.csv'
        null_path = tmp_path / 'null.csv'
        bad_real.to_csv(bad_path, index=False)
        null_df.to_csv(null_path, index=False)

        import sys as _sys
        old_argv = _sys.argv[:]
        _sys.argv = [
            'compare_attributions.py',
            '--real', str(bad_path),
            '--null', str(null_path),
            '--output-dir', str(tmp_path / 'out3'),
        ]
        try:
            with pytest.raises(ValueError, match='mean_attribution'):
                compare_mod.main()
        finally:
            _sys.argv = old_argv

    def test_fails_if_mean_attribution_missing_in_null(self, tmp_path: Path) -> None:
        """Script must raise ValueError if null file lacks mean_attribution."""
        rng = np.random.default_rng(6)
        real_df = _make_raw_df(rng.normal(size=100))
        bad_null = pd.DataFrame({
            'chromosome': ['1'] * 50,
            'some_other_column': rng.normal(size=50),
        })

        real_path = tmp_path / 'real.csv'
        bad_path = tmp_path / 'bad_null.csv'
        real_df.to_csv(real_path, index=False)
        bad_null.to_csv(bad_path, index=False)

        import sys as _sys
        old_argv = _sys.argv[:]
        _sys.argv = [
            'compare_attributions.py',
            '--real', str(real_path),
            '--null', str(bad_path),
            '--output-dir', str(tmp_path / 'out4'),
        ]
        try:
            with pytest.raises(ValueError, match='mean_attribution'):
                compare_mod.main()
        finally:
            _sys.argv = old_argv


# ---------------------------------------------------------------------------
# correct_chrx_bias.py no longer accepts --null-rankings
# ---------------------------------------------------------------------------

class TestCorrectChrxBiasNoNullRankings:

    def test_null_rankings_flag_rejected(self) -> None:
        """
        Invoking correct_chrx_bias.py with --null-rankings should exit with
        an argparse error (exit code 2).
        """
        script = (
            Path(__file__).parent.parent / 'scripts' / 'correct_chrx_bias.py'
        )
        result = subprocess.run(
            [sys.executable, str(script), '--null-rankings', '/some/path.csv',
             '--rankings', '/other.csv', '--output-dir', '/tmp/out'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, (
            f"Expected exit code 2 from argparse error, got {result.returncode}. "
            f"stderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# correct_chrx_bias.py preserves significance columns from compare_attributions
# ---------------------------------------------------------------------------

class TestCorrectChrxBiasPreservesSignificanceColumns:

    def test_significance_columns_preserved(self, tmp_path: Path) -> None:
        """
        compute_chromosome_zscores() must pass through empirical_p_variant and
        fdr_variant when they are present in the input DataFrame.
        """
        from src.data.genome import get_genome_build
        rng = np.random.default_rng(77)
        n = 100
        df = pd.DataFrame({
            'chromosome': [str(rng.integers(1, 23)) for _ in range(n)],
            'position': rng.integers(1_000_000, 250_000_000, size=n),
            'mean_attribution': rng.normal(0.5, 1, size=n),
            'gene_name': [f'GENE_{i}' for i in range(n)],
            'empirical_p_variant': rng.uniform(0, 1, size=n),
            'fdr_variant': rng.uniform(0, 1, size=n),
        })
        df['chromosome'] = df['chromosome'].astype(str)
        build = get_genome_build('GRCh37')
        corrected = chrx_mod.compute_chromosome_zscores(df, build)
        assert 'empirical_p_variant' in corrected.columns, \
            "empirical_p_variant must be preserved after chrX correction"
        assert 'fdr_variant' in corrected.columns, \
            "fdr_variant must be preserved after chrX correction"
        # Values should be identical (column is not modified by z-scoring)
        pd.testing.assert_series_equal(
            corrected['empirical_p_variant'].reset_index(drop=True),
            df['empirical_p_variant'].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            corrected['fdr_variant'].reset_index(drop=True),
            df['fdr_variant'].reset_index(drop=True),
        )

    def test_z_attribution_added(self, tmp_path: Path) -> None:
        """compute_chromosome_zscores() adds z_attribution, corrected_rank, is_sex_chrom."""
        from src.data.genome import get_genome_build
        rng = np.random.default_rng(88)
        n = 50
        df = pd.DataFrame({
            'chromosome': (['1'] * 25) + (['2'] * 25),
            'position': rng.integers(1_000_000, 200_000_000, size=n),
            'mean_attribution': rng.normal(0, 1, size=n),
        })
        df['chromosome'] = df['chromosome'].astype(str)
        build = get_genome_build('GRCh37')
        corrected = chrx_mod.compute_chromosome_zscores(df, build)
        for col in ('z_attribution', 'corrected_rank', 'is_sex_chrom'):
            assert col in corrected.columns, f"Column '{col}' missing from corrected output"


# ---------------------------------------------------------------------------
# compare_attributions.py --project-dir routing
# ---------------------------------------------------------------------------

class TestCompareAttributionsProjectDir:

    def test_project_dir_routes_output_correctly(self, tmp_path: Path) -> None:
        """
        Using --project-dir should place output in
        {project_dir}/real_experiments/{LEVEL}/attributions/.
        The LEVEL is inferred from the --real path.
        """
        rng = np.random.default_rng(55)
        project_dir = tmp_path / 'CohortX'
        level = 'L2'
        real_dir = project_dir / 'real_experiments' / level / 'attributions'
        real_dir.mkdir(parents=True)

        real_df = _make_raw_df(rng.normal(0.5, 1, size=200))
        null_df = _make_raw_df(rng.normal(0, 1, size=300))

        real_path = real_dir / 'sieve_variant_rankings.csv'
        null_path = tmp_path / 'null.csv'
        real_df.to_csv(real_path, index=False)
        null_df.to_csv(null_path, index=False)

        import sys as _sys
        old_argv = _sys.argv[:]
        _sys.argv = [
            'compare_attributions.py',
            '--real', str(real_path),
            '--null', str(null_path),
            '--project-dir', str(project_dir),
        ]
        try:
            compare_mod.main()
        finally:
            _sys.argv = old_argv

        expected_out = real_dir
        assert (expected_out / 'variant_rankings_with_significance.csv').exists(), \
            "variant_rankings_with_significance.csv should be in real_experiments/L2/attributions/"
        assert (expected_out / 'gene_rankings_with_significance.csv').exists()
        assert (expected_out / 'significance_summary.yaml').exists()

    def test_infer_level_from_path_real_experiments(self) -> None:
        """_infer_level_from_path extracts level from real_experiments paths."""
        path = '/data/CohortX/real_experiments/L3/attributions/sieve_variant_rankings.csv'
        assert compare_mod._infer_level_from_path(path) == 'L3'

    def test_infer_level_from_path_null_baselines(self) -> None:
        """_infer_level_from_path extracts level from null_baselines paths."""
        path = '/data/CohortX/null_baselines/L1/attributions/sieve_variant_rankings.csv'
        assert compare_mod._infer_level_from_path(path) == 'L1'

    def test_infer_level_from_path_raises_on_unknown(self) -> None:
        """_infer_level_from_path raises ValueError when level cannot be found."""
        path = '/data/CohortX/some_other_dir/sieve_variant_rankings.csv'
        with pytest.raises(ValueError, match='Cannot infer annotation level'):
            compare_mod._infer_level_from_path(path)

    def test_project_dir_and_output_dir_mutually_exclusive(
        self, tmp_path: Path,
    ) -> None:
        """Passing both --project-dir and --output-dir should fail with exit code 2."""
        script = (
            Path(__file__).parent.parent / 'scripts' / 'compare_attributions.py'
        )
        result = subprocess.run(
            [sys.executable, str(script),
             '--real', '/some/real.csv',
             '--null', '/some/null.csv',
             '--output-dir', '/some/out',
             '--project-dir', '/some/project'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, (
            f"Expected exit code 2 from argparse, got {result.returncode}. "
            f"stderr: {result.stderr}"
        )
