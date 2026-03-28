"""
Tests for the cross-cohort gene-set burden validation pipeline.

Covers:
1. Gene list generation (generate_sieve_gene_list.py)
2. Burden extraction (extract_validation_burden.py)
3. Enrichment statistics (test_burden_enrichment.py)
4. Consequence stratification
5. Gene matching diagnostics
6. Integration test with test VCF
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_sieve_gene_list import generate_gene_list
from scripts.extract_validation_burden import (
    classify_consequence,
    extract_burden_from_vcf,
    load_sieve_genes,
)
from scripts.test_burden_enrichment import (
    compute_burden_for_gene_set,
    logistic_regression_z,
    mannwhitney_test,
    run_enrichment_test,
)
from src.data.vcf_parser import load_phenotypes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path("test_data/small")
TEST_VCF = TEST_DATA_DIR / "test_data.vcf.gz"
TEST_PHENOTYPES = TEST_DATA_DIR / "test_data_phenotypes.tsv"


@pytest.fixture
def mock_variant_rankings() -> pd.DataFrame:
    """Create mock variant-level rankings."""
    return pd.DataFrame({
        "gene_name": ["GENE_A", "GENE_A", "GENE_B", "GENE_C", "GENE_C", "GENE_C",
                       "GENE_D", "GENE_E", "GENE_F", "GENE_X"],
        "chromosome": ["1", "1", "2", "3", "3", "3", "5", "7", "11", "X"],
        "position": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "z_attribution": [3.5, 2.0, 4.0, 1.5, 3.0, 2.5, 1.0, 0.5, 0.2, 5.0],
        "mean_attribution": [0.8, 0.5, 0.9, 0.3, 0.7, 0.6, 0.2, 0.1, 0.05, 1.0],
        "exceeds_null_p01": [True, False, True, False, True, False, False, False, False, True],
        "exceeds_null_p05": [True, True, True, False, True, True, False, False, False, True],
        "is_sex_chrom": [False, False, False, False, False, False, False, False, False, True],
    })


@pytest.fixture
def mock_gene_list(tmp_path: Path) -> Path:
    """Create a mock gene list TSV file."""
    df = pd.DataFrame({
        "gene_name": ["NMNAT1", "RBP7", "ARID1A", "EPHA2", "KIF1B"],
        "gene_rank": [1, 2, 3, 4, 5],
        "gene_score": [3.5, 3.0, 2.5, 2.0, 1.5],
        "n_variants": [5, 3, 4, 2, 6],
        "chromosome": ["1", "1", "1", "1", "1"],
    })
    path = tmp_path / "sieve_genes.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def mock_gene_matrix() -> pd.DataFrame:
    """Create a mock gene burden matrix."""
    rng = np.random.default_rng(42)
    n_samples = 50
    n_genes = 20
    genes = [f"GENE_{i}" for i in range(n_genes)]
    samples = [f"sample_{i}" for i in range(n_samples)]
    data = rng.poisson(lam=2, size=(n_samples, n_genes))

    # Inject signal: first 5 genes have higher burden in cases (first 25 samples)
    data[:25, :5] += rng.poisson(lam=3, size=(25, 5))

    return pd.DataFrame(data, index=samples, columns=genes)


# ---------------------------------------------------------------------------
# Test gene list generation
# ---------------------------------------------------------------------------

class TestGenerateGeneList:

    def test_basic_aggregation_max(self, mock_variant_rankings):
        result = generate_gene_list(
            mock_variant_rankings,
            score_column="z_attribution",
            exclude_sex_chroms=False,
            aggregation="max",
        )
        assert "gene_name" in result.columns
        assert "gene_rank" in result.columns
        assert "gene_score" in result.columns
        assert "n_variants" in result.columns
        # GENE_X has highest z (5.0), should be rank 1
        top = result.iloc[0]
        assert top["gene_name"] == "GENE_X"
        assert top["gene_score"] == 5.0

    def test_aggregation_mean(self, mock_variant_rankings):
        result = generate_gene_list(
            mock_variant_rankings,
            score_column="z_attribution",
            exclude_sex_chroms=False,
            aggregation="mean",
        )
        # GENE_A: mean of (3.5, 2.0) = 2.75
        gene_a = result[result["gene_name"] == "GENE_A"].iloc[0]
        assert abs(gene_a["gene_score"] - 2.75) < 1e-6

    def test_exclude_sex_chroms(self, mock_variant_rankings):
        result = generate_gene_list(
            mock_variant_rankings,
            exclude_sex_chroms=True,
        )
        # GENE_X is on chrX, should be excluded
        assert "GENE_X" not in result["gene_name"].values

    def test_include_sex_chroms(self, mock_variant_rankings):
        result = generate_gene_list(
            mock_variant_rankings,
            exclude_sex_chroms=False,
        )
        assert "GENE_X" in result["gene_name"].values

    def test_null_threshold_filter(self, mock_variant_rankings):
        result = generate_gene_list(
            mock_variant_rankings,
            exclude_sex_chroms=False,
            min_null_threshold="p01",
        )
        # Only variants with exceeds_null_p01=True should contribute
        # GENE_A has 1 passing variant (z=3.5), GENE_B has 1 (z=4.0),
        # GENE_C has 1 (z=3.0), GENE_X has 1 (z=5.0)
        assert len(result) == 4
        assert "GENE_D" not in result["gene_name"].values

    def test_ranking_order(self, mock_variant_rankings):
        result = generate_gene_list(
            mock_variant_rankings,
            exclude_sex_chroms=False,
        )
        # Ranks should be 1, 2, 3, ...
        assert list(result["gene_rank"]) == list(range(1, len(result) + 1))
        # Scores should be descending
        scores = list(result["gene_score"])
        assert scores == sorted(scores, reverse=True)

    def test_variant_count(self, mock_variant_rankings):
        result = generate_gene_list(
            mock_variant_rankings,
            exclude_sex_chroms=False,
        )
        gene_c = result[result["gene_name"] == "GENE_C"].iloc[0]
        assert gene_c["n_variants"] == 3

    def test_missing_score_column_raises(self, mock_variant_rankings):
        with pytest.raises(ValueError, match="Score column"):
            generate_gene_list(
                mock_variant_rankings,
                score_column="nonexistent_column",
            )


# ---------------------------------------------------------------------------
# Test consequence classification
# ---------------------------------------------------------------------------

class TestConsequenceClassification:

    def test_lof(self):
        assert classify_consequence("stop_gained") == "lof"
        assert classify_consequence("frameshift_variant") == "lof"
        assert classify_consequence("splice_acceptor_variant") == "lof"

    def test_missense(self):
        assert classify_consequence("missense_variant") == "missense"
        assert classify_consequence("missense_variant&splice_region_variant") == "missense"

    def test_synonymous(self):
        assert classify_consequence("synonymous_variant") == "synonymous"

    def test_other(self):
        assert classify_consequence("intron_variant") == "other"
        assert classify_consequence("5_prime_UTR_variant") == "other"
        assert classify_consequence("unknown") == "other"


# ---------------------------------------------------------------------------
# Test gene matching and loading
# ---------------------------------------------------------------------------

class TestGeneLoading:

    def test_load_gene_list_tsv(self, mock_gene_list):
        df = load_sieve_genes(mock_gene_list)
        assert len(df) == 5
        assert "gene_name" in df.columns
        assert "gene_rank" in df.columns
        # Should be sorted by rank
        assert list(df["gene_rank"]) == [1, 2, 3, 4, 5]

    def test_case_insensitive_matching(self):
        """Gene matching should be case-insensitive (via .upper())."""
        # This tests the convention used in extract_validation_burden
        target = {"GENE_A", "GENE_B"}
        test_gene = "gene_a"
        assert test_gene.upper() in target


# ---------------------------------------------------------------------------
# Test enrichment statistics
# ---------------------------------------------------------------------------

class TestEnrichmentStatistics:

    def test_logistic_regression_z(self):
        rng = np.random.default_rng(42)
        n = 100
        labels = np.array([0] * 50 + [1] * 50)
        # Create burden with known signal
        burden = rng.normal(0, 1, n)
        burden[50:] += 2  # cases have higher burden

        z, p = logistic_regression_z(burden, labels)
        # Should detect the signal
        assert abs(z) > 1.5
        assert p < 0.05

    def test_logistic_regression_no_signal(self):
        rng = np.random.default_rng(42)
        n = 100
        labels = np.array([0] * 50 + [1] * 50)
        burden = rng.normal(0, 1, n)  # No signal

        z, p = logistic_regression_z(burden, labels)
        assert p > 0.05

    def test_mannwhitney(self):
        cases_burden = np.array([5, 6, 7, 8, 9, 10])
        controls_burden = np.array([1, 2, 3, 4, 5, 6])
        labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        burden = np.concatenate([controls_burden, cases_burden])

        u, p = mannwhitney_test(burden, labels)
        assert p < 0.05

    def test_mannwhitney_no_signal(self):
        rng = np.random.default_rng(42)
        labels = np.array([0] * 50 + [1] * 50)
        burden = rng.normal(5, 1, 100)

        u, p = mannwhitney_test(burden, labels)
        assert p > 0.05

    def test_compute_burden_for_gene_set(self, mock_gene_matrix):
        gene_set = {"GENE_0", "GENE_1", "GENE_2"}
        burden = compute_burden_for_gene_set(mock_gene_matrix, gene_set)
        assert len(burden) == len(mock_gene_matrix)
        # Should equal sum of those 3 columns
        expected = mock_gene_matrix[["GENE_0", "GENE_1", "GENE_2"]].sum(axis=1).values
        np.testing.assert_array_equal(burden, expected)

    def test_compute_burden_empty_set(self, mock_gene_matrix):
        burden = compute_burden_for_gene_set(mock_gene_matrix, set())
        assert burden.sum() == 0

    def test_compute_burden_missing_genes(self, mock_gene_matrix):
        gene_set = {"GENE_0", "NONEXISTENT_GENE"}
        burden = compute_burden_for_gene_set(mock_gene_matrix, gene_set)
        expected = mock_gene_matrix[["GENE_0"]].sum(axis=1).values
        np.testing.assert_array_equal(burden, expected)

    def test_empirical_p_value(self):
        """Test that enrichment test detects injected signal."""
        rng = np.random.default_rng(42)
        n_samples = 200
        n_genes = 30
        genes = [f"GENE_{i}" for i in range(n_genes)]
        samples = [f"sample_{i}" for i in range(n_samples)]

        # Baseline burden for all genes
        data = rng.poisson(lam=3, size=(n_samples, n_genes))
        # Moderate signal in first 5 genes for cases (first 100 samples)
        data[:100, :5] += rng.poisson(lam=2, size=(100, 5))

        gene_matrix = pd.DataFrame(data, index=samples, columns=genes)
        labels = np.array([1] * 100 + [0] * 100)

        sieve_genes = {f"GENE_{i}" for i in range(5)}
        bg_genes = list(gene_matrix.columns)

        result = run_enrichment_test(
            gene_matrix=gene_matrix,
            labels=labels,
            sieve_genes=sieve_genes,
            background_genes=bg_genes,
            n_permutations=200,
            seed=42,
        )

        # Signal was injected in first 5 genes for cases
        assert result["observed"]["mean_cases"] > result["observed"]["mean_controls"]
        # Empirical p should be low (moderate signal)
        assert result["permutation"]["empirical_p"] < 0.1


# ---------------------------------------------------------------------------
# Test burden extraction with real test VCF
# ---------------------------------------------------------------------------

class TestBurdenExtraction:

    @pytest.fixture
    def phenotypes(self):
        return load_phenotypes(TEST_PHENOTYPES)

    def test_extract_burden_basic(self, phenotypes, mock_gene_list):
        """Test burden extraction with the test VCF and known genes."""
        gene_df = load_sieve_genes(mock_gene_list)
        target_genes = set(gene_df["gene_name"].str.upper())
        target_gene_sets = {5: target_genes}

        burden_dfs, summaries, _, _ = extract_burden_from_vcf(
            vcf_path=TEST_VCF,
            phenotypes=phenotypes,
            target_gene_sets=target_gene_sets,
            genome_build_name="GRCh37",
            min_gq=20,
            consequence_stratify=False,
            include_sex_chroms=False,
            compute_full_matrix=False,
        )

        assert 5 in burden_dfs
        df = burden_dfs[5]
        assert "sample_id" in df.columns
        assert "phenotype" in df.columns
        assert "total_burden" in df.columns
        assert len(df) == len(phenotypes)
        # At least some burden should be non-zero for known genes
        assert df["total_burden"].sum() > 0

    def test_extract_with_consequence_stratification(self, phenotypes, mock_gene_list):
        """Test that consequence stratification produces extra columns."""
        gene_df = load_sieve_genes(mock_gene_list)
        target_genes = set(gene_df["gene_name"].str.upper())
        target_gene_sets = {5: target_genes}

        burden_dfs, _, _, _ = extract_burden_from_vcf(
            vcf_path=TEST_VCF,
            phenotypes=phenotypes,
            target_gene_sets=target_gene_sets,
            consequence_stratify=True,
            compute_full_matrix=False,
        )

        df = burden_dfs[5]
        assert "missense_burden" in df.columns
        assert "lof_burden" in df.columns
        assert "synonymous_burden" in df.columns
        assert "other_burden" in df.columns

    def test_full_gene_matrix(self, phenotypes, mock_gene_list):
        """Test full gene matrix construction."""
        gene_df = load_sieve_genes(mock_gene_list)
        target_genes = set(gene_df["gene_name"].str.upper())
        target_gene_sets = {5: target_genes}

        _, _, full_matrix, metadata = extract_burden_from_vcf(
            vcf_path=TEST_VCF,
            phenotypes=phenotypes,
            target_gene_sets=target_gene_sets,
            compute_full_matrix=True,
        )

        assert full_matrix is not None
        assert metadata is not None
        assert full_matrix.shape[0] == len(phenotypes)
        assert full_matrix.shape[1] > 0
        assert "n_genes" in metadata
        assert "sample_ids" in metadata

    def test_gene_matching_diagnostics(self, phenotypes):
        """Test that missing genes are reported in summary."""
        # Use genes we know don't exist in the test VCF
        fake_genes = {"FAKEGENE1", "FAKEGENE2", "NMNAT1"}
        target_gene_sets = {3: fake_genes}

        _, summaries, _, _ = extract_burden_from_vcf(
            vcf_path=TEST_VCF,
            phenotypes=phenotypes,
            target_gene_sets=target_gene_sets,
            compute_full_matrix=False,
        )

        summary = summaries[3]
        # NMNAT1 should be found, the fakes should not
        assert summary["n_sieve_genes_missing"] >= 2
        assert "FAKEGENE1" in summary["missing_genes"]
        assert "FAKEGENE2" in summary["missing_genes"]

    def test_sex_chrom_exclusion(self, phenotypes, mock_gene_list):
        """Test that sex chrom variants are excluded by default."""
        gene_df = load_sieve_genes(mock_gene_list)
        target_genes = set(gene_df["gene_name"].str.upper())
        target_gene_sets = {5: target_genes}

        # Run with and without sex chroms
        _, _, matrix_excl, _ = extract_burden_from_vcf(
            vcf_path=TEST_VCF,
            phenotypes=phenotypes,
            target_gene_sets=target_gene_sets,
            include_sex_chroms=False,
            compute_full_matrix=True,
        )

        _, _, matrix_incl, _ = extract_burden_from_vcf(
            vcf_path=TEST_VCF,
            phenotypes=phenotypes,
            target_gene_sets=target_gene_sets,
            include_sex_chroms=True,
            compute_full_matrix=True,
        )

        # Including sex chroms should give >= genes
        assert matrix_incl.shape[1] >= matrix_excl.shape[1]


# ---------------------------------------------------------------------------
# Integration test: end-to-end mini pipeline
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_end_to_end(self, tmp_path):
        """Run the full pipeline on the test VCF with a small number of permutations."""
        phenotypes = load_phenotypes(TEST_PHENOTYPES)

        # Step 1: Create a mock gene list from genes we know are in the test VCF
        known_genes = ["NMNAT1", "RBP7", "ARID1A", "EPHA2", "KIF1B",
                        "FUCA1", "MTOR", "SDHB", "EPB41", "CASP9"]
        gene_list_df = pd.DataFrame({
            "gene_name": known_genes,
            "gene_rank": list(range(1, len(known_genes) + 1)),
            "gene_score": np.linspace(3.0, 1.0, len(known_genes)),
            "n_variants": [3] * len(known_genes),
            "chromosome": ["1"] * len(known_genes),
        })
        gene_list_path = tmp_path / "sieve_genes.tsv"
        gene_list_df.to_csv(gene_list_path, sep="\t", index=False)

        # Step 2: Extract burden with full matrix
        target_gene_sets = {
            5: set(g.upper() for g in known_genes[:5]),
            10: set(g.upper() for g in known_genes),
        }

        burden_dfs, summaries, full_matrix, metadata = extract_burden_from_vcf(
            vcf_path=TEST_VCF,
            phenotypes=phenotypes,
            target_gene_sets=target_gene_sets,
            consequence_stratify=True,
            compute_full_matrix=True,
        )

        # Verify burden outputs
        assert 5 in burden_dfs and 10 in burden_dfs
        assert full_matrix is not None
        assert full_matrix.shape[0] == len(phenotypes)

        # Save burden files (needed by enrichment script)
        burden_dir = tmp_path / "burden"
        burden_dir.mkdir()
        for k, df in burden_dfs.items():
            df.to_csv(burden_dir / f"burden_topK{k}.tsv", sep="\t", index=False)
        full_matrix.to_parquet(burden_dir / "gene_burden_matrix.parquet")

        # Step 3: Run enrichment with small permutations
        # Align samples
        common_samples = [s for s in full_matrix.index if s in phenotypes]
        matrix_aligned = full_matrix.loc[common_samples]
        matrix_aligned.columns = [c.upper() for c in matrix_aligned.columns]
        labels = np.array([phenotypes[s] for s in common_samples])
        bg_genes = list(matrix_aligned.columns)

        sieve_genes_top5 = set(g.upper() for g in known_genes[:5])

        result = run_enrichment_test(
            gene_matrix=matrix_aligned,
            labels=labels,
            sieve_genes=sieve_genes_top5,
            background_genes=bg_genes,
            n_permutations=100,
            seed=42,
        )

        # Verify enrichment output structure
        assert "observed" in result
        assert "permutation" in result
        assert "logistic_z" in result["observed"]
        assert "empirical_p" in result["permutation"]
        assert 0 <= result["permutation"]["empirical_p"] <= 1
        assert "null_z_values" in result
        assert len(result["null_z_values"]) == 100
