"""Tests for non-linear classifier validation of SIEVE gene sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.validate_nonlinear_classifier as nonlinear


def make_xor_signal_dataset(
    n_samples: int = 80,
    n_genes: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Create a dataset with a strong non-linear XOR signal in the first 10 genes."""
    if n_samples % 4 != 0:
        raise ValueError("n_samples must be divisible by 4 for the XOR construction")

    rng = np.random.default_rng(seed)
    samples = [f"sample_{idx}" for idx in range(n_samples)]
    genes = [f"GENE_{idx}" for idx in range(n_genes)]
    data = rng.poisson(lam=0.1, size=(n_samples, n_genes))

    quarter = n_samples // 4
    y = np.array([1] * (2 * quarter) + [0] * (2 * quarter), dtype=np.int64)
    xor_patterns = np.array(
        [[1, 0]] * quarter
        + [[0, 1]] * quarter
        + [[0, 0]] * quarter
        + [[1, 1]] * quarter,
        dtype=np.int64,
    )

    data[:, 0] = xor_patterns[:, 0]
    data[:, 1] = xor_patterns[:, 1]
    for gene_idx in range(2, 10):
        data[:, gene_idx] = xor_patterns[:, gene_idx % 2]

    burden_matrix = pd.DataFrame(data, index=samples, columns=genes)
    sieve_df = pd.DataFrame(
        {
            "gene_name": genes[:10],
            "gene_rank": np.arange(1, 11),
            "gene_score": np.linspace(10.0, 1.0, 10),
            "n_variants": [2] * 10,
            "chromosome": ["1"] * 10,
        }
    )
    return burden_matrix, y, sieve_df


def make_null_dataset(
    n_samples: int = 120,
    n_genes: int = 100,
    seed: int = 123,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Create a burden matrix with no phenotype signal."""
    rng = np.random.default_rng(seed)
    samples = [f"sample_{idx}" for idx in range(n_samples)]
    genes = [f"GENE_{idx}" for idx in range(n_genes)]
    data = rng.poisson(lam=0.15, size=(n_samples, n_genes))
    y = np.array([0, 1] * (n_samples // 2), dtype=np.int64)

    burden_matrix = pd.DataFrame(data, index=samples, columns=genes)
    sieve_df = pd.DataFrame(
        {
            "gene_name": genes[:10],
            "gene_rank": np.arange(1, 11),
            "gene_score": np.linspace(10.0, 1.0, 10),
            "n_variants": [1] * 10,
            "chromosome": ["1"] * 10,
        }
    )
    return burden_matrix, y, sieve_df


def write_phenotypes(path: Path, sample_ids: list[str], labels: np.ndarray) -> None:
    """Write a PLINK-style phenotype file."""
    lines = [
        f"{sample_id}\t{2 if label == 1 else 1}"
        for sample_id, label in zip(sample_ids, labels)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestNonlinearValidation:

    def test_random_forest_detects_xor_signal_above_null(self, tmp_path: Path) -> None:
        burden_matrix, y, sieve_df = make_xor_signal_dataset(n_genes=500)
        fold_indices = nonlinear.generate_fixed_folds(y, cv_folds=4, seed=42, n_repeats=2)

        result = nonlinear.run_validation(
            burden_matrix=burden_matrix,
            burden_values=burden_matrix.to_numpy(dtype=np.float64, copy=True),
            sieve_genes_df=sieve_df,
            y=y,
            sample_ids=np.asarray(burden_matrix.index, dtype=object),
            level="L1",
            top_k_requested=10,
            top_k_used=10,
            classifier_name="rf",
            fold_indices=fold_indices,
            n_permutations=20,
            seed=42,
            n_jobs=1,
            output_dir=tmp_path,
            consequence="total",
        )

        assert result
        assert result["observed_auc"] > 0.80
        assert result["observed_auc"] > np.mean(result["null_aucs"]) + 0.15
        assert result["empirical_p"] < 0.10

    def test_random_gene_set_on_null_data_is_not_significant(self, tmp_path: Path) -> None:
        burden_matrix, y, sieve_df = make_null_dataset()
        fold_indices = nonlinear.generate_fixed_folds(y, cv_folds=4, seed=7, n_repeats=2)

        result = nonlinear.run_validation(
            burden_matrix=burden_matrix,
            burden_values=burden_matrix.to_numpy(dtype=np.float64, copy=True),
            sieve_genes_df=sieve_df,
            y=y,
            sample_ids=np.asarray(burden_matrix.index, dtype=object),
            level="L0",
            top_k_requested=10,
            top_k_used=10,
            classifier_name="rf",
            fold_indices=fold_indices,
            n_permutations=20,
            seed=7,
            n_jobs=1,
            output_dir=tmp_path,
            consequence="total",
        )

        assert result
        assert abs(result["observed_auc"] - np.mean(result["null_aucs"])) < 0.10
        assert 0.05 <= result["empirical_p"] <= 0.95

    def test_same_fold_indices_are_reused_for_observed_and_null(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        burden_matrix, y, sieve_df = make_xor_signal_dataset(n_samples=40, n_genes=40, seed=9)
        fold_indices = nonlinear.generate_fixed_folds(y, cv_folds=4, seed=9, n_repeats=2)
        seen_fold_ids: list[int] = []

        def fake_evaluate_gene_set(
            X: np.ndarray,
            y_array: np.ndarray,
            received_fold_indices: nonlinear.FoldIndices,
            classifier_name: str,
            model_seed: int,
            warn_context: str = "",
        ) -> np.ndarray:
            del X, y_array, classifier_name, model_seed, warn_context
            seen_fold_ids.append(id(received_fold_indices))
            return np.full(len(received_fold_indices), 0.5, dtype=np.float64)

        monkeypatch.setattr(nonlinear, "evaluate_gene_set", fake_evaluate_gene_set)

        result = nonlinear.run_validation(
            burden_matrix=burden_matrix,
            burden_values=burden_matrix.to_numpy(dtype=np.float64, copy=True),
            sieve_genes_df=sieve_df,
            y=y,
            sample_ids=np.asarray(burden_matrix.index, dtype=object),
            level="L2",
            top_k_requested=10,
            top_k_used=10,
            classifier_name="rf",
            fold_indices=fold_indices,
            n_permutations=5,
            seed=9,
            n_jobs=1,
            output_dir=tmp_path,
            consequence="total",
        )

        assert result
        assert len(seen_fold_ids) == 6
        assert set(seen_fold_ids) == {id(fold_indices)}

    def test_cli_outputs_yaml_tsv_npz_png_and_csv(self, tmp_path: Path) -> None:
        burden_matrix, y, sieve_df = make_xor_signal_dataset(n_samples=40, n_genes=30, seed=21)
        burden_path = tmp_path / "gene_burden_matrix.parquet"
        phenotypes_path = tmp_path / "phenotypes.tsv"
        sieve_dir = tmp_path / "sieve_levels"
        output_dir = tmp_path / "nonlinear_validation"

        burden_with_sample_id = burden_matrix.reset_index().rename(columns={"index": "sample_id"})
        burden_with_sample_id.to_parquet(burden_path)
        write_phenotypes(phenotypes_path, burden_matrix.index.tolist(), y)

        sieve_dir.mkdir()
        sieve_df.to_csv(sieve_dir / "sieve_genes_L0.tsv", sep="\t", index=False)
        sieve_df.iloc[::-1].assign(gene_rank=np.arange(1, len(sieve_df) + 1)).to_csv(
            sieve_dir / "sieve_genes_L1.tsv",
            sep="\t",
            index=False,
        )

        nonlinear.main(
            [
                "--burden-matrix",
                str(burden_path),
                "--sieve-genes",
                str(sieve_dir),
                "--phenotypes",
                str(phenotypes_path),
                "--output-dir",
                str(output_dir),
                "--top-k",
                "5",
                "--n-permutations",
                "4",
                "--cv-folds",
                "3",
                "--seed",
                "21",
                "--n-jobs",
                "1",
                "--classifiers",
                "both",
                "--also-export-csv",
            ]
        )

        primary_yaml = output_dir / "nonlinear_validation_L0_topK5.yaml"
        secondary_yaml = output_dir / "nonlinear_validation_L0_topK5_lr.yaml"
        null_npz = output_dir / "null_aucs_L0_topK5.npz"
        plot_png = output_dir / "validation_plot_L0_topK5.png"
        summary_tsv = output_dir / "nonlinear_validation_summary.tsv"
        heatmap_png = output_dir / "nonlinear_validation_heatmap.png"
        report_md = output_dir / "nonlinear_validation_report.md"
        csv_path = output_dir / "csv" / "feature_matrix_total_L0_top5.csv"

        assert primary_yaml.exists()
        assert secondary_yaml.exists()
        assert null_npz.exists()
        assert plot_png.exists()
        assert summary_tsv.exists()
        assert heatmap_png.exists()
        assert report_md.exists()
        assert csv_path.exists()

        yaml_payload = yaml.safe_load(primary_yaml.read_text(encoding="utf-8"))
        assert "parameters" in yaml_payload
        assert "observed" in yaml_payload
        assert "null_distribution" in yaml_payload
        assert "linear_baseline" in yaml_payload

        summary_df = pd.read_csv(summary_tsv, sep="\t")
        expected_columns = {
            "level",
            "top_k",
            "k_effective",
            "classifier",
            "observed_auc",
            "observed_std",
            "null_mean_auc",
            "null_std_auc",
            "empirical_p",
            "percentile_rank",
            "bonferroni_significant",
        }
        assert expected_columns.issubset(summary_df.columns)
        assert set(summary_df["classifier"]) == {"rf", "lr"}

        csv_df = pd.read_csv(csv_path)
        assert csv_df.columns[0] == "sample_id"
        assert csv_df.columns[-1] == "phenotype"
        assert csv_df.shape[0] == burden_matrix.shape[0]
        assert csv_df.shape[1] == 7
