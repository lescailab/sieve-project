"""Tests for non-linear classifier validation of SIEVE gene sets."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.validate_nonlinear_classifier as nonlinear


def make_xor_signal_dataset(
    n_samples: int = 80,
    n_genes: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create a burden matrix with a strong XOR-like signal in the first 10 genes."""
    if n_samples % 4 != 0:
        raise ValueError("n_samples must be divisible by 4 for the XOR construction")

    rng = np.random.default_rng(seed)
    samples = [f"sample_{idx}" for idx in range(n_samples)]
    genes = [f"GENE_{idx}" for idx in range(n_genes)]
    data = rng.poisson(lam=0.1, size=(n_samples, n_genes))

    quarter = n_samples // 4
    labels = np.array([1] * (2 * quarter) + [0] * (2 * quarter), dtype=np.int64)
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
    return burden_matrix, labels


def make_gene_rankings(
    genes: list[str],
    signal_genes: list[str],
    reverse_tail: bool = False,
) -> pd.DataFrame:
    """Create a corrected gene-ranking file with optional significance columns."""
    tail_genes = [gene for gene in genes if gene not in signal_genes]
    if reverse_tail:
        tail_genes = tail_genes[::-1]

    ordered = signal_genes + tail_genes
    gene_z = np.linspace(4.0, -1.0, len(ordered))
    fdr = np.linspace(0.001, 0.50, len(ordered))

    return pd.DataFrame(
        {
            "gene_name": ordered,
            "gene_rank": np.arange(1, len(ordered) + 1),
            "gene_z_score": gene_z,
            "num_variants": np.ones(len(ordered), dtype=int),
            "empirical_p_gene": np.clip(fdr / 2, 0, 1),
            "fdr_gene": fdr,
        }
    )


def write_labels(path: Path, sample_ids: list[str], labels: np.ndarray) -> None:
    """Write a phenotype file in standard SIEVE format."""
    lines = [
        f"{sample_id}\t{2 if label == 1 else 1}"
        for sample_id, label in zip(sample_ids, labels)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_inputs(
    tmp_path: Path,
    levels: tuple[str, ...] = ("L0", "L1"),
) -> tuple[Path, Path, Path]:
    """Create a burden matrix, label file, and per-level corrected rankings."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    burden_matrix, labels = make_xor_signal_dataset(n_samples=40, n_genes=40, seed=21)
    burden_path = tmp_path / "gene_burden_matrix.parquet"
    label_path = tmp_path / "labels.tsv"
    rankings_root = tmp_path / "rankings"

    burden_with_sample_id = burden_matrix.reset_index().rename(columns={"index": "sample_id"})
    burden_with_sample_id.to_parquet(burden_path)
    write_labels(label_path, burden_matrix.index.tolist(), labels)

    rankings_root.mkdir()
    signal_genes = burden_matrix.columns[:10].tolist()
    all_genes = burden_matrix.columns.tolist()

    for level_index, level in enumerate(levels):
        level_dir = rankings_root / level
        level_dir.mkdir()
        rankings_df = make_gene_rankings(
            genes=all_genes,
            signal_genes=signal_genes,
            reverse_tail=bool(level_index % 2),
        )
        rankings_df.to_csv(
            level_dir / "gene_rankings_with_significance.csv",
            index=False,
        )

    return burden_path, label_path, rankings_root


class TestNonlinearValidation:

    def test_output_tsv_has_exact_columns_and_shared_null_stats(self, tmp_path: Path) -> None:
        """Summary TSV must match the new schema and share null stats per k_effective group."""
        burden_path, label_path, rankings_root = prepare_inputs(tmp_path)
        output_tsv = tmp_path / "nonlinear_validation_summary.tsv"

        nonlinear.main(
            [
                "--real-rankings-dir",
                str(rankings_root),
                "--burden-matrix",
                str(burden_path),
                "--labels",
                str(label_path),
                "--output-tsv",
                str(output_tsv),
                "--top-k",
                "5,10",
                "--classifiers",
                "rf,lr",
                "--levels",
                "L0,L1",
                "--n-permutations",
                "6",
                "--cv-folds",
                "3",
                "--n-cores",
                "1",
                "--seed",
                "21",
            ]
        )

        summary_df = pd.read_csv(output_tsv, sep="\t")
        assert list(summary_df.columns) == nonlinear.SUMMARY_COLUMNS
        assert len(summary_df) == 2 * 2 * 2

        for (_, _, _), group in summary_df.groupby(["top_k", "classifier", "k_effective"]):
            assert group["null_mean_auc"].nunique() == 1
            assert group["null_std_auc"].nunique() == 1

        sorted_fdr = summary_df.sort_values("empirical_p")["fdr_bh"].to_numpy()
        assert np.all(np.diff(sorted_fdr) >= -1e-12)

    def test_same_seed_produces_byte_identical_output(self, tmp_path: Path) -> None:
        """Running twice with the same seed should produce byte-identical TSV output."""
        burden_path, label_path, rankings_root = prepare_inputs(tmp_path / "inputs")
        output_a = tmp_path / "run_a.tsv"
        output_b = tmp_path / "run_b.tsv"

        common_args = [
            "--real-rankings-dir",
            str(rankings_root),
            "--burden-matrix",
            str(burden_path),
            "--labels",
            str(label_path),
            "--top-k",
            "5",
            "--classifiers",
            "rf",
            "--levels",
            "L0,L1",
            "--n-permutations",
            "5",
            "--cv-folds",
            "3",
            "--n-cores",
            "1",
            "--seed",
            "123",
        ]

        nonlinear.main(common_args + ["--output-tsv", str(output_a)])
        nonlinear.main(common_args + ["--output-tsv", str(output_b)])

        assert output_a.read_bytes() == output_b.read_bytes()

    def test_cli_requires_top_k_or_fdr_threshold(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Argparse must reject invocations that omit both --top-k and --fdr-threshold."""
        with pytest.raises(SystemExit):
            nonlinear.parse_args(
                [
                    "--real-rankings-dir",
                    "rankings",
                    "--burden-matrix",
                    "burden.parquet",
                    "--labels",
                    "labels.tsv",
                    "--output-tsv",
                    "out.tsv",
                    "--classifiers",
                    "rf",
                ]
            )

        stderr = capsys.readouterr().err
        assert "--top-k" in stderr or "--fdr-threshold" in stderr

    def test_cli_rejects_top_k_and_fdr_threshold_together(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Argparse must reject invocations with both --top-k and --fdr-threshold."""
        with pytest.raises(SystemExit):
            nonlinear.parse_args(
                [
                    "--real-rankings-dir",
                    "rankings",
                    "--burden-matrix",
                    "burden.parquet",
                    "--labels",
                    "labels.tsv",
                    "--output-tsv",
                    "out.tsv",
                    "--classifiers",
                    "rf",
                    "--top-k",
                    "100",
                    "--fdr-threshold",
                    "0.05",
                ]
            )

    def test_fdr_threshold_out_of_range_exits(self, tmp_path: Path) -> None:
        """main() must reject --fdr-threshold values outside (0, 1]."""
        burden_path, label_path, rankings_root = prepare_inputs(tmp_path)

        for bad_value in ["-0.1", "0", "1.1", "2.0"]:
            with pytest.raises(SystemExit):
                nonlinear.main(
                    [
                        "--real-rankings-dir",
                        str(rankings_root),
                        "--burden-matrix",
                        str(burden_path),
                        "--labels",
                        str(label_path),
                        "--output-tsv",
                        str(tmp_path / "out.tsv"),
                        "--fdr-threshold",
                        bad_value,
                        "--classifiers",
                        "rf",
                        "--levels",
                        "L0",
                        "--n-permutations",
                        "4",
                        "--cv-folds",
                        "3",
                        "--n-cores",
                        "1",
                    ]
                )

    def test_custom_grid_row_count(self, tmp_path: Path) -> None:
        """Custom top-k and classifier grids must produce the expected row count."""
        burden_path, label_path, rankings_root = prepare_inputs(
            tmp_path,
            levels=("L0", "L1", "L2"),
        )
        output_tsv = tmp_path / "custom_grid.tsv"

        nonlinear.main(
            [
                "--real-rankings-dir",
                str(rankings_root),
                "--burden-matrix",
                str(burden_path),
                "--labels",
                str(label_path),
                "--output-tsv",
                str(output_tsv),
                "--top-k",
                "5,8",
                "--classifiers",
                "rf",
                "--levels",
                "L0,L1,L2",
                "--n-permutations",
                "4",
                "--cv-folds",
                "3",
                "--n-cores",
                "1",
                "--seed",
                "99",
            ]
        )

        summary_df = pd.read_csv(output_tsv, sep="\t")
        assert len(summary_df) == 2 * 1 * 3
        assert set(summary_df["top_k"]) == {5, 8}
        assert set(summary_df["classifier"]) == {"rf"}
        assert set(summary_df["level"]) == {"L0", "L1", "L2"}

    def test_levels_with_different_k_effective_draw_separate_shared_nulls(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Levels with missing ranked genes must draw nulls at their own k_effective."""
        burden_path, label_path, rankings_root = prepare_inputs(tmp_path, levels=("L0", "L1"))
        output_tsv = tmp_path / "k_effective.tsv"

        l1_path = rankings_root / "L1" / "gene_rankings_with_significance.csv"
        l1_rankings = pd.read_csv(l1_path)
        l1_rankings.loc[0, "gene_name"] = "MISSING_GENE_A"
        l1_rankings.loc[1, "gene_name"] = "MISSING_GENE_B"
        l1_rankings.to_csv(l1_path, index=False)

        seen_set_sizes: list[int] = []
        original_draw_random_gene_sets = nonlinear.draw_random_gene_sets

        def recording_draw_random_gene_sets(
            gene_universe_size: int,
            set_size: int,
            n_permutations: int,
            seed: int,
        ) -> list[np.ndarray]:
            seen_set_sizes.append(set_size)
            return original_draw_random_gene_sets(
                gene_universe_size=gene_universe_size,
                set_size=set_size,
                n_permutations=n_permutations,
                seed=seed,
            )

        monkeypatch.setattr(
            nonlinear,
            "draw_random_gene_sets",
            recording_draw_random_gene_sets,
        )

        nonlinear.main(
            [
                "--real-rankings-dir",
                str(rankings_root),
                "--burden-matrix",
                str(burden_path),
                "--labels",
                str(label_path),
                "--output-tsv",
                str(output_tsv),
                "--top-k",
                "5",
                "--classifiers",
                "rf",
                "--levels",
                "L0,L1",
                "--n-permutations",
                "4",
                "--cv-folds",
                "3",
                "--n-cores",
                "1",
                "--seed",
                "17",
            ]
        )

        summary_df = pd.read_csv(output_tsv, sep="\t")
        assert set(summary_df["k_effective"]) == {3, 5}
        assert seen_set_sizes.count(3) == 1
        assert seen_set_sizes.count(5) == 1

    def test_fdr_threshold_end_to_end(self, tmp_path: Path) -> None:
        """FDR-threshold mode should produce valid output with fdr_threshold populated."""
        burden_path, label_path, rankings_root = prepare_inputs(tmp_path)
        output_tsv = tmp_path / "fdr_validation_summary.tsv"

        nonlinear.main(
            [
                "--real-rankings-dir",
                str(rankings_root),
                "--burden-matrix",
                str(burden_path),
                "--labels",
                str(label_path),
                "--output-tsv",
                str(output_tsv),
                "--fdr-threshold",
                "0.25",
                "--classifiers",
                "rf",
                "--levels",
                "L0,L1",
                "--n-permutations",
                "4",
                "--cv-folds",
                "3",
                "--n-cores",
                "1",
                "--seed",
                "42",
            ]
        )

        summary_df = pd.read_csv(output_tsv, sep="\t")
        assert list(summary_df.columns) == nonlinear.SUMMARY_COLUMNS
        assert len(summary_df) > 0
        assert (summary_df["fdr_threshold"] == 0.25).all()
        # All levels should appear (both have genes with fdr < 0.25)
        assert set(summary_df["level"]) == {"L0", "L1"}

    def test_fdr_threshold_skips_levels_with_no_passing_genes(self, tmp_path: Path) -> None:
        """Levels where no genes pass the FDR threshold should be skipped."""
        burden_path, label_path, rankings_root = prepare_inputs(tmp_path, levels=("L0", "L1"))
        output_tsv = tmp_path / "fdr_skip.tsv"

        # Set all FDR values in L1 to high values so nothing passes
        l1_path = rankings_root / "L1" / "gene_rankings_with_significance.csv"
        l1_rankings = pd.read_csv(l1_path)
        l1_rankings["fdr_gene"] = 0.99
        l1_rankings.to_csv(l1_path, index=False)

        nonlinear.main(
            [
                "--real-rankings-dir",
                str(rankings_root),
                "--burden-matrix",
                str(burden_path),
                "--labels",
                str(label_path),
                "--output-tsv",
                str(output_tsv),
                "--fdr-threshold",
                "0.25",
                "--classifiers",
                "rf",
                "--levels",
                "L0,L1",
                "--n-permutations",
                "4",
                "--cv-folds",
                "3",
                "--n-cores",
                "1",
                "--seed",
                "42",
            ]
        )

        summary_df = pd.read_csv(output_tsv, sep="\t")
        assert len(summary_df) > 0
        # L1 should be skipped since no genes pass fdr < 0.25
        assert set(summary_df["level"]) == {"L0"}

    def test_thread_environment_variables_are_pinned(self) -> None:
        """BLAS/OpenMP thread counts must be pinned to one at import time."""
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"
        assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
        assert os.environ["NUMEXPR_NUM_THREADS"] == "1"

    def test_parallel_null_distribution_scales_with_cores(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The shared permutation phase should scale materially with outer-loop parallelism."""
        available_cores = os.cpu_count() or 1
        if available_cores < 8:
            pytest.skip("Parallel scaling test requires at least 8 CPU cores")
        if os.environ.get("CI"):
            pytest.skip("Timing-based scaling check is skipped on CI runners")

        def fake_evaluate_random_gene_set_mean_auc(
            burden_values: np.ndarray,
            labels: np.ndarray,
            feature_indices: np.ndarray,
            fold_indices: object,
            classifier_name: str,
            model_seed: int,
        ) -> float:
            del burden_values, labels, feature_indices, fold_indices, classifier_name, model_seed
            time.sleep(0.2)
            return 0.5

        monkeypatch.setattr(
            nonlinear,
            "evaluate_random_gene_set_mean_auc",
            fake_evaluate_random_gene_set_mean_auc,
        )

        burden_values = np.zeros((20, 100), dtype=np.float64)
        labels = np.array([0, 1] * 10, dtype=np.int64)
        fold_indices = [(np.arange(10), np.arange(10, 20))]
        random_gene_sets = [np.arange(5, dtype=np.int64) for _ in range(80)]

        start_single = time.perf_counter()
        nonlinear.run_shared_null_distribution(
            burden_values=burden_values,
            labels=labels,
            random_gene_sets=random_gene_sets,
            fold_indices=fold_indices,
            classifier_name="rf",
            seed=7,
            n_cores=1,
            parallel_verbose=0,
        )
        single_elapsed = time.perf_counter() - start_single

        start_parallel = time.perf_counter()
        nonlinear.run_shared_null_distribution(
            burden_values=burden_values,
            labels=labels,
            random_gene_sets=random_gene_sets,
            fold_indices=fold_indices,
            classifier_name="rf",
            seed=7,
            n_cores=8,
            parallel_verbose=0,
        )
        parallel_elapsed = time.perf_counter() - start_parallel

        assert parallel_elapsed < single_elapsed, (
            "Expected the multi-core run to beat the single-core run, but it did not"
        )
        assert parallel_elapsed <= single_elapsed / 2, (
            f"Expected at least 2x speedup with 8 cores, but got "
            f"{single_elapsed / parallel_elapsed:.2f}x"
        )
