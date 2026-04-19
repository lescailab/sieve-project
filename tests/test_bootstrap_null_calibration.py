"""Tests for bootstrap rank-based null calibration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.bootstrap_null_calibration as calibration


N_SAMPLES = 100
N_VARIANTS = 500


def _build_variant_catalog() -> pd.DataFrame:
    """Return a deterministic variant catalogue for synthetic tests."""
    chromosomes = np.array(
        ["1"] * 200 + ["2"] * 200 + ["X"] * 50 + ["Y"] * 50,
        dtype=object,
    )
    positions = np.arange(1, N_VARIANTS + 1) * 10

    gene_names: list[str] = []
    gene_ids: list[int] = []
    for idx in range(N_VARIANTS):
        if idx < 3:
            gene_names.append("UNDERPOWERED")
            gene_ids.append(0)
        elif idx < 15:
            gene_names.append("POWERED")
            gene_ids.append(1)
        else:
            gene_id = 2 + (idx - 15) // 10
            gene_names.append(f"GENE_{gene_id}")
            gene_ids.append(gene_id)

    return pd.DataFrame(
        {
            "chromosome": chromosomes,
            "position": positions,
            "gene_name": gene_names,
            "gene_id": gene_ids,
        }
    )


def _write_null_attributions(
    path: Path,
    catalog: pd.DataFrame,
    score_matrix: np.ndarray,
) -> None:
    """Write a mock explain.py-style attributions.npz file."""
    variant_scores = []
    metadata = []
    positions = catalog["position"].to_numpy(dtype=int)
    gene_ids = catalog["gene_id"].to_numpy(dtype=int)
    chromosomes = catalog["chromosome"].to_numpy(dtype=object)

    for sample_idx in range(score_matrix.shape[0]):
        variant_scores.append(score_matrix[sample_idx].astype(float))
        metadata.append(
            {
                "positions": positions.copy(),
                "gene_ids": gene_ids.copy(),
                "chromosomes": chromosomes.copy(),
                "sample_idx": sample_idx,
                "sample_id": f"S{sample_idx:03d}",
                "label": int(sample_idx % 2 == 0),
            }
        )

    np.savez(
        path,
        variant_scores=np.array(variant_scores, dtype=object),
        metadata=np.array(metadata, dtype=object),
    )


def _build_real_rankings(
    catalog: pd.DataFrame,
    null_means: np.ndarray,
    *,
    boost_indices: list[int] | None = None,
    prefix_chr: bool = False,
    add_missing_variant: bool = False,
    permute_null_means: bool = False,
) -> pd.DataFrame:
    """Build a synthetic real rankings dataframe."""
    real_df = catalog.copy()
    real_df["num_samples"] = N_SAMPLES
    real_df["mean_attribution"] = np.array(null_means, copy=True)

    if permute_null_means:
        rng = np.random.default_rng(2024)
        real_df["mean_attribution"] = rng.permutation(real_df["mean_attribution"].to_numpy())

    if boost_indices:
        boost_values = 20.0 + np.arange(len(boost_indices), 0, -1)
        real_df.loc[boost_indices, "mean_attribution"] = boost_values

    if prefix_chr:
        real_df["chromosome"] = real_df["chromosome"].map(lambda chrom: f"chr{chrom}")

    if add_missing_variant:
        real_df = pd.concat(
            [
                real_df,
                pd.DataFrame(
                    [
                        {
                            "chromosome": "chr9" if prefix_chr else "9",
                            "position": 9_999_999,
                            "gene_name": "MISSING_NULL",
                            "gene_id": 9_999,
                            "num_samples": N_SAMPLES,
                            "mean_attribution": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    return real_df


def _run_bootstrap(
    tmp_path: Path,
    *,
    real_df: pd.DataFrame,
    null_path: Path,
    n_bootstrap: int = 100,
    seed: int = 42,
    exclude_sex_chroms: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the bootstrap script and return its three outputs."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    real_path = tmp_path / "real_rankings.csv"
    output_path = tmp_path / "rank_calibrated.csv"
    gene_path = tmp_path / "gene_stats.csv"
    summary_path = tmp_path / "summary.yaml"
    metadata_path = tmp_path / "analysis_metadata.yaml"

    real_df.to_csv(real_path, index=False)
    metadata_path.write_text(
        yaml.safe_dump({"n_samples": N_SAMPLES}, sort_keys=False),
        encoding="utf-8",
    )

    argv = [
        "--real-rankings",
        str(real_path),
        "--null-attributions",
        str(null_path),
        "--output",
        str(output_path),
        "--output-gene-stats",
        str(gene_path),
        "--output-summary",
        str(summary_path),
        "--n-bootstrap",
        str(n_bootstrap),
        "--seed",
        str(seed),
        "--n-jobs",
        "1",
    ]
    if exclude_sex_chroms:
        argv.append("--exclude-sex-chroms")

    exit_code = calibration.main(argv)
    assert exit_code == 0

    return (
        pd.read_csv(output_path),
        pd.read_csv(gene_path),
        yaml.safe_load(summary_path.read_text(encoding="utf-8")),
    )


@pytest.fixture(scope="module")
def synthetic_null_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Create a shared synthetic null attributions file."""
    base_dir = tmp_path_factory.mktemp("bootstrap_null_fixture")
    catalog = _build_variant_catalog()

    rng = np.random.default_rng(123)
    base_effects = rng.normal(loc=0.0, scale=0.2, size=N_VARIANTS)
    base_effects[:20] = -2.0
    base_effects[20:40] = 2.0
    noise = rng.normal(loc=0.0, scale=0.15, size=(N_SAMPLES, N_VARIANTS))
    score_matrix = base_effects[None, :] + noise

    null_path = base_dir / "attributions.npz"
    _write_null_attributions(null_path, catalog, score_matrix)

    return {
        "catalog": catalog,
        "null_means": score_matrix.mean(axis=0),
        "null_path": null_path,
    }


def test_bootstrap_reproducibility(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Running with the same seed should reproduce identical outputs."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=list(range(20)),
    )

    out_a, gene_a, summary_a = _run_bootstrap(
        tmp_path / "run_a",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
        seed=7,
    )
    out_b, gene_b, summary_b = _run_bootstrap(
        tmp_path / "run_b",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
        seed=7,
    )

    pd.testing.assert_frame_equal(out_a, out_b)
    pd.testing.assert_frame_equal(gene_a, gene_b)
    assert summary_a == summary_b


@pytest.mark.parametrize("n_bootstrap", [100, 500])
def test_resolution_floor(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
    n_bootstrap: int,
) -> None:
    """Empirical p-values must respect the Phipson-Smyth resolution floor."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=list(range(20)),
    )
    out_df, _, summary = _run_bootstrap(
        tmp_path / f"resolution_{n_bootstrap}",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=n_bootstrap,
    )

    expected_floor = 1.0 / (n_bootstrap + 1)
    assert out_df["p_rank_boot"].min() >= expected_floor - 1e-12
    assert np.isclose(
        summary["per_variant"]["p_rank_boot_resolution_floor"],
        expected_floor,
    )
    assert (
        out_df["at_resolution_floor"]
        == np.isclose(out_df["p_rank_boot"], expected_floor)
    ).all()


def test_fdr_monotonicity(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """BH-FDR must remain monotone when sorted by p-value."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=list(range(20)),
    )
    out_df, _, _ = _run_bootstrap(
        tmp_path / "fdr_monotonicity",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
    )

    sorted_out = out_df.sort_values("p_rank_boot").reset_index(drop=True)
    diffs = np.diff(sorted_out["fdr_rank_boot"].to_numpy(dtype=float))
    assert np.all(diffs >= -1e-12)


def test_null_signal_case(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """A null-like real ranking should not produce a large discovery fraction."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
    )
    out_df, _, _ = _run_bootstrap(
        tmp_path / "null_case",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
        seed=11,
    )

    discovery_fraction = float((out_df["fdr_rank_boot"] < 0.05).mean())
    assert discovery_fraction <= 0.10


def test_strong_signal_case(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Strongly boosted variants should sit at the empirical resolution floor."""
    signal_indices = list(range(20))
    signal_positions = set(
        synthetic_null_dataset["catalog"].iloc[signal_indices]["position"].tolist()
    )
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=signal_indices,
    )
    out_df, _, _ = _run_bootstrap(
        tmp_path / "strong_signal",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=500,
    )

    signal_rows = out_df[out_df["position"].isin(signal_positions)].copy()
    assert len(signal_rows) == 20
    assert signal_rows["at_resolution_floor"].all()
    assert (signal_rows["fdr_rank_boot"] < 0.05).all()


def test_gene_wilcoxon_underpowered_flag(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Genes below the variant threshold should be flagged and skipped."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=list(range(20)),
    )
    _, gene_df, _ = _run_bootstrap(
        tmp_path / "gene_stats",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
    )

    underpowered = gene_df.loc[gene_df["gene_name"] == "UNDERPOWERED"].iloc[0]
    powered = gene_df.loc[gene_df["gene_name"] == "POWERED"].iloc[0]
    assert bool(underpowered["underpowered"]) is True
    assert pd.isna(underpowered["wilcoxon_p"])
    assert bool(powered["underpowered"]) is False
    assert not pd.isna(powered["wilcoxon_p"])


def test_exclude_sex_chroms(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Sex chromosomes should be removed from both real and null inputs."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=list(range(20)),
    )
    out_df, _, summary = _run_bootstrap(
        tmp_path / "exclude_sex",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
        exclude_sex_chroms=True,
    )

    assert not out_df["chromosome"].isin(["X", "Y"]).any()
    assert summary["excluded_sex_chroms"] is True
    assert summary["n_real_variants_removed_sex_chroms"] > 0
    assert summary["n_null_rows_removed_sex_chroms"] > 0


def test_missing_variant_in_null(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Variants absent from the null universe should be assigned worst null ranks."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        add_missing_variant=True,
    )

    with caplog.at_level(logging.WARNING):
        out_df, _, summary = _run_bootstrap(
            tmp_path / "missing_null",
            real_df=real_df,
            null_path=synthetic_null_dataset["null_path"],
            n_bootstrap=100,
        )

    missing_row = out_df.loc[out_df["gene_name"] == "MISSING_NULL"].iloc[0]
    assert summary["n_real_variants_missing_from_null"] == 1
    assert missing_row["median_rank_null_boot"] > len(synthetic_null_dataset["catalog"])
    assert missing_row["fdr_rank_boot"] >= 0.05
    assert "absent from the null universe" in caplog.text


def test_delta_rank_sign(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Signal variants should have positive delta_rank while null-like ones stay near zero."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=[0],
    )
    out_df, _, _ = _run_bootstrap(
        tmp_path / "delta_rank",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
    )

    signal_row = out_df.loc[out_df["position"] == 10].iloc[0]
    background_row = out_df.loc[out_df["position"] == 1010].iloc[0]
    assert signal_row["delta_rank"] > 0
    assert abs(background_row["delta_rank"]) < 10


def test_chromosome_normalisation(
    tmp_path: Path,
    synthetic_null_dataset: dict[str, object],
) -> None:
    """Real 'chrN' labels should still match null 'N' labels."""
    real_df = _build_real_rankings(
        synthetic_null_dataset["catalog"],
        synthetic_null_dataset["null_means"],
        boost_indices=[0],
        prefix_chr=True,
    )
    out_df, _, summary = _run_bootstrap(
        tmp_path / "chrom_norm",
        real_df=real_df,
        null_path=synthetic_null_dataset["null_path"],
        n_bootstrap=100,
    )

    assert summary["n_real_variants_missing_from_null"] == 0
    assert "1" in set(out_df["chromosome"].astype(str))
    assert "chr1" not in set(out_df["chromosome"].astype(str))
