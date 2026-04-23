"""Tests for repair_gene_delta_rank.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.repair_gene_delta_rank as repair


N_GENES = 10
N_VARIANTS_PER_GENE = 5


def _make_variant_df(genes: list[str], rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Build a synthetic variant-level DataFrame with delta_rank."""
    if rng is None:
        rng = np.random.default_rng(42)
    records = []
    for gene in genes:
        for v in range(N_VARIANTS_PER_GENE):
            records.append(
                {
                    "gene_name": gene,
                    "chromosome": "1",
                    "position": len(records) + 1,
                    "delta_rank": float(rng.uniform(-5.0, 10.0)),
                }
            )
    return pd.DataFrame(records)


def _make_gene_stats_df(genes: list[str]) -> pd.DataFrame:
    """Build a minimal gene-stats DataFrame without gene_delta_rank."""
    return pd.DataFrame(
        {
            "gene_name": genes,
            "n_variants_real": [N_VARIANTS_PER_GENE] * len(genes),
            "wilcoxon_p": [0.05] * len(genes),
            "fdr_gene_wilcoxon": [0.10] * len(genes),
        }
    )


def _write_csvs(
    tmp_path: Path,
    genes: list[str],
    *,
    extra_genes_in_stats: list[str] | None = None,
) -> tuple[Path, Path]:
    """Write variant-rankings and gene-stats CSVs to tmp_path."""
    rng = np.random.default_rng(99)
    variant_df = _make_variant_df(genes, rng)
    stats_genes = genes + (extra_genes_in_stats or [])
    gene_stats_df = _make_gene_stats_df(stats_genes)

    variant_path = tmp_path / "rank_calibrated.csv"
    gene_stats_path = tmp_path / "gene_stats.csv"
    variant_df.to_csv(variant_path, index=False)
    gene_stats_df.to_csv(gene_stats_path, index=False)
    return variant_path, gene_stats_path


def test_repair_produces_expected_gene_delta_rank_max(tmp_path: Path) -> None:
    """Patched output must have gene_delta_rank equal to manual per-gene max."""
    genes = [f"GENE_{i}" for i in range(N_GENES)]
    variant_path, gene_stats_path = _write_csvs(tmp_path, genes)
    output_path = tmp_path / "gene_stats_repaired.csv"

    exit_code = repair.main(
        [
            "--variant-rankings", str(variant_path),
            "--gene-stats", str(gene_stats_path),
            "--output", str(output_path),
            "--aggregation", "max",
        ]
    )
    assert exit_code == 0

    patched = pd.read_csv(output_path)
    variant_df = pd.read_csv(variant_path)
    manual_max = variant_df.groupby("gene_name")["delta_rank"].max()

    for gene in genes:
        expected = float(manual_max[gene])
        actual = float(patched.loc[patched["gene_name"] == gene, "gene_delta_rank"].iloc[0])
        assert abs(actual - expected) < 1e-10, f"Mismatch for {gene}: {actual} != {expected}"


def test_repair_preserves_all_existing_gene_stats_columns(tmp_path: Path) -> None:
    """Every original column in the gene-stats CSV must be present unchanged."""
    genes = [f"GENE_{i}" for i in range(N_GENES)]
    variant_path, gene_stats_path = _write_csvs(tmp_path, genes)
    output_path = tmp_path / "gene_stats_repaired.csv"

    original = pd.read_csv(gene_stats_path)
    repair.main(
        [
            "--variant-rankings", str(variant_path),
            "--gene-stats", str(gene_stats_path),
            "--output", str(output_path),
            "--aggregation", "max",
        ]
    )
    patched = pd.read_csv(output_path)

    for col in original.columns:
        assert col in patched.columns, f"Original column '{col}' missing from patched output"
        pd.testing.assert_series_equal(
            original[col].reset_index(drop=True),
            patched[col].reset_index(drop=True),
            check_names=False,
        )


def test_repair_refuses_to_overwrite_input(tmp_path: Path) -> None:
    """Passing the same path for --gene-stats and --output must exit with an error."""
    genes = [f"GENE_{i}" for i in range(N_GENES)]
    variant_path, gene_stats_path = _write_csvs(tmp_path, genes)

    with pytest.raises(SystemExit):
        repair.main(
            [
                "--variant-rankings", str(variant_path),
                "--gene-stats", str(gene_stats_path),
                "--output", str(gene_stats_path),
                "--aggregation", "max",
            ]
        )


def test_repair_refuses_when_target_column_already_present(tmp_path: Path) -> None:
    """gene-stats CSV containing gene_delta_rank must trigger SystemExit."""
    genes = [f"GENE_{i}" for i in range(N_GENES)]
    variant_path, gene_stats_path = _write_csvs(tmp_path, genes)

    gene_stats_df = pd.read_csv(gene_stats_path)
    gene_stats_df["gene_delta_rank"] = 0.0
    gene_stats_df.to_csv(gene_stats_path, index=False)

    output_path = tmp_path / "gene_stats_repaired.csv"
    with pytest.raises(SystemExit):
        repair.main(
            [
                "--variant-rankings", str(variant_path),
                "--gene-stats", str(gene_stats_path),
                "--output", str(output_path),
                "--aggregation", "max",
            ]
        )


def test_repair_fails_when_too_many_genes_missing_from_variant_file(tmp_path: Path) -> None:
    """More than 1% of genes missing from variant file must trigger SystemExit."""
    genes = [f"GENE_{i}" for i in range(N_GENES)]
    missing_genes = [f"MISSING_{i}" for i in range(N_GENES * 10)]
    variant_path, gene_stats_path = _write_csvs(
        tmp_path, genes, extra_genes_in_stats=missing_genes
    )
    output_path = tmp_path / "gene_stats_repaired.csv"

    with pytest.raises(SystemExit):
        repair.main(
            [
                "--variant-rankings", str(variant_path),
                "--gene-stats", str(gene_stats_path),
                "--output", str(output_path),
                "--aggregation", "max",
            ]
        )


def test_dry_run_does_not_write_output(tmp_path: Path) -> None:
    """With --dry-run, no files are created at --output or at <output>.repair_log.txt."""
    genes = [f"GENE_{i}" for i in range(N_GENES)]
    variant_path, gene_stats_path = _write_csvs(tmp_path, genes)
    output_path = tmp_path / "gene_stats_repaired.csv"
    log_path = tmp_path / "gene_stats_repaired.repair_log.txt"

    exit_code = repair.main(
        [
            "--variant-rankings", str(variant_path),
            "--gene-stats", str(gene_stats_path),
            "--output", str(output_path),
            "--aggregation", "max",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not output_path.exists(), "--dry-run must not write the output CSV"
    assert not log_path.exists(), "--dry-run must not write the repair log"
