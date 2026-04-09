"""Tests for ablation comparison ranking semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

import scripts.compare_ablation_rankings as ablation


def write_variant_rankings(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a small ranking CSV for testing."""
    headers = [
        "variant_id",
        "gene_name",
        "chromosome",
        "position",
        "empirical_p_variant",
        "fdr_variant",
        "z_attribution",
    ]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row[header]) for header in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_rankings_sorts_empirical_p_ascending(tmp_path: Path) -> None:
    """Lower empirical p-values should receive better ranks."""
    csv_path = tmp_path / "L0_sieve_variant_rankings.csv"
    write_variant_rankings(
        csv_path,
        [
            {
                "variant_id": "1:100_A",
                "gene_name": "GENE1",
                "chromosome": "1",
                "position": 100,
                "empirical_p_variant": 0.20,
                "fdr_variant": 0.30,
                "z_attribution": 6.0,
            },
            {
                "variant_id": "1:200_B",
                "gene_name": "GENE2",
                "chromosome": "1",
                "position": 200,
                "empirical_p_variant": 0.01,
                "fdr_variant": 0.02,
                "z_attribution": 1.0,
            },
            {
                "variant_id": "1:300_C",
                "gene_name": "GENE3",
                "chromosome": "1",
                "position": 300,
                "empirical_p_variant": 0.10,
                "fdr_variant": 0.15,
                "z_attribution": 4.0,
            },
        ],
    )

    rows, resolved_col, was_explicit = ablation.load_rankings(
        csv_path,
        score_column="empirical_p_variant",
    )

    assert resolved_col == "empirical_p_variant"
    assert was_explicit is True
    assert [row["variant_id"] for row in rows] == [
        "1:200_B",
        "1:300_C",
        "1:100_A",
    ]
    assert [row["rank"] for row in rows] == [1, 2, 3]


def test_main_defaults_to_empirical_p_variant(tmp_path: Path, monkeypatch) -> None:
    """The CLI should default to null-contrast empirical p-value ranking."""
    ranking_dir = tmp_path / "rankings"
    ranking_dir.mkdir()

    write_variant_rankings(
        ranking_dir / "L0_sieve_variant_rankings.csv",
        [
            {
                "variant_id": "1:100_A",
                "gene_name": "GENE1",
                "chromosome": "1",
                "position": 100,
                "empirical_p_variant": 0.30,
                "fdr_variant": 0.40,
                "z_attribution": 9.0,
            },
            {
                "variant_id": "1:200_B",
                "gene_name": "GENE2",
                "chromosome": "1",
                "position": 200,
                "empirical_p_variant": 0.01,
                "fdr_variant": 0.02,
                "z_attribution": 1.0,
            },
        ],
    )
    write_variant_rankings(
        ranking_dir / "L1_sieve_variant_rankings.csv",
        [
            {
                "variant_id": "1:100_A",
                "gene_name": "GENE1",
                "chromosome": "1",
                "position": 100,
                "empirical_p_variant": 0.02,
                "fdr_variant": 0.03,
                "z_attribution": 2.0,
            },
            {
                "variant_id": "1:300_C",
                "gene_name": "GENE3",
                "chromosome": "1",
                "position": 300,
                "empirical_p_variant": 0.50,
                "fdr_variant": 0.60,
                "z_attribution": 10.0,
            },
        ],
    )

    out_comparison = tmp_path / "summary.yaml"
    out_jaccard = tmp_path / "jaccard.tsv"
    out_level_specific = tmp_path / "level_specific.tsv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_ablation_rankings.py",
            "--ranking-dir",
            str(ranking_dir),
            "--top-k",
            "1",
            "--out-comparison",
            str(out_comparison),
            "--out-jaccard",
            str(out_jaccard),
            "--out-level-specific",
            str(out_level_specific),
        ],
    )

    exit_code = ablation.main()

    assert exit_code == 0
    summary = yaml.safe_load(out_comparison.read_text(encoding="utf-8"))
    assert summary["score_column"] == "empirical_p_variant"
    assert summary["score_sort_order"] == "ascending"


def test_load_rankings_pushes_invalid_empirical_p_to_the_bottom(tmp_path: Path) -> None:
    """Malformed p-values must not be promoted to the best ranks."""
    csv_path = tmp_path / "L0_sieve_variant_rankings.csv"
    csv_path.write_text(
        "\n".join(
            [
                "variant_id,gene_name,chromosome,position,empirical_p_variant,fdr_variant,z_attribution",
                "1:100_A,GENE1,1,100,0.02,0.05,5.0",
                "1:200_B,GENE2,1,200,NA,0.10,4.0",
                "1:300_C,GENE3,1,300,0.10,0.15,3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, _, _ = ablation.load_rankings(csv_path, score_column="empirical_p_variant")

    assert [row["variant_id"] for row in rows] == [
        "1:100_A",
        "1:300_C",
        "1:200_B",
    ]


def test_main_fails_when_significance_column_is_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """The default ablation comparison should reject raw-only ranking files."""
    ranking_dir = tmp_path / "rankings"
    ranking_dir.mkdir()

    raw_csv = "\n".join(
        [
            "variant_id,gene_name,chromosome,position,z_attribution",
            "1:100_A,GENE1,1,100,5.0",
            "1:200_B,GENE2,1,200,4.0",
        ]
    )
    (ranking_dir / "L0_sieve_variant_rankings.csv").write_text(raw_csv + "\n", encoding="utf-8")
    (ranking_dir / "L1_sieve_variant_rankings.csv").write_text(raw_csv + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_ablation_rankings.py",
            "--ranking-dir",
            str(ranking_dir),
            "--out-comparison",
            str(tmp_path / "summary.yaml"),
            "--out-jaccard",
            str(tmp_path / "jaccard.tsv"),
            "--out-level-specific",
            str(tmp_path / "level_specific.tsv"),
        ],
    )

    exit_code = ablation.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "empirical_p_variant" in captured.err
