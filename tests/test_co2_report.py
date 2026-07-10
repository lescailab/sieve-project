"""Tests for consolidated CodeCarbon reporting."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import co2_report


FIELDNAMES = [
    "timestamp",
    "project_name",
    "run_id",
    "duration",
    "emissions",
    "cpu_energy",
    "gpu_energy",
    "ram_energy",
    "energy_consumed",
    "country_name",
    "country_iso_code",
    "region",
    "cloud_provider",
    "cloud_region",
    "cpu_count",
    "cpu_model",
    "gpu_count",
    "gpu_model",
    "ram_total_size",
    "tracking_mode",
]


def _row(
    run_id: str,
    project_name: str,
    *,
    duration: float,
    emissions: float,
    cpu_energy: float,
    gpu_energy: float,
    ram_energy: float,
    country: str = "Example country",
    cpu_model: str = "Example CPU",
    gpu_model: str = "Example GPU",
) -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T12:00:00",
        "project_name": project_name,
        "run_id": run_id,
        "duration": duration,
        "emissions": emissions,
        "cpu_energy": cpu_energy,
        "gpu_energy": gpu_energy,
        "ram_energy": ram_energy,
        "energy_consumed": cpu_energy + gpu_energy + ram_energy,
        "country_name": country,
        "country_iso_code": "EXA",
        "region": "",
        "cloud_provider": "",
        "cloud_region": "",
        "cpu_count": "8",
        "cpu_model": cpu_model,
        "gpu_count": "1",
        "gpu_model": gpu_model,
        "ram_total_size": "32",
        "tracking_mode": "machine",
    }


def _write_emissions(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_discovery_normalization_deduplication_and_exact_totals(tmp_path, capsys):
    training_csv = _write_emissions(
        tmp_path / "training" / "co2footprint" / "emissions.csv",
        [
            _row(
                "run-training",
                "sieve-training",
                duration=3600,
                emissions=0.002,
                cpu_energy=0.01,
                gpu_energy=0.02,
                ram_energy=0.005,
            )
        ],
    )
    _write_emissions(
        tmp_path / "explain" / "co2footprint" / "emissions.csv",
        [
            _row(
                "run-explain",
                "sieve-explain",
                duration=1800,
                emissions=0.001,
                cpu_energy=0.005,
                gpu_energy=0.01,
                ram_energy=0.0025,
                country="Another country",
                gpu_model="Another GPU",
            ),
            _row(
                "run-training",
                "sieve-training",
                duration=3600,
                emissions=0.002,
                cpu_energy=0.01,
                gpu_energy=0.02,
                ram_energy=0.005,
            ),
        ],
    )

    files = co2_report.discover_emissions_files([tmp_path, training_csv])
    assert len(files) == 2

    rows = co2_report.normalize_emissions_files(files)
    assert len(rows) == 2
    assert {row["stage"] for row in rows} == {"training", "explain"}
    assert sum(row["duration_seconds"] for row in rows) == pytest.approx(5400)
    assert sum(row["energy_consumed_kwh"] for row in rows) == pytest.approx(0.0525)
    assert sum(row["emissions_kg_co2eq"] for row in rows) == pytest.approx(0.003)
    assert sum(row["emissions_g_co2eq"] for row in rows) == pytest.approx(3.0)
    assert "duplicate run_id" in capsys.readouterr().err

    markdown = co2_report.build_markdown_report(rows)
    assert "| explain | 1 |" in markdown
    assert "| training | 1 |" in markdown
    assert "3.000 g CO2eq" in markdown
    assert "Another GPU" in markdown


def test_missing_optional_component_energy_is_normalized_to_zero(tmp_path):
    path = tmp_path / "co2footprint" / "emissions.csv"
    rows = [
        {
            **_row(
                "run-1",
                "sieve-training",
                duration=10,
                emissions=0.0001,
                cpu_energy=0.001,
                gpu_energy=0,
                ram_energy=0,
            ),
            "gpu_energy": "",
            "ram_energy": "",
        }
    ]
    _write_emissions(path, rows)

    normalized = co2_report.normalize_emissions_files([path])
    assert normalized[0]["gpu_energy_kwh"] == 0.0
    assert normalized[0]["ram_energy_kwh"] == 0.0


def test_malformed_and_non_sieve_rows_are_skipped(tmp_path, capsys):
    path = _write_emissions(
        tmp_path / "co2footprint" / "emissions.csv",
        [
            _row(
                "bad-duration",
                "sieve-training",
                duration="not-a-number",
                emissions=0.1,
                cpu_energy=0.1,
                gpu_energy=0,
                ram_energy=0,
            ),
            _row(
                "other-project",
                "unrelated",
                duration=10,
                emissions=0.1,
                cpu_energy=0.1,
                gpu_energy=0,
                ram_energy=0,
            ),
        ],
    )

    assert co2_report.normalize_emissions_files([path]) == []
    stderr = capsys.readouterr().err
    assert "invalid duration" in stderr
    assert "project_name must begin with 'sieve-'" in stderr


def test_main_writes_csv_and_markdown(tmp_path):
    input_dir = tmp_path / "analysis"
    _write_emissions(
        input_dir / "co2footprint" / "emissions.csv",
        [
            _row(
                "run-1",
                "sieve-training",
                duration=60,
                emissions=0.0002,
                cpu_energy=0.001,
                gpu_energy=0.002,
                ram_energy=0.0005,
            )
        ],
    )
    output_dir = tmp_path / "report"

    exit_code = co2_report.main(
        ["--input", str(input_dir), "--output-dir", str(output_dir)]
    )

    assert exit_code == 0
    runs_path = output_dir / "co2_footprint_runs.csv"
    report_path = output_dir / "co2_footprint_report.md"
    assert runs_path.exists()
    assert report_path.exists()
    with runs_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["emissions_kg_co2eq"] == "0.0002"
    assert "SIEVE CO2 Footprint Report" in report_path.read_text(encoding="utf-8")


def test_main_returns_nonzero_when_no_valid_measurements(tmp_path, capsys):
    exit_code = co2_report.main(
        ["--input", str(tmp_path), "--output-dir", str(tmp_path / "report")]
    )

    assert exit_code == 1
    assert "No valid SIEVE CodeCarbon measurements" in capsys.readouterr().err
