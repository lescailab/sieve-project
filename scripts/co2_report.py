#!/usr/bin/env python3
"""Compile local CodeCarbon CSV files into a consolidated SIEVE report."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


RUN_COLUMNS = [
    "stage",
    "project_name",
    "run_id",
    "timestamp",
    "duration_seconds",
    "energy_consumed_kwh",
    "emissions_kg_co2eq",
    "emissions_g_co2eq",
    "cpu_energy_kwh",
    "gpu_energy_kwh",
    "ram_energy_kwh",
    "cpu_model",
    "gpu_model",
    "cpu_count",
    "gpu_count",
    "ram_total_size_gb",
    "country_name",
    "country_iso_code",
    "region",
    "cloud_provider",
    "cloud_region",
    "tracking_mode",
    "source_file",
]

NUMERIC_FIELDS = {
    "duration_seconds": ("duration",),
    "energy_consumed_kwh": ("energy_consumed",),
    "emissions_kg_co2eq": ("emissions",),
    "cpu_energy_kwh": ("cpu_energy",),
    "gpu_energy_kwh": ("gpu_energy",),
    "ram_energy_kwh": ("ram_energy",),
}


def _warn(message: str) -> None:
    print(f"WARNING: CO2 report {message}", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compile SIEVE CodeCarbon measurements into Markdown and CSV reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help=(
            "CodeCarbon emissions CSV or a directory to scan recursively for "
            "co2footprint/emissions.csv (repeatable)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for co2_footprint_runs.csv and co2_footprint_report.md",
    )
    return parser.parse_args(argv)


def discover_emissions_files(inputs: Iterable[str | Path]) -> list[Path]:
    """Resolve explicit CSVs and recursively discover standard footprint files."""
    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw_input in inputs:
        input_path = Path(raw_input)
        if input_path.is_file():
            candidates = [input_path]
        elif input_path.is_dir():
            candidates = sorted(input_path.rglob("co2footprint/emissions.csv"))
            if not candidates:
                _warn(f"found no co2footprint/emissions.csv below {input_path}")
        else:
            _warn(f"input does not exist: {input_path}")
            continue

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(candidate)
    return discovered


def _first_value(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _parse_float(
    row: dict[str, str],
    names: tuple[str, ...],
    *,
    required: bool,
) -> float:
    value = _first_value(row, *names)
    if not value:
        if required:
            raise ValueError(f"missing {names[0]}")
        return 0.0
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"invalid {names[0]} value {value!r}") from error


def normalize_emissions_files(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Read and normalize valid SIEVE CodeCarbon rows from one or more files."""
    normalized: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    _warn(f"skipping empty CSV: {path}")
                    continue

                for line_number, raw_row in enumerate(reader, start=2):
                    project_name = _first_value(raw_row, "project_name")
                    if not project_name.startswith("sieve-") or len(project_name) <= len("sieve-"):
                        _warn(
                            f"skipping {path}:{line_number}: project_name must begin with 'sieve-'"
                        )
                        continue

                    run_id = _first_value(raw_row, "run_id", "run-id")
                    timestamp = _first_value(raw_row, "timestamp")
                    if not run_id or not timestamp:
                        _warn(f"skipping {path}:{line_number}: missing run_id or timestamp")
                        continue
                    if run_id in seen_run_ids:
                        _warn(f"ignoring duplicate run_id {run_id} from {path}:{line_number}")
                        continue

                    try:
                        numbers = {
                            output_name: _parse_float(
                                raw_row,
                                input_names,
                                required=output_name
                                in {
                                    "duration_seconds",
                                    "energy_consumed_kwh",
                                    "emissions_kg_co2eq",
                                },
                            )
                            for output_name, input_names in NUMERIC_FIELDS.items()
                        }
                    except ValueError as error:
                        _warn(f"skipping {path}:{line_number}: {error}")
                        continue

                    record: dict[str, Any] = {
                        "stage": project_name.removeprefix("sieve-"),
                        "project_name": project_name,
                        "run_id": run_id,
                        "timestamp": timestamp,
                        **numbers,
                        "emissions_g_co2eq": numbers["emissions_kg_co2eq"] * 1000.0,
                        "cpu_model": _first_value(raw_row, "cpu_model"),
                        "gpu_model": _first_value(raw_row, "gpu_model"),
                        "cpu_count": _first_value(raw_row, "cpu_count"),
                        "gpu_count": _first_value(raw_row, "gpu_count"),
                        "ram_total_size_gb": _first_value(
                            raw_row, "ram_total_size", "ram_total_size_gb"
                        ),
                        "country_name": _first_value(raw_row, "country_name"),
                        "country_iso_code": _first_value(raw_row, "country_iso_code"),
                        "region": _first_value(raw_row, "region"),
                        "cloud_provider": _first_value(raw_row, "cloud_provider"),
                        "cloud_region": _first_value(raw_row, "cloud_region"),
                        "tracking_mode": _first_value(raw_row, "tracking_mode"),
                        "source_file": str(path),
                    }
                    seen_run_ids.add(run_id)
                    normalized.append(record)
        except (OSError, csv.Error) as error:
            _warn(f"could not read {path}: {error}")

    return normalized


def write_normalized_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write normalized run-level measurements."""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _totals(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    fields = [
        "duration_seconds",
        "energy_consumed_kwh",
        "emissions_kg_co2eq",
        "emissions_g_co2eq",
        "cpu_energy_kwh",
        "gpu_energy_kwh",
        "ram_energy_kwh",
    ]
    return {field: sum(float(row[field]) for row in rows) for field in fields}


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_markdown_report(rows: list[dict[str, Any]]) -> str:
    """Build the human-readable consolidated footprint report."""
    overall = _totals(rows)
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stage[str(row["stage"])].append(row)

    lines = [
        "# SIEVE CO2 Footprint Report",
        "",
        "## Overview",
        "",
        f"- Measured runs: {len(rows)}",
        f"- Runtime: {overall['duration_seconds'] / 3600.0:.3f} hours",
        f"- Energy consumed: {overall['energy_consumed_kwh']:.6f} kWh",
        f"- Emissions: {overall['emissions_g_co2eq']:.3f} g CO2eq "
        f"({overall['emissions_kg_co2eq']:.6f} kg CO2eq)",
        "",
        "## Totals by stage",
        "",
        "| Stage | Runs | Runtime (h) | Total energy (kWh) | CPU (kWh) | GPU (kWh) | RAM (kWh) | CO2eq (g) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for stage in sorted(by_stage):
        stage_rows = by_stage[stage]
        totals = _totals(stage_rows)
        lines.append(
            f"| {_escape_markdown(stage)} | {len(stage_rows)} | "
            f"{totals['duration_seconds'] / 3600.0:.3f} | "
            f"{totals['energy_consumed_kwh']:.6f} | "
            f"{totals['cpu_energy_kwh']:.6f} | "
            f"{totals['gpu_energy_kwh']:.6f} | "
            f"{totals['ram_energy_kwh']:.6f} | "
            f"{totals['emissions_g_co2eq']:.3f} |"
        )
    lines.append(
        f"| **Overall** | **{len(rows)}** | "
        f"**{overall['duration_seconds'] / 3600.0:.3f}** | "
        f"**{overall['energy_consumed_kwh']:.6f}** | "
        f"**{overall['cpu_energy_kwh']:.6f}** | "
        f"**{overall['gpu_energy_kwh']:.6f}** | "
        f"**{overall['ram_energy_kwh']:.6f}** | "
        f"**{overall['emissions_g_co2eq']:.3f}** |"
    )

    lines.extend(
        [
            "",
            "## Individual runs",
            "",
            "| Timestamp | Stage | Run ID | Runtime (s) | Energy (kWh) | CO2eq (g) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda value: (str(value["timestamp"]), str(value["run_id"]))):
        lines.append(
            f"| {_escape_markdown(row['timestamp'])} | {_escape_markdown(row['stage'])} | "
            f"{_escape_markdown(row['run_id'])} | {row['duration_seconds']:.3f} | "
            f"{row['energy_consumed_kwh']:.6f} | {row['emissions_g_co2eq']:.3f} |"
        )

    environment_counts = Counter(
        (
            row["cpu_model"] or "Not reported",
            row["gpu_model"] or "Not reported",
            row["country_name"] or row["country_iso_code"] or "Not reported",
            row["region"] or row["cloud_region"] or "Not reported",
            row["tracking_mode"] or "Not reported",
        )
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Measurement environments",
            "",
            "| CPU | GPU | Country | Region/cloud region | Tracking mode | Runs |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for environment, count in sorted(environment_counts.items(), key=lambda item: item[0]):
        cpu, gpu, country, region, tracking_mode = environment
        lines.append(
            f"| {_escape_markdown(cpu)} | {_escape_markdown(gpu)} | "
            f"{_escape_markdown(country)} | {_escape_markdown(region)} | "
            f"{_escape_markdown(tracking_mode)} | {count} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation caveats",
            "",
            "- These values estimate operational CPU, GPU, and RAM energy and emissions; "
            "they do not represent full hardware lifecycle or facility emissions.",
            "- Machine tracking can include unrelated activity on shared hosts. Dedicated "
            "compute allocations provide cleaner estimates.",
            "- CPU and Apple Silicon measurements may use fallback estimates when RAPL or "
            "powermetrics access is unavailable.",
            "- Component energy values absent from a source row are represented as zero in "
            "the normalized report.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """Write normalized CSV and Markdown report files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "co2_footprint_runs.csv"
    markdown_path = output_dir / "co2_footprint_report.md"
    write_normalized_csv(rows, csv_path)
    markdown_path.write_text(build_markdown_report(rows), encoding="utf-8")
    return csv_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    """Compile requested CodeCarbon measurements."""
    args = parse_args(argv)
    files = discover_emissions_files(args.inputs)
    rows = normalize_emissions_files(files)
    if not rows:
        print("ERROR: No valid SIEVE CodeCarbon measurements were found.", file=sys.stderr)
        return 1

    csv_path, markdown_path = write_reports(rows, Path(args.output_dir))
    print(f"Normalized footprint data saved to {csv_path}")
    print(f"CO2 footprint report saved to {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
