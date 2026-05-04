"""Shared fixture helpers for aggregate_gene_interactions tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch


@dataclass
class _Variant:
    chrom: str
    pos: int
    gene: str


@dataclass
class _Sample:
    sample_id: str
    label: int
    variants: list[_Variant] = field(default_factory=list)


_GENES = ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E", "GENE_F"]
_CHROM = {
    "GENE_A": "1", "GENE_B": "1", "GENE_C": "1",
    "GENE_D": "2", "GENE_E": "2", "GENE_F": "2",
}
_POS = {g: 100 * (i + 1) for i, g in enumerate(_GENES)}

_VARIANT_ROWS = [
    {"gene_name": "GENE_A", "chromosome": "1", "position": 100, "z_attribution": 5.0, "delta_rank": 1.0},
    {"gene_name": "GENE_B", "chromosome": "1", "position": 200, "z_attribution": 4.0, "delta_rank": 0.5},
    {"gene_name": "GENE_C", "chromosome": "1", "position": 300, "z_attribution": 3.0, "delta_rank": 0.3},
    {"gene_name": "GENE_D", "chromosome": "2", "position": 100, "z_attribution": 2.0, "delta_rank": 8.0},
    {"gene_name": "GENE_E", "chromosome": "2", "position": 200, "z_attribution": 1.0, "delta_rank": 7.0},
    {"gene_name": "GENE_F", "chromosome": "2", "position": 300, "z_attribution": 0.5, "delta_rank": 6.0},
]

_GENE_ROWS = [
    {"gene_name": "GENE_A", "gene_z_score": 5.0, "gene_delta_rank": 1.0, "gene_rank": 1},
    {"gene_name": "GENE_B", "gene_z_score": 4.0, "gene_delta_rank": 0.5, "gene_rank": 2},
    {"gene_name": "GENE_C", "gene_z_score": 3.0, "gene_delta_rank": 0.3, "gene_rank": 3},
    {"gene_name": "GENE_D", "gene_z_score": 2.0, "gene_delta_rank": 8.0, "gene_rank": 4},
    {"gene_name": "GENE_E", "gene_z_score": 1.0, "gene_delta_rank": 7.0, "gene_rank": 5},
    {"gene_name": "GENE_F", "gene_z_score": 0.5, "gene_delta_rank": 6.0, "gene_rank": 6},
]


def make_pt(tmp_path: Path, genes: list[str] | None = None) -> Path:
    """Save a minimal preprocessed .pt payload; every sample carries all genes."""
    if genes is None:
        genes = _GENES
    chrom_map = {g: _CHROM.get(g, "1") for g in genes}
    pos_map = {g: _POS.get(g, 100) for g in genes}
    samples = [
        _Sample(
            sample_id=f"S{i}",
            label=int(i < 5),
            variants=[_Variant(chrom=chrom_map[g], pos=pos_map[g], gene=g) for g in genes],
        )
        for i in range(10)
    ]
    pt_path = tmp_path / "preprocessed.pt"
    torch.save({"samples": samples}, pt_path)
    return pt_path


def make_variant_rankings(
    tmp_path: Path,
    include_delta_rank: bool = True,
    rows: list[dict] | None = None,
    filename: str = "variant_rankings.csv",
) -> Path:
    """Write variant rankings CSV."""
    df = pd.DataFrame(rows if rows is not None else _VARIANT_ROWS)
    if not include_delta_rank and "delta_rank" in df.columns:
        df = df.drop(columns=["delta_rank"])
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


def make_gene_rankings(
    tmp_path: Path,
    include_gene_delta_rank: bool = True,
    rows: list[dict] | None = None,
    filename: str = "gene_rankings.csv",
) -> Path:
    """Write gene rankings CSV."""
    df = pd.DataFrame(rows if rows is not None else _GENE_ROWS)
    if not include_gene_delta_rank and "gene_delta_rank" in df.columns:
        df = df.drop(columns=["gene_delta_rank"])
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


def make_null_rankings(tmp_path: Path) -> Path:
    """Write a minimal null rankings CSV with uniformly low z_attribution scores."""
    rows = [
        {"gene_name": f"NULL_G{i}", "chromosome": "1", "position": i * 100,
         "z_attribution": -10.0}
        for i in range(20)
    ]
    path = tmp_path / "null_rankings.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def genes_in_pairs(out_dir: Path, suffix: str = "") -> set[str]:
    """Return the set of genes appearing in the pair CSV (with optional suffix)."""
    fname = f"gene_pair_interactions{suffix}.csv"
    df = pd.read_csv(out_dir / fname)
    return set(df["gene_a"].tolist()) | set(df["gene_b"].tolist())
