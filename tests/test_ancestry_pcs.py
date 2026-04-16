"""
Tests for ancestry-PC covariate plumbing.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.covariates import attach_pc_covariates_to_samples, load_pc_map
from src.data.vcf_parser import SampleVariants, VariantRecord
from src.encoding.chunked_dataset import ChunkedVariantDataset, collate_chunks
from src.encoding.levels import AnnotationLevel
from src.models.chunked_sieve import build_sample_covariates


def _make_variant() -> VariantRecord:
    return VariantRecord(
        chrom='1',
        pos=100,
        ref='A',
        alt='T',
        gene='GENE1',
        consequence='missense_variant',
        genotype=1,
        annotations={'sift': 0.01, 'polyphen': 0.9},
    )


def _make_samples():
    variant = _make_variant()
    return [
        SampleVariants('S1', label=1, variants=[variant], sex='M'),
        SampleVariants('S2', label=0, variants=[variant], sex='F'),
    ]


def _write_pc_map(path: Path, rows: list[str]) -> None:
    path.write_text("sample_id\tPC1\tPC2\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_load_pc_map_reads_requested_columns(tmp_path):
    """PC loader extracts the requested number of PCs as float32 vectors."""
    pc_map_path = tmp_path / "pcs.tsv"
    _write_pc_map(pc_map_path, ["S1\t0.1\t0.2", "S2\t-0.4\t0.5"])

    pc_map = load_pc_map(pc_map_path, num_pcs=2)

    assert set(pc_map) == {"S1", "S2"}
    np.testing.assert_allclose(pc_map["S1"], np.array([0.1, 0.2], dtype=np.float32))
    assert pc_map["S1"].dtype == np.float32


def test_load_pc_map_raises_when_num_pcs_exceeds_columns(tmp_path):
    """Requesting more PCs than present in the file raises a clear error."""
    pc_map_path = tmp_path / "pcs.tsv"
    _write_pc_map(pc_map_path, ["S1\t0.1\t0.2"])

    with pytest.raises(ValueError, match="missing required columns"):
        load_pc_map(pc_map_path, num_pcs=3)


def test_attach_pc_covariates_requires_all_samples(tmp_path):
    """Samples missing from the PC map are rejected explicitly."""
    pc_map_path = tmp_path / "pcs.tsv"
    _write_pc_map(pc_map_path, ["S1\t0.1\t0.2"])
    samples = _make_samples()
    pc_map = load_pc_map(pc_map_path, num_pcs=2)

    with pytest.raises(ValueError, match="S2"):
        attach_pc_covariates_to_samples(samples, pc_map=pc_map, include_sex=True)


def test_chunked_dataset_collates_combined_covariates(tmp_path):
    """Chunk collation exposes the full covariate tensor: sex first, then PCs."""
    pc_map_path = tmp_path / "pcs.tsv"
    _write_pc_map(pc_map_path, ["S1\t0.1\t0.2", "S2\t-0.4\t0.5"])
    samples = _make_samples()
    pc_map = load_pc_map(pc_map_path, num_pcs=2)
    attach_pc_covariates_to_samples(samples, pc_map=pc_map, include_sex=True)

    dataset = ChunkedVariantDataset(samples, AnnotationLevel.L0)
    batch = collate_chunks([dataset[0], dataset[1]])

    assert "covariates" in batch
    assert batch["covariates"].shape == (2, 3)
    torch.testing.assert_close(
        batch["covariates"],
        torch.tensor([[1.0, 0.1, 0.2], [0.0, -0.4, 0.5]], dtype=torch.float32),
    )


def test_build_sample_covariates_prefers_full_covariate_tensor():
    """Helper returns the provided multi-covariate tensor unchanged."""
    covariates = torch.tensor([[1.0, 0.1, 0.2], [0.0, -0.4, 0.5]], dtype=torch.float32)
    built = build_sample_covariates(
        batch_sex=torch.tensor([1.0, 0.0], dtype=torch.float32),
        num_covariates=3,
        num_samples=2,
        device=torch.device('cpu'),
        batch_covariates=covariates,
    )
    torch.testing.assert_close(built, covariates)
