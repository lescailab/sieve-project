"""
Tests for the chunking logic in validate_epistasis.py.

Verifies that:
- Large samples are chunked to chunk_size with correct index remapping
- Small samples pass through unchanged
- Interactions spanning more than chunk_size variants are skipped
"""

import pytest
import numpy as np

from src.data import VariantRecord, SampleVariants
from src.encoding import AnnotationLevel, VariantDataset
from src.encoding.sparse_tensor import build_variant_tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_variant(pos: int, gene: str = 'GENE1') -> VariantRecord:
    return VariantRecord(
        chrom='1', pos=pos, ref='A', alt='T',
        gene=gene, consequence='missense_variant',
        genotype=1, annotations={},
    )


def _make_sample(n_variants: int, gene: str = 'GENE1') -> SampleVariants:
    """Create a sample with n_variants at positions 1..n_variants."""
    variants = [_make_variant(pos=i + 1, gene=gene) for i in range(n_variants)]
    return SampleVariants('test_sample', label=1, variants=variants)


def _chunk_sample(
    sv: SampleVariants,
    v1_idx: int,
    v2_idx: int,
    chunk_size: int,
    dataset: VariantDataset,
):
    """
    Replicate the chunking logic from validate_epistasis.py.

    Returns (chunk_tensor, chunk_v1_idx, chunk_v2_idx, was_chunked,
             chunk_start_idx, chunk_end_idx) or None if the pair
    cannot fit in a single chunk.
    """
    n_variants = len(sv.variants)

    if n_variants > chunk_size:
        lo = min(v1_idx, v2_idx)
        hi = max(v1_idx, v2_idx)
        span = hi - lo + 1

        if span > chunk_size:
            return None  # cannot fit both

        pad = (chunk_size - span) // 2
        start = max(0, lo - pad)
        end = min(n_variants, start + chunk_size)
        start = max(0, end - chunk_size)

        chunk_variants = sv.variants[start:end]
        chunk_sv = SampleVariants(
            sample_id=sv.sample_id,
            label=sv.label,
            variants=chunk_variants,
        )
        chunk_tensor = build_variant_tensor(
            chunk_sv, dataset.annotation_level,
            dataset.gene_index, impute_value=dataset.impute_value,
        )
        chunk_v1_idx = v1_idx - start
        chunk_v2_idx = v2_idx - start
        return chunk_tensor, chunk_v1_idx, chunk_v2_idx, True, start, end
    else:
        chunk_tensor = build_variant_tensor(
            sv, dataset.annotation_level,
            dataset.gene_index, impute_value=dataset.impute_value,
        )
        return chunk_tensor, v1_idx, v2_idx, False, 0, n_variants


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChunkingLogic:
    """Tests for the sample chunking window selection."""

    def test_small_sample_not_chunked(self):
        """Samples within chunk_size should not be chunked."""
        sv = _make_sample(100)
        dataset = VariantDataset([sv], AnnotationLevel.L0)
        result = _chunk_sample(sv, v1_idx=10, v2_idx=50, chunk_size=200, dataset=dataset)

        chunk_tensor, cv1, cv2, was_chunked, start, end = result
        assert not was_chunked
        assert cv1 == 10
        assert cv2 == 50
        assert start == 0
        assert end == 100
        assert chunk_tensor['features'].shape[0] == 100

    def test_large_sample_is_chunked(self):
        """Samples above chunk_size should be chunked."""
        sv = _make_sample(500)
        dataset = VariantDataset([sv], AnnotationLevel.L0)
        chunk_size = 100
        result = _chunk_sample(sv, v1_idx=200, v2_idx=210, chunk_size=chunk_size, dataset=dataset)

        chunk_tensor, cv1, cv2, was_chunked, start, end = result
        assert was_chunked
        assert end - start == chunk_size
        assert chunk_tensor['features'].shape[0] == chunk_size

    def test_remapped_indices_point_to_correct_variants(self):
        """After chunking, remapped indices must match the original positions."""
        sv = _make_sample(500)
        dataset = VariantDataset([sv], AnnotationLevel.L0)
        v1_idx, v2_idx = 200, 230

        result = _chunk_sample(sv, v1_idx=v1_idx, v2_idx=v2_idx, chunk_size=100, dataset=dataset)
        chunk_tensor, cv1, cv2, was_chunked, start, end = result

        # The remapped indices should reference the same genomic positions
        original_pos1 = sv.variants[v1_idx].pos
        original_pos2 = sv.variants[v2_idx].pos
        chunk_pos = chunk_tensor['positions'].tolist()
        assert chunk_pos[cv1] == original_pos1
        assert chunk_pos[cv2] == original_pos2

    def test_chunk_centred_on_pair(self):
        """The chunk window should be roughly centred on the target pair."""
        sv = _make_sample(1000)
        dataset = VariantDataset([sv], AnnotationLevel.L0)
        v1_idx, v2_idx = 500, 510
        chunk_size = 100

        result = _chunk_sample(sv, v1_idx=v1_idx, v2_idx=v2_idx, chunk_size=chunk_size, dataset=dataset)
        _, _, _, _, start, end = result

        midpoint = (v1_idx + v2_idx) / 2
        chunk_midpoint = (start + end) / 2
        # Midpoints should be close (within half the chunk)
        assert abs(midpoint - chunk_midpoint) < chunk_size / 2

    def test_chunk_at_start_boundary(self):
        """Targets near the start should not produce negative indices."""
        sv = _make_sample(500)
        dataset = VariantDataset([sv], AnnotationLevel.L0)
        result = _chunk_sample(sv, v1_idx=2, v2_idx=5, chunk_size=100, dataset=dataset)

        chunk_tensor, cv1, cv2, was_chunked, start, end = result
        assert was_chunked
        assert start == 0
        assert cv1 >= 0
        assert cv2 >= 0

    def test_chunk_at_end_boundary(self):
        """Targets near the end should not exceed variant count."""
        sv = _make_sample(500)
        dataset = VariantDataset([sv], AnnotationLevel.L0)
        result = _chunk_sample(sv, v1_idx=495, v2_idx=498, chunk_size=100, dataset=dataset)

        chunk_tensor, cv1, cv2, was_chunked, start, end = result
        assert was_chunked
        assert end == 500
        assert cv1 < chunk_tensor['features'].shape[0]
        assert cv2 < chunk_tensor['features'].shape[0]

    def test_span_exceeds_chunk_size_returns_none(self):
        """When targets are further apart than chunk_size, return None."""
        sv = _make_sample(500)
        dataset = VariantDataset([sv], AnnotationLevel.L0)
        result = _chunk_sample(sv, v1_idx=10, v2_idx=200, chunk_size=100, dataset=dataset)

        assert result is None

    def test_span_equals_chunk_size_fits(self):
        """Edge case: span exactly equals chunk_size should still fit."""
        sv = _make_sample(500)
        dataset = VariantDataset([sv], AnnotationLevel.L0)
        # span = 99 - 0 + 1 = 100, equals chunk_size
        result = _chunk_sample(sv, v1_idx=150, v2_idx=249, chunk_size=100, dataset=dataset)

        chunk_tensor, cv1, cv2, was_chunked, start, end = result
        assert was_chunked
        assert chunk_tensor['features'].shape[0] == 100

    def test_n_variants_in_chunk_matches(self):
        """The chunk tensor size must equal the configured chunk_size for large samples."""
        sv = _make_sample(5000)
        dataset = VariantDataset([sv], AnnotationLevel.L0)
        chunk_size = 3000

        result = _chunk_sample(sv, v1_idx=2000, v2_idx=2100, chunk_size=chunk_size, dataset=dataset)
        chunk_tensor, _, _, was_chunked, _, _ = result

        assert was_chunked
        assert chunk_tensor['features'].shape[0] == chunk_size
