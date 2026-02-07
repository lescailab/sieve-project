"""
Unit tests for sex covariate support across the SIEVE pipeline.

Tests cover:
- SampleVariants sex field storage and propagation
- ChunkedVariantDataset sex encoding and collation
- PhenotypeClassifier with covariates
- ChunkedSIEVEModel with sex covariates end-to-end
- Backward compatibility when no covariates are used
"""
import pytest
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.vcf_parser import SampleVariants, VariantRecord
from src.encoding.chunked_dataset import (
    ChunkedVariantDataset,
    collate_chunks,
    _encode_sex,
)
from src.encoding.levels import AnnotationLevel
from src.models.classifier import PhenotypeClassifier, AttentionPoolingClassifier
from src.models.sieve import SIEVE
from src.models.chunked_sieve import ChunkedSIEVEModel
from src.training.loss import SIEVELoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_variant(gene: str = 'GENE1') -> VariantRecord:
    return VariantRecord(
        chrom='1', pos=100, ref='A', alt='T',
        gene=gene, consequence='missense_variant',
        genotype=1, annotations={'sift': 0.05, 'polyphen': 0.9},
    )


def _make_samples():
    """Create a small set of samples with sex annotations."""
    var = _make_variant()
    return [
        SampleVariants('s1', label=1, variants=[var], sex='M'),
        SampleVariants('s2', label=0, variants=[var], sex='F'),
        SampleVariants('s3', label=1, variants=[var], sex='M'),
        SampleVariants('s4', label=0, variants=[var], sex=None),
    ]


# ---------------------------------------------------------------------------
# Tests: _encode_sex
# ---------------------------------------------------------------------------

class TestEncodeSex:
    def test_male(self):
        assert _encode_sex('M') == 1.0

    def test_female(self):
        assert _encode_sex('F') == 0.0

    def test_none(self):
        assert _encode_sex(None) == -1.0

    def test_unknown(self):
        assert _encode_sex('AMBIGUOUS') == -1.0


# ---------------------------------------------------------------------------
# Tests: SampleVariants sex field
# ---------------------------------------------------------------------------

class TestSampleVariantsSex:
    def test_default_is_none(self):
        sv = SampleVariants('s', 0, [])
        assert sv.sex is None

    def test_stores_sex(self):
        sv = SampleVariants('s', 0, [], sex='M')
        assert sv.sex == 'M'

    def test_repr_includes_sex(self):
        sv = SampleVariants('s', 0, [], sex='F')
        assert 'sex=F' in repr(sv)

    def test_repr_omits_sex_when_none(self):
        sv = SampleVariants('s', 0, [])
        assert 'sex' not in repr(sv)


# ---------------------------------------------------------------------------
# Tests: ChunkedVariantDataset sex propagation
# ---------------------------------------------------------------------------

class TestChunkedDatasetSex:
    def test_chunk_info_contains_sex(self):
        samples = _make_samples()
        ds = ChunkedVariantDataset(samples, AnnotationLevel.L0)
        for info in ds.chunk_info:
            assert 'sex' in info

    def test_sex_encoding_in_chunk_info(self):
        samples = _make_samples()
        ds = ChunkedVariantDataset(samples, AnnotationLevel.L0)
        # s1=M -> 1.0, s2=F -> 0.0, s3=M -> 1.0, s4=None -> -1.0
        expected = [1.0, 0.0, 1.0, -1.0]
        actual = [info['sex'] for info in ds.chunk_info]
        assert actual == expected

    def test_getitem_returns_sex(self):
        samples = _make_samples()
        ds = ChunkedVariantDataset(samples, AnnotationLevel.L0)
        item = ds[0]
        assert 'sex' in item
        assert item['sex'] == 1.0  # s1 is Male


# ---------------------------------------------------------------------------
# Tests: collate_chunks with sex
# ---------------------------------------------------------------------------

class TestCollateChunksSex:
    def test_collated_batch_has_sex_tensor(self):
        samples = _make_samples()
        ds = ChunkedVariantDataset(samples, AnnotationLevel.L0)
        items = [ds[i] for i in range(len(ds))]
        batch = collate_chunks(items)
        assert 'sex' in batch
        assert isinstance(batch['sex'], torch.Tensor)
        assert batch['sex'].dtype == torch.float32
        assert batch['sex'].shape == (len(ds),)

    def test_sex_values_correct(self):
        samples = _make_samples()
        ds = ChunkedVariantDataset(samples, AnnotationLevel.L0)
        items = [ds[i] for i in range(len(ds))]
        batch = collate_chunks(items)
        expected = torch.tensor([1.0, 0.0, 1.0, -1.0])
        assert torch.equal(batch['sex'], expected)


# ---------------------------------------------------------------------------
# Tests: PhenotypeClassifier with covariates
# ---------------------------------------------------------------------------

class TestClassifierCovariates:
    def test_no_covariates_backward_compat(self):
        clf = PhenotypeClassifier(num_genes=10, latent_dim=8)
        x = torch.randn(2, 10, 8)
        out = clf(x)
        assert out.shape == (2, 1)

    def test_with_covariates(self):
        clf = PhenotypeClassifier(num_genes=10, latent_dim=8, num_covariates=1)
        x = torch.randn(2, 10, 8)
        cov = torch.tensor([[1.0], [0.0]])
        out = clf(x, covariates=cov)
        assert out.shape == (2, 1)

    def test_covariates_affect_output(self):
        """Verify that different covariate values produce different outputs."""
        clf = PhenotypeClassifier(num_genes=10, latent_dim=8, num_covariates=1)
        clf.eval()
        x = torch.randn(1, 10, 8)
        x_dup = x.clone()
        cov_m = torch.tensor([[1.0]])
        cov_f = torch.tensor([[0.0]])
        with torch.no_grad():
            out_m = clf(x, covariates=cov_m)
            out_f = clf(x_dup, covariates=cov_f)
        # Different covariates should (almost certainly) produce different logits
        assert not torch.allclose(out_m, out_f)

    def test_attention_pooling_with_covariates(self):
        clf = AttentionPoolingClassifier(
            num_genes=10, latent_dim=8, num_covariates=1
        )
        x = torch.randn(2, 10, 8)
        cov = torch.tensor([[1.0], [0.0]])
        out = clf(x, covariates=cov)
        assert out.shape == (2, 1)


# ---------------------------------------------------------------------------
# Tests: SIEVE model with covariates
# ---------------------------------------------------------------------------

class TestSIEVECovariates:
    def test_sieve_forward_no_covariates(self):
        model = SIEVE(input_dim=1, num_genes=5, latent_dim=8, num_covariates=0)
        model.eval()
        f = torch.randn(2, 3, 1)
        p = torch.randint(0, 1000, (2, 3))
        g = torch.randint(0, 5, (2, 3))
        m = torch.ones(2, 3, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = model(f, p, g, m)
        assert logits.shape == (2, 1)

    def test_sieve_forward_with_covariates(self):
        model = SIEVE(input_dim=1, num_genes=5, latent_dim=8, num_covariates=1)
        model.eval()
        f = torch.randn(2, 3, 1)
        p = torch.randint(0, 1000, (2, 3))
        g = torch.randint(0, 5, (2, 3))
        m = torch.ones(2, 3, dtype=torch.bool)
        cov = torch.tensor([[1.0], [0.0]])
        with torch.no_grad():
            logits, _ = model(f, p, g, m, covariates=cov)
        assert logits.shape == (2, 1)


# ---------------------------------------------------------------------------
# Tests: ChunkedSIEVEModel with sex covariates (end-to-end)
# ---------------------------------------------------------------------------

class TestChunkedModelSexCovariate:
    @pytest.fixture
    def device(self):
        return torch.device('cpu')

    @pytest.fixture
    def chunked_model_with_cov(self):
        base = SIEVE(input_dim=10, num_genes=100, latent_dim=8, num_covariates=1)
        return ChunkedSIEVEModel(base, aggregation_method='mean')

    @pytest.fixture
    def chunked_model_no_cov(self):
        base = SIEVE(input_dim=10, num_genes=100, latent_dim=8, num_covariates=0)
        return ChunkedSIEVEModel(base, aggregation_method='mean')

    @pytest.fixture
    def sample_batch_with_sex(self, device):
        batch_size = 5
        num_variants = 10
        feature_dim = 10
        return {
            'features': torch.randn(batch_size, num_variants, feature_dim),
            'positions': torch.randint(0, 1000000, (batch_size, num_variants)),
            'gene_ids': torch.randint(0, 100, (batch_size, num_variants)),
            'mask': torch.ones(batch_size, num_variants, dtype=torch.bool),
            'labels': torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
            'sex': torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),  # chunk sex
            'chunk_indices': torch.tensor([0, 1, 0, 1, 2], dtype=torch.long),
            'total_chunks': torch.tensor([2, 2, 2, 3, 3], dtype=torch.long),
            'original_sample_indices': torch.tensor([0, 0, 1, 2, 2], dtype=torch.long),
        }

    def test_train_step_with_sex(self, chunked_model_with_cov, sample_batch_with_sex, device):
        chunked_model_with_cov.to(device)
        criterion = SIEVELoss(lambda_attr=0.0)
        loss, preds = chunked_model_with_cov.train_step(
            sample_batch_with_sex, criterion, device
        )
        assert loss.requires_grad
        assert preds.shape[0] == 3  # 3 unique samples
        loss.backward()

    def test_train_step_backward_compat_no_sex(self, chunked_model_no_cov, device):
        """Model without covariates still works when batch has sex field."""
        batch = {
            'features': torch.randn(3, 10, 10),
            'positions': torch.randint(0, 100, (3, 10)),
            'gene_ids': torch.randint(0, 100, (3, 10)),
            'mask': torch.ones(3, 10, dtype=torch.bool),
            'labels': torch.tensor([0, 1, 0], dtype=torch.long),
            'sex': torch.tensor([1.0, 0.0, -1.0]),
            'chunk_indices': torch.tensor([0, 0, 0], dtype=torch.long),
            'total_chunks': torch.tensor([1, 1, 1], dtype=torch.long),
            'original_sample_indices': torch.tensor([0, 1, 2], dtype=torch.long),
        }
        criterion = SIEVELoss(lambda_attr=0.0)
        loss, preds = chunked_model_no_cov.train_step(batch, criterion, device)
        assert preds.shape[0] == 3
        loss.backward()

    def test_train_step_no_sex_in_batch(self, chunked_model_with_cov, device):
        """Model with covariates gracefully handles missing sex in batch."""
        batch = {
            'features': torch.randn(2, 10, 10),
            'positions': torch.randint(0, 100, (2, 10)),
            'gene_ids': torch.randint(0, 100, (2, 10)),
            'mask': torch.ones(2, 10, dtype=torch.bool),
            'labels': torch.tensor([0, 1], dtype=torch.long),
            # no 'sex' key
            'chunk_indices': torch.tensor([0, 0], dtype=torch.long),
            'total_chunks': torch.tensor([1, 1], dtype=torch.long),
            'original_sample_indices': torch.tensor([0, 1], dtype=torch.long),
        }
        criterion = SIEVELoss(lambda_attr=0.0)
        # Should not crash, covariates will be None
        loss, preds = chunked_model_with_cov.train_step(batch, criterion, device)
        assert preds.shape[0] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
