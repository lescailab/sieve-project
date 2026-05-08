"""
Tests for the classifier_type switch in SIEVE and create_sieve_model.

Covers:
- Default (flatten) matches explicit flatten construction
- attention_pool produces AttentionPoolingClassifier with small param count
- Forward pass works for both types with covariates
- Invalid value raises ValueError
- Config round-trip via create_sieve_model
- Backward-compat checkpoint load with flatten head (strict=True)
- ChunkedSIEVEModel forward pass with attention_pool head
"""

import io
import pytest
import torch

from src.models.classifier import AttentionPoolingClassifier, PhenotypeClassifier
from src.models.chunked_sieve import ChunkedSIEVEModel
from src.models.sieve import SIEVE, create_sieve_model


NUM_GENES = 10
LATENT_DIM = 8
HIDDEN_DIM = 16
BATCH = 2
NUM_VARIANTS = 20


def _make_sieve(classifier_type="flatten", num_covariates=0, num_genes=NUM_GENES):
    return SIEVE(
        input_dim=1,
        num_genes=num_genes,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_attention_layers=1,
        classifier_hidden_dim=HIDDEN_DIM,
        dropout=0.0,
        num_covariates=num_covariates,
        classifier_type=classifier_type,
    )


def _dummy_batch(num_genes=NUM_GENES, num_covariates=0):
    features = torch.zeros(BATCH, NUM_VARIANTS, 1)
    positions = torch.zeros(BATCH, NUM_VARIANTS, dtype=torch.long)
    gene_ids = torch.zeros(BATCH, NUM_VARIANTS, dtype=torch.long)
    mask = torch.ones(BATCH, NUM_VARIANTS, dtype=torch.bool)
    covariates = None
    if num_covariates > 0:
        covariates = torch.zeros(BATCH, num_covariates)
    return features, positions, gene_ids, mask, covariates


class TestDefaultIsFlatten:
    def test_default_classifier_is_phenotype_classifier(self):
        model = _make_sieve()
        assert isinstance(model.classifier, PhenotypeClassifier)

    def test_default_classifier_type_attr(self):
        model = _make_sieve()
        assert model.classifier_type == "flatten"


class TestExplicitFlatten:
    def test_explicit_flatten_same_class(self):
        model = _make_sieve(classifier_type="flatten")
        assert isinstance(model.classifier, PhenotypeClassifier)
        assert model.classifier_type == "flatten"

    def test_explicit_flatten_param_count_matches_default(self):
        default_model = _make_sieve()
        explicit_model = _make_sieve(classifier_type="flatten")
        default_params = sum(p.numel() for p in default_model.classifier.parameters())
        explicit_params = sum(p.numel() for p in explicit_model.classifier.parameters())
        assert default_params == explicit_params


class TestAttentionPool:
    def test_attention_pool_class(self):
        model = _make_sieve(classifier_type="attention_pool")
        assert isinstance(model.classifier, AttentionPoolingClassifier)
        assert model.classifier_type == "attention_pool"

    def test_attention_pool_small_param_count(self):
        model = SIEVE(
            input_dim=1,
            num_genes=1000,
            latent_dim=32,
            hidden_dim=64,
            num_attention_layers=1,
            classifier_hidden_dim=256,
            dropout=0.0,
            classifier_type="attention_pool",
        )
        cls_params = sum(p.numel() for p in model.classifier.parameters())
        assert cls_params < 20_000, (
            f"attention_pool classifier params should be <20K, got {cls_params}"
        )

    def test_flatten_larger_than_attention_pool(self):
        kwargs = dict(
            input_dim=1, num_genes=1000, latent_dim=32, hidden_dim=64,
            num_attention_layers=1, classifier_hidden_dim=256, dropout=0.0,
        )
        flatten_model = SIEVE(**kwargs, classifier_type="flatten")
        pool_model = SIEVE(**kwargs, classifier_type="attention_pool")
        flatten_cls = sum(p.numel() for p in flatten_model.classifier.parameters())
        pool_cls = sum(p.numel() for p in pool_model.classifier.parameters())
        assert flatten_cls > pool_cls


class TestForwardPass:
    @pytest.mark.parametrize("classifier_type", ["flatten", "attention_pool"])
    def test_forward_with_covariates(self, classifier_type):
        model = _make_sieve(classifier_type=classifier_type, num_covariates=1)
        features, positions, gene_ids, mask, covariates = _dummy_batch(num_covariates=1)
        logits, _ = model(features, positions, gene_ids, mask, covariates=covariates)
        assert logits.shape == (BATCH, 1)

    @pytest.mark.parametrize("classifier_type", ["flatten", "attention_pool"])
    def test_forward_without_covariates(self, classifier_type):
        model = _make_sieve(classifier_type=classifier_type, num_covariates=0)
        features, positions, gene_ids, mask, _ = _dummy_batch()
        logits, _ = model(features, positions, gene_ids, mask)
        assert logits.shape == (BATCH, 1)


class TestInvalidValue:
    def test_unknown_classifier_type_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier_type"):
            SIEVE(
                input_dim=1,
                num_genes=NUM_GENES,
                latent_dim=LATENT_DIM,
                classifier_type="nonsense",
            )


class TestConfigRoundTrip:
    def test_attention_pool_via_create_sieve_model(self):
        config = {
            "input_dim": 1,
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_attention_layers": 1,
            "classifier_hidden_dim": HIDDEN_DIM,
            "classifier_type": "attention_pool",
        }
        model = create_sieve_model(config, num_genes=NUM_GENES)
        assert isinstance(model.classifier, AttentionPoolingClassifier)

    def test_no_key_gives_phenotype_classifier(self):
        config = {
            "input_dim": 1,
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_attention_layers": 1,
            "classifier_hidden_dim": HIDDEN_DIM,
        }
        model = create_sieve_model(config, num_genes=NUM_GENES)
        assert isinstance(model.classifier, PhenotypeClassifier)


class TestCheckpointCompat:
    def test_flatten_strict_load(self):
        model_a = _make_sieve(classifier_type="flatten")
        buf = io.BytesIO()
        torch.save(model_a.state_dict(), buf)
        buf.seek(0)
        state = torch.load(buf, map_location="cpu", weights_only=True)

        model_b = _make_sieve(classifier_type="flatten")
        model_b.load_state_dict(state, strict=True)

    def test_flatten_checkpoint_contains_expected_key(self):
        model = _make_sieve(classifier_type="flatten")
        keys = set(model.state_dict().keys())
        assert any("classifier.classifier.0.weight" in k for k in keys)

    def test_attention_pool_checkpoint_contains_attention_weights(self):
        model = _make_sieve(classifier_type="attention_pool")
        keys = set(model.state_dict().keys())
        assert any("classifier.attention_weights.weight" in k for k in keys)


class TestChunkedSIEVECompat:
    def test_chunked_attention_pool_forward(self):
        base = _make_sieve(classifier_type="attention_pool", num_covariates=0)
        chunked = ChunkedSIEVEModel(base, aggregation_method="mean")

        features = torch.zeros(BATCH, NUM_VARIANTS, 1)
        positions = torch.zeros(BATCH, NUM_VARIANTS, dtype=torch.long)
        gene_ids = torch.zeros(BATCH, NUM_VARIANTS, dtype=torch.long)
        mask = torch.ones(BATCH, NUM_VARIANTS, dtype=torch.bool)
        original_sample_indices = torch.tensor([0, 1])
        chunk_indices = torch.zeros(BATCH, dtype=torch.long)
        total_chunks = torch.ones(BATCH, dtype=torch.long)

        logits, _ = chunked(
            features, positions, gene_ids, mask,
            chunk_indices=chunk_indices,
            total_chunks=total_chunks,
            original_sample_indices=original_sample_indices,
        )
        assert logits.shape[0] == BATCH


class TestModelSummary:
    def test_summary_includes_classifier_type_flatten(self):
        model = _make_sieve(classifier_type="flatten")
        summary = model.get_model_summary()
        assert summary["classifier_type"] == "flatten"

    def test_summary_includes_classifier_type_attention_pool(self):
        model = _make_sieve(classifier_type="attention_pool")
        summary = model.get_model_summary()
        assert summary["classifier_type"] == "attention_pool"
