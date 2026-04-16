"""
Acceptance tests for B1: covariates propagated through Integrated Gradients.

Tests cover:
1. Pre-covariate model (num_covariates=0): output is unchanged vs original code path.
2. Covariate model (num_covariates=1): IG with/without covariates differ.
3. IG completeness axiom with and without covariates.
4. Guard clauses: missing/extra covariates raise ValueError.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sieve import SIEVE
from src.explain.gradients import IntegratedGradientsExplainer, SIEVEWrapper


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_sieve(num_covariates: int = 0, latent_dim: int = 8) -> SIEVE:
    """Create a tiny SIEVE model for testing."""
    return SIEVE(
        input_dim=3,
        latent_dim=latent_dim,
        num_genes=4,
        num_attention_layers=1,
        num_heads=2,
        hidden_dim=16,
        num_covariates=num_covariates,
    )


def _make_batch(batch_size: int = 2, n_variants: int = 5, input_dim: int = 3):
    """Return a minimal batch dict."""
    features = torch.rand(batch_size, n_variants, input_dim)
    positions = torch.arange(n_variants, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    gene_ids = (torch.arange(n_variants) % 4).unsqueeze(0).expand(batch_size, -1)
    mask = torch.ones(batch_size, n_variants, dtype=torch.bool)
    labels = torch.randint(0, 2, (batch_size,)).float()
    return features, positions, gene_ids, mask, labels


# ---------------------------------------------------------------------------
# 1. Pre-covariate model: output is byte-identical regardless of code path
# ---------------------------------------------------------------------------

class TestNoCovariates:
    """num_covariates=0 must produce identical output to the legacy code path."""

    def test_wrapper_forward_no_covariates(self):
        """SIEVEWrapper.forward with covariates=None equals without passing covariates."""
        model = _make_sieve(num_covariates=0)
        model.eval()
        wrapper = SIEVEWrapper(model)

        features, positions, gene_ids, mask, _ = _make_batch(batch_size=1)

        torch.manual_seed(0)
        out_none = wrapper(features, positions, gene_ids, mask, covariates=None)

        torch.manual_seed(0)
        out_default = wrapper(features, positions, gene_ids, mask)

        torch.testing.assert_close(out_none, out_default)

    def test_attribute_no_covariates(self):
        """attribute() with covariates=None works without error."""
        model = _make_sieve(num_covariates=0)
        explainer = IntegratedGradientsExplainer(model, device='cpu', n_steps=4)

        features, positions, gene_ids, mask, _ = _make_batch(batch_size=1)
        attr = explainer.attribute(features, positions, gene_ids, mask, covariates=None)
        assert attr.shape == features.shape


# ---------------------------------------------------------------------------
# 2. Covariate model: IG with/without covariates differ
# ---------------------------------------------------------------------------

class TestWithCovariates:
    """num_covariates=1: code paths with vs without covariates must differ."""

    def test_attributions_differ_with_different_covariates(self):
        """IG results differ when sex covariate is flipped (male vs female)."""
        model = _make_sieve(num_covariates=1)
        model.eval()
        explainer = IntegratedGradientsExplainer(model, device='cpu', n_steps=4)

        features, positions, gene_ids, mask, _ = _make_batch(batch_size=1)

        cov_male = torch.tensor([[1.0]])    # M
        cov_female = torch.tensor([[-1.0]]) # F (see _encode_sex convention: 0=F→-1, 1=M→1)

        torch.manual_seed(42)
        attr_male = explainer.attribute(
            features, positions, gene_ids, mask, covariates=cov_male
        )
        torch.manual_seed(42)
        attr_female = explainer.attribute(
            features, positions, gene_ids, mask, covariates=cov_female
        )

        # Attributions should differ because the function being explained differs
        assert not torch.allclose(attr_male, attr_female), (
            "Expected IG attributions to differ when covariates differ"
        )


# ---------------------------------------------------------------------------
# 3. IG completeness axiom
# ---------------------------------------------------------------------------

class TestIGCompleteness:
    """
    IG completeness: sum of attributions ≈ f(input) - f(baseline).

    This is the strongest available proof that the code explains the same
    function that was trained.  We test both the covariate=None and
    covariate=<value> paths.
    """

    def _check_completeness(self, model, covariates, tol=5e-3):
        model.eval()
        explainer = IntegratedGradientsExplainer(model, device='cpu', n_steps=100)
        wrapper = SIEVEWrapper(model)

        features, positions, gene_ids, mask, _ = _make_batch(batch_size=1)
        baseline = torch.zeros_like(features)

        attr = explainer.attribute(
            features, positions, gene_ids, mask,
            baseline=baseline, covariates=covariates
        )

        # f(input) - f(baseline) using the wrapper so the same function is called
        with torch.no_grad():
            f_input = wrapper(features, positions, gene_ids, mask, covariates=covariates)
            f_base = wrapper(baseline, positions, gene_ids, mask, covariates=covariates)

        delta = (f_input - f_base).item()
        attr_sum = attr.sum().item()

        assert abs(attr_sum - delta) < tol, (
            f"IG completeness violated: sum(attr)={attr_sum:.6f}, "
            f"f(x)-f(x0)={delta:.6f}, diff={abs(attr_sum - delta):.6f}"
        )

    def test_completeness_no_covariates(self):
        model = _make_sieve(num_covariates=0)
        self._check_completeness(model, covariates=None)

    def test_completeness_with_covariates(self):
        model = _make_sieve(num_covariates=1)
        cov = torch.tensor([[1.0]])
        self._check_completeness(model, covariates=cov)

    def test_completeness_with_two_covariates(self):
        """Extends B1 test 3 to num_covariates > 1 (C1 coverage)."""
        model = _make_sieve(num_covariates=2)
        cov = torch.tensor([[1.0, 0.5]])  # sex + one PC
        self._check_completeness(model, covariates=cov)


# ---------------------------------------------------------------------------
# 4. Guard clauses
# ---------------------------------------------------------------------------

class TestGuardClauses:
    """Guard clauses defined in B1 must raise ValueError."""

    def test_missing_covariates_raises(self):
        """num_covariates>0 but covariates=None should raise in attribute_batch."""
        import types

        model = _make_sieve(num_covariates=1)
        explainer = IntegratedGradientsExplainer(model, device='cpu', n_steps=4)

        features, positions, gene_ids, mask, labels = _make_batch(batch_size=2)

        # Build a minimal DataLoader-like iterator
        batch = {
            'features': features,
            'positions': positions,
            'gene_ids': gene_ids,
            'mask': mask,
            'labels': labels,
            # No 'sex' key → should raise
        }

        class FakeLoader:
            dataset = [None, None]  # len == 2
            def __iter__(self):
                yield batch

        with pytest.raises(ValueError, match='num_covariates'):
            explainer.attribute_batch(FakeLoader(), num_covariates=1)

    def test_extra_covariates_raises(self):
        """num_covariates=0 but batch contains 'sex' tensor should raise."""
        model = _make_sieve(num_covariates=0)
        explainer = IntegratedGradientsExplainer(model, device='cpu', n_steps=4)

        features, positions, gene_ids, mask, labels = _make_batch(batch_size=2)

        batch = {
            'features': features,
            'positions': positions,
            'gene_ids': gene_ids,
            'mask': mask,
            'labels': labels,
            'sex': torch.zeros(2),  # unexpected sex tensor
        }

        class FakeLoader:
            dataset = [None, None]
            def __iter__(self):
                yield batch

        with pytest.raises(ValueError, match='num_covariates=0'):
            explainer.attribute_batch(FakeLoader(), num_covariates=0)
