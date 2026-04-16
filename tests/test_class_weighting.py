"""
Acceptance tests for B2: class weighting in train.py.

Tests cover:
1. Balanced labels + 'auto': pos_weight is None (no change to loss).
2. Imbalanced labels + 'auto': pos_weight is applied and logged.
3. '--class-weighting off' always disables.
4. '--class-weighting on' always enables.
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import _resolve_pos_weight, save_fold_config


class TestResolveposWeight:
    """Unit tests for _resolve_pos_weight helper."""

    # ---- balanced cohort ----

    def test_balanced_auto_returns_none(self):
        """Balanced cohort (case fraction 0.5) with 'auto' → no weighting."""
        labels = np.array([0, 1] * 50)  # exactly 0.5
        result = _resolve_pos_weight(labels, 'auto')
        assert result is None

    def test_balanced_near_threshold_auto_returns_none(self):
        """Case fraction 0.45 (inside [0.4, 0.6]) → no weighting."""
        n = 100
        n_cases = 45
        labels = np.array([1] * n_cases + [0] * (n - n_cases))
        result = _resolve_pos_weight(labels, 'auto')
        assert result is None

    # ---- imbalanced cohort ----

    def test_imbalanced_auto_returns_tensor(self, capsys):
        """Imbalanced cohort (case fraction 0.2) with 'auto' → returns weight."""
        n = 100
        n_cases = 20
        labels = np.array([1] * n_cases + [0] * (n - n_cases))
        result = _resolve_pos_weight(labels, 'auto')
        assert result is not None
        assert isinstance(result, torch.Tensor)
        # pos_weight = n_total / (2 * n_positive) = 100 / 40 = 2.5
        assert abs(result.item() - 2.5) < 1e-5

    def test_imbalanced_auto_logs_message(self, capsys):
        """Imbalanced cohort + 'auto' prints the applied pos_weight."""
        labels = np.array([1] * 10 + [0] * 90)
        _resolve_pos_weight(labels, 'auto')
        captured = capsys.readouterr()
        assert 'pos_weight' in captured.out

    # ---- explicit on/off ----

    def test_off_always_returns_none(self):
        """'off' disables weighting even for very imbalanced cohort."""
        labels = np.array([1] * 5 + [0] * 95)
        assert _resolve_pos_weight(labels, 'off') is None

    def test_on_balanced_still_returns_weight(self):
        """'on' enables weighting even for a balanced cohort."""
        labels = np.array([0, 1] * 50)
        result = _resolve_pos_weight(labels, 'on')
        assert result is not None
        # For exactly balanced: weight = 1.0
        assert abs(result.item() - 1.0) < 1e-5

    def test_on_imbalanced_returns_expected_weight(self):
        """'on' with imbalanced data returns expected inverse-frequency weight."""
        labels = np.array([1] * 30 + [0] * 70)
        result = _resolve_pos_weight(labels, 'on')
        assert result is not None
        expected = 100 / (2 * 30)  # ≈ 1.667
        assert abs(result.item() - expected) < 1e-4

    def test_fold_config_records_class_weighting_keys(self, tmp_path):
        """Fold config.yaml includes the documented class-weighting metadata."""
        args = argparse.Namespace(
            experiment_name="test",
            level="L3",
            latent_dim=32,
            hidden_dim=64,
            num_attention_layers=1,
            num_heads=4,
            chunk_size=3000,
            chunk_overlap=0,
            aggregation_method="mean",
            lr=1e-5,
            lambda_attr=0.1,
            batch_size=16,
            gradient_accumulation_steps=1,
            gradient_clip=None,
            early_stopping=10,
            epochs=5,
            seed=42,
            genome_build="GRCh37",
            sex_map=None,
            pc_map=None,
            num_pcs=0,
            pc_map_sha256=None,
            preprocessed_data="/tmp/preprocessed.pt",
            vcf=None,
            phenotypes=None,
            _fold_pos_weight=torch.tensor(2.5),
        )
        fold_dir = tmp_path / "fold_0"
        fold_dir.mkdir()
        save_fold_config(fold_dir, fold_idx=0, args=args)
        with open(fold_dir / "config.yaml") as handle:
            config = yaml.safe_load(handle)
        assert "class_weighting_applied" in config
        assert "class_weighting_pos_weight" in config
        assert config["class_weighting_applied"] is True
        assert abs(config["class_weighting_pos_weight"] - 2.5) < 1e-6
