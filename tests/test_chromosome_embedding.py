"""
Tests for the chromosome-aware and chromosome-blind attention builds.

The chromosome-aware model (default) carries a chromosome embedding and a
cross-chromosome row in the relative-position bias. ``--no-chromosome-embedding``
builds the chromosome-blind attention of earlier checkpoints, which must load
strictly and bucket every pair by coordinate difference.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scripts import train as train_script
from scripts.train import _run_training
from src.models.attention import PositionAwareSparseAttention
from src.models.chunked_sieve import ChunkedSIEVEModel
from src.models.sieve import SIEVE, create_sieve_model, resolve_chromosome_aware
from test_split_file import _labels, _make_samples, _parse

NUM_HEADS = 2
NUM_BUCKETS = 32


def _has_chrom_embedding(state_dict) -> bool:
    return any('chrom_embedding' in key for key in state_dict)


def _bias_rows(state_dict) -> set:
    return {
        tensor.shape for key, tensor in state_dict.items()
        if key.endswith('position_bias.weight')
    }


class TestAttentionLayer:
    def test_default_layer_is_chromosome_aware(self):
        layer = PositionAwareSparseAttention(latent_dim=8, num_heads=NUM_HEADS, num_chromosomes=3)
        assert layer.chrom_embedding is not None
        assert layer.position_bias.weight.shape == (NUM_BUCKETS + 1, NUM_HEADS)

    def test_blind_layer_matches_earlier_architecture(self):
        layer = PositionAwareSparseAttention(
            latent_dim=8, num_heads=NUM_HEADS, num_chromosomes=3, chromosome_aware=False,
        )
        assert layer.chrom_embedding is None
        assert layer.position_bias.weight.shape == (NUM_BUCKETS, NUM_HEADS)

    def test_blind_layer_buckets_cross_chromosome_pairs_by_position(self):
        """Same coordinate on two chromosomes gets the same bias as on one chromosome."""
        torch.manual_seed(0)
        layer = PositionAwareSparseAttention(
            latent_dim=8, num_heads=NUM_HEADS, chromosome_aware=False, dropout=0.0,
        ).eval()
        with torch.no_grad():
            layer.position_bias.weight.copy_(torch.randn_like(layer.position_bias.weight))
        x = torch.randn(1, 3, 8)
        positions = torch.tensor([[100, 100, 5000]])
        same_chrom = torch.tensor([[0, 0, 0]])
        diff_chrom = torch.tensor([[0, 1, 2]])

        _, attn_same = layer(x, positions, return_attention=True, chrom_ids=same_chrom)
        _, attn_diff = layer(x, positions, return_attention=True, chrom_ids=diff_chrom)
        _, attn_none = layer(x, positions, return_attention=True)
        torch.testing.assert_close(attn_same, attn_diff)
        torch.testing.assert_close(attn_same, attn_none)

    def test_aware_layer_separates_cross_chromosome_pairs(self):
        torch.manual_seed(0)
        layer = PositionAwareSparseAttention(
            latent_dim=8, num_heads=NUM_HEADS, dropout=0.0,
        ).eval()
        with torch.no_grad():
            layer.position_bias.weight.copy_(torch.randn_like(layer.position_bias.weight))
        x = torch.randn(1, 3, 8)
        positions = torch.tensor([[100, 100, 5000]])
        _, attn_same = layer(x, positions, return_attention=True, chrom_ids=torch.tensor([[0, 0, 0]]))
        _, attn_diff = layer(x, positions, return_attention=True, chrom_ids=torch.tensor([[0, 1, 2]]))
        assert not torch.allclose(attn_same, attn_diff)


class TestCheckpointRebuild:
    @staticmethod
    def _config(**extra):
        config = {'input_dim': 10, 'latent_dim': 8, 'hidden_dim': 8,
                  'num_heads': NUM_HEADS, 'num_attention_layers': 1,
                  'num_chromosomes': 3}
        config.update(extra)
        return config

    def test_blind_checkpoint_loads_strictly_without_config_key(self):
        """A chromosome-blind checkpoint whose config predates the key is rebuilt exactly."""
        blind = ChunkedSIEVEModel(
            SIEVE(input_dim=10, num_genes=4, latent_dim=8, hidden_dim=8,
                  num_heads=NUM_HEADS, num_attention_layers=1, chromosome_aware=False),
        )
        state_dict = blind.state_dict()
        config = self._config()  # no 'chromosome_aware' entry
        assert resolve_chromosome_aware(config, state_dict) is False

        config['chromosome_aware'] = resolve_chromosome_aware(config, state_dict)
        rebuilt = ChunkedSIEVEModel(create_sieve_model(config, num_genes=4))
        rebuilt.load_state_dict(state_dict, strict=True)

    def test_aware_checkpoint_resolves_to_aware(self):
        aware = SIEVE(input_dim=10, num_genes=4, latent_dim=8, hidden_dim=8,
                      num_heads=NUM_HEADS, num_attention_layers=1, num_chromosomes=3)
        assert resolve_chromosome_aware(self._config(), aware.state_dict()) is True

    def test_aware_checkpoint_without_embedding_resolves_to_aware(self):
        """33-row bias without a chromosome embedding is still the aware layout."""
        aware = SIEVE(input_dim=10, num_genes=4, latent_dim=8, hidden_dim=8,
                      num_heads=NUM_HEADS, num_attention_layers=1)
        assert resolve_chromosome_aware(self._config(), aware.state_dict()) is True

    def test_config_key_takes_precedence(self):
        aware = SIEVE(input_dim=10, num_genes=4, latent_dim=8, hidden_dim=8,
                      num_heads=NUM_HEADS, num_attention_layers=1)
        config = self._config(chromosome_aware=False)
        assert resolve_chromosome_aware(config, aware.state_dict()) is False


@pytest.fixture
def preprocessed(tmp_path):
    path = tmp_path / 'preprocessed.pt'
    torch.save({'samples': _make_samples(_labels()), 'metadata': {}}, path)
    return path


class TestTrainingBuilds:
    def _run(self, monkeypatch, tmp_path, preprocessed, extra):
        out = tmp_path / 'out' / 'run'
        out.mkdir(parents=True)
        args = _parse(monkeypatch, [
            '--level', 'L3', '--preprocessed-data', str(preprocessed),
            '--epochs', '3', '--lr', '0.01', '--batch-size', '4', '--latent-dim', '8',
            '--hidden-dim', '8', '--num-heads', str(NUM_HEADS), '--num-attention-layers', '1',
            '--device', 'cpu', '--class-weighting', 'off', *extra,
        ])
        args.experiment_name = 'run'
        train_script.set_seed(args.seed)
        _run_training(args, out)
        return out

    @staticmethod
    def _state(path: Path):
        return torch.load(path, weights_only=False)['model_state_dict']

    @staticmethod
    def _yaml(path: Path):
        with open(path) as f:
            return yaml.safe_load(f)

    @pytest.mark.parametrize('flag, aware', [([], True), (['--no-chromosome-embedding'], False)])
    def test_single_split(self, monkeypatch, tmp_path, preprocessed, flag, aware):
        out = self._run(monkeypatch, tmp_path, preprocessed, ['--val-split', '0.34', *flag])
        state = self._state(out / 'best_model.pt')
        assert _has_chrom_embedding(state) is aware
        assert _bias_rows(state) == {(NUM_BUCKETS + aware, NUM_HEADS)}
        assert self._yaml(out / 'config.yaml')['chromosome_aware'] is aware
        assert self._yaml(out / 'results.yaml')['chromosome_aware'] is aware

    @pytest.mark.parametrize('flag, aware', [([], True), (['--no-chromosome-embedding'], False)])
    def test_cross_validation(self, monkeypatch, tmp_path, preprocessed, flag, aware):
        out = self._run(monkeypatch, tmp_path, preprocessed, ['--cv', '2', *flag])
        assert self._yaml(out / 'config.yaml')['chromosome_aware'] is aware
        assert self._yaml(out / 'cv_results.yaml')['chromosome_aware'] is aware
        for fold in ('fold_0', 'fold_1'):
            state = self._state(out / fold / 'best_model.pt')
            assert _has_chrom_embedding(state) is aware
            assert _bias_rows(state) == {(NUM_BUCKETS + aware, NUM_HEADS)}
            assert self._yaml(out / fold / 'config.yaml')['chromosome_aware'] is aware
