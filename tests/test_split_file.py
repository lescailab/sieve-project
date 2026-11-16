"""
Tests for reusing a recorded train/validation split via ``--split-file``.

A ``fold_info.yaml`` written by ``save_fold_info`` is fed back to
``train.py`` so that a single-split run trains on exactly the partition of an
earlier cross-validation fold.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import train as train_script
from scripts.train import (
    _run_training,
    _single_split_indices,
    load_split_file,
    save_fold_info,
)
from src.data.covariates import compute_file_sha256
from src.data.vcf_parser import SampleVariants, VariantRecord


N_SAMPLES = 12
TRAIN_IDX = [0, 2, 3, 5, 6, 8, 9, 11]
VAL_IDX = [1, 4, 7, 10]


def _labels() -> np.ndarray:
    """Alternating case/control labels; the fixed split has 2+2 in validation."""
    return np.array([1, 0] * (N_SAMPLES // 2))


def _write_fold_info(path: Path, train_idx, val_idx, labels=None) -> Path:
    """Write a fold_info.yaml with the same writer used by CV training."""
    path.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    save_fold_info(
        fold_dir=path,
        fold_idx=4,
        n_folds=5,
        seed=42,
        train_indices=list(train_idx),
        val_indices=list(val_idx),
        labels=_labels() if labels is None else labels,
        fold_metrics={'auc': 0.6, 'accuracy': 0.6},
        training_started=now,
        training_completed=now,
    )
    return path / 'fold_info.yaml'


def _edit_fold_info(path: Path, **updates) -> None:
    with open(path) as f:
        info = yaml.safe_load(f)
    info.update(updates)
    with open(path, 'w') as f:
        yaml.dump(info, f, sort_keys=False)


def _parse(monkeypatch, argv):
    monkeypatch.setattr(sys, 'argv', ['train.py', *argv])
    return train_script.parse_args()


def _make_samples(labels: np.ndarray):
    samples = []
    for i, label in enumerate(labels):
        variants = [
            VariantRecord(
                chrom=chrom,
                pos=1000 * (j + 1) + i,
                ref='A',
                alt='T',
                gene=gene,
                consequence='missense_variant',
                genotype=2 if label else 1,  # separable, so validation AUC > 0
                annotations={'sift': 0.05 * j, 'polyphen': 0.1 * j},
            )
            for j, (chrom, gene) in enumerate([('1', 'GENE1'), ('1', 'GENE1'), ('2', 'GENE2')])
        ]
        samples.append(SampleVariants(f'S{i}', label=int(label), variants=variants))
    return samples


@pytest.fixture
def split_file(tmp_path):
    return _write_fold_info(tmp_path / 'cv_run' / 'fold_4', TRAIN_IDX, VAL_IDX)


@pytest.fixture
def preprocessed(tmp_path):
    path = tmp_path / 'preprocessed.pt'
    torch.save({'samples': _make_samples(_labels()), 'metadata': {}}, path)
    return path


class TestLoadSplitFile:
    def test_returns_recorded_indices(self, split_file):
        train_idx, val_idx = load_split_file(str(split_file), N_SAMPLES)
        assert train_idx.tolist() == TRAIN_IDX
        assert val_idx.tolist() == VAL_IDX

    def test_overlapping_indices_raise(self, tmp_path):
        path = _write_fold_info(tmp_path / 'f', TRAIN_IDX + [1], VAL_IDX)
        with pytest.raises(ValueError, match='both'):
            load_split_file(str(path), N_SAMPLES)

    def test_incomplete_coverage_raises(self, tmp_path):
        path = _write_fold_info(tmp_path / 'f', TRAIN_IDX[:-1], VAL_IDX)
        with pytest.raises(ValueError, match='cover'):
            load_split_file(str(path), N_SAMPLES)

    def test_dataset_size_mismatch_raises(self, split_file):
        with pytest.raises(ValueError, match='cover'):
            load_split_file(str(split_file), N_SAMPLES + 1)

    def test_size_differing_from_file_raises(self, split_file):
        _edit_fold_info(split_file, n_val_samples=len(VAL_IDX) + 1)
        with pytest.raises(ValueError, match='n_val_samples'):
            load_split_file(str(split_file), N_SAMPLES)

    def test_label_permuted_dataset_loads_same_split(self, split_file):
        """Case counts per partition change under permutation; the split does not."""
        args = type('Args', (), {'split_file': str(split_file)})()
        permuted = np.random.default_rng(0).permutation(_labels())
        permuted[VAL_IDX] = 0  # validation case count now differs from the file
        train_idx, val_idx = _single_split_indices(permuted, args)
        assert train_idx.tolist() == TRAIN_IDX
        assert val_idx.tolist() == VAL_IDX


class TestDefaultSplit:
    def test_without_split_file_matches_train_test_split(self, monkeypatch):
        from sklearn.model_selection import train_test_split

        labels = np.array([1] * 20 + [0] * 30)
        args = _parse(monkeypatch, ['--level', 'L3', '--preprocessed-data', 'x.pt',
                                    '--val-split', '0.2', '--seed', '7'])
        assert args.split_file is None
        train_idx, val_idx = _single_split_indices(labels, args)
        exp_train, exp_val = train_test_split(
            np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=7,
        )
        np.testing.assert_array_equal(train_idx, exp_train)
        np.testing.assert_array_equal(val_idx, exp_val)


class TestArgumentValidation:
    def test_split_file_with_cv_raises(self, monkeypatch, split_file):
        monkeypatch.setattr(sys, 'argv', [
            'train.py', '--level', 'L3', '--preprocessed-data', 'x.pt',
            '--cv', '5', '--split-file', str(split_file),
        ])
        with pytest.raises(ValueError, match='--cv'):
            train_script.main()

    def test_missing_split_file_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, 'argv', [
            'train.py', '--level', 'L3', '--preprocessed-data', 'x.pt',
            '--split-file', str(tmp_path / 'absent.yaml'),
        ])
        with pytest.raises(FileNotFoundError):
            train_script.main()


class TestTrainingRun:
    def _run(self, monkeypatch, tmp_path, preprocessed, extra):
        out = tmp_path / 'out' / 'run'
        out.mkdir(parents=True)
        args = _parse(monkeypatch, [
            '--level', 'L3', '--preprocessed-data', str(preprocessed),
            '--epochs', '3', '--lr', '0.01', '--batch-size', '4', '--latent-dim', '8',
            '--hidden-dim', '8', '--num-heads', '2', '--num-attention-layers', '1',
            '--device', 'cpu', '--class-weighting', 'off', *extra,
        ])
        args.experiment_name = 'run'
        train_script.set_seed(args.seed)
        _run_training(args, out)
        return out

    def test_config_and_results_record_split_file(self, monkeypatch, tmp_path,
                                                  preprocessed, split_file, capsys):
        out = self._run(monkeypatch, tmp_path, preprocessed,
                        ['--split-file', str(split_file.resolve())])
        sha = compute_file_sha256(str(split_file))

        with open(out / 'config.yaml') as f:
            config = yaml.safe_load(f)
        assert config['split_file'] == str(split_file.resolve())
        assert config['split_file_sha256'] == sha

        with open(out / 'results.yaml') as f:
            results = yaml.safe_load(f)
        assert results['split_file'] == str(split_file.resolve())
        assert results['split_file_sha256'] == sha

        log = capsys.readouterr().out
        assert f'Val:   {len(VAL_IDX)} samples (2 cases, 2 controls' in log

        state = torch.load(out / 'best_model.pt', weights_only=False)['model_state_dict']
        assert any('chrom_embedding' in k for k in state)

    def test_run_without_split_file_records_null(self, monkeypatch, tmp_path, preprocessed):
        out = self._run(monkeypatch, tmp_path, preprocessed, ['--val-split', '0.34'])
        with open(out / 'config.yaml') as f:
            config = yaml.safe_load(f)
        assert config['split_file'] is None
        assert config['split_file_sha256'] is None
        with open(out / 'results.yaml') as f:
            results = yaml.safe_load(f)
        assert 'split_file' not in results
