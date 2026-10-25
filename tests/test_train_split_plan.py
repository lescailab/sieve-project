"""Tests for train.py's generic external-split hook (``--split-plan``).

``--split-plan`` only supplies sample-level train/validation indices. SIEVE
applies structural safety checks, never writes or hashes the file, and never
compares it with earlier output state. Without it, training generates the
historical stratified splits.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import train


def _args(
    *,
    cv: int | None = None,
    split_plan: str | Path | None = None,
    seed: int = 42,
    val_split: float = 0.5,
):
    return argparse.Namespace(
        cv=cv,
        val_split=val_split,
        seed=seed,
        split_plan=None if split_plan is None else str(split_plan),
    )


def _labels() -> np.ndarray:
    return np.array([0, 0, 1, 1, 0, 1])


def _write(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _cv_file(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "indices.yaml",
        {
            "mode": "cv",
            "folds": [
                {"train_indices": [5, 1, 3, 2], "val_indices": [0, 4]},
                {"train_indices": [0, 4, 2, 5], "val_indices": [1, 3]},
            ],
        },
    )


def _single_file(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "indices.yaml",
        {"mode": "single_split", "train_indices": [4, 0, 2, 3], "val_indices": [5, 1]},
    )


def _as_lists(splits) -> list[tuple[list[int], list[int]]]:
    return [(train_idx.tolist(), val_idx.tolist()) for train_idx, val_idx in splits]


def _forbid(name: str):
    def forbidden(*args, **kwargs):
        raise AssertionError(f"{name} must not run when --split-plan is supplied")

    return forbidden


def test_parser_help_describes_external_indices():
    parser = train.build_arg_parser()
    help_text = " ".join(parser.format_help().split())

    assert "--split-plan" in help_text
    assert "externally supplied" in help_text
    assert train.parse_args(["--level", "L3"]).split_plan is None


def test_without_split_plan_cv_uses_stratified_folds(monkeypatch):
    expected = [(np.array([1, 3]), np.array([0, 2])), (np.array([0, 2]), np.array([1, 3]))]
    calls = []

    def fake_create_stratified_folds(labels, n_folds, random_state):
        calls.append((labels.copy(), n_folds, random_state))
        return expected

    monkeypatch.setattr(train, "create_stratified_folds", fake_create_stratified_folds)

    splits = train.resolve_training_splits(_args(cv=2, seed=99), _labels())

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], _labels())
    assert calls[0][1:] == (2, 99)
    assert splits is expected


def test_without_split_plan_single_split_uses_stratified_train_test_split(monkeypatch):
    calls = []

    def fake_train_test_split(indices, *, test_size, stratify, random_state):
        calls.append((indices.copy(), test_size, stratify.copy(), random_state))
        return np.array([0, 2]), np.array([1, 3])

    monkeypatch.setattr(train, "train_test_split", fake_train_test_split)

    splits = train.resolve_training_splits(_args(seed=77, val_split=0.25), _labels())

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.arange(6))
    assert calls[0][1] == 0.25
    np.testing.assert_array_equal(calls[0][2], _labels())
    assert calls[0][3] == 77
    assert _as_lists(splits) == [([0, 2], [1, 3])]


def test_without_split_plan_real_stratified_generation_is_seeded_and_unchanged():
    labels = np.array([0, 1] * 10)

    first = train.resolve_training_splits(_args(cv=5, seed=3), labels)
    second = train.resolve_training_splits(_args(cv=5, seed=3), labels)
    reference = train.create_stratified_folds(labels, n_folds=5, random_state=3)

    assert _as_lists(first) == _as_lists(second) == _as_lists(reference)


def test_external_cv_indices_are_used_exactly(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "create_stratified_folds", _forbid("create_stratified_folds"))
    monkeypatch.setattr(train, "train_test_split", _forbid("train_test_split"))

    splits = train.resolve_training_splits(_args(cv=2, split_plan=_cv_file(tmp_path)), _labels())

    assert _as_lists(splits) == [([5, 1, 3, 2], [0, 4]), ([0, 4, 2, 5], [1, 3])]
    assert all(part.dtype == np.int64 for pair in splits for part in pair)


def test_external_single_split_indices_are_used_exactly(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "create_stratified_folds", _forbid("create_stratified_folds"))
    monkeypatch.setattr(train, "train_test_split", _forbid("train_test_split"))

    splits = train.resolve_training_splits(_args(split_plan=_single_file(tmp_path)), _labels())

    assert _as_lists(splits) == [([4, 0, 2, 3], [5, 1])]


def test_seed_and_val_split_do_not_change_external_indices(tmp_path):
    path = _single_file(tmp_path)

    baseline = train.resolve_training_splits(_args(split_plan=path), _labels())
    changed = train.resolve_training_splits(
        _args(split_plan=path, seed=12345, val_split=0.9), _labels()
    )

    assert _as_lists(changed) == _as_lists(baseline)


def test_external_split_writes_no_files(tmp_path, monkeypatch):
    indices_dir = tmp_path / "indices"
    indices_dir.mkdir()
    path = _cv_file(indices_dir)
    working_dir = tmp_path / "cwd"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    before = path.read_bytes()

    train.resolve_training_splits(_args(cv=2, split_plan=path), _labels())

    assert path.read_bytes() == before
    assert sorted(p.name for p in indices_dir.iterdir()) == ["indices.yaml"]
    assert list(working_dir.iterdir()) == []


def test_training_writes_and_validates_no_split_plan_yaml():
    source = inspect.getsource(train._run_training)

    assert source.count("resolve_training_splits(args, labels)") == 1
    assert "split_plan.yaml" not in source
    assert "split_plan_metadata" not in source
    assert "dataset_provenance" not in source
    assert not hasattr(train, "prepare_training_split_plan")


@pytest.mark.parametrize(
    ("content", "cv", "message"),
    [
        ("- 1\n- 2\n", 2, "YAML mapping"),
        ({"mode": "single_split", "train_indices": [0], "val_indices": [1]}, 2, "mode must be"),
        ({"mode": "cv", "folds": []}, 2, "has 0 folds but --cv is 2"),
        ({"mode": "cv", "folds": "x"}, 2, "folds must be a list"),
        ({"mode": "cv", "folds": [[0], [1]]}, 2, "must be a mapping"),
        (
            {
                "mode": "cv",
                "folds": [
                    {"train_indices": [0, 1], "val_indices": [2], "fold_index": 0},
                    {"train_indices": [2], "val_indices": [0, 1]},
                ],
            },
            2,
            "unexpected keys",
        ),
        (
            {"mode": "cv", "seed": 1, "folds": [{"train_indices": [0], "val_indices": [1]}]},
            1,
            "unexpected keys",
        ),
        ({"mode": "single_split", "train_indices": [0, 1]}, None, "val_indices must be"),
        ({"mode": "single_split", "train_indices": [], "val_indices": [1]}, None, "non-empty"),
        ({"mode": "single_split", "train_indices": [0, 6], "val_indices": [1]}, None, "outside"),
        ({"mode": "single_split", "train_indices": [-1], "val_indices": [1]}, None, "outside"),
        ({"mode": "single_split", "train_indices": [0, 0], "val_indices": [1]}, None, "duplicate"),
        ({"mode": "single_split", "train_indices": [0, 1], "val_indices": [1]}, None, "overlap"),
        ({"mode": "single_split", "train_indices": [True], "val_indices": [1]}, None, "integers"),
        ({"mode": "single_split", "train_indices": [0.0], "val_indices": [1]}, None, "integers"),
        ({"mode": "single_split", "train_indices": ["0"], "val_indices": [1]}, None, "integers"),
    ],
)
def test_external_split_structural_validation(tmp_path, content, cv, message):
    path = tmp_path / "indices.yaml"
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        _write(path, content)

    with pytest.raises(ValueError, match=message):
        train.resolve_training_splits(_args(cv=cv, split_plan=path), _labels())


def test_external_split_does_not_validate_scientific_properties(tmp_path):
    # Unstratified folds (all cases in one validation set) are structurally
    # valid; stratification is the caller's responsibility.
    path = _write(
        tmp_path / "indices.yaml",
        {
            "mode": "cv",
            "folds": [
                {"train_indices": [0, 1, 4], "val_indices": [2, 3, 5]},
                {"train_indices": [2, 3, 5], "val_indices": [0, 1, 4]},
            ],
        },
    )

    splits = train.resolve_training_splits(_args(cv=2, split_plan=path), _labels())

    assert _as_lists(splits) == [([0, 1, 4], [2, 3, 5]), ([2, 3, 5], [0, 1, 4])]


def test_fold_info_records_the_indices_actually_used(tmp_path):
    [(train_idx, val_idx), _] = train.resolve_training_splits(
        _args(cv=2, split_plan=_cv_file(tmp_path)), _labels()
    )
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()

    train.save_fold_info(
        fold_dir=fold_dir,
        fold_idx=0,
        n_folds=2,
        seed=42,
        train_indices=train_idx,
        val_indices=val_idx,
        labels=_labels(),
        fold_metrics={"auc": 0.5, "accuracy": 0.5},
        training_started=train.datetime(2026, 1, 1, tzinfo=train.timezone.utc),
        training_completed=train.datetime(2026, 1, 1, tzinfo=train.timezone.utc),
    )

    saved = yaml.safe_load((fold_dir / "fold_info.yaml").read_text(encoding="utf-8"))
    assert saved["train_sample_indices"] == [5, 1, 3, 2]
    assert saved["val_sample_indices"] == [0, 4]
