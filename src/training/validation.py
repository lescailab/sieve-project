"""
Cross-validation utilities for SIEVE.

This module provides utilities for k-fold cross-validation with stratification
to preserve case-control ratios in each fold.

Key functions:
- create_stratified_folds: Create stratified k-fold splits
- get_train_val_loaders: Create train/val DataLoaders from fold indices

Author: Francesco Lescai
"""

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from src.encoding import VariantDataset, collate_samples


def create_stratified_folds(
    labels: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified k-fold splits.

    Stratification ensures that each fold has approximately the same
    proportion of cases and controls as the full dataset.

    Args:
        labels: Binary labels [num_samples]
        n_folds: Number of folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        List of (train_indices, val_indices) tuples for each fold

    Example:
        >>> labels = np.array([0, 0, 1, 1, 0, 1])
        >>> folds = create_stratified_folds(labels, n_folds=3)
        >>> len(folds)
        3
        >>> train_idx, val_idx = folds[0]
        >>> len(train_idx) + len(val_idx)
        6
    """
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    folds = []
    indices = np.arange(len(labels))

    for train_idx, val_idx in skf.split(indices, labels):
        folds.append((train_idx, val_idx))

    return folds


def get_train_val_loaders(
    dataset: VariantDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    max_variants_per_batch: int = 3000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from fold indices.

    Args:
        dataset: Full VariantDataset
        train_indices: Indices for training set
        val_indices: Indices for validation set
        batch_size: Batch size (default: 32)
        num_workers: Number of workers for data loading (default: 0)

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> dataset = VariantDataset(...)
        >>> train_idx = np.array([0, 1, 2, 3])
        >>> val_idx = np.array([4, 5])
        >>> train_loader, val_loader = get_train_val_loaders(
        ...     dataset, train_idx, val_idx, batch_size=2
        ... )
        >>> len(train_loader)
        2
        >>> len(val_loader)
        1
    """
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create custom collate function with max_variants limit
    def collate_fn(batch):
        return collate_samples(batch, max_variants_per_batch=max_variants_per_batch)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )

    return train_loader, val_loader


def print_fold_stats(
    fold_idx: int,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
) -> None:
    """
    Print statistics for a cross-validation fold.

    Args:
        fold_idx: Fold index (0-based)
        train_labels: Training labels for this fold
        val_labels: Validation labels for this fold
    """
    train_cases = train_labels.sum()
    train_controls = len(train_labels) - train_cases
    train_ratio = train_cases / len(train_labels) if len(train_labels) > 0 else 0

    val_cases = val_labels.sum()
    val_controls = len(val_labels) - val_cases
    val_ratio = val_cases / len(val_labels) if len(val_labels) > 0 else 0

    print(f"Fold {fold_idx + 1}:")
    print(f"  Train: {len(train_labels)} samples "
          f"({train_cases} cases, {train_controls} controls, {train_ratio:.1%} case rate)")
    print(f"  Val:   {len(val_labels)} samples "
          f"({val_cases} cases, {val_controls} controls, {val_ratio:.1%} case rate)")


def get_nested_cv_splits(
    labels: np.ndarray,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]]:
    """
    Create nested cross-validation splits.

    Nested CV is used for hyperparameter tuning without data leakage.
    The outer loop is for model evaluation, the inner loop for hyperparameter selection.

    Args:
        labels: Binary labels [num_samples]
        n_outer_folds: Number of outer folds (default: 5)
        n_inner_folds: Number of inner folds (default: 3)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        List of (outer_train_idx, outer_test_idx, inner_folds) tuples
        where inner_folds is a list of (inner_train_idx, inner_val_idx) tuples

    Example:
        >>> labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        >>> nested_splits = get_nested_cv_splits(labels, n_outer_folds=2, n_inner_folds=2)
        >>> len(nested_splits)
        2
        >>> outer_train, outer_test, inner_folds = nested_splits[0]
        >>> len(inner_folds)
        2
    """
    # Create outer folds
    outer_folds = create_stratified_folds(labels, n_outer_folds, random_state)

    nested_splits = []

    for outer_train_idx, outer_test_idx in outer_folds:
        # Get labels for outer training set
        outer_train_labels = labels[outer_train_idx]

        # Create inner folds from outer training set
        inner_folds = []
        inner_skf = StratifiedKFold(
            n_splits=n_inner_folds,
            shuffle=True,
            random_state=random_state,
        )

        for inner_train_idx, inner_val_idx in inner_skf.split(
            np.arange(len(outer_train_labels)), outer_train_labels
        ):
            # Map inner indices back to original dataset indices
            inner_train_global = outer_train_idx[inner_train_idx]
            inner_val_global = outer_train_idx[inner_val_idx]
            inner_folds.append((inner_train_global, inner_val_global))

        nested_splits.append((outer_train_idx, outer_test_idx, inner_folds))

    return nested_splits
