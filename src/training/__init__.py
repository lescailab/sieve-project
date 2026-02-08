"""
Training infrastructure for SIEVE.

This module provides training loops, loss functions, and cross-validation utilities
for the SIEVE model.

Key components:
- loss: Loss functions including attribution regularization
- trainer: Training loop with checkpointing and early stopping
- validation: Cross-validation utilities

Author: Francesco Lescai
"""

from .loss import SIEVELoss, attribution_sparsity_loss, compute_class_weights
from .trainer import Trainer
from .validation import (
    create_stratified_folds,
    get_train_val_loaders,
    print_fold_stats,
)

__all__ = [
    'SIEVELoss',
    'attribution_sparsity_loss',
    'compute_class_weights',
    'Trainer',
    'create_stratified_folds',
    'get_train_val_loaders',
    'print_fold_stats',
]
