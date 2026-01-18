"""
Loss functions for SIEVE training.

This module implements the loss functions for training SIEVE models:
1. Binary cross-entropy for case-control classification
2. Attribution sparsity regularization for interpretability

The combined loss encourages the model to:
- Accurately predict phenotype (classification loss)
- Rely on a sparse set of variants (attribution sparsity)

Author: Lescai Lab
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


class SIEVELoss(nn.Module):
    """
    Combined loss for SIEVE training.

    Loss = classification_loss + λ_attr * attribution_sparsity_loss

    The classification loss is binary cross-entropy with logits.
    The attribution sparsity loss encourages the model to focus on
    a small number of informative variants.

    Args:
        lambda_attr: Weight for attribution sparsity term (default: 0.0)
        pos_weight: Positive class weight for imbalanced datasets (default: None)

    Attributes:
        lambda_attr: Attribution regularization weight
        bce_loss: Binary cross-entropy loss function
    """

    def __init__(
        self,
        lambda_attr: float = 0.0,
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.lambda_attr = lambda_attr
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        variant_embeddings: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Model predictions [batch_size, 1]
            labels: True labels [batch_size]
            variant_embeddings: Variant embeddings for attribution [batch_size, num_variants, latent_dim]
                Required if lambda_attr > 0
            mask: Valid variant mask [batch_size, num_variants]
                Required if lambda_attr > 0

        Returns:
            Dictionary containing:
                - 'total': Total loss (scalar)
                - 'classification': Classification loss (scalar)
                - 'attribution_sparsity': Attribution sparsity loss (scalar, 0 if lambda_attr=0)

        Raises:
            ValueError: If lambda_attr > 0 but variant_embeddings or mask not provided
        """
        # Compute classification loss
        logits = logits.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        labels = labels.float()
        classification_loss = self.bce_loss(logits, labels)

        # Initialize total loss
        total_loss = classification_loss
        attr_loss = torch.tensor(0.0, device=logits.device)

        # Add attribution sparsity if requested
        if self.lambda_attr > 0:
            if variant_embeddings is None or mask is None:
                raise ValueError(
                    "variant_embeddings and mask required when lambda_attr > 0"
                )

            attr_loss = attribution_sparsity_loss(
                variant_embeddings=variant_embeddings,
                logits=logits,
                mask=mask,
            )
            total_loss = total_loss + self.lambda_attr * attr_loss

        return {
            'total': total_loss,
            'classification': classification_loss,
            'attribution_sparsity': attr_loss,
        }


def attribution_sparsity_loss(
    variant_embeddings: Tensor,
    logits: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    Compute attribution sparsity loss.

    This loss encourages the model to rely on a sparse set of variants
    by penalizing the sum of L2-norm magnitudes of variant embeddings.

    NOTE: This is a simplified version for Phase 1. Full gradient-based
    attribution regularization will be implemented in Phase 2.

    Args:
        variant_embeddings: Variant embeddings [batch_size, num_variants, latent_dim]
        logits: Model predictions [batch_size] (unused in simplified version)
        mask: Valid variant mask [batch_size, num_variants]

    Returns:
        Mean L1 norm of embedding magnitudes across batch (scalar)

    Note:
        Phase 1 uses simple L1 regularization on embeddings.
        Phase 2 will implement true gradient-based attribution sparsity.
    """
    # Compute L2 norm of embeddings for each variant
    # embedding_magnitudes shape: [batch_size, num_variants]
    embedding_magnitudes = torch.norm(variant_embeddings, p=2, dim=-1)

    # Mask out invalid variants
    embedding_magnitudes = embedding_magnitudes * mask.float()

    # Compute L1 sparsity penalty (sum of magnitudes)
    # Normalize by number of valid variants per sample
    num_valid_variants = mask.sum(dim=1).float().clamp(min=1.0)  # Avoid division by zero
    sparsity_per_sample = embedding_magnitudes.sum(dim=1) / num_valid_variants

    # Average across batch
    return sparsity_per_sample.mean()


def compute_class_weights(labels: Tensor) -> Tensor:
    """
    Compute class weights for imbalanced datasets.

    Uses inverse frequency weighting:
        weight_pos = n_total / (2 * n_positive)

    Args:
        labels: Binary labels [num_samples]

    Returns:
        Positive class weight (scalar tensor)

    Example:
        >>> labels = torch.tensor([0, 0, 0, 1])  # 3 controls, 1 case
        >>> weight = compute_class_weights(labels)
        >>> weight
        tensor(2.0)  # 4 / (2 * 1) = 2.0
    """
    n_total = len(labels)
    n_positive = labels.sum().item()
    n_negative = n_total - n_positive

    if n_positive == 0 or n_negative == 0:
        # Perfectly balanced or missing one class
        return torch.tensor(1.0)

    # Inverse frequency weighting
    pos_weight = n_total / (2.0 * n_positive)

    return torch.tensor(pos_weight)
