"""
Phenotype classifier for SIEVE.

This module implements the final classification head that predicts phenotype
(case vs control) from gene-level embeddings, with optional covariate support
(e.g. sex) to control for confounders.

Author: Lescai Lab
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class PhenotypeClassifier(nn.Module):
    """
    Binary phenotype classifier from gene embeddings.

    Takes gene-level embeddings (and optional covariates) and produces a single
    logit for binary classification (case vs control).

    Architecture:
        Flatten gene embeddings
        → Linear(num_genes * latent_dim [+ num_covariates] → hidden_dim)
        → ReLU
        → Dropout
        → Linear(hidden_dim → 1)

    When ``num_covariates > 0``, covariate values are concatenated to the
    flattened gene embeddings before the first linear layer.  This is the
    standard approach for adjusting deep-learning phenotype classifiers for
    confounders such as biological sex.

    Parameters
    ----------
    num_genes : int
        Number of genes
    latent_dim : int
        Dimension of gene embeddings
    hidden_dim : int
        Hidden layer dimension (default: 256)
    dropout : float
        Dropout probability (default: 0.3)
    num_covariates : int
        Number of additional covariates concatenated before classification
        (default: 0, backward-compatible)

    Examples
    --------
    >>> classifier = PhenotypeClassifier(num_genes=100, latent_dim=64, num_covariates=1)
    >>> gene_emb = torch.randn(2, 100, 64)
    >>> covariates = torch.tensor([[1.0], [0.0]])  # sex
    >>> logits = classifier(gene_emb, covariates=covariates)
    >>> logits.shape
    torch.Size([2, 1])
    """

    def __init__(
        self,
        num_genes: int,
        latent_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_covariates: int = 0,
    ):
        super().__init__()

        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_covariates = num_covariates

        input_dim = num_genes * latent_dim + num_covariates

        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        gene_embeddings: Tensor,
        covariates: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict phenotype from gene embeddings.

        Parameters
        ----------
        gene_embeddings : Tensor
            Gene embeddings, shape (batch, num_genes, latent_dim)
        covariates : Tensor, optional
            Sample-level covariates, shape (batch, num_covariates).
            Required when ``num_covariates > 0``.

        Returns
        -------
        Tensor
            Logits, shape (batch, 1)
            Use with BCEWithLogitsLoss (includes sigmoid)
        """
        x = self.flatten(gene_embeddings)
        if self.num_covariates > 0:
            if covariates is not None:
                if covariates.dim() != 2:
                    raise ValueError(
                        f"covariates must be 2D (batch, num_covariates), got {covariates.shape}"
                    )
                if covariates.shape[0] != x.shape[0]:
                    raise ValueError(
                        "covariates batch dimension must match gene_embeddings batch size "
                        f"({covariates.shape[0]} vs {x.shape[0]})"
                    )
                if covariates.shape[1] != self.num_covariates:
                    raise ValueError(
                        "covariates feature dimension must match num_covariates "
                        f"({covariates.shape[1]} vs {self.num_covariates})"
                    )
                x = torch.cat([x, covariates], dim=1)
            else:
                # Pad with zeros when covariates expected but not provided
                zeros = torch.zeros(
                    x.shape[0], self.num_covariates,
                    dtype=x.dtype, device=x.device,
                )
                x = torch.cat([x, zeros], dim=1)
        return self.classifier(x)


class AttentionPoolingClassifier(nn.Module):
    """
    Alternative classifier using attention pooling over genes.

    Instead of flattening, this uses attention to weight gene importance
    before classification. This can be more parameter-efficient for large
    numbers of genes.

    Parameters
    ----------
    num_genes : int
        Number of genes
    latent_dim : int
        Dimension of gene embeddings
    hidden_dim : int
        Hidden layer dimension (default: 256)
    dropout : float
        Dropout probability (default: 0.3)
    num_covariates : int
        Number of additional covariates concatenated before classification
        (default: 0, backward-compatible)

    Examples
    --------
    >>> classifier = AttentionPoolingClassifier(num_genes=100, latent_dim=64, num_covariates=1)
    >>> gene_emb = torch.randn(2, 100, 64)
    >>> covariates = torch.tensor([[1.0], [0.0]])
    >>> logits = classifier(gene_emb, covariates=covariates)
    >>> logits.shape
    torch.Size([2, 1])
    """

    def __init__(
        self,
        num_genes: int,
        latent_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_covariates: int = 0,
    ):
        super().__init__()

        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_covariates = num_covariates

        # Attention pooling
        self.attention_weights = nn.Linear(latent_dim, 1)

        # Classifier input = pooled embedding + optional covariates
        classifier_input_dim = latent_dim + num_covariates

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        gene_embeddings: Tensor,
        covariates: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict phenotype using attention pooling.

        Parameters
        ----------
        gene_embeddings : Tensor
            Gene embeddings, shape (batch, num_genes, latent_dim)
        covariates : Tensor, optional
            Sample-level covariates, shape (batch, num_covariates).

        Returns
        -------
        Tensor
            Logits, shape (batch, 1)
        """
        # Compute attention weights
        # Shape: (batch, num_genes, 1)
        attn_scores = self.attention_weights(gene_embeddings)

        # Softmax over genes
        # Shape: (batch, num_genes, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum over genes
        # Shape: (batch, latent_dim)
        pooled = (gene_embeddings * attn_weights).sum(dim=1)

        # Concatenate covariates if provided
        if self.num_covariates > 0:
            if covariates is not None:
                if covariates.dim() != 2:
                    raise ValueError(
                        f"covariates must be 2D (batch, num_covariates), got {covariates.shape}"
                    )
                if covariates.shape[0] != pooled.shape[0]:
                    raise ValueError(
                        "covariates batch dimension must match gene_embeddings batch size "
                        f"({covariates.shape[0]} vs {pooled.shape[0]})"
                    )
                if covariates.shape[1] != self.num_covariates:
                    raise ValueError(
                        "covariates feature dimension must match num_covariates "
                        f"({covariates.shape[1]} vs {self.num_covariates})"
                    )
                pooled = torch.cat([pooled, covariates], dim=1)
            else:
                zeros = torch.zeros(
                    pooled.shape[0], self.num_covariates,
                    dtype=pooled.dtype, device=pooled.device,
                )
                pooled = torch.cat([pooled, zeros], dim=1)

        # Classify
        # Shape: (batch, 1)
        logits = self.classifier(pooled)

        return logits
