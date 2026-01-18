"""
Phenotype classifier for SIEVE.

This module implements the final classification head that predicts phenotype
(case vs control) from gene-level embeddings.

Author: Lescai Lab
"""

import torch
import torch.nn as nn
from torch import Tensor


class PhenotypeClassifier(nn.Module):
    """
    Binary phenotype classifier from gene embeddings.

    Takes gene-level embeddings and produces a single logit for binary
    classification (case vs control).

    Architecture:
        Flatten gene embeddings
        → Linear(num_genes * latent_dim → hidden_dim)
        → ReLU
        → Dropout
        → Linear(hidden_dim → 1)

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

    Examples
    --------
    >>> classifier = PhenotypeClassifier(num_genes=100, latent_dim=64, hidden_dim=256)
    >>> gene_emb = torch.randn(2, 100, 64)  # [batch, genes, latent_dim]
    >>> logits = classifier(gene_emb)
    >>> logits.shape
    torch.Size([2, 1])
    """

    def __init__(
        self,
        num_genes: int,
        latent_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        input_dim = num_genes * latent_dim

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),  # Flatten gene embeddings
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gene_embeddings: Tensor) -> Tensor:
        """
        Predict phenotype from gene embeddings.

        Parameters
        ----------
        gene_embeddings : Tensor
            Gene embeddings, shape (batch, num_genes, latent_dim)

        Returns
        -------
        Tensor
            Logits, shape (batch, 1)
            Use with BCEWithLogitsLoss (includes sigmoid)
        """
        return self.classifier(gene_embeddings)


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

    Examples
    --------
    >>> classifier = AttentionPoolingClassifier(num_genes=100, latent_dim=64)
    >>> gene_emb = torch.randn(2, 100, 64)
    >>> logits = classifier(gene_emb)
    >>> logits.shape
    torch.Size([2, 1])
    """

    def __init__(
        self,
        num_genes: int,
        latent_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Attention pooling
        self.attention_weights = nn.Linear(latent_dim, 1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gene_embeddings: Tensor) -> Tensor:
        """
        Predict phenotype using attention pooling.

        Parameters
        ----------
        gene_embeddings : Tensor
            Gene embeddings, shape (batch, num_genes, latent_dim)

        Returns
        -------
        Tensor
            Logits, shape (batch, 1)
        """
        batch_size = gene_embeddings.shape[0]

        # Compute attention weights
        # Shape: (batch, num_genes, 1)
        attn_scores = self.attention_weights(gene_embeddings)

        # Softmax over genes
        # Shape: (batch, num_genes, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum over genes
        # Shape: (batch, latent_dim)
        pooled = (gene_embeddings * attn_weights).sum(dim=1)

        # Classify
        # Shape: (batch, 1)
        logits = self.classifier(pooled)

        return logits
