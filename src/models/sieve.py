"""
SIEVE: Sparse Interpretable Exome Variant Explainer

This module implements the complete SIEVE model that combines:
1. Variant encoding (feature projection)
2. Position-aware sparse attention (the innovation)
3. Gene aggregation (variant → gene level)
4. Phenotype classification (case vs control)

Author: Lescai Lab
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .encoder import VariantEncoder
from .attention import MultiLayerAttention
from .aggregation import EfficientGeneAggregator
from .classifier import PhenotypeClassifier


class SIEVE(nn.Module):
    """
    Complete SIEVE model for variant discovery.

    This integrates all components into a single end-to-end model:
    1. Encode variant features to latent space
    2. Apply position-aware attention layers
    3. Aggregate variants to gene level
    4. Classify phenotype

    Parameters
    ----------
    input_dim : int
        Input feature dimension (depends on annotation level)
    num_genes : int
        Number of unique genes in dataset
    latent_dim : int
        Latent embedding dimension (default: 64)
    hidden_dim : int
        Hidden layer dimension for encoder (default: 128)
    num_heads : int
        Number of attention heads (default: 4)
    num_attention_layers : int
        Number of attention layers (default: 2)
    classifier_hidden_dim : int
        Hidden dimension for classifier (default: 256)
    dropout : float
        Dropout probability (default: 0.1)
    aggregation : str
        Gene aggregation method: 'max', 'mean', 'sum' (default: 'max')
    num_position_buckets : int
        Number of relative position buckets (default: 32)
    max_distance : int
        Maximum genomic distance for bucketing (default: 100,000 bp)

    Attributes
    ----------
    variant_encoder : VariantEncoder
        Encodes variant features to latent space
    attention : MultiLayerAttention
        Position-aware attention layers
    gene_aggregator : EfficientGeneAggregator
        Aggregates variants to genes
    classifier : PhenotypeClassifier
        Predicts phenotype from gene embeddings

    Examples
    --------
    >>> model = SIEVE(input_dim=71, num_genes=100, latent_dim=64)
    >>> features = torch.randn(2, 50, 71)  # [batch, variants, features]
    >>> positions = torch.randint(100, 10000, (2, 50))
    >>> gene_ids = torch.randint(0, 100, (2, 50))
    >>> mask = torch.ones(2, 50, dtype=torch.bool)
    >>> logits, attn_weights = model(features, positions, gene_ids, mask, return_attention=True)
    >>> logits.shape
    torch.Size([2, 1])
    """

    def __init__(
        self,
        input_dim: int,
        num_genes: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_attention_layers: int = 2,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.1,
        aggregation: str = 'max',
        num_position_buckets: int = 32,
        max_distance: int = 100000,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_genes = num_genes
        self.latent_dim = latent_dim

        # 1. Variant encoder
        self.variant_encoder = VariantEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # 2. Position-aware attention
        self.attention = MultiLayerAttention(
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_attention_layers,
            dropout=dropout,
            num_position_buckets=num_position_buckets,
            max_distance=max_distance,
        )

        # 3. Gene aggregation
        self.gene_aggregator = EfficientGeneAggregator(
            num_genes=num_genes,
            latent_dim=latent_dim,
            aggregation=aggregation,
        )

        # 4. Phenotype classifier
        self.classifier = PhenotypeClassifier(
            num_genes=num_genes,
            latent_dim=latent_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        variant_features: Tensor,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_intermediate: bool = False,
        return_embeddings: bool = False
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Forward pass through SIEVE model.

        Parameters
        ----------
        variant_features : Tensor
            Variant features, shape (batch, num_variants, input_dim)
        positions : Tensor
            Genomic positions, shape (batch, num_variants)
        gene_ids : Tensor
            Gene assignments, shape (batch, num_variants)
        mask : Optional[Tensor]
            Validity mask, shape (batch, num_variants)
            True = real variant, False = padding
        return_attention : bool
            Whether to return attention weights (for explainability)
        return_intermediate : bool
            Whether to return intermediate representations
        return_embeddings : bool
            If True, return gene embeddings instead of logits (for chunked aggregation)

        Returns
        -------
        output : Tensor
            If return_embeddings=False: Phenotype prediction logits, shape (batch, 1)
            If return_embeddings=True: Gene embeddings, shape (batch, num_genes, latent_dim)
        intermediates : Optional[Dict]
            Dictionary containing intermediate outputs if requested:
            - 'variant_embeddings': After encoder
            - 'attended_embeddings': After attention
            - 'gene_embeddings': After aggregation
            - 'attention_weights': Attention weights from each layer (if return_attention=True)
        """
        intermediates = {} if (return_attention or return_intermediate or return_embeddings) else None

        # 1. Encode variants
        variant_embeddings = self.variant_encoder(variant_features)
        if return_intermediate or return_embeddings:
            intermediates['variant_embeddings'] = variant_embeddings

        # 2. Apply attention
        attended_embeddings, attention_weights = self.attention(
            variant_embeddings,
            positions,
            mask,
            return_attention=return_attention
        )
        if return_intermediate or return_embeddings:
            intermediates['attended_embeddings'] = attended_embeddings
        if return_attention:
            intermediates['attention_weights'] = attention_weights

        # 3. Aggregate to genes
        gene_embeddings = self.gene_aggregator(
            attended_embeddings,
            gene_ids,
            mask
        )
        if return_intermediate or return_embeddings:
            intermediates['gene_embeddings'] = gene_embeddings

        # 4. Return embeddings or classify
        if return_embeddings:
            return gene_embeddings, intermediates
        else:
            logits = self.classifier(gene_embeddings)
            return logits, intermediates

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of model architecture.

        Returns
        -------
        Dict[str, Any]
            Model configuration and parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'input_dim': self.input_dim,
            'num_genes': self.num_genes,
            'latent_dim': self.latent_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_params': sum(p.numel() for p in self.variant_encoder.parameters()),
            'attention_params': sum(p.numel() for p in self.attention.parameters()),
            'aggregator_params': sum(p.numel() for p in self.gene_aggregator.parameters()),
            'classifier_params': sum(p.numel() for p in self.classifier.parameters()),
        }

    def get_attention_patterns(
        self,
        variant_features: Tensor,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """
        Extract attention patterns for explainability.

        Parameters
        ----------
        variant_features : Tensor
            Variant features, shape (batch, num_variants, input_dim)
        positions : Tensor
            Genomic positions, shape (batch, num_variants)
        gene_ids : Tensor
            Gene assignments, shape (batch, num_variants)
        mask : Optional[Tensor]
            Validity mask, shape (batch, num_variants)

        Returns
        -------
        List[Tensor]
            Attention weights from each layer
            Each tensor: (batch, num_heads, num_variants, num_variants)
        """
        with torch.no_grad():
            _, intermediates = self.forward(
                variant_features,
                positions,
                gene_ids,
                mask,
                return_attention=True
            )
            return intermediates['attention_weights']


def create_sieve_model(
    config: Dict,
    num_genes: int
) -> SIEVE:
    """
    Create SIEVE model from configuration dictionary.

    Parameters
    ----------
    config : Dict
        Configuration dictionary with model hyperparameters
    num_genes : int
        Number of unique genes in dataset

    Returns
    -------
    SIEVE
        Initialized SIEVE model

    Examples
    --------
    >>> config = {
    ...     'input_dim': 71,
    ...     'latent_dim': 64,
    ...     'num_heads': 4,
    ...     'num_attention_layers': 2,
    ... }
    >>> model = create_sieve_model(config, num_genes=100)
    """
    return SIEVE(
        input_dim=config.get('input_dim'),
        num_genes=num_genes,
        latent_dim=config.get('latent_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_heads=config.get('num_heads', 4),
        num_attention_layers=config.get('num_attention_layers', 2),
        classifier_hidden_dim=config.get('classifier_hidden_dim', 256),
        dropout=config.get('dropout', 0.1),
        aggregation=config.get('aggregation', 'max'),
        num_position_buckets=config.get('num_position_buckets', 32),
        max_distance=config.get('max_distance', 100000),
    )
