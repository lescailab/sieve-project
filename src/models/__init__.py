"""
Model architecture module for SIEVE.

This module contains all neural network components for SIEVE:
- VariantEncoder: Feature projection
- PositionAwareSparseAttention: Core innovation (position-aware attention)
- GeneAggregator: Variant to gene-level aggregation
- PhenotypeClassifier: Binary classification head
- SIEVE: Complete end-to-end model

Author: Lescai Lab
"""

from .encoder import VariantEncoder
from .attention import PositionAwareSparseAttention, MultiLayerAttention
from .aggregation import GeneAggregator, EfficientGeneAggregator
from .classifier import PhenotypeClassifier, AttentionPoolingClassifier
from .sieve import SIEVE, create_sieve_model
from .chunked_sieve import ChunkedSIEVEModel

__all__ = [
    # Encoder
    'VariantEncoder',

    # Attention
    'PositionAwareSparseAttention',
    'MultiLayerAttention',

    # Aggregation
    'GeneAggregator',
    'EfficientGeneAggregator',

    # Classifier
    'PhenotypeClassifier',
    'AttentionPoolingClassifier',

    # Full model
    'SIEVE',
    'create_sieve_model',

    # Chunked processing
    'ChunkedSIEVEModel',
]
