"""
Explainability module for SIEVE.

This module provides tools for interpreting SIEVE models and discovering
disease-associated variants through:
- Integrated gradients for variant-level attribution
- Attention weight analysis for epistatic interactions
- Variant ranking and prioritization
- Visualization of discovered variants

Author: Lescai Lab
"""

from .gradients import IntegratedGradientsExplainer
from .attention_analysis import AttentionAnalyzer
from .variant_ranking import VariantRanker

__all__ = [
    'IntegratedGradientsExplainer',
    'AttentionAnalyzer',
    'VariantRanker',
]
