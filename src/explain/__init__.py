"""
Explainability module for SIEVE.

This module provides tools for interpreting SIEVE models and discovering
disease-associated variants through:
- Integrated gradients for variant-level attribution
- Attention weight analysis for epistatic interactions
- SHAP-based epistasis detection and validation
- Variant ranking and prioritization
- Biological validation against databases
- Visualization of discovered variants

Author: Francesco Lescai
"""

from .gradients import IntegratedGradientsExplainer
from .attention_analysis import AttentionAnalyzer
from .variant_ranking import VariantRanker
from .shap_epistasis import SHAPEpistasisDetector
from .biological_validation import BiologicalValidator

__all__ = [
    'IntegratedGradientsExplainer',
    'AttentionAnalyzer',
    'VariantRanker',
    'SHAPEpistasisDetector',
    'BiologicalValidator',
]
