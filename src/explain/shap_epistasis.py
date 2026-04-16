"""
Deprecated: module kept for backward compatibility.

Import :class:`~src.explain.counterfactual_epistasis.CounterfactualEpistasisDetector`
directly from :mod:`src.explain.counterfactual_epistasis` instead.
"""

import warnings

from .counterfactual_epistasis import CounterfactualEpistasisDetector

warnings.warn(
    "src.explain.shap_epistasis is deprecated. "
    "Import CounterfactualEpistasisDetector from "
    "src.explain.counterfactual_epistasis instead.",
    DeprecationWarning,
    stacklevel=2,
)

#: Deprecated alias for :class:`CounterfactualEpistasisDetector`.
SHAPEpistasisDetector = CounterfactualEpistasisDetector
