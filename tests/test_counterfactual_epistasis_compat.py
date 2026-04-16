"""
Compatibility tests for the CounterfactualEpistasisDetector rename.
"""

import importlib
import sys
import warnings

from src.explain import CounterfactualEpistasisDetector


def test_new_export_is_available():
    """The new detector name is exported from src.explain."""
    assert CounterfactualEpistasisDetector.__name__ == "CounterfactualEpistasisDetector"


def test_deprecated_import_still_works_with_warning():
    """The legacy module path still works and emits DeprecationWarning."""
    sys.modules.pop("src.explain.shap_epistasis", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module("src.explain.shap_epistasis")

    assert module.SHAPEpistasisDetector is CounterfactualEpistasisDetector
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
