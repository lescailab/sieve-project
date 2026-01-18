# src/data/__init__.py
"""
Data processing module for SIEVE.

This module handles VCF parsing, annotation extraction, and dataset construction.
"""

from .vcf_parser import (
    VariantRecord,
    SampleVariants,
    harmonize_contig,
    parse_vcf_cyvcf2,
    build_sample_variants,
    load_phenotypes
)

from .annotation import (
    CONSEQUENCE_SEVERITY,
    map_consequence_to_severity,
    normalize_sift_score,
    normalize_polyphen_score,
    impute_missing_score,
    get_lof_variants,
    get_missense_variants,
    get_synonymous_variants,
    compute_annotation_statistics,
    extract_variant_features
)

__all__ = [
    # VCF Parser
    'VariantRecord',
    'SampleVariants',
    'harmonize_contig',
    'parse_vcf_cyvcf2',
    'build_sample_variants',
    'load_phenotypes',

    # Annotation utilities
    'CONSEQUENCE_SEVERITY',
    'map_consequence_to_severity',
    'normalize_sift_score',
    'normalize_polyphen_score',
    'impute_missing_score',
    'get_lof_variants',
    'get_missense_variants',
    'get_synonymous_variants',
    'compute_annotation_statistics',
    'extract_variant_features',
]
