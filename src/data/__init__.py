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

__all__ = [
    'VariantRecord',
    'SampleVariants', 
    'harmonize_contig',
    'parse_vcf_cyvcf2',
    'build_sample_variants',
    'load_phenotypes'
]
