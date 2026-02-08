"""
Annotation extraction and processing utilities for SIEVE.

This module provides functions for extracting and processing specific
annotations from VEP-annotated variants, including:
- Consequence severity mapping
- SIFT and PolyPhen score extraction and normalization
- Missing value imputation
- Annotation quality metrics

Author: Francesco Lescai
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# Consequence severity mapping (used for feature encoding)
CONSEQUENCE_SEVERITY = {
    # HIGH impact (LoF variants)
    'transcript_ablation': 4,
    'splice_acceptor_variant': 4,
    'splice_donor_variant': 4,
    'stop_gained': 4,
    'frameshift_variant': 4,
    'stop_lost': 4,
    'start_lost': 4,
    'transcript_amplification': 4,

    # MODERATE impact (missense, inframe indels)
    'missense_variant': 3,
    'inframe_insertion': 3,
    'inframe_deletion': 3,
    'protein_altering_variant': 3,

    # LOW impact (synonymous, UTR)
    'synonymous_variant': 2,
    'stop_retained_variant': 2,
    'incomplete_terminal_codon_variant': 2,
    'splice_region_variant': 2,
    'splice_polypyrimidine_tract_variant': 2,
    'coding_sequence_variant': 2,
    '5_prime_UTR_variant': 2,
    '3_prime_UTR_variant': 2,
    'start_retained_variant': 2,

    # MODIFIER impact (intron, intergenic, etc.)
    'intron_variant': 1,
    'intergenic_variant': 1,
    'upstream_gene_variant': 1,
    'downstream_gene_variant': 1,
    'non_coding_transcript_variant': 1,
    'non_coding_transcript_exon_variant': 1,
    'mature_miRNA_variant': 1,
    'regulatory_region_variant': 1,
    'TF_binding_site_variant': 1,
    'TFBS_ablation': 1,
    'TFBS_amplification': 1,

    # Unknown
    'unknown': 0,
}


def map_consequence_to_severity(consequence: str) -> int:
    """
    Map VEP consequence term to ordinal severity score.

    Severity scale:
    - 4: HIGH (LoF: stop_gained, frameshift, splice donor/acceptor)
    - 3: MODERATE (missense, inframe indels)
    - 2: LOW (synonymous, UTR variants, splice region)
    - 1: MODIFIER (intron, intergenic, upstream/downstream)
    - 0: unknown

    Handles compound consequences (e.g., "missense_variant&splice_region_variant")
    by taking the maximum severity.

    Parameters
    ----------
    consequence : str
        VEP consequence term (may be compound with & separator)

    Returns
    -------
    int
        Severity score (0-4)

    Examples
    --------
    >>> map_consequence_to_severity('missense_variant')
    3
    >>> map_consequence_to_severity('stop_gained')
    4
    >>> map_consequence_to_severity('missense_variant&splice_region_variant')
    3
    >>> map_consequence_to_severity('intron_variant')
    1
    """
    # Handle compound consequences
    consequences = consequence.split('&')

    max_severity = 0
    for csq in consequences:
        severity = CONSEQUENCE_SEVERITY.get(csq, 0)
        max_severity = max(max_severity, severity)

    return max_severity


def normalize_sift_score(sift_score: Optional[float]) -> Optional[float]:
    """
    Normalize SIFT score to [0, 1] where higher = more deleterious.

    SIFT scores:
    - Original range: [0, 1]
    - Original interpretation: lower = more deleterious (0 = deleterious)
    - Normalized: higher = more deleterious (1 = deleterious)
    - Transformation: normalized = 1 - original

    Parameters
    ----------
    sift_score : Optional[float]
        Raw SIFT score [0, 1] or None

    Returns
    -------
    Optional[float]
        Normalized score [0, 1] where 1 = deleterious, or None

    Examples
    --------
    >>> normalize_sift_score(0.05)  # Deleterious
    0.95
    >>> normalize_sift_score(0.8)   # Tolerated
    0.2
    >>> normalize_sift_score(None)
    None
    """
    if sift_score is None:
        return None

    # Invert so higher = more deleterious
    return 1.0 - sift_score


def normalize_polyphen_score(polyphen_score: Optional[float]) -> Optional[float]:
    """
    Normalize PolyPhen score to [0, 1] where higher = more deleterious.

    PolyPhen scores:
    - Original range: [0, 1]
    - Original interpretation: higher = more deleterious (1 = probably damaging)
    - Already in correct direction, no transformation needed

    Parameters
    ----------
    polyphen_score : Optional[float]
        Raw PolyPhen score [0, 1] or None

    Returns
    -------
    Optional[float]
        Score [0, 1] where 1 = probably damaging, or None

    Examples
    --------
    >>> normalize_polyphen_score(0.999)  # Probably damaging
    0.999
    >>> normalize_polyphen_score(0.05)   # Benign
    0.05
    >>> normalize_polyphen_score(None)
    None
    """
    # PolyPhen already in correct direction
    return polyphen_score


def impute_missing_score(
    score: Optional[float],
    all_scores: List[Optional[float]],
    method: str = 'median'
) -> float:
    """
    Impute missing functional score using cohort statistics.

    Parameters
    ----------
    score : Optional[float]
        Score to impute if None
    all_scores : List[Optional[float]]
        All scores in cohort (for computing statistics)
    method : str
        Imputation method: 'median', 'mean', 'neutral'
        - 'median': Use median of non-missing scores
        - 'mean': Use mean of non-missing scores
        - 'neutral': Use neutral value (0.5)

    Returns
    -------
    float
        Imputed score

    Examples
    --------
    >>> scores = [0.1, 0.3, None, 0.7, None, 0.9]
    >>> impute_missing_score(None, scores, method='median')
    0.5
    >>> impute_missing_score(0.7, scores, method='median')
    0.7
    >>> impute_missing_score(None, scores, method='neutral')
    0.5
    """
    if score is not None:
        return score

    if method == 'neutral':
        return 0.5

    # Filter out None values
    valid_scores = [s for s in all_scores if s is not None]

    if not valid_scores:
        return 0.5  # Default neutral if no valid scores

    if method == 'median':
        return float(np.median(valid_scores))
    elif method == 'mean':
        return float(np.mean(valid_scores))
    else:
        raise ValueError(f"Unknown imputation method: {method}")


def get_lof_variants(variants: List) -> List:
    """
    Filter variants to only high-impact (LoF) variants.

    Parameters
    ----------
    variants : List[VariantRecord]
        List of variant records

    Returns
    -------
    List[VariantRecord]
        Filtered list containing only LoF variants

    Examples
    --------
    >>> from src.data.vcf_parser import VariantRecord
    >>> vars = [
    ...     VariantRecord('1', 100, 'A', 'T', 'GENE1', 'missense_variant', 1, {}),
    ...     VariantRecord('1', 200, 'C', 'G', 'GENE2', 'stop_gained', 1, {}),
    ... ]
    >>> lof = get_lof_variants(vars)
    >>> len(lof)
    1
    >>> lof[0].consequence
    'stop_gained'
    """
    return [
        v for v in variants
        if map_consequence_to_severity(v.consequence) == 4
    ]


def get_missense_variants(variants: List) -> List:
    """
    Filter variants to only missense variants.

    Parameters
    ----------
    variants : List[VariantRecord]
        List of variant records

    Returns
    -------
    List[VariantRecord]
        Filtered list containing only missense variants
    """
    return [
        v for v in variants
        if 'missense_variant' in v.consequence
    ]


def get_synonymous_variants(variants: List) -> List:
    """
    Filter variants to only synonymous variants.

    Parameters
    ----------
    variants : List[VariantRecord]
        List of variant records

    Returns
    -------
    List[VariantRecord]
        Filtered list containing only synonymous variants
    """
    return [
        v for v in variants
        if 'synonymous_variant' in v.consequence
    ]


def compute_annotation_statistics(variants: List) -> Dict[str, any]:
    """
    Compute summary statistics for variant annotations in a cohort.

    Parameters
    ----------
    variants : List[VariantRecord]
        All variant records from all samples

    Returns
    -------
    Dict[str, any]
        Dictionary containing:
        - 'n_variants': Total number of variants
        - 'n_lof': Number of LoF variants
        - 'n_missense': Number of missense variants
        - 'n_synonymous': Number of synonymous variants
        - 'sift_available': Fraction with SIFT scores
        - 'polyphen_available': Fraction with PolyPhen scores
        - 'sift_median': Median SIFT score
        - 'polyphen_median': Median PolyPhen score

    Examples
    --------
    >>> from src.data.vcf_parser import VariantRecord
    >>> vars = [
    ...     VariantRecord('1', 100, 'A', 'T', 'G1', 'missense_variant', 1,
    ...                   {'sift': 0.05, 'polyphen': 0.9}),
    ...     VariantRecord('1', 200, 'C', 'G', 'G2', 'stop_gained', 1, {}),
    ... ]
    >>> stats = compute_annotation_statistics(vars)
    >>> stats['n_variants']
    2
    >>> stats['n_lof']
    1
    >>> stats['sift_available']
    0.5
    """
    n_variants = len(variants)

    if n_variants == 0:
        return {
            'n_variants': 0,
            'n_lof': 0,
            'n_missense': 0,
            'n_synonymous': 0,
            'sift_available': 0.0,
            'polyphen_available': 0.0,
            'sift_median': None,
            'polyphen_median': None,
        }

    # Count by consequence type
    n_lof = len(get_lof_variants(variants))
    n_missense = len(get_missense_variants(variants))
    n_synonymous = len(get_synonymous_variants(variants))

    # SIFT availability
    sift_scores = [
        v.annotations.get('sift')
        for v in variants
        if v.annotations.get('sift') is not None
    ]
    sift_available = len(sift_scores) / n_variants if n_variants > 0 else 0.0
    sift_median = float(np.median(sift_scores)) if sift_scores else None

    # PolyPhen availability
    polyphen_scores = [
        v.annotations.get('polyphen')
        for v in variants
        if v.annotations.get('polyphen') is not None
    ]
    polyphen_available = len(polyphen_scores) / n_variants if n_variants > 0 else 0.0
    polyphen_median = float(np.median(polyphen_scores)) if polyphen_scores else None

    return {
        'n_variants': n_variants,
        'n_lof': n_lof,
        'n_missense': n_missense,
        'n_synonymous': n_synonymous,
        'sift_available': sift_available,
        'polyphen_available': polyphen_available,
        'sift_median': sift_median,
        'polyphen_median': polyphen_median,
    }


def extract_variant_features(
    variant,
    include_sift: bool = True,
    include_polyphen: bool = True,
    sift_impute_value: float = 0.5,
    polyphen_impute_value: float = 0.5
) -> Dict[str, any]:
    """
    Extract all relevant features from a VariantRecord for model input.

    Parameters
    ----------
    variant : VariantRecord
        Variant record to extract features from
    include_sift : bool
        Whether to include SIFT score
    include_polyphen : bool
        Whether to include PolyPhen score
    sift_impute_value : float
        Value to use for missing SIFT scores
    polyphen_impute_value : float
        Value to use for missing PolyPhen scores

    Returns
    -------
    Dict[str, any]
        Feature dictionary with keys:
        - 'chrom': Chromosome
        - 'pos': Position
        - 'gene': Gene symbol
        - 'consequence': Consequence string
        - 'consequence_severity': Ordinal severity (0-4)
        - 'genotype': Dosage (0, 1, 2)
        - 'sift': SIFT score (if included)
        - 'polyphen': PolyPhen score (if included)

    Examples
    --------
    >>> from src.data.vcf_parser import VariantRecord
    >>> var = VariantRecord('1', 100, 'A', 'T', 'BRCA1', 'missense_variant', 1,
    ...                     {'sift': 0.05, 'polyphen': 0.9})
    >>> features = extract_variant_features(var)
    >>> features['consequence_severity']
    3
    >>> features['sift']
    0.95
    """
    features = {
        'chrom': variant.chrom,
        'pos': variant.pos,
        'gene': variant.gene,
        'consequence': variant.consequence,
        'consequence_severity': map_consequence_to_severity(variant.consequence),
        'genotype': variant.genotype,
    }

    if include_sift:
        sift_raw = variant.annotations.get('sift')
        if sift_raw is not None:
            features['sift'] = normalize_sift_score(sift_raw)
        else:
            features['sift'] = sift_impute_value

    if include_polyphen:
        polyphen_raw = variant.annotations.get('polyphen')
        if polyphen_raw is not None:
            features['polyphen'] = normalize_polyphen_score(polyphen_raw)
        else:
            features['polyphen'] = polyphen_impute_value

    return features
