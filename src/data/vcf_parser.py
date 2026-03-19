"""
VCF Parser for SIEVE Project.

This module provides functionality for parsing multi-sample VCF files annotated
with Ensembl VEP. It handles:
- CSQ field parsing with sanitization
- Contig harmonization (chr1 vs 1)
- Multi-allelic sites
- Multiple transcript annotations per variant

Author: Francesco Lescai
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cyvcf2
import numpy as np


@dataclass
class VariantRecord:
    """
    Represents a single variant for a single sample.

    Attributes
    ----------
    chrom : str
        Chromosome (harmonized, without 'chr' prefix)
    pos : int
        Genomic position (1-based)
    ref : str
        Reference allele
    alt : str
        Alternate allele
    gene : str
        Gene symbol from VEP annotation
    consequence : str
        VEP consequence term (e.g., 'missense_variant')
    genotype : int
        Genotype dosage: 0 (hom-ref), 1 (het), 2 (hom-alt)
    annotations : Dict[str, any]
        Additional annotations (SIFT, PolyPhen, etc.)
    """

    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str
    consequence: str
    genotype: int
    annotations: Dict[str, any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"VariantRecord({self.chrom}:{self.pos} {self.ref}>{self.alt} "
            f"{self.gene} {self.consequence} GT={self.genotype})"
        )


@dataclass
class SampleVariants:
    """
    Contains all variants for a single sample with associated metadata.

    Attributes
    ----------
    sample_id : str
        Sample identifier
    label : int
        Phenotype label: 0=control, 1=case
    variants : List[VariantRecord]
        List of variants carried by this sample
    sex : Optional[str]
        Biological sex: 'M', 'F', or None if unknown.
        Used as a covariate in the classifier when sex imbalance
        exists between cases and controls.
    """

    sample_id: str
    label: int
    variants: List[VariantRecord] = field(default_factory=list)
    sex: Optional[str] = None

    def __repr__(self) -> str:
        sex_str = f", sex={self.sex}" if self.sex else ""
        return (
            f"SampleVariants({self.sample_id}, label={self.label}, "
            f"n_variants={len(self.variants)}{sex_str})"
        )


def harmonize_contig(contig: str) -> str:
    """
    Harmonize chromosome notation by removing 'chr' prefix.

    .. deprecated::
        Use :func:`src.data.genome.normalise_chrom` instead, which also
        handles numeric sex chromosome aliases (23 -> X, 24 -> Y).

    Parameters
    ----------
    contig : str
        Chromosome identifier (e.g., 'chr1', '1', 'chrX', 'X')

    Returns
    -------
    str
        Harmonized contig without 'chr' prefix (e.g., '1', 'X')

    Examples
    --------
    >>> harmonize_contig('chr1')
    '1'
    >>> harmonize_contig('1')
    '1'
    >>> harmonize_contig('chrX')
    'X'
    """
    from src.data.genome import normalise_chrom, get_genome_build
    return normalise_chrom(contig, get_genome_build('GRCh37'))


def parse_csq_field(csq_string: str, alt_allele: str) -> List[Dict[str, str]]:
    """
    Parse VEP CSQ field with critical sanitization.

    The CSQ field requires careful handling:
    1. Remove spaces, quotes, and parentheses that can corrupt parsing
    2. Filter annotations to match the specific alternate allele
    3. Split by comma to get per-transcript annotations
    4. Split each transcript annotation by pipe to get fields

    This implements the critical CSQ parsing fix documented in CLAUDE.md.

    Parameters
    ----------
    csq_string : str
        Raw CSQ field value from VCF INFO
    alt_allele : str
        The alternate allele to filter annotations for

    Returns
    -------
    List[Dict[str, str]]
        List of parsed annotation dictionaries, one per transcript.
        Each dict has field indices as keys (e.g., '0', '1', '3', '36', '37').

    Notes
    -----
    CSQ field format (pipe-delimited):
    - Field 0: Allele
    - Field 1: Consequence
    - Field 3: SYMBOL (gene name)
    - Field 36: SIFT
    - Field 37: PolyPhen
    - ... many more fields

    Examples
    --------
    >>> csq = "A|missense_variant|MODERATE|BRCA1|...|tolerated(0.5)|benign(0.1)"
    >>> parse_csq_field(csq, 'A')
    [{'0': 'A', '1': 'missense_variant', '3': 'BRCA1', ...}]
    """
    # Critical sanitization step from CLAUDE.md
    sanitized = (
        csq_string
        .replace(" ", "")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
    )

    # Split by comma to get per-transcript annotations
    transcript_annotations = [
        x for x in sanitized.split(',')
        if x.startswith(alt_allele + '|')
    ]

    # Parse each transcript annotation
    parsed_annotations = []
    for annotation in transcript_annotations:
        fields = annotation.split('|')
        # Store as dict with field indices
        annotation_dict = {str(i): field_val for i, field_val in enumerate(fields)}
        parsed_annotations.append(annotation_dict)

    return parsed_annotations


def extract_sift_score(sift_string: str) -> Optional[float]:
    """
    Extract SIFT score from VEP annotation string.

    SIFT format: 'prediction(score)' or 'prediction_with_confidence(score)'
    After sanitization (parentheses removed): 'predictionscore'

    Parameters
    ----------
    sift_string : str
        SIFT field from CSQ (after sanitization)

    Returns
    -------
    Optional[float]
        SIFT score (0-1), or None if missing

    Examples
    --------
    >>> extract_sift_score('tolerated0.42')
    0.42
    >>> extract_sift_score('deleterious0.01')
    0.01
    >>> extract_sift_score('')
    None
    """
    if not sift_string or sift_string == '':
        return None

    # After sanitization, format is 'predictionscore'
    # Extract trailing number
    try:
        # Find the last contiguous digit sequence (possibly with decimal point)
        import re
        match = re.search(r'(\d+\.?\d*)$', sift_string)
        if match:
            return float(match.group(1))
    except (ValueError, AttributeError):
        pass

    return None


def extract_polyphen_score(polyphen_string: str) -> Optional[float]:
    """
    Extract PolyPhen score from VEP annotation string.

    PolyPhen format: 'prediction(score)' or 'prediction_with_confidence(score)'
    After sanitization (parentheses removed): 'predictionscore'

    Parameters
    ----------
    polyphen_string : str
        PolyPhen field from CSQ (after sanitization)

    Returns
    -------
    Optional[float]
        PolyPhen score (0-1), or None if missing

    Examples
    --------
    >>> extract_polyphen_score('benign0.001')
    0.001
    >>> extract_polyphen_score('probably_damaging0.999')
    0.999
    >>> extract_polyphen_score('')
    None
    """
    if not polyphen_string or polyphen_string == '':
        return None

    # Same logic as SIFT
    try:
        import re
        match = re.search(r'(\d+\.?\d*)$', polyphen_string)
        if match:
            return float(match.group(1))
    except (ValueError, AttributeError):
        pass

    return None


def select_canonical_annotation(annotations: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Select the most severe/canonical annotation from multiple transcripts.

    Priority (highest to lowest):
    1. Canonical transcript (CANONICAL field = 'YES')
    2. Most severe consequence
    3. First annotation (arbitrary but deterministic)

    Parameters
    ----------
    annotations : List[Dict[str, str]]
        List of parsed CSQ annotations

    Returns
    -------
    Dict[str, str]
        Selected canonical annotation

    Notes
    -----
    Consequence severity ranking (HIGH > MODERATE > LOW > MODIFIER):
    - HIGH: stop_gained, frameshift_variant, splice_acceptor_variant, splice_donor_variant
    - MODERATE: missense_variant, inframe_deletion, inframe_insertion
    - LOW: synonymous_variant, stop_retained_variant
    - MODIFIER: intron_variant, upstream_gene_variant, downstream_gene_variant
    """
    if not annotations:
        return {}

    # Try to find canonical transcript (field 24 in VEP CSQ)
    for ann in annotations:
        if ann.get('24', '') == 'YES':
            return ann

    # Fallback: select by consequence severity
    severity_order = {
        'stop_gained': 4,
        'frameshift_variant': 4,
        'splice_acceptor_variant': 4,
        'splice_donor_variant': 4,
        'start_lost': 4,
        'stop_lost': 4,
        'missense_variant': 3,
        'inframe_deletion': 3,
        'inframe_insertion': 3,
        'protein_altering_variant': 3,
        'synonymous_variant': 2,
        'stop_retained_variant': 2,
        'splice_region_variant': 2,
        'intron_variant': 1,
        'upstream_gene_variant': 1,
        'downstream_gene_variant': 1,
        '5_prime_UTR_variant': 1,
        '3_prime_UTR_variant': 1,
    }

    def get_severity(ann: Dict[str, str]) -> int:
        consequence = ann.get('1', '')
        # Handle compound consequences (e.g., "missense_variant&splice_region_variant")
        consequences = consequence.split('&')
        max_severity = 0
        for csq in consequences:
            max_severity = max(max_severity, severity_order.get(csq, 0))
        return max_severity

    # Sort by severity (descending) and return most severe
    sorted_annotations = sorted(annotations, key=get_severity, reverse=True)
    return sorted_annotations[0]


def load_phenotypes(phenotype_file: Path) -> Dict[str, int]:
    """
    Load phenotype labels from TSV file.

    Expected format:
    - Column 1: sample_id
    - Column 2: phenotype (1=control, 2=case)
    - Tab-delimited
    - No header
    - Comments starting with '#' are ignored

    Converts phenotype encoding from 1/2 to 0/1:
    - Input 1 (control) -> Output 0
    - Input 2 (case) -> Output 1

    Parameters
    ----------
    phenotype_file : Path
        Path to phenotype TSV file

    Returns
    -------
    Dict[str, int]
        Mapping from sample_id to label (0=control, 1=case)

    Raises
    ------
    ValueError
        If phenotype values are not 1 or 2
        If file format is invalid

    Examples
    --------
    >>> phenotypes = load_phenotypes(Path('phenotypes.tsv'))
    >>> phenotypes['sample1']
    1  # case
    >>> phenotypes['sample743']
    0  # control
    """
    phenotypes = {}

    with open(phenotype_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse tab-delimited
            parts = line.split('\t')
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid phenotype file format at line {line_num}: "
                    f"expected 2 columns, got {len(parts)}"
                )

            sample_id = parts[0]
            try:
                pheno_value = int(parts[1])
            except ValueError:
                raise ValueError(
                    f"Invalid phenotype value at line {line_num}: "
                    f"'{parts[1]}' is not an integer"
                )

            # Convert from 1/2 encoding to 0/1 encoding
            if pheno_value == 1:
                label = 0  # control
            elif pheno_value == 2:
                label = 1  # case
            else:
                raise ValueError(
                    f"Invalid phenotype value at line {line_num}: "
                    f"expected 1 or 2, got {pheno_value}"
                )

            phenotypes[sample_id] = label

    return phenotypes


def parse_vcf_cyvcf2(
    vcf_path: Path,
    phenotypes: Dict[str, int],
    genome_build: Optional['GenomeBuild'] = None,
    sex_map: Optional[Dict[str, str]] = None,
    max_variants_per_sample: Optional[int] = None,
    min_gq: int = 20,
) -> Iterator[SampleVariants]:
    """
    Parse multi-sample VCF file and yield SampleVariants for each sample.

    This is a memory-efficient iterator that processes variants one at a time
    and yields complete SampleVariants objects sample-by-sample.

    Parameters
    ----------
    vcf_path : Path
        Path to VCF file (can be .vcf.gz)
    phenotypes : Dict[str, int]
        Mapping from sample_id to phenotype label (0=control, 1=case)
    genome_build : GenomeBuild, optional
        Genome build for contig normalisation and PAR coordinates.
        Defaults to GRCh37 if not provided.
    sex_map : Dict[str, str], optional
        Mapping from sample_id to sex ('M' or 'F'). When provided,
        enables ploidy-aware dosage encoding on sex chromosomes:
        hemizygous males on chrX (non-PAR) get dosage doubled (0/2).
    max_variants_per_sample : Optional[int]
        If specified, limit number of variants per sample (for debugging)
    min_gq : int
        Minimum genotype quality threshold (default: 20)

    Yields
    ------
    SampleVariants
        One SampleVariants object per sample in the VCF

    Notes
    -----
    - Only processes samples present in the phenotypes dictionary
    - Filters out low-quality genotypes (GQ < min_gq)
    - Only includes non-reference genotypes (GT != 0/0)
    - Handles multi-allelic sites correctly
    - Selects canonical transcript annotation per variant
    - When sex_map is provided, applies ploidy correction:
      - Male chrX non-PAR: dosage * 2 (hemizygous alt -> 2)
      - Female chrY: variant skipped (data quality issue)

    Warnings
    --------
    - Samples in VCF but not in phenotypes are skipped with a warning
    - Variants without CSQ annotation are skipped with a warning
    """
    from src.data.genome import (
        get_genome_build,
        is_in_par,
        normalise_chrom,
    )

    if genome_build is None:
        genome_build = get_genome_build('GRCh37')

    vcf = cyvcf2.VCF(str(vcf_path))
    samples = vcf.samples

    # -------------------------------------------------------------------
    # Validate that VCF contains VEP CSQ annotations in the header
    # -------------------------------------------------------------------
    has_csq = False
    for header_line in vcf.raw_header.split('\n'):
        if header_line.startswith('##INFO=<ID=CSQ'):
            has_csq = True
            break

    if not has_csq:
        available_info = [
            line.split('ID=')[1].split(',')[0]
            for line in vcf.raw_header.split('\n')
            if line.startswith('##INFO=<ID=')
        ]
        raise ValueError(
            f"VCF file '{vcf_path}' does not contain VEP CSQ annotations.\n"
            f"SIEVE requires VCF files annotated with Ensembl VEP.\n"
            f"Run VEP before preprocessing:\n"
            f"\n"
            f"  vep --input_file {vcf_path} \\\n"
            f"      --output_file annotated.vcf \\\n"
            f"      --vcf --symbol --canonical --sift b --polyphen b \\\n"
            f"      --assembly GRCh37 --offline --cache\n"
            f"\n"
            f"Available INFO fields in this VCF: {available_info}"
        )

    # Validate that all samples have phenotypes
    missing_phenotypes = [s for s in samples if s not in phenotypes]
    if missing_phenotypes:
        print(f"Warning: {len(missing_phenotypes)} samples in VCF lack phenotype data")
        print(f"  Missing samples: {missing_phenotypes[:5]}...")

    if sex_map:
        n_with_sex = sum(1 for s in samples if s in sex_map)
        print(f"Ploidy-aware mode: {n_with_sex}/{len(samples)} samples have sex info")

    # Initialize storage for each sample
    sample_variants: Dict[str, List[VariantRecord]] = {
        sample: [] for sample in samples if sample in phenotypes
    }

    # Iterate through variants
    variant_count = 0
    csq_missing_count = 0
    csq_allele_mismatch_count = 0
    assigned_count = 0
    ploidy_corrections = 0
    skipped_female_y = 0
    for variant in vcf:
        variant_count += 1

        chrom = normalise_chrom(variant.CHROM, genome_build)
        pos = variant.POS
        ref = variant.REF
        alts = variant.ALT  # List of alternate alleles

        # Pre-compute sex-chromosome flags for this variant
        is_non_par_x = (chrom == 'X') and not is_in_par(pos, 'X', genome_build)
        is_y = (chrom == 'Y')

        # Get CSQ field
        try:
            csq_raw = variant.INFO.get('CSQ')
            if not csq_raw:
                csq_missing_count += 1
                continue  # Skip variants without VEP annotation
        except KeyError:
            csq_missing_count += 1
            continue

        # Process each alternate allele
        for alt_idx, alt in enumerate(alts):
            # Parse CSQ for this specific allele
            csq_annotations = parse_csq_field(csq_raw, alt)

            if not csq_annotations:
                csq_allele_mismatch_count += 1
                continue  # No annotations for this allele

            # Select canonical annotation
            canonical_ann = select_canonical_annotation(csq_annotations)

            # Extract key fields
            gene = canonical_ann.get('3', 'UNKNOWN')
            consequence = canonical_ann.get('1', 'unknown')
            sift_raw = canonical_ann.get('36', '')
            polyphen_raw = canonical_ann.get('37', '')

            # Parse SIFT and PolyPhen scores
            sift_score = extract_sift_score(sift_raw)
            polyphen_score = extract_polyphen_score(polyphen_raw)

            # Process genotypes for each sample
            genotypes = variant.genotypes  # List of [allele1, allele2, phased]
            gq_values = variant.gt_quals  # Genotype qualities

            for sample_idx, sample in enumerate(samples):
                # Skip samples without phenotypes
                if sample not in phenotypes:
                    continue

                # Get sample sex (default to UNKNOWN if no sex_map)
                sample_sex = sex_map.get(sample, 'UNKNOWN') if sex_map else 'UNKNOWN'

                # Skip female Y variants (data quality issue)
                if is_y and sample_sex == 'F':
                    skipped_female_y += 1
                    continue

                # Get genotype for this sample
                gt = genotypes[sample_idx]
                allele1, allele2, phased = gt[0], gt[1], gt[2]

                # Skip if low quality
                if gq_values is not None and gq_values[sample_idx] < min_gq:
                    continue

                # Count dosage of this specific alternate allele
                # alt_idx + 1 because 0 = ref, 1 = first alt, 2 = second alt, etc.
                target_allele = alt_idx + 1
                dosage = 0
                if allele1 == target_allele:
                    dosage += 1
                if allele2 >= 0 and allele2 == target_allele:
                    dosage += 1

                # Apply ploidy correction for hemizygous males on chrX
                if is_non_par_x and sample_sex == 'M':
                    dosage = dosage * 2
                    if dosage > 0:
                        ploidy_corrections += 1

                # Skip reference genotypes
                if dosage == 0:
                    continue

                # Create VariantRecord
                var_record = VariantRecord(
                    chrom=chrom,
                    pos=pos,
                    ref=ref,
                    alt=alt,
                    gene=gene,
                    consequence=consequence,
                    genotype=dosage,
                    annotations={
                        'sift': sift_score,
                        'polyphen': polyphen_score,
                    }
                )

                sample_variants[sample].append(var_record)
                assigned_count += 1

    print(f"Processed {variant_count} variants from VCF")
    print(f"  Variants without CSQ annotation: {csq_missing_count}")
    print(f"  Allele mismatches in CSQ: {csq_allele_mismatch_count}")
    print(f"  Variant-sample assignments: {assigned_count}")
    if sex_map:
        print(f"  Ploidy corrections applied: {ploidy_corrections}")
        print(f"  Female Y variants skipped: {skipped_female_y}")

    # -------------------------------------------------------------------
    # Fail fast if no variants were assigned to any sample
    # -------------------------------------------------------------------
    if assigned_count == 0:
        diagnostics = []
        if csq_missing_count == variant_count:
            diagnostics.append(
                "ALL variants lacked CSQ values despite the header declaring "
                "the field. The VCF may have been filtered or re-header'd "
                "after VEP annotation."
            )
        elif csq_missing_count > 0:
            diagnostics.append(
                f"{csq_missing_count}/{variant_count} variants had no CSQ "
                f"annotation (partial VEP run?)."
            )
        if csq_allele_mismatch_count > 0:
            diagnostics.append(
                f"{csq_allele_mismatch_count} alleles had CSQ annotations "
                f"that did not match the ALT allele. This may indicate an "
                f"allele representation mismatch between VEP and the VCF."
            )
        if csq_missing_count == 0 and csq_allele_mismatch_count == 0:
            diagnostics.append(
                "CSQ annotations were parsed successfully but all genotypes "
                "were either reference (0/0) or below the GQ threshold "
                f"(min_gq={min_gq})."
            )
        raise ValueError(
            f"Preprocessing produced zero variant-sample assignments from "
            f"{variant_count} VCF records.\n"
            + "\n".join(f"  - {d}" for d in diagnostics)
        )

    # Yield SampleVariants for each sample
    for sample in samples:
        if sample not in phenotypes:
            continue

        variants = sample_variants[sample]

        # Apply max_variants limit if specified
        if max_variants_per_sample is not None:
            variants = variants[:max_variants_per_sample]

        # Propagate sex information if available
        sample_sex = None
        if sex_map and sample in sex_map:
            s = sex_map[sample]
            if s in ('M', 'F'):
                sample_sex = s

        yield SampleVariants(
            sample_id=sample,
            label=phenotypes[sample],
            variants=variants,
            sex=sample_sex,
        )


def build_sample_variants(
    vcf_path: Path,
    phenotype_file: Path,
    genome_build: Optional['GenomeBuild'] = None,
    sex_map: Optional[Dict[str, str]] = None,
    max_variants_per_sample: Optional[int] = None,
    min_gq: int = 20,
) -> List[SampleVariants]:
    """
    Convenience function to load VCF and phenotypes, returning list of SampleVariants.

    Parameters
    ----------
    vcf_path : Path
        Path to VCF file
    phenotype_file : Path
        Path to phenotype TSV file
    genome_build : GenomeBuild, optional
        Genome build for contig normalisation and PAR coordinates.
        Defaults to GRCh37.
    sex_map : Dict[str, str], optional
        Mapping from sample_id to sex ('M' or 'F') for ploidy-aware encoding.
    max_variants_per_sample : Optional[int]
        If specified, limit variants per sample
    min_gq : int
        Minimum genotype quality

    Returns
    -------
    List[SampleVariants]
        List of SampleVariants, one per sample

    Examples
    --------
    >>> from pathlib import Path
    >>> samples = build_sample_variants(
    ...     Path('test_data.vcf.gz'),
    ...     Path('phenotypes.tsv')
    ... )
    >>> print(f"Loaded {len(samples)} samples")
    >>> print(f"First sample: {samples[0]}")
    """
    # Load phenotypes
    phenotypes = load_phenotypes(phenotype_file)
    print(f"Loaded phenotypes for {len(phenotypes)} samples")
    print(f"  Cases: {sum(1 for v in phenotypes.values() if v == 1)}")
    print(f"  Controls: {sum(1 for v in phenotypes.values() if v == 0)}")

    # Parse VCF
    sample_variants = list(
        parse_vcf_cyvcf2(
            vcf_path,
            phenotypes,
            genome_build=genome_build,
            sex_map=sex_map,
            max_variants_per_sample=max_variants_per_sample,
            min_gq=min_gq,
        )
    )

    print(f"\nLoaded {len(sample_variants)} samples with variants")

    # Print summary statistics
    variant_counts = [len(sv.variants) for sv in sample_variants]
    print(f"Variant count statistics:")
    print(f"  Mean: {np.mean(variant_counts):.1f}")
    print(f"  Median: {np.median(variant_counts):.1f}")
    print(f"  Min: {np.min(variant_counts)}")
    print(f"  Max: {np.max(variant_counts)}")

    return sample_variants
