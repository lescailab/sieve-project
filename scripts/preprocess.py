#!/usr/bin/env python3
"""
Preprocessing script for SIEVE.

This script parses VCF files and saves the processed variant data to disk,
allowing faster training iterations without re-parsing VCF each time.

Supports ploidy-aware dosage encoding when a sex map is provided or
when sex inference is run automatically from the VCF.

Usage:
    python scripts/preprocess.py --vcf data.vcf.gz --phenotypes pheno.tsv --output preprocessed.pt
    python scripts/preprocess.py --vcf data.vcf.gz --phenotypes pheno.tsv --output preprocessed.pt \
        --sex-map sex_inference/sample_sex.tsv --genome-build GRCh37

Author: Lescai Lab
"""

import argparse
import sys
from pathlib import Path
import time

import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import build_sample_variants
from src.data.genome import get_genome_build


def load_sex_map(sex_map_path: str) -> dict:
    """
    Load sex map from TSV file produced by infer_sex.py.

    Parameters
    ----------
    sex_map_path : str
        Path to sample_sex.tsv.

    Returns
    -------
    dict
        Mapping from sample_id to sex ('M' or 'F').
        Only includes definitive assignments.
    """
    df = pd.read_csv(sex_map_path, sep='\t')
    sex_map = {}
    for _, row in df.iterrows():
        sex = row['inferred_sex']
        if sex in ('M', 'F'):
            sex_map[row['sample_id']] = sex
    return sex_map


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess VCF data for SIEVE training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--vcf', type=str, required=True,
                        help='Path to VCF file')
    parser.add_argument('--phenotypes', type=str, required=True,
                        help='Path to phenotypes file (TSV with sample_id, phenotype)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for preprocessed data (.pt file)')
    parser.add_argument('--max-variants-per-sample', type=int, default=None,
                        help='Maximum variants per sample (for debugging)')
    parser.add_argument('--min-gq', type=int, default=20,
                        help='Minimum genotype quality')
    parser.add_argument('--genome-build', type=str, default='GRCh37',
                        help='Reference genome build (GRCh37 or GRCh38)')
    parser.add_argument('--sex-map', type=str, default=None,
                        help='Path to sample_sex.tsv for ploidy-aware encoding')

    return parser.parse_args()


def main():
    """Main preprocessing function."""
    args = parse_args()

    build = get_genome_build(args.genome_build)

    print(f"SIEVE Data Preprocessing")
    print(f"=" * 60)
    print(f"VCF file: {args.vcf}")
    print(f"Phenotypes: {args.phenotypes}")
    print(f"Output: {args.output}")
    print(f"Genome build: {build.name}")
    print(f"=" * 60)

    # Load sex map if provided
    sex_map = None
    if args.sex_map:
        print(f"\nLoading sex map from {args.sex_map}")
        sex_map = load_sex_map(args.sex_map)
        print(f"  {len(sex_map)} samples with definitive sex assignment")
        n_m = sum(1 for v in sex_map.values() if v == 'M')
        n_f = sum(1 for v in sex_map.values() if v == 'F')
        print(f"  Males: {n_m}, Females: {n_f}")

    # Load data
    print(f"\nLoading data from VCF...")
    start_time = time.time()

    all_samples = build_sample_variants(
        vcf_path=args.vcf,
        phenotype_file=args.phenotypes,
        genome_build=build,
        sex_map=sex_map,
        max_variants_per_sample=args.max_variants_per_sample,
        min_gq=args.min_gq,
    )

    load_time = time.time() - start_time
    print(f"Loaded {len(all_samples)} samples in {load_time:.1f} seconds")

    # Check for empty samples
    if len(all_samples) == 0:
        print("ERROR: No samples loaded. Check that:")
        print("  1. VCF file exists and is readable")
        print("  2. Phenotype file exists and sample IDs match VCF")
        print("  3. VCF contains valid variants")
        sys.exit(1)

    # Print statistics
    n_cases = sum(1 for s in all_samples if s.label == 1)
    n_controls = len(all_samples) - n_cases
    print(f"  Cases: {n_cases}")
    print(f"  Controls: {n_controls}")

    variant_counts = [len(s.variants) for s in all_samples]
    if len(variant_counts) > 0:
        print(f"  Variants per sample: mean={sum(variant_counts)/len(variant_counts):.1f}, "
              f"min={min(variant_counts)}, max={max(variant_counts)}")
    else:
        print(f"  Variants per sample: 0 (no variants found)")

    # Save preprocessed data
    print(f"\nSaving preprocessed data...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with metadata
    save_dict = {
        'samples': all_samples,
        'metadata': {
            'vcf_path': args.vcf,
            'phenotype_path': args.phenotypes,
            'genome_build': build.name,
            'ploidy_aware': sex_map is not None,
            'sex_map_path': args.sex_map,
            'num_samples': len(all_samples),
            'num_cases': n_cases,
            'num_controls': n_controls,
            'max_variants_per_sample': args.max_variants_per_sample,
            'min_gq': args.min_gq,
            'preprocessing_time': load_time,
        }
    }

    torch.save(save_dict, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved to {output_path} ({file_size_mb:.1f} MB)")

    print(f"\nPreprocessing complete!")
    print(f"Use this file with --preprocessed-data {args.output} in training")


if __name__ == '__main__':
    main()
