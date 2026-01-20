#!/usr/bin/env python3
"""
Preprocessing script for SIEVE.

This script parses VCF files and saves the processed variant data to disk,
allowing faster training iterations without re-parsing VCF each time.

Usage:
    python scripts/preprocess.py --vcf data.vcf.gz --phenotypes pheno.tsv --output preprocessed.pt

Author: Lescai Lab
"""

import argparse
import sys
from pathlib import Path
import time

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import build_sample_variants


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

    return parser.parse_args()


def main():
    """Main preprocessing function."""
    args = parse_args()

    print(f"SIEVE Data Preprocessing")
    print(f"=" * 60)
    print(f"VCF file: {args.vcf}")
    print(f"Phenotypes: {args.phenotypes}")
    print(f"Output: {args.output}")
    print(f"=" * 60)

    # Load data
    print(f"\nLoading data from VCF...")
    start_time = time.time()

    all_samples = build_sample_variants(
        vcf_path=args.vcf,
        phenotype_file=args.phenotypes,
        max_variants_per_sample=args.max_variants_per_sample,
        min_gq=args.min_gq,
    )

    load_time = time.time() - start_time
    print(f"Loaded {len(all_samples)} samples in {load_time:.1f} seconds")

    # Print statistics
    n_cases = sum(1 for s in all_samples if s.label == 1)
    n_controls = len(all_samples) - n_cases
    print(f"  Cases: {n_cases}")
    print(f"  Controls: {n_controls}")

    variant_counts = [len(s.variants) for s in all_samples]
    print(f"  Variants per sample: mean={sum(variant_counts)/len(variant_counts):.1f}, "
          f"min={min(variant_counts)}, max={max(variant_counts)}")

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
