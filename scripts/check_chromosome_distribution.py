#!/usr/bin/env python3
"""
Diagnostic script to check chromosome distribution in preprocessed data.

This will tell us if the VCF parsing is actually broken or if there's
another issue causing the Manhattan plot to only show chr1 and chr2.

Usage:
    python scripts/check_chromosome_distribution.py --preprocessed data/preprocessed.pt
"""

import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Check chromosome distribution in preprocessed data')
    parser.add_argument('--preprocessed', type=str, required=True,
                        help='Path to preprocessed .pt file')
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("Chromosome Distribution Diagnostic")
    print("="*60)

    # Load preprocessed data
    print(f"\nLoading: {args.preprocessed}")
    data = torch.load(args.preprocessed, weights_only=False)

    samples = data['samples']
    print(f"Loaded {len(samples)} samples")

    # Count variants per chromosome
    chrom_counts = Counter()
    position_counts = defaultdict(lambda: defaultdict(int))  # chrom -> position -> count
    gene_counts = defaultdict(set)  # chrom -> set of genes

    total_variants = 0
    for sample in samples:
        for variant in sample.variants:
            chrom = variant.chrom
            pos = variant.pos
            gene = variant.gene

            chrom_counts[chrom] += 1
            position_counts[chrom][pos] += 1
            gene_counts[chrom].add(gene)
            total_variants += 1

    print(f"\nTotal variants across all samples: {total_variants:,}")
    print(f"\nChromosomes found: {len(chrom_counts)}")

    # Print detailed chromosome statistics
    print("\n" + "="*60)
    print("Chromosome Distribution:")
    print("="*60)
    print(f"{'Chromosome':<12} {'Variants':<12} {'Unique Pos':<12} {'Genes':<12} {'% of Total':<12}")
    print("-"*60)

    # Sort chromosomes properly (1, 2, ... 22, X, Y, MT)
    def chrom_sort_key(chrom):
        if chrom.isdigit():
            return (0, int(chrom))
        elif chrom == 'X':
            return (1, 0)
        elif chrom == 'Y':
            return (1, 1)
        elif chrom in ['MT', 'M']:
            return (1, 2)
        else:
            return (2, chrom)

    sorted_chroms = sorted(chrom_counts.keys(), key=chrom_sort_key)

    for chrom in sorted_chroms:
        count = chrom_counts[chrom]
        unique_pos = len(position_counts[chrom])
        genes = len(gene_counts[chrom])
        pct = 100.0 * count / total_variants if total_variants > 0 else 0
        print(f"{chrom:<12} {count:<12,} {unique_pos:<12,} {genes:<12,} {pct:<12.2f}%")

    # Check for suspicious patterns
    print("\n" + "="*60)
    print("Diagnostic Results:")
    print("="*60)

    if len(chrom_counts) == 0:
        print("❌ CRITICAL: No chromosomes found! VCF parsing completely failed.")
        return 1

    if len(chrom_counts) <= 2:
        print(f"❌ CRITICAL: Only {len(chrom_counts)} chromosome(s) found!")
        print("   This indicates a serious VCF parsing issue.")
        print("   Expected: 22 autosomes + X/Y/MT")
        return 1

    if len(chrom_counts) < 22:
        print(f"⚠️  WARNING: Only {len(chrom_counts)} chromosomes found (expected ~22-25)")
        print("   Some chromosomes may be missing from the VCF file")
        missing_expected = set(str(i) for i in range(1, 23)) - set(sorted_chroms)
        if missing_expected:
            print(f"   Missing autosomes: {sorted(missing_expected, key=int)}")

    # Check if most variants are on chr1 or chr2
    chr1_chr2_count = chrom_counts.get('1', 0) + chrom_counts.get('2', 0)
    chr1_chr2_pct = 100.0 * chr1_chr2_count / total_variants if total_variants > 0 else 0

    if chr1_chr2_pct > 50:
        print(f"⚠️  WARNING: {chr1_chr2_pct:.1f}% of variants are on chr1+chr2")
        print("   This is suspicious - typically should be ~15-20%")
        print("   Possible causes:")
        print("   - VCF file was filtered to only chr1 and chr2")
        print("   - VCF parsing is somehow filtering other chromosomes")

    if len(chrom_counts) >= 22:
        print("✅ OK: Found expected number of chromosomes")

    # Print sample statistics
    print("\n" + "="*60)
    print("Sample Statistics:")
    print("="*60)

    variants_per_sample = [len(s.variants) for s in samples]
    if variants_per_sample:
        print(f"Variants per sample:")
        print(f"  Mean: {sum(variants_per_sample)/len(variants_per_sample):.1f}")
        print(f"  Min: {min(variants_per_sample)}")
        print(f"  Max: {max(variants_per_sample)}")

        # Sample a few variants from a few samples
        print("\nSample of variants from first sample:")
        if len(samples) > 0 and len(samples[0].variants) > 0:
            for i, var in enumerate(samples[0].variants[:10]):
                print(f"  {var.chrom}:{var.pos} {var.ref}>{var.alt} {var.gene} ({var.consequence})")
            if len(samples[0].variants) > 10:
                print(f"  ... and {len(samples[0].variants) - 10} more")

    return 0


if __name__ == '__main__':
    sys.exit(main())
