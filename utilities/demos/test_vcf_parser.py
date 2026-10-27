#!/usr/bin/env python3
"""
Quick test script to verify VCF parser works on test data.
"""

from pathlib import Path
from src.data.vcf_parser import build_sample_variants

def main():
    print("=" * 80)
    print("Testing VCF Parser on Real Test Data")
    print("=" * 80)

    # Paths to test data
    vcf_path = Path("test_data/small/test_data.vcf.gz")
    pheno_path = Path("test_data/small/test_data_phenotypes.tsv")

    print(f"\nVCF file: {vcf_path}")
    print(f"Phenotype file: {pheno_path}")
    print(f"VCF exists: {vcf_path.exists()}")
    print(f"Phenotype exists: {pheno_path.exists()}")

    # Load data
    print("\n" + "-" * 80)
    print("Loading data...")
    print("-" * 80)

    samples = build_sample_variants(
        vcf_path=vcf_path,
        phenotype_file=pheno_path,
        min_gq=20,
    )

    print("\n" + "-" * 80)
    print("Sample Summary")
    print("-" * 80)

    # Group by phenotype
    cases = [s for s in samples if s.label == 1]
    controls = [s for s in samples if s.label == 0]

    print(f"Total samples: {len(samples)}")
    print(f"Cases (label=1): {len(cases)}")
    print(f"Controls (label=0): {len(controls)}")

    # Show first case sample details
    if cases:
        print("\n" + "-" * 80)
        print("Example Case Sample")
        print("-" * 80)
        case = cases[0]
        print(f"Sample ID: {case.sample_id}")
        print(f"Label: {case.label}")
        print(f"Number of variants: {len(case.variants)}")

        if case.variants:
            print(f"\nFirst 5 variants:")
            for i, var in enumerate(case.variants[:5], 1):
                print(f"  {i}. {var.chrom}:{var.pos} {var.ref}>{var.alt}")
                print(f"     Gene: {var.gene}")
                print(f"     Consequence: {var.consequence}")
                print(f"     Genotype: {var.genotype}")
                if var.annotations.get('sift') is not None:
                    print(f"     SIFT: {var.annotations['sift']:.3f}")
                if var.annotations.get('polyphen') is not None:
                    print(f"     PolyPhen: {var.annotations['polyphen']:.3f}")
                print()

    # Show first control sample details
    if controls:
        print("\n" + "-" * 80)
        print("Example Control Sample")
        print("-" * 80)
        control = controls[0]
        print(f"Sample ID: {control.sample_id}")
        print(f"Label: {control.label}")
        print(f"Number of variants: {len(control.variants)}")

        if control.variants:
            print(f"\nFirst 3 variants:")
            for i, var in enumerate(control.variants[:3], 1):
                print(f"  {i}. {var}")

    # Consequence distribution
    print("\n" + "-" * 80)
    print("Consequence Distribution (all samples)")
    print("-" * 80)

    from collections import Counter
    all_consequences = []
    for sample in samples:
        for var in sample.variants:
            # Get first consequence if compound
            csq = var.consequence.split('&')[0]
            all_consequences.append(csq)

    csq_counts = Counter(all_consequences)
    for csq, count in csq_counts.most_common(10):
        print(f"  {csq}: {count}")

    # Gene distribution
    print("\n" + "-" * 80)
    print("Top 10 Genes (by variant count)")
    print("-" * 80)

    all_genes = []
    for sample in samples:
        for var in sample.variants:
            all_genes.append(var.gene)

    gene_counts = Counter(all_genes)
    for gene, count in gene_counts.most_common(10):
        print(f"  {gene}: {count}")

    # SIFT/PolyPhen availability
    print("\n" + "-" * 80)
    print("Annotation Availability")
    print("-" * 80)

    total_variants = sum(len(s.variants) for s in samples)
    sift_present = sum(
        1 for s in samples for v in s.variants
        if v.annotations.get('sift') is not None
    )
    polyphen_present = sum(
        1 for s in samples for v in s.variants
        if v.annotations.get('polyphen') is not None
    )

    print(f"Total variant instances: {total_variants}")
    print(f"With SIFT: {sift_present} ({100*sift_present/total_variants:.1f}%)")
    print(f"With PolyPhen: {polyphen_present} ({100*polyphen_present/total_variants:.1f}%)")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
