#!/usr/bin/env python3
"""
Biological validation of SIEVE discoveries.

Takes variant and gene rankings from explain.py and validates them against:
- ClinVar (known pathogenic variants)
- GWAS Catalog (disease associations)
- Gene Ontology (functional enrichment)

Usage:
    python scripts/validate_discoveries.py \
        --variant-rankings results/sieve_variant_rankings.csv \
        --gene-rankings results/sieve_gene_rankings.csv \
        --output-dir results/validation \
        --clinvar data/clinvar.tsv \
        --gwas data/gwas_catalog.tsv

Author: Lescai Lab
"""

import argparse
from pathlib import Path
import yaml
import pandas as pd
import json

from src.explain.biological_validation import BiologicalValidator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate SIEVE discoveries against biological databases',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required inputs
    parser.add_argument('--variant-rankings', type=str, required=True,
                        help='Path to variant rankings CSV')
    parser.add_argument('--gene-rankings', type=str, required=True,
                        help='Path to gene rankings CSV')

    # Database files (optional)
    parser.add_argument('--clinvar-db', '--clinvar', type=str, dest='clinvar',
                        help='Path to ClinVar database (TSV format)')
    parser.add_argument('--gwas-db', '--gwas', type=str, dest='gwas',
                        help='Path to GWAS Catalog (TSV format)')
    parser.add_argument('--go-annotations', '--go-mapping', type=str, dest='go_mapping',
                        help='Path to gene-to-GO mapping (JSON format)')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for validation results')

    # Options
    parser.add_argument('--top-k-variants', type=int, default=100,
                        help='Number of top variants to validate')
    parser.add_argument('--top-k-genes', type=int, default=50,
                        help='Number of top genes to validate')
    parser.add_argument('--disease-terms', type=str, nargs='+',
                        help='Disease terms for GWAS filtering (e.g., "diabetes" "obesity")')
    parser.add_argument('--genome-build', '--reference-genome', type=str,
                        default='GRCh37', dest='genome_build',
                        help='Reference genome build (GRCh37 or GRCh38)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("SIEVE Discovery Validation")
    print("="*60)

    # Load rankings
    print("\nLoading rankings...")
    variant_rankings = pd.read_csv(args.variant_rankings)
    gene_rankings = pd.read_csv(args.gene_rankings)

    print(f"  Variants: {len(variant_rankings)}")
    print(f"  Genes: {len(gene_rankings)}")

    # Initialize validator
    validator = BiologicalValidator(reference_genome=args.genome_build)

    # === CLINVAR VALIDATION ===
    clinvar_validation = None
    if args.clinvar:
        print("\n" + "="*60)
        print("ClinVar Validation")
        print("="*60)

        clinvar_df = validator.load_clinvar(args.clinvar)
        if not clinvar_df.empty:
            clinvar_validation = validator.validate_variants_against_clinvar(
                variant_rankings=variant_rankings,
                clinvar_df=clinvar_df,
                top_k=args.top_k_variants
            )

            # Save results
            clinvar_path = output_dir / 'clinvar_validation.csv'
            clinvar_validation.to_csv(clinvar_path, index=False)
            print(f"\nClinVar validation saved to {clinvar_path}")

            # Show top pathogenic matches
            if 'clinvar_significance' in clinvar_validation.columns:
                pathogenic = clinvar_validation[
                    clinvar_validation['clinvar_significance'].str.contains(
                        'Pathogenic', case=False, na=False
                    )
                ]
                if len(pathogenic) > 0:
                    print(f"\nTop pathogenic variants discovered:")
                    print(pathogenic[['position', 'gene_id', 'mean_attribution', 'clinvar_significance']].head(10))

    # === GWAS VALIDATION ===
    gwas_validation = None
    if args.gwas:
        print("\n" + "="*60)
        print("GWAS Catalog Validation")
        print("="*60)

        gwas_df = validator.load_gwas_catalog(args.gwas)
        if not gwas_df.empty:
            gwas_validation = validator.validate_genes_against_gwas(
                gene_rankings=gene_rankings,
                gwas_df=gwas_df,
                disease_terms=args.disease_terms,
                top_k=args.top_k_genes
            )

            # Save results
            gwas_path = output_dir / 'gwas_validation.csv'
            gwas_validation.to_csv(gwas_path, index=False)
            print(f"\nGWAS validation saved to {gwas_path}")

            # Show top GWAS matches
            gwas_genes = gwas_validation[gwas_validation['in_gwas']]
            if len(gwas_genes) > 0:
                print(f"\nTop GWAS genes discovered:")
                print(gwas_genes[['gene_id', 'gene_score', 'gwas_studies', 'gwas_traits']].head(10))

            # Enrichment test
            if len(gwas_genes) > 0:
                gwas_gene_set = set(gwas_df['gene'].unique()) if 'gene' in gwas_df.columns else set()
                if gwas_gene_set:
                    enrichment = validator.compute_enrichment(
                        discovered_genes=list(gene_rankings['gene_id'].head(args.top_k_genes)),
                        database_genes=gwas_gene_set,
                        total_genes=20000  # Approximate human gene count
                    )

                    print(f"\nGWAS Enrichment Analysis:")
                    print(f"  Overlap: {enrichment['overlap']}/{enrichment['n_discovered']}")
                    print(f"  Fold enrichment: {enrichment['fold_enrichment']:.2f}")
                    print(f"  P-value: {enrichment['p_value']:.2e}")
                    print(f"  Significant: {'Yes' if enrichment['is_significant'] else 'No'}")

    # === GO ENRICHMENT ===
    go_enrichment = None
    if args.go_mapping:
        print("\n" + "="*60)
        print("Gene Ontology Enrichment")
        print("="*60)

        # Load GO mapping (maps gene symbols to GO terms)
        with open(args.go_mapping) as f:
            gene_to_go = json.load(f)

        print(f"Loaded GO annotations for {len(gene_to_go)} genes")

        # Get top gene names (not gene IDs!)
        if 'gene_name' in gene_rankings.columns:
            top_genes = list(gene_rankings['gene_name'].head(args.top_k_genes))
        else:
            print("ERROR: gene_name column missing in gene rankings - skipping GO enrichment")
            top_genes = []

        if top_genes:
            go_enrichment = validator.perform_go_enrichment(
                gene_list=top_genes,
                gene_to_go=gene_to_go,
                min_genes_per_term=3,
                max_genes_per_term=500
            )

        if go_enrichment is not None and not go_enrichment.empty:
            # Save results
            go_path = output_dir / 'go_enrichment.csv'
            go_enrichment.to_csv(go_path, index=False)
            print(f"GO enrichment saved to {go_path}")

            # Show top enriched terms
            significant = go_enrichment[go_enrichment['fdr'] < 0.05]
            print(f"\nFound {len(significant)} significantly enriched GO terms (FDR < 0.05)")
            if len(significant) > 0:
                print("\nTop enriched GO terms:")
                print(significant[['go_term', 'overlap', 'fold_enrichment', 'fdr']].head(10))
        else:
            print("No significant GO enrichment found")

    # === VALIDATION SUMMARY ===
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    summary = validator.create_validation_summary(
        variant_rankings=variant_rankings,
        gene_rankings=gene_rankings,
        clinvar_validation=clinvar_validation,
        gwas_validation=gwas_validation,
        go_enrichment=go_enrichment
    )

    # Print summary
    print("\nOverall Statistics:")
    print(f"  Total variants analyzed: {summary['total_variants']}")
    print(f"  Total genes analyzed: {summary['total_genes']}")

    if 'clinvar' in summary:
        print(f"\nClinVar Validation:")
        print(f"  Variants checked: {summary['clinvar']['n_variants_checked']}")
        print(f"  In ClinVar: {summary['clinvar']['n_in_clinvar']} ({summary['clinvar']['pct_in_clinvar']:.1f}%)")
        print(f"  Pathogenic: {summary['clinvar']['n_pathogenic']}")

    if 'gwas' in summary:
        print(f"\nGWAS Validation:")
        print(f"  Genes checked: {summary['gwas']['n_genes_checked']}")
        print(f"  In GWAS: {summary['gwas']['n_in_gwas']} ({summary['gwas']['pct_in_gwas']:.1f}%)")
        print(f"  Total GWAS studies: {summary['gwas']['total_studies']}")

    if 'go_enrichment' in summary:
        print(f"\nGO Enrichment:")
        print(f"  Terms tested: {summary['go_enrichment']['n_terms_tested']}")
        print(f"  Significant terms (FDR < 0.05): {summary['go_enrichment']['n_significant']}")
        if summary['go_enrichment']['top_term']:
            print(f"  Top term: {summary['go_enrichment']['top_term']}")
            print(f"  P-value: {summary['go_enrichment']['top_term_pvalue']:.2e}")

    # Save summary
    summary_path = output_dir / 'validation_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"\nValidation summary saved to {summary_path}")
    print("="*60)


if __name__ == '__main__':
    main()
