#!/usr/bin/env python3
"""
Fix ranking outputs by adding chromosome and gene_name columns.

This script reads the variant/gene ranking CSVs and adds missing columns
by mapping gene_ids back to the preprocessed data.

Usage:
    python scripts/fix_ranking_outputs.py \
        --preprocessed-data data/preprocessed.pt \
        --variant-rankings results/explainability/sieve_variant_rankings.csv \
        --gene-rankings results/explainability/sieve_gene_rankings.csv \
        --output-dir results/explainability_fixed/
"""

import argparse
from pathlib import Path
import pandas as pd
import torch
from collections import defaultdict


def build_gene_id_mapping(samples):
    """Build mapping from gene_id to (gene_name, chromosomes, positions)."""
    gene_map = defaultdict(lambda: {
        'gene_name': None,
        'chromosomes': set(),
        'positions': set()
    })

    for sample in samples:
        for variant in sample.variants:
            # Get gene_id (we'll need to map gene symbol to ID)
            gene_symbol = variant.gene
            chrom = variant.chrom
            pos = variant.pos

            # For now, store by gene_symbol
            if gene_symbol not in gene_map:
                gene_map[gene_symbol] = {
                    'gene_name': gene_symbol,
                    'chromosomes': set(),
                    'positions': set()
                }

            gene_map[gene_symbol]['chromosomes'].add(chrom)
            gene_map[gene_symbol]['positions'].add(pos)

    return gene_map


def build_position_gene_mapping(samples):
    """Build mapping from (position, gene_id) to (chromosome, gene_name).

    CRITICAL: Positions are NOT unique across chromosomes!
    Position 12345 can exist on chr1, chr2, chr3, etc.
    We must use (position, gene_id) as the key, not just position.
    """
    # First build gene_index (same as in encoding)
    gene_symbols = sorted(set(v.gene for s in samples for v in s.variants))
    gene_index = {gene: idx for idx, gene in enumerate(gene_symbols)}

    # Now build the mapping using (pos, gene_id) as key
    pos_gene_map = {}

    for sample in samples:
        for variant in sample.variants:
            pos = variant.pos
            chrom = variant.chrom
            gene_symbol = variant.gene
            gene_id = gene_index[gene_symbol]

            key = (pos, gene_id)
            if key not in pos_gene_map:
                pos_gene_map[key] = (chrom, gene_symbol)

    return pos_gene_map, gene_index


def build_gene_index_reverse(samples):
    """Build reverse mapping from gene_id (index) back to gene_symbol."""
    # Collect all unique gene symbols
    gene_symbols = sorted(set(v.gene for s in samples for v in s.variants))

    # Create gene_index like in the encoding
    gene_index = {gene: idx for idx, gene in enumerate(gene_symbols)}

    # Reverse it
    index_to_gene = {idx: gene for gene, idx in gene_index.items()}

    return index_to_gene, gene_index


def fix_variant_rankings(variant_df, preprocessed_data):
    """Add chromosome and gene_name columns to variant rankings."""
    samples = preprocessed_data['samples']

    # Build mappings with CORRECT key: (position, gene_id)
    pos_gene_to_info, gene_index = build_position_gene_mapping(samples)
    index_to_gene = {idx: gene for gene, idx in gene_index.items()}

    # Add columns
    chromosomes = []
    gene_names = []

    for _, row in variant_df.iterrows():
        position = int(row['position'])
        gene_id = int(row['gene_id'])

        # Use (position, gene_id) as key - this is unique!
        key = (position, gene_id)
        if key in pos_gene_to_info:
            chrom, gene_name = pos_gene_to_info[key]
        else:
            chrom = 'unknown'
            gene_name = index_to_gene.get(gene_id, f'GENE_{gene_id}')

        chromosomes.append(chrom)
        gene_names.append(gene_name)

    # Insert columns after position
    variant_df.insert(1, 'chromosome', chromosomes)
    variant_df.insert(3, 'gene_name', gene_names)

    # Reorder columns: chromosome, position, gene_name, gene_id, ...
    cols = list(variant_df.columns)
    cols.remove('chromosome')
    cols.remove('position')
    cols.remove('gene_name')
    cols.remove('gene_id')

    new_cols = ['chromosome', 'position', 'gene_name', 'gene_id'] + cols
    variant_df = variant_df[new_cols]

    return variant_df


def fix_gene_rankings(gene_df, preprocessed_data):
    """Add gene_name column to gene rankings."""
    samples = preprocessed_data['samples']

    # Build reverse gene index
    index_to_gene, gene_index = build_gene_index_reverse(samples)

    # Map gene_ids to chromosomes and gene_names
    gene_map = defaultdict(lambda: {'chroms': set(), 'name': None})
    for sample in samples:
        for variant in sample.variants:
            gene_symbol = variant.gene
            if gene_symbol in gene_index:
                gene_id = gene_index[gene_symbol]
                gene_map[gene_id]['chroms'].add(variant.chrom)
                gene_map[gene_id]['name'] = gene_symbol

    # Add columns
    gene_names = []
    chromosomes = []

    for _, row in gene_df.iterrows():
        gene_id = int(row['gene_id'])

        if gene_id in gene_map and gene_map[gene_id]['name']:
            gene_name = gene_map[gene_id]['name']
            # Get most common chromosome
            chroms = gene_map[gene_id]['chroms']
            chrom = sorted(chroms)[0] if chroms else 'unknown'
        else:
            gene_name = index_to_gene.get(gene_id, f'GENE_{gene_id}')
            chrom = 'unknown'

        gene_names.append(gene_name)
        chromosomes.append(chrom)

    # Insert columns
    gene_df.insert(0, 'chromosome', chromosomes)
    gene_df.insert(1, 'gene_name', gene_names)

    # Reorder: chromosome, gene_name, gene_id, ...
    cols = list(gene_df.columns)
    cols.remove('chromosome')
    cols.remove('gene_name')
    cols.remove('gene_id')

    new_cols = ['chromosome', 'gene_name', 'gene_id'] + cols
    gene_df = gene_df[new_cols]

    return gene_df


def main():
    parser = argparse.ArgumentParser(description='Fix ranking outputs by adding chromosome and gene_name')
    parser.add_argument('--preprocessed-data', required=True, help='Path to preprocessed.pt file')
    parser.add_argument('--variant-rankings', required=True, help='Path to variant rankings CSV')
    parser.add_argument('--gene-rankings', help='Path to gene rankings CSV (optional)')
    parser.add_argument('--output-dir', required=True, help='Output directory for fixed files')
    args = parser.parse_args()

    # Load preprocessed data
    print(f"Loading preprocessed data from {args.preprocessed_data}...")
    preprocessed = torch.load(args.preprocessed_data, weights_only=False)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fix variant rankings
    print(f"\nFixing variant rankings...")
    variant_df = pd.read_csv(args.variant_rankings)
    print(f"  Original columns: {list(variant_df.columns)}")
    print(f"  Rows: {len(variant_df)}")

    variant_df_fixed = fix_variant_rankings(variant_df, preprocessed)

    output_path = output_dir / Path(args.variant_rankings).name
    variant_df_fixed.to_csv(output_path, index=False)
    print(f"  Fixed columns: {list(variant_df_fixed.columns)}")
    print(f"  Saved to: {output_path}")

    # Fix gene rankings if provided
    if args.gene_rankings:
        print(f"\nFixing gene rankings...")
        gene_df = pd.read_csv(args.gene_rankings)
        print(f"  Original columns: {list(gene_df.columns)}")
        print(f"  Rows: {len(gene_df)}")

        gene_df_fixed = fix_gene_rankings(gene_df, preprocessed)

        output_path = output_dir / Path(args.gene_rankings).name
        gene_df_fixed.to_csv(output_path, index=False)
        print(f"  Fixed columns: {list(gene_df_fixed.columns)}")
        print(f"  Saved to: {output_path}")

    print(f"\n{'='*60}")
    print("SUCCESS!")
    print("="*60)
    print(f"\nFixed files saved to: {output_dir}")
    print("\nExample output:")
    print(variant_df_fixed.head(3).to_string())


if __name__ == '__main__':
    main()
