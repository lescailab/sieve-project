#!/usr/bin/env python3
"""
Download and parse ClinVar database for variant validation.

Downloads ClinVar VCF file and extracts pathogenic/likely pathogenic variants
into a TSV format required by SIEVE biological validation.

Usage:
    python utilities/download_clinvar.py --output data/clinvar_grch37.tsv
    python utilities/download_clinvar.py --genome GRCh38 --output data/clinvar_grch38.tsv

Output format (TSV):
    chrom | pos | ref | alt | gene | clinical_significance

Author: Lescai Lab
"""

import argparse
import gzip
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, TextIO
import urllib.request
import shutil

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    sys.exit(1)

# ClinVar VCF URLs (NCBI FTP)
CLINVAR_URLS = {
    'GRCh37': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/clinvar.vcf.gz',
    'GRCh38': 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz',
}

# Clinical significance categories to extract
PATHOGENIC_CATEGORIES = {
    'Pathogenic',
    'Likely_pathogenic',
    'Pathogenic/Likely_pathogenic',
}


def download_with_progress(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    print(f"Downloading from {url}...")

    def reporthook(block_num, block_size, total_size):
        """Progress callback."""
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100.0 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, output_path, reporthook)
        print()  # Newline after progress
        print(f"Downloaded to {output_path}")
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        raise


def parse_info_field(info: str, key: str) -> Optional[str]:
    """Extract value from VCF INFO field."""
    for item in info.split(';'):
        if item.startswith(f"{key}="):
            return item[len(key)+1:]
    return None


def parse_clinvar_vcf(vcf_path: Path, output_path: Path, max_variants: Optional[int] = None) -> None:
    """
    Parse ClinVar VCF and extract pathogenic variants to TSV.

    Parameters
    ----------
    vcf_path : Path
        Path to ClinVar VCF file (can be gzipped)
    output_path : Path
        Path to output TSV file
    max_variants : Optional[int]
        Maximum number of variants to process (for testing)
    """
    print("\nParsing ClinVar VCF...")

    # Determine if file is gzipped
    is_gzipped = vcf_path.suffix == '.gz'

    # Open file
    if is_gzipped:
        file_handle = gzip.open(vcf_path, 'rt')
    else:
        file_handle = open(vcf_path, 'r')

    variants = []
    total_lines = 0
    pathogenic_count = 0

    try:
        for line in file_handle:
            total_lines += 1

            # Skip header lines
            if line.startswith('#'):
                continue

            # Progress update
            if total_lines % 100000 == 0:
                print(f"  Processed {total_lines:,} lines, found {pathogenic_count:,} pathogenic variants...")

            # Parse VCF line
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue

            chrom = fields[0]
            pos = fields[1]
            ref = fields[3]
            alt = fields[4]
            info = fields[7]

            # Remove 'chr' prefix if present (normalize to GRCh37 style)
            if chrom.startswith('chr'):
                chrom = chrom[3:]

            # Extract clinical significance
            clnsig = parse_info_field(info, 'CLNSIG')
            if not clnsig:
                continue

            # Check if pathogenic
            is_pathogenic = any(cat in clnsig for cat in PATHOGENIC_CATEGORIES)
            if not is_pathogenic:
                continue

            # Extract gene name
            gene = parse_info_field(info, 'GENEINFO')
            if gene:
                # GENEINFO format: "GENE:ID|GENE:ID"
                gene = gene.split(':')[0].split('|')[0]
            else:
                gene = 'Unknown'

            # Handle multiple ALT alleles
            for alt_allele in alt.split(','):
                variants.append({
                    'chrom': chrom,
                    'pos': int(pos),
                    'ref': ref,
                    'alt': alt_allele,
                    'gene': gene,
                    'clinical_significance': clnsig
                })
                pathogenic_count += 1

            # Stop if max reached
            if max_variants and pathogenic_count >= max_variants:
                break

    finally:
        file_handle.close()

    print(f"\nProcessed {total_lines:,} total lines")
    print(f"Found {pathogenic_count:,} pathogenic/likely pathogenic variants")

    # Create DataFrame and save
    df = pd.DataFrame(variants)

    if len(df) == 0:
        print("WARNING: No pathogenic variants found!")
        return

    # Sort by chromosome and position
    df = df.sort_values(['chrom', 'pos'])

    # Save to TSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"\nSaved {len(df):,} variants to {output_path}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Unique chromosomes: {df['chrom'].nunique()}")
    print(f"  Unique genes: {df['gene'].nunique()}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    # Show top genes
    top_genes = df['gene'].value_counts().head(10)
    print(f"\nTop 10 genes with most pathogenic variants:")
    for gene, count in top_genes.items():
        print(f"  {gene}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and parse ClinVar database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download GRCh37 ClinVar (default)
  python utilities/download_clinvar.py --output data/clinvar_grch37.tsv

  # Download GRCh38 ClinVar
  python utilities/download_clinvar.py --genome GRCh38 --output data/clinvar_grch38.tsv

  # Use existing VCF file
  python utilities/download_clinvar.py --vcf clinvar.vcf.gz --output data/clinvar.tsv

  # Test mode (first 1000 pathogenic variants)
  python utilities/download_clinvar.py --output test.tsv --max-variants 1000
        """
    )

    parser.add_argument('--genome', type=str, default='GRCh37',
                        choices=['GRCh37', 'GRCh38'],
                        help='Reference genome version (default: GRCh37)')
    parser.add_argument('--vcf', type=str,
                        help='Path to existing ClinVar VCF file (skips download)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output TSV file path')
    parser.add_argument('--max-variants', type=int,
                        help='Maximum pathogenic variants to extract (for testing)')
    parser.add_argument('--keep-vcf', action='store_true',
                        help='Keep downloaded VCF file after processing')

    args = parser.parse_args()

    output_path = Path(args.output)

    print("="*60)
    print("ClinVar Database Download and Parser")
    print("="*60)
    print(f"Reference genome: {args.genome}")
    print(f"Output file: {output_path}")
    if args.max_variants:
        print(f"Max variants: {args.max_variants:,}")
    print()

    # Determine VCF path
    if args.vcf:
        # Use provided VCF
        vcf_path = Path(args.vcf)
        if not vcf_path.exists():
            print(f"ERROR: VCF file not found: {vcf_path}")
            sys.exit(1)
        print(f"Using existing VCF: {vcf_path}")
        temp_vcf = None
    else:
        # Download ClinVar
        url = CLINVAR_URLS[args.genome]

        if args.keep_vcf:
            # Download to same directory as output
            vcf_path = output_path.parent / f"clinvar_{args.genome.lower()}.vcf.gz"
            download_with_progress(url, vcf_path)
            temp_vcf = None
        else:
            # Download to temp file
            temp_vcf = tempfile.NamedTemporaryFile(suffix='.vcf.gz', delete=False)
            temp_vcf.close()
            vcf_path = Path(temp_vcf.name)
            download_with_progress(url, vcf_path)

    try:
        # Parse VCF to TSV
        parse_clinvar_vcf(vcf_path, output_path, max_variants=args.max_variants)

        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"\nClinVar database ready: {output_path}")
        print("\nYou can now use this file with:")
        print(f"  python scripts/validate_discoveries.py \\")
        print(f"    --clinvar-db {output_path} \\")
        print(f"    ...")

    finally:
        # Clean up temp file if needed
        if temp_vcf and vcf_path.exists() and not args.keep_vcf:
            print(f"\nCleaning up temporary VCF file...")
            vcf_path.unlink()


if __name__ == '__main__':
    main()
