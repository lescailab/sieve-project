#!/usr/bin/env python3
"""
Download and parse GWAS Catalog for gene-disease association validation.

Downloads the NHGRI-EBI GWAS Catalog and extracts gene-disease associations
into a TSV format required by SIEVE biological validation.

Usage:
    python utilities/download_gwas_catalog.py --output data/gwas_catalog.tsv
    python utilities/download_gwas_catalog.py --genome GRCh38 --output data/gwas_grch38.tsv

Output format (TSV):
    gene | disease_trait | chr | pos | snp_id | p_value | risk_allele

Author: Francesco Lescai
"""

import argparse
import gzip
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional
import urllib.request

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    sys.exit(1)

# GWAS Catalog URLs
# The main GWAS catalog file includes both GRCh37 and GRCh38 coordinates
# We filter based on the genome parameter after download
#
# Primary: EBI GWAS Catalog API (verified 2026-01-28 by user)
# Source: https://www.ebi.ac.uk/gwas/docs/file-downloads
# NOTE: This endpoint does NOT specify genome version in URL - file contains both builds
GWAS_URL = 'https://www.ebi.ac.uk/gwas/api/search/downloads/associations/v1.0?split=false'

# NOTE: If this URL fails, visit https://www.ebi.ac.uk/gwas/docs/file-downloads for current links
# The GWAS Catalog does not maintain separate files per genome build - coordinates for
# both GRCh37 and GRCh38 are included in the same file and we extract based on --genome parameter


def download_with_progress(url: str, output_path: Path) -> None:
    """Download file with progress bar."""
    print(f"Downloading GWAS Catalog from {url}...")

    def reporthook(block_num, block_size, total_size):
        """Progress callback."""
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100.0 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()
        else:
            # Size unknown
            downloaded = block_num * block_size
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Downloaded: {mb_downloaded:.1f} MB")
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


def extract_if_zip(file_path: Path) -> Path:
    """Check if file is a ZIP and extract it. Returns path to TSV file."""
    import zipfile

    # Check if it's a ZIP file by reading the magic number
    with open(file_path, 'rb') as f:
        magic = f.read(4)

    if magic.startswith(b'PK\x03\x04'):  # ZIP file signature
        print(f"Detected ZIP file, extracting...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # List contents
            files = zip_ref.namelist()
            print(f"ZIP contains {len(files)} file(s):")
            for fname in files:
                print(f"  - {fname}")

            # Find the TSV file (should be the associations file)
            tsv_files = [f for f in files if f.endswith('.tsv')]
            if not tsv_files:
                raise ValueError(f"No .tsv file found in ZIP archive. Contents: {files}")

            tsv_file = tsv_files[0]
            print(f"Extracting {tsv_file}...")

            # Extract to same directory
            extract_dir = file_path.parent
            zip_ref.extract(tsv_file, extract_dir)

            extracted_path = extract_dir / tsv_file
            print(f"Extracted to {extracted_path}")

            return extracted_path
    else:
        # Not a ZIP, return as-is
        return file_path


def parse_gwas_catalog(gwas_path: Path, output_path: Path, genome: str = 'GRCh37',
                      min_pvalue: float = 5e-8) -> Path:
    """
    Parse GWAS Catalog and extract gene-disease associations.

    Parameters
    ----------
    gwas_path : Path
        Path to downloaded GWAS catalog TSV (or ZIP containing TSV)
    output_path : Path
        Path to output TSV file
    genome : str
        Reference genome version ('GRCh37' or 'GRCh38')
    min_pvalue : float
        Minimum p-value threshold (default: 5e-8, genome-wide significance)

    Returns
    -------
    Path
        Path to extracted TSV file (same as input if not a ZIP, different if extracted)
    """
    print("\nParsing GWAS Catalog...")
    print(f"Filtering for p-value < {min_pvalue}")

    # Check if file is a ZIP and extract if needed
    gwas_path = extract_if_zip(gwas_path)

    # Read GWAS catalog
    # The GWAS API endpoint returns a TSV file, but format may vary
    # IMPORTANT: GWAS Catalog uses latin-1 encoding, NOT UTF-8
    print("Attempting to read GWAS catalog file...")

    # Try to detect the actual format
    try:
        # First, try as TSV with C parser
        df = pd.read_csv(
            gwas_path,
            sep='\t',
            encoding='latin-1',
            low_memory=False,
            on_bad_lines='warn',
            engine='c'
        )
        print(f"Loaded {len(df):,} total associations")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"C parser failed: {e}")
        print("Trying Python parser (slower but more robust)...")

        try:
            # Try with Python parser with minimal quoting
            df = pd.read_csv(
                gwas_path,
                sep='\t',
                encoding='latin-1',
                engine='python',
                on_bad_lines='skip',
                quoting=3  # QUOTE_NONE - don't interpret quotes at all
            )
            print(f"Loaded {len(df):,} total associations with Python parser")
        except Exception as e2:
            print(f"Python parser also failed: {e2}")
            print("Trying final fallback: line-by-line reading with error tolerance...")

            try:
                # Ultimate fallback: Read raw lines and parse manually
                import csv
                rows = []
                with open(gwas_path, 'r', encoding='latin-1', errors='replace') as f:
                    reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                    header = next(reader)

                    for i, row in enumerate(reader):
                        try:
                            # Only keep rows that have the right number of columns (±2)
                            if abs(len(row) - len(header)) <= 2:
                                # Pad or truncate to match header length
                                if len(row) < len(header):
                                    row.extend([''] * (len(header) - len(row)))
                                elif len(row) > len(header):
                                    row = row[:len(header)]
                                rows.append(row)
                        except Exception:
                            continue  # Skip problematic rows

                        if i > 0 and i % 100000 == 0:
                            print(f"  Processed {i:,} lines, kept {len(rows):,} valid rows...")

                df = pd.DataFrame(rows, columns=header)
                print(f"Successfully loaded {len(df):,} associations using manual parsing")
                print(f"  (skipped {i - len(rows):,} malformed rows)")
            except Exception as e3:
                print(f"ERROR: All parsing methods failed: {e3}")
                print(f"\nFile location: {gwas_path}")
                print("Please check the file format manually:")
                print(f"  head -20 {gwas_path}")
                print(f"  wc -l {gwas_path}")
                raise
    except Exception as e:
        print(f"ERROR: Unexpected error reading GWAS catalog: {e}")
        print(f"\nFile location: {gwas_path}")
        raise

    # Debug: Print column information
    print(f"\nFile has {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10

    # Select relevant columns based on what's available
    # GWAS Catalog columns (may vary by version):
    # - MAPPED_GENE (or REPORTED GENE(S))
    # - DISEASE/TRAIT
    # - CHR_ID (chromosome)
    # - CHR_POS (position, depends on genome)
    # - SNPS (SNP ID)
    # - P-VALUE
    # - STRONGEST SNP-RISK ALLELE

    # Determine column names (they may vary)
    gene_col = 'MAPPED_GENE' if 'MAPPED_GENE' in df.columns else 'REPORTED GENE(S)'
    trait_col = 'DISEASE/TRAIT'
    chr_col = 'CHR_ID'
    snp_col = 'SNPS'
    pvalue_col = 'P-VALUE'

    # Position column depends on genome version
    if genome == 'GRCh38':
        pos_col = 'CHR_POS' if 'CHR_POS' in df.columns else 'BP'
    else:
        pos_col = 'CHR_POS' if 'CHR_POS' in df.columns else 'BP'

    # Risk allele column
    risk_col = 'STRONGEST SNP-RISK ALLELE' if 'STRONGEST SNP-RISK ALLELE' in df.columns else 'RISK ALLELE FREQUENCY'

    # Check required columns exist
    required_cols = [gene_col, trait_col, pvalue_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Convert p-value to numeric
    df[pvalue_col] = pd.to_numeric(df[pvalue_col], errors='coerce')

    # Filter by p-value
    df_filtered = df[df[pvalue_col] < min_pvalue].copy()
    print(f"After p-value filter: {len(df_filtered):,} associations")

    # Remove rows with missing genes
    df_filtered = df_filtered[df_filtered[gene_col].notna()]
    print(f"After removing missing genes: {len(df_filtered):,} associations")

    # Extract gene-disease associations
    associations = []

    for idx, row in df_filtered.iterrows():
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx+1:,} associations...")

        genes = str(row[gene_col])
        trait = row[trait_col]

        # Split multiple genes (often separated by commas, hyphens, or " - ")
        gene_list = []
        for separator in [',', ' - ', '-']:
            if separator in genes:
                gene_list = [g.strip() for g in genes.split(separator)]
                break
        if not gene_list:
            gene_list = [genes.strip()]

        # Get other info
        chrom = row.get(chr_col, '')
        pos = row.get(pos_col, '')
        snp = row.get(snp_col, '')
        pval = row[pvalue_col]
        risk = row.get(risk_col, '')

        # Create entry for each gene
        for gene in gene_list:
            gene = gene.strip()
            if gene and gene != 'NR':  # NR = "Not Reported"
                associations.append({
                    'gene': gene,
                    'disease_trait': trait,
                    'chr': chrom,
                    'pos': pos,
                    'snp_id': snp,
                    'p_value': pval,
                    'risk_allele': risk
                })

    print(f"\nExtracted {len(associations):,} gene-disease associations")

    # Create DataFrame
    result_df = pd.DataFrame(associations)

    if len(result_df) == 0:
        print("WARNING: No associations found after filtering!")
        return

    # Remove duplicates
    result_df = result_df.drop_duplicates(subset=['gene', 'disease_trait'])
    print(f"After removing duplicates: {len(result_df):,} unique gene-disease pairs")

    # Save to TSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, sep='\t', index=False)
    print(f"\nSaved to {output_path}")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Unique genes: {result_df['gene'].nunique()}")
    print(f"  Unique disease traits: {result_df['disease_trait'].nunique()}")
    print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    # Show top genes
    top_genes = result_df['gene'].value_counts().head(10)
    print(f"\nTop 10 genes with most disease associations:")
    for gene, count in top_genes.items():
        print(f"  {gene}: {count}")

    # Return the extracted TSV path for cleanup
    return gwas_path


def main():
    parser = argparse.ArgumentParser(
        description='Download and parse GWAS Catalog',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download GWAS Catalog (default: GRCh37, genome-wide significant hits)
  python utilities/download_gwas_catalog.py --output data/gwas_catalog.tsv

  # Use less stringent p-value threshold
  python utilities/download_gwas_catalog.py --output data/gwas.tsv --min-pvalue 1e-5

  # Use existing GWAS file
  python utilities/download_gwas_catalog.py --gwas gwas.tsv --output data/parsed_gwas.tsv
        """
    )

    parser.add_argument('--genome', type=str, default='GRCh37',
                        choices=['GRCh37', 'GRCh38'],
                        help='Reference genome version (default: GRCh37)')
    parser.add_argument('--gwas', type=str,
                        help='Path to existing GWAS catalog file (skips download)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output TSV file path')
    parser.add_argument('--min-pvalue', type=float, default=5e-8,
                        help='Minimum p-value threshold (default: 5e-8)')
    parser.add_argument('--keep-raw', action='store_true',
                        help='Keep downloaded raw GWAS file after processing')

    args = parser.parse_args()

    output_path = Path(args.output)

    print("="*60)
    print("GWAS Catalog Download and Parser")
    print("="*60)
    print(f"Reference genome: {args.genome}")
    print(f"P-value threshold: {args.min_pvalue}")
    print(f"Output file: {output_path}")
    print()

    # Determine GWAS file path
    if args.gwas:
        # Use provided file
        gwas_path = Path(args.gwas)
        if not gwas_path.exists():
            print(f"ERROR: GWAS file not found: {gwas_path}")
            sys.exit(1)
        print(f"Using existing GWAS file: {gwas_path}")
        temp_gwas = None
    else:
        # Download GWAS Catalog
        url = GWAS_URL

        if args.keep_raw:
            gwas_path = output_path.parent / f"gwas_catalog_raw.tsv"
            temp_gwas = None
        else:
            temp_gwas = tempfile.NamedTemporaryFile(suffix='.tsv', delete=False)
            temp_gwas.close()
            gwas_path = Path(temp_gwas.name)

        # Download from GWAS Catalog API
        try:
            download_with_progress(url, gwas_path)
        except Exception as e:
            print(f"\nERROR: Download failed: {e}")
            print(f"  URL: {url}")
            print("\nTroubleshooting:")
            print("1. Check your internet connection")
            print("2. Visit https://www.ebi.ac.uk/gwas/docs/file-downloads for current download URLs")
            print("3. Download manually and use --gwas parameter:")
            print(f"   python utilities/download_gwas_catalog.py --gwas <downloaded_file> --output {output_path}")
            sys.exit(1)

    extracted_tsv = None
    try:
        # Parse GWAS catalog (may extract from ZIP)
        extracted_tsv = parse_gwas_catalog(gwas_path, output_path, genome=args.genome,
                                          min_pvalue=args.min_pvalue)

        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"\nGWAS Catalog database ready: {output_path}")
        print("\nYou can now use this file with:")
        print(f"  python scripts/validate_discoveries.py \\")
        print(f"    --gwas-db {output_path} \\")
        print(f"    ...")

    finally:
        # Clean up temp files if needed
        if not args.keep_raw:
            if temp_gwas and gwas_path.exists():
                print(f"\nCleaning up temporary ZIP file...")
                gwas_path.unlink()

            # Also clean up extracted TSV if it's different from the ZIP
            if extracted_tsv and extracted_tsv != gwas_path and extracted_tsv.exists():
                print(f"Cleaning up extracted TSV file...")
                extracted_tsv.unlink()


if __name__ == '__main__':
    main()
