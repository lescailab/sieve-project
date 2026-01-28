#!/usr/bin/env python3
"""
Download and parse Gene Ontology annotations for enrichment analysis.

Downloads GO annotations from Ensembl BioMart and creates a JSON mapping
of gene symbols to GO terms for use with SIEVE biological validation.

Usage:
    python utilities/download_gene_ontology.py --output data/gene_to_go.json
    python utilities/download_gene_ontology.py --species human --output data/human_go.json

Output format (JSON):
    {
        "GENE1": ["GO:0001234", "GO:0005678", ...],
        "GENE2": ["GO:0009876", ...],
        ...
    }

Author: Lescai Lab
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set
import urllib.request
import urllib.parse

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    sys.exit(1)

# Ensembl BioMart URLs
BIOMART_URL = 'http://www.ensembl.org/biomart/martservice'

# GO annotation file URLs (alternative sources)
GO_ANNOTATION_URLS = {
    'human': 'http://current.geneontology.org/annotations/goa_human.gaf.gz',
    'mouse': 'http://current.geneontology.org/annotations/mgi.gaf.gz',
}

# GO aspect filters
GO_ASPECTS = {
    'all': ['C', 'F', 'P'],
    'biological_process': ['P'],
    'molecular_function': ['F'],
    'cellular_component': ['C'],
}


def download_with_progress(url: str, output_path: Path, method: str = 'GET',
                           data: bytes = None) -> None:
    """Download file with progress bar."""
    print(f"Downloading from {url[:80]}...")

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
            downloaded = block_num * block_size
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  Downloaded: {mb_downloaded:.1f} MB")
            sys.stdout.flush()

    try:
        if method == 'POST' and data:
            # For BioMart POST requests
            req = urllib.request.Request(url, data=data, method='POST')
            with urllib.request.urlopen(req) as response:
                with open(output_path, 'wb') as out_file:
                    out_file.write(response.read())
        else:
            urllib.request.urlretrieve(url, output_path, reporthook)

        print()  # Newline after progress
        print(f"Downloaded to {output_path}")
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        raise


def download_from_biomart(species: str, output_path: Path) -> None:
    """
    Download GO annotations from Ensembl BioMart.

    Parameters
    ----------
    species : str
        Species name ('human' or 'mouse')
    output_path : Path
        Path to save downloaded file
    """
    # BioMart XML query
    # This queries for gene symbol and GO term IDs
    dataset = 'hsapiens_gene_ensembl' if species == 'human' else 'mmusculus_gene_ensembl'

    xml_query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="0" count="" datasetConfigVersion="0.6">
    <Dataset name="{dataset}" interface="default">
        <Attribute name="external_gene_name" />
        <Attribute name="go_id" />
        <Attribute name="name_1006" />
        <Attribute name="namespace_1003" />
    </Dataset>
</Query>"""

    # URL encode the query
    query_url = f"{BIOMART_URL}?query={urllib.parse.quote(xml_query)}"

    # Download
    download_with_progress(query_url, output_path)


def download_from_goa(species: str, output_path: Path) -> None:
    """
    Download GO annotations from Gene Ontology Annotation (GOA) database.

    Parameters
    ----------
    species : str
        Species name ('human' or 'mouse')
    output_path : Path
        Path to save downloaded file
    """
    if species not in GO_ANNOTATION_URLS:
        raise ValueError(f"Unsupported species: {species}. Available: {list(GO_ANNOTATION_URLS.keys())}")

    url = GO_ANNOTATION_URLS[species]
    download_with_progress(url, output_path)


def parse_biomart_file(biomart_path: Path, go_aspects: List[str]) -> Dict[str, Set[str]]:
    """
    Parse BioMart download to extract gene-GO mappings.

    Parameters
    ----------
    biomart_path : Path
        Path to BioMart TSV file
    go_aspects : List[str]
        GO aspects to include ('C', 'F', 'P')

    Returns
    -------
    gene_to_go : Dict[str, Set[str]]
        Mapping of gene symbols to GO term IDs
    """
    print("\nParsing BioMart file...")

    try:
        df = pd.read_csv(biomart_path, sep='\t')
        print(f"Loaded {len(df):,} annotations")
    except Exception as e:
        print(f"ERROR: Failed to read BioMart file: {e}")
        raise

    # Expected columns: Gene name, GO term accession, GO term name, GO domain
    gene_col = 'Gene name' if 'Gene name' in df.columns else df.columns[0]
    go_col = 'GO term accession' if 'GO term accession' in df.columns else df.columns[1]
    aspect_col = 'GO domain' if 'GO domain' in df.columns else df.columns[3]

    # Filter out rows with missing GO terms
    df = df[df[go_col].notna()]
    df = df[df[gene_col].notna()]

    # Filter by GO aspect if specified
    if go_aspects and aspect_col in df.columns:
        df = df[df[aspect_col].isin(go_aspects)]
        print(f"After filtering by GO aspects: {len(df):,} annotations")

    # Build gene to GO mapping
    gene_to_go = {}
    for gene, go_term in zip(df[gene_col], df[go_col]):
        gene = str(gene).strip()
        go_term = str(go_term).strip()

        if gene and go_term and go_term.startswith('GO:'):
            if gene not in gene_to_go:
                gene_to_go[gene] = set()
            gene_to_go[gene].add(go_term)

    print(f"\nExtracted annotations for {len(gene_to_go):,} genes")

    return gene_to_go


def parse_gaf_file(gaf_path: Path, go_aspects: List[str]) -> Dict[str, Set[str]]:
    """
    Parse GAF (Gene Association File) format.

    Parameters
    ----------
    gaf_path : Path
        Path to GAF file (can be gzipped)
    go_aspects : List[str]
        GO aspects to include ('C', 'F', 'P')

    Returns
    -------
    gene_to_go : Dict[str, Set[str]]
        Mapping of gene symbols to GO term IDs
    """
    print("\nParsing GAF file...")

    import gzip

    # Open file (gzipped or not)
    if gaf_path.suffix == '.gz':
        file_handle = gzip.open(gaf_path, 'rt')
    else:
        file_handle = open(gaf_path, 'r')

    gene_to_go = {}
    line_count = 0

    try:
        for line in file_handle:
            # Skip comment lines
            if line.startswith('!'):
                continue

            line_count += 1
            if line_count % 100000 == 0:
                print(f"  Processed {line_count:,} lines...")

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            # GAF format columns:
            # 0: DB, 1: DB Object ID, 2: DB Object Symbol, 3: Qualifier,
            # 4: GO ID, 5: DB:Reference, 6: Evidence Code, 7: With (or) From,
            # 8: Aspect, 9: DB Object Name, ...

            gene_symbol = fields[2]
            go_id = fields[4]
            aspect = fields[8]

            # Filter by aspect
            if go_aspects and aspect not in go_aspects:
                continue

            # Add to mapping
            if gene_symbol and go_id.startswith('GO:'):
                if gene_symbol not in gene_to_go:
                    gene_to_go[gene_symbol] = set()
                gene_to_go[gene_symbol].add(go_id)

    finally:
        file_handle.close()

    print(f"\nProcessed {line_count:,} total lines")
    print(f"Extracted annotations for {len(gene_to_go):,} genes")

    return gene_to_go


def main():
    parser = argparse.ArgumentParser(
        description='Download and parse Gene Ontology annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download GO annotations for human genes (all aspects)
  python utilities/download_gene_ontology.py --output data/gene_to_go.json

  # Download only biological process annotations
  python utilities/download_gene_ontology.py \\
    --output data/go_biological_process.json \\
    --aspect biological_process

  # Use GOA database instead of BioMart
  python utilities/download_gene_ontology.py \\
    --output data/gene_to_go.json \\
    --source goa

  # Parse existing annotation file
  python utilities/download_gene_ontology.py \\
    --gaf goa_human.gaf.gz \\
    --output data/gene_to_go.json
        """
    )

    parser.add_argument('--species', type=str, default='human',
                        choices=['human', 'mouse'],
                        help='Species (default: human)')
    parser.add_argument('--source', type=str, default='biomart',
                        choices=['biomart', 'goa'],
                        help='Annotation source (default: biomart)')
    parser.add_argument('--aspect', type=str, default='all',
                        choices=['all', 'biological_process', 'molecular_function', 'cellular_component'],
                        help='GO aspect to include (default: all)')
    parser.add_argument('--gaf', type=str,
                        help='Path to existing GAF file (skips download)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--keep-raw', action='store_true',
                        help='Keep downloaded raw file after processing')
    parser.add_argument('--min-genes', type=int, default=1,
                        help='Minimum genes per GO term (default: 1)')

    args = parser.parse_args()

    output_path = Path(args.output)
    go_aspects = GO_ASPECTS[args.aspect]

    print("="*60)
    print("Gene Ontology Annotation Download and Parser")
    print("="*60)
    print(f"Species: {args.species}")
    print(f"Source: {args.source}")
    print(f"GO aspects: {args.aspect} {go_aspects}")
    print(f"Output file: {output_path}")
    print()

    # Determine annotation file path
    if args.gaf:
        # Use provided file
        anno_path = Path(args.gaf)
        if not anno_path.exists():
            print(f"ERROR: Annotation file not found: {anno_path}")
            sys.exit(1)
        print(f"Using existing annotation file: {anno_path}")
        temp_file = None
        source = 'gaf'  # Force GAF parser
    else:
        # Download annotations
        if args.keep_raw:
            # Download to same directory as output
            if args.source == 'biomart':
                anno_path = output_path.parent / f"go_annotations_{args.species}_biomart.tsv"
            else:
                anno_path = output_path.parent / f"go_annotations_{args.species}_goa.gaf.gz"
        else:
            # Download to temp file
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.tsv' if args.source == 'biomart' else '.gaf.gz',
                delete=False
            )
            temp_file.close()
            anno_path = Path(temp_file.name)

        # Download
        if args.source == 'biomart':
            download_from_biomart(args.species, anno_path)
            source = 'biomart'
        else:
            download_from_goa(args.species, anno_path)
            source = 'gaf'

    try:
        # Parse annotation file
        if source == 'biomart':
            gene_to_go = parse_biomart_file(anno_path, go_aspects)
        else:
            gene_to_go = parse_gaf_file(anno_path, go_aspects)

        # Convert sets to lists for JSON serialization
        gene_to_go_list = {gene: sorted(list(go_terms))
                           for gene, go_terms in gene_to_go.items()
                           if len(go_terms) >= args.min_genes}

        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(gene_to_go_list, f, indent=2)

        print(f"\nSaved to {output_path}")

        # Print summary statistics
        total_genes = len(gene_to_go_list)
        total_annotations = sum(len(terms) for terms in gene_to_go_list.values())
        avg_terms_per_gene = total_annotations / total_genes if total_genes > 0 else 0

        print("\nSummary:")
        print(f"  Genes with GO annotations: {total_genes:,}")
        print(f"  Total annotations: {total_annotations:,}")
        print(f"  Average GO terms per gene: {avg_terms_per_gene:.1f}")
        print(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

        # Show distribution
        term_counts = [len(terms) for terms in gene_to_go_list.values()]
        if term_counts:
            print(f"\nGO terms per gene distribution:")
            print(f"  Min: {min(term_counts)}")
            print(f"  Median: {sorted(term_counts)[len(term_counts)//2]}")
            print(f"  Max: {max(term_counts)}")

        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"\nGene Ontology database ready: {output_path}")
        print("\nYou can now use this file with:")
        print(f"  python scripts/validate_discoveries.py \\")
        print(f"    --go-annotations {output_path} \\")
        print(f"    ...")

    finally:
        # Clean up temp file if needed
        if temp_file and anno_path.exists() and not args.keep_raw:
            print(f"\nCleaning up temporary file...")
            anno_path.unlink()


if __name__ == '__main__':
    main()
