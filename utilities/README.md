# SIEVE Utilities

This directory contains utility scripts for downloading and preparing external databases used in Phase 3 biological validation.

## Overview

Phase 3C biological validation requires three external databases:
1. **ClinVar** - Pathogenic variant database
2. **GWAS Catalog** - Gene-disease associations
3. **Gene Ontology** - GO term annotations for enrichment analysis

These scripts automatically download, parse, and format the data for use with SIEVE.

## ⚠️ IMPORTANT: Genome Version Selection

**All scripts support genome version selection to match your VCF data.**

- **ClinVar**: `--genome {GRCh37, GRCh38}` (default: GRCh37)
- **GWAS Catalog**: `--genome {GRCh37, GRCh38}` (default: GRCh37)
- **Gene Ontology**: `--species {human, mouse}` (default: human)

**You must use the same genome build across all databases and match your VCF reference genome.**

### For GRCh37/hg19 Data (default)
```bash
python utilities/download_clinvar.py --genome GRCh37 --output data/clinvar_grch37.tsv
python utilities/download_gwas_catalog.py --genome GRCh37 --output data/gwas_grch37.tsv
python utilities/download_gene_ontology.py --species human --output data/gene_to_go.json
```

### For GRCh38/hg38 Data
```bash
python utilities/download_clinvar.py --genome GRCh38 --output data/clinvar_grch38.tsv
python utilities/download_gwas_catalog.py --genome GRCh38 --output data/gwas_grch38.tsv
python utilities/download_gene_ontology.py --species human --output data/gene_to_go.json
```

**Note**: Gene Ontology annotations are genome-agnostic (gene symbols → GO terms), so the same GO file works for both genome builds.

---

## Quick Start

**For GRCh37 data** (most common for research cohorts):

```bash
# 1. Download ClinVar (GRCh37)
python utilities/download_clinvar.py --genome GRCh37 --output data/clinvar_grch37.tsv

# 2. Download GWAS Catalog (GRCh37)
python utilities/download_gwas_catalog.py --genome GRCh37 --output data/gwas_grch37.tsv

# 3. Download Gene Ontology annotations (genome-agnostic)
python utilities/download_gene_ontology.py --species human --output data/gene_to_go.json
```

Then use with Phase 3C validation:
```bash
python scripts/validate_discoveries.py \
    --variant-rankings results/sieve_variant_rankings.csv \
    --gene-rankings results/sieve_gene_rankings.csv \
    --clinvar-db data/clinvar_grch37.tsv \
    --gwas-db data/gwas_grch37.tsv \
    --go-annotations data/gene_to_go.json \
    --output-dir results/biological_validation
```

**For GRCh38 data**, simply change `--genome GRCh37` to `--genome GRCh38` in the first two commands.

---

## 1. ClinVar Downloader (`download_clinvar.py`)

Downloads and parses the ClinVar database of clinically relevant genetic variants.

### Usage

```bash
# Basic usage (GRCh37)
python utilities/download_clinvar.py --output data/clinvar_grch37.tsv

# GRCh38 genome
python utilities/download_clinvar.py --genome GRCh38 --output data/clinvar_grch38.tsv

# Use existing VCF file
python utilities/download_clinvar.py --vcf clinvar.vcf.gz --output data/clinvar.tsv

# Test mode (first 1000 pathogenic variants only)
python utilities/download_clinvar.py --output test_clinvar.tsv --max-variants 1000

# Keep downloaded VCF file for later use
python utilities/download_clinvar.py --output data/clinvar.tsv --keep-vcf
```

### Output Format

TSV file with columns:
- `chrom` - Chromosome (without 'chr' prefix)
- `pos` - Genomic position
- `ref` - Reference allele
- `alt` - Alternate allele
- `gene` - Gene symbol
- `clinical_significance` - Clinical interpretation (Pathogenic/Likely pathogenic)

### Parameters

- `--genome` - Reference genome (GRCh37 or GRCh38, default: GRCh37)
- `--vcf` - Use existing VCF file instead of downloading
- `--output` - Output TSV file path (required)
- `--max-variants` - Limit number of pathogenic variants (for testing)
- `--keep-vcf` - Keep downloaded VCF file

### Expected Output

```
Loaded 250,000+ pathogenic/likely pathogenic variants
Unique genes: 5,000+
File size: ~50 MB
```

### Download Time

- **First run**: 5-10 minutes (downloads ~100 MB VCF file)
- **Subsequent runs with --keep-vcf**: <1 minute (parses existing file)

---

## 2. GWAS Catalog Downloader (`download_gwas_catalog.py`)

Downloads and parses the NHGRI-EBI GWAS Catalog of genome-wide association studies.

### Usage

```bash
# Basic usage (genome-wide significant hits)
python utilities/download_gwas_catalog.py --output data/gwas_catalog.tsv

# More lenient p-value threshold
python utilities/download_gwas_catalog.py \
    --output data/gwas_lenient.tsv \
    --min-pvalue 1e-5

# Use existing GWAS file
python utilities/download_gwas_catalog.py \
    --gwas gwas_download.tsv \
    --output data/parsed_gwas.tsv

# Keep raw download
python utilities/download_gwas_catalog.py \
    --output data/gwas.tsv \
    --keep-raw
```

### Output Format

TSV file with columns:
- `gene` - Gene symbol
- `disease_trait` - Disease or trait name
- `chr` - Chromosome
- `pos` - Genomic position
- `snp_id` - SNP identifier (rs number)
- `p_value` - Association p-value
- `risk_allele` - Risk allele information

### Parameters

- `--genome` - Reference genome (GRCh37 or GRCh38, default: GRCh37)
- `--gwas` - Use existing GWAS file instead of downloading
- `--output` - Output TSV file path (required)
- `--min-pvalue` - P-value threshold (default: 5e-8, genome-wide significance)
- `--keep-raw` - Keep raw download

### Expected Output

```
Extracted 50,000+ gene-disease associations
Unique genes: 10,000+
Unique disease traits: 5,000+
File size: ~10 MB
```

### Download Time

- **First run**: 2-5 minutes (downloads ~30 MB catalog)
- **Subsequent runs with --keep-raw**: <1 minute

---

## 3. Gene Ontology Downloader (`download_gene_ontology.py`)

Downloads and parses Gene Ontology annotations for enrichment analysis.

### Usage

```bash
# Basic usage (all GO aspects)
python utilities/download_gene_ontology.py --output data/gene_to_go.json

# Biological process only
python utilities/download_gene_ontology.py \
    --output data/go_biological_process.json \
    --aspect biological_process

# Molecular function only
python utilities/download_gene_ontology.py \
    --output data/go_molecular_function.json \
    --aspect molecular_function

# Use GOA database instead of BioMart
python utilities/download_gene_ontology.py \
    --output data/gene_to_go.json \
    --source goa

# Parse existing GAF file
python utilities/download_gene_ontology.py \
    --gaf goa_human.gaf.gz \
    --output data/gene_to_go.json
```

### Output Format

JSON file mapping gene symbols to GO term IDs:
```json
{
  "TP53": ["GO:0000785", "GO:0003677", "GO:0006355", ...],
  "BRCA1": ["GO:0000724", "GO:0003677", "GO:0006281", ...],
  ...
}
```

### Parameters

- `--species` - Species (human or mouse, default: human)
- `--source` - Annotation source (biomart or goa, default: biomart)
- `--aspect` - GO aspect (all, biological_process, molecular_function, cellular_component)
- `--gaf` - Use existing GAF file instead of downloading
- `--output` - Output JSON file path (required)
- `--keep-raw` - Keep raw download
- `--min-genes` - Minimum genes per GO term (default: 1)

### GO Aspects

- **biological_process** (P) - Biological processes (e.g., cell division, apoptosis)
- **molecular_function** (F) - Molecular activities (e.g., DNA binding, kinase activity)
- **cellular_component** (C) - Cellular locations (e.g., nucleus, mitochondrion)
- **all** - All three aspects combined (recommended)

### Expected Output

```
Genes with GO annotations: 20,000+
Total annotations: 200,000+
Average GO terms per gene: 10-15
File size: ~5 MB
```

### Download Time

- **BioMart source**: 1-3 minutes (direct API query)
- **GOA source**: 2-5 minutes (downloads ~50 MB GAF file)

---

## Troubleshooting

### Genome Version Mismatch

**CRITICAL**: Ensure all databases use the same genome build as your VCF data.

**Symptoms of mismatch**:
- Variants in ClinVar but not matching your positions
- No overlap between GWAS hits and your variants
- Validation failing with "0 variants found"

**Solution**:
1. Check your VCF reference genome (usually in VCF header):
   ```bash
   zgrep "^##reference" your_data.vcf.gz
   # Example output: ##reference=file:///human_g1k_v37.fasta (= GRCh37)
   # Example output: ##reference=GRCh38 (= GRCh38)
   ```

2. Check chromosome naming in your VCF:
   - GRCh37 typically: `1, 2, 3, ..., X, Y` or `chr1, chr2, ...`
   - GRCh38 typically: `chr1, chr2, chr3, ..., chrX, chrY`

3. Re-download databases with correct `--genome` parameter:
   ```bash
   # If your VCF is GRCh38
   python utilities/download_clinvar.py --genome GRCh38 --output data/clinvar_grch38.tsv
   python utilities/download_gwas_catalog.py --genome GRCh38 --output data/gwas_grch38.tsv
   ```

**Note**: SIEVE automatically handles chromosome naming (removes "chr" prefix if present), so `chr1` and `1` are treated as equivalent.

### Download Failures

If downloads fail due to network issues:

1. **Use existing files**: Download manually and use `--vcf`, `--gwas`, or `--gaf` parameters
2. **Alternative sources**: Try different `--source` for GO annotations
3. **Retry**: Network timeouts are temporary, try again later

### Parse Errors

If parsing fails:

1. **Check file format**: Ensure downloaded file is not corrupted
2. **Update URLs**: Database URLs may change, check script comments for latest URLs
3. **File permissions**: Ensure write permissions for output directory

### Memory Issues

For very large downloads:

1. **Use --max-variants**: Limit ClinVar to subset (e.g., --max-variants 10000)
2. **Filter GO terms**: Use specific --aspect instead of 'all'
3. **Increase threshold**: Use higher --min-pvalue for GWAS

---

## Data Sources

### ClinVar
- **Provider**: NCBI
- **URL**: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/
- **Update frequency**: Weekly
- **License**: Public domain

### GWAS Catalog
- **Provider**: NHGRI-EBI
- **URL**: https://www.ebi.ac.uk/gwas/
- **Update frequency**: Monthly
- **License**: Creative Commons

### Gene Ontology
- **Provider**: Gene Ontology Consortium
- **URL**: http://geneontology.org/
- **Update frequency**: Daily
- **License**: Creative Commons

---

## Reference Genome Builds

### GRCh37 vs GRCh38

**GRCh37** (also known as hg19):
- Released: 2009
- Most common in research datasets and older cohorts
- UCSC nomenclature: hg19
- Chromosome naming: `1, 2, 3, ..., X, Y` or `chr1, chr2, ...`
- Use this if: Your VCF was aligned to b37, hg19, or hs37d5

**GRCh38** (also known as hg38):
- Released: 2013
- Current reference genome
- UCSC nomenclature: hg38
- Chromosome naming: `chr1, chr2, chr3, ..., chrX, chrY`
- Use this if: Your VCF was aligned to GRCh38 or hg38

### How Database Genome Versions Differ

**ClinVar**:
- Separate VCF files for GRCh37 and GRCh38
- Variant positions differ between builds (due to assembly changes)
- Always use the version matching your VCF alignment

**GWAS Catalog**:
- Single file contains both GRCh37 and GRCh38 coordinates
- Script extracts positions for the specified genome build
- Positions can differ by several kilobases between builds

**Gene Ontology**:
- Genome-independent (maps gene symbols to GO terms)
- Same file works for both GRCh37 and GRCh38
- Only species matters (human vs mouse)

### Position Coordinate Differences

Example: Same variant in different builds:
```
GRCh37: chr1:12345678 A>G
GRCh38: chr1:12456789 A>G  (positions differ!)
```

This is why **using the wrong genome build will result in NO overlaps** with your variant discoveries.

---

## Updating Databases

To keep databases current, re-run the download scripts periodically:

```bash
# Recommended: quarterly updates
# Create dated versions with genome build in filename
python utilities/download_clinvar.py \
    --genome GRCh37 \
    --output data/clinvar_grch37_2026_01.tsv

python utilities/download_gwas_catalog.py \
    --genome GRCh37 \
    --output data/gwas_grch37_2026_01.tsv

python utilities/download_gene_ontology.py \
    --species human \
    --output data/go_human_2026_01.json
```

**Best practice**: Include genome build in filenames to avoid confusion:
- ✅ `clinvar_grch37_2026_01.tsv`
- ✅ `gwas_grch38_latest.tsv`
- ❌ `clinvar.tsv` (ambiguous)

---

## Testing

Test the utilities with small datasets before full downloads:

```bash
# Test ClinVar (first 1000 variants, specify genome)
python utilities/download_clinvar.py \
    --genome GRCh37 \
    --output test_data/test_clinvar_grch37.tsv \
    --max-variants 1000

# Test GWAS (lenient threshold, fewer results, specify genome)
python utilities/download_gwas_catalog.py \
    --genome GRCh37 \
    --output test_data/test_gwas_grch37.tsv \
    --min-pvalue 1e-10

# Test GO (single aspect, specify species)
python utilities/download_gene_ontology.py \
    --species human \
    --output test_data/test_go_human.json \
    --aspect biological_process
```

---

## Integration with SIEVE

Once databases are downloaded with the correct genome build, use them with Phase 3C validation:

**For GRCh37 data**:
```bash
# Complete Phase 3C validation pipeline
python scripts/validate_discoveries.py \
    --variant-rankings results/sieve_variant_rankings.csv \
    --gene-rankings results/sieve_gene_rankings.csv \
    --clinvar-db data/clinvar_grch37.tsv \
    --gwas-db data/gwas_grch37.tsv \
    --go-annotations data/gene_to_go.json \
    --output-dir results/biological_validation \
    --min-go-overlap 3 \
    --fdr-threshold 0.05
```

**For GRCh38 data**:
```bash
python scripts/validate_discoveries.py \
    --variant-rankings results/sieve_variant_rankings.csv \
    --gene-rankings results/sieve_gene_rankings.csv \
    --clinvar-db data/clinvar_grch38.tsv \
    --gwas-db data/gwas_grch38.tsv \
    --go-annotations data/gene_to_go.json \
    --output-dir results/biological_validation \
    --min-go-overlap 3 \
    --fdr-threshold 0.05
```

**Important**: Ensure the genome build of your databases matches your VCF reference genome. Mismatched builds will result in zero overlaps.

See `PHASE3_GUIDE.md` for complete validation workflow.

---

## Contributing

If database URLs change or formats are updated, please:
1. Update the URL constants at the top of each script
2. Update parsing logic if needed
3. Add version information to comments
4. Test with current database versions

---

## Support

For issues or questions:
- Check script help: `python utilities/<script>.py --help`
- Review error messages and logs
- Verify network connectivity and firewall settings
- Check database provider websites for changes

---

Last updated: 2026-01-28
