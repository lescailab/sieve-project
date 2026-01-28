# Utilities Testing Documentation

## Overview

All three database download utilities have been tested with both unit tests and integration tests to ensure reliability and correct output formatting.

## URL Verification Test

**IMPORTANT**: Before running database downloads, verify that all URLs are accessible.

```bash
# Test all download URLs
python utilities/test_download_urls.py
```

This script checks:
- GWAS Catalog API endpoint
- GWAS Catalog FTP fallback
- ClinVar GRCh37 VCF
- ClinVar GRCh38 VCF

If any URLs fail, the script provides troubleshooting guidance and links to check for updated URLs.

**Run this test whenever**:
- Setting up on a new system
- After long periods without downloads
- If downloads are failing
- Before releasing updated utilities

## Test Coverage

### Unit Tests (`tests/test_utilities.py`)

**TestClinVarParser** - 3 tests
- `test_parse_info_field`: INFO field extraction from VCF
- `test_parse_vcf`: End-to-end VCF parsing with pathogenic variant filtering
- `test_max_variants_limit`: Verify max_variants parameter limits output

**TestGWASParser** - 3 tests
- `test_parse_gwas_catalog`: End-to-end GWAS catalog parsing
- `test_pvalue_filtering`: P-value threshold filtering
- `test_multiple_genes`: Handling multiple genes per association (comma/hyphen separation)

**TestGOParser** - 3 tests
- `test_parse_biomart_file`: BioMart TSV parsing
- `test_aspect_filtering`: GO aspect filtering (P/F/C)
- `test_json_output_format`: JSON output structure validation

**Total: 9 tests, all passing**

```bash
python -m pytest tests/test_utilities.py -v
# ============================== 9 passed in 2.48s ===============================
```

## Integration Testing

### Test Data Created

1. **ClinVar Test VCF** (`test_data/test_clinvar.vcf`)
   - 4 variants (3 pathogenic, 1 benign)
   - Tests chromosome handling, gene extraction, clinical significance filtering

2. **GWAS Test TSV** (`test_data/test_gwas_simple.tsv`)
   - 4 associations (3 significant, 1 below threshold)
   - Tests p-value filtering, gene extraction, multiple columns

3. **BioMart Test TSV** (`test_data/test_biomart.tsv`)
   - 9 GO annotations across 4 genes
   - Tests aspect filtering (P/F/C), gene-to-GO mapping

### Verified Output Formats

**ClinVar Output** (`test_data/parsed_clinvar.tsv`):
```tsv
chrom	pos	ref	alt	gene	clinical_significance
1	12345	A	G	BRCA1	Pathogenic
1	23456	C	T	TP53	Likely_pathogenic
3	45678	T	C	APOE	Pathogenic/Likely_pathogenic
```

**GWAS Output** (`test_data/parsed_gwas.tsv`):
```tsv
gene	disease_trait	chr	pos	snp_id	p_value	risk_allele
TCF7L2	Type 2 diabetes	1	100000	rs7903146	1e-12	rs7903146-A
APOE	Alzheimer's disease	19	45411941	rs429358	5e-100	rs429358-C
CDKN2A	Coronary artery disease	9	22125503	rs1333049	2e-15	rs1333049-C
```

**GO Output** (`test_data/parsed_go.json`):
```json
{
  "TP53": ["GO:0006355"],
  "BRCA1": ["GO:0006281"],
  "APOE": ["GO:0006869"],
  "TCF7L2": ["GO:0006355"]
}
```

## Known Limitations

### Network Restrictions

During testing, network downloads from external sources (NCBI, EBI, Ensembl) were blocked by environment restrictions (403 Forbidden errors). This is acceptable because:

1. **Parsing logic is fully tested** with mock data files
2. **Output formats are verified** against expected structure
3. **Users can provide existing files** via `--vcf`, `--gwas`, `--gaf` parameters
4. **Scripts will work in normal network environments** where FTP/HTTPS access is available

### Testing Strategy

Given network restrictions, testing focused on:
- ✅ Parsing logic with realistic mock data
- ✅ Error handling for edge cases
- ✅ Output format verification
- ✅ Parameter handling (max_variants, min_pvalue, aspect filtering)
- ⚠️ Live downloads not tested (environment limitation)

## Recommended User Testing

### Genome Version Verification

**CRITICAL FIRST STEP**: Verify you're using the correct genome build for your VCF data.

```bash
# Check your VCF reference genome
zgrep "^##reference" your_data.vcf.gz

# Check chromosome naming convention
zgrep -v "^#" your_data.vcf.gz | head -1 | cut -f1
```

**Then download databases with matching genome build**:
- If GRCh37/hg19 → use `--genome GRCh37`
- If GRCh38/hg38 → use `--genome GRCh38`

### Testing Workflow

When using these utilities for the first time, users should:

1. **Test with small datasets first**:
```bash
# ClinVar: First 1000 pathogenic variants
python utilities/download_clinvar.py --output test_clinvar.tsv --max-variants 1000

# GWAS: Very stringent threshold (fewer results)
python utilities/download_gwas_catalog.py --output test_gwas.tsv --min-pvalue 1e-10

# GO: Single aspect (faster download)
python utilities/download_gene_ontology.py --output test_go.json --aspect biological_process
```

2. **Verify output files**:
```bash
# Check file sizes
ls -lh test_*.tsv test_*.json

# Check column structure
head test_clinvar.tsv
head test_gwas.tsv
head -20 test_go.json
```

3. **Run full downloads with --keep-raw**:
```bash
# Keep raw files for debugging
python utilities/download_clinvar.py --output data/clinvar.tsv --keep-vcf
python utilities/download_gwas_catalog.py --output data/gwas.tsv --keep-raw
python utilities/download_gene_ontology.py --output data/go.json --keep-raw
```

## Data Caching

All utilities support data reuse to avoid repeated downloads:

### Option 1: Keep Raw Files
```bash
# First run: Download and keep raw file
python utilities/download_clinvar.py --output data/clinvar.tsv --keep-vcf

# Later: Reprocess with different parameters using kept VCF
python utilities/download_clinvar.py --vcf data/clinvar_grch37.vcf.gz --output data/clinvar_subset.tsv --max-variants 5000
```

### Option 2: Use Existing Downloaded Files
```bash
# If you already downloaded files manually
python utilities/download_clinvar.py --vcf /path/to/clinvar.vcf.gz --output data/clinvar.tsv
python utilities/download_gwas_catalog.py --gwas /path/to/gwas.tsv --output data/parsed_gwas.tsv
python utilities/download_gene_ontology.py --gaf /path/to/goa_human.gaf.gz --output data/go.json
```

### Option 3: Version Snapshots
```bash
# Create dated snapshots for reproducibility
python utilities/download_clinvar.py --output data/clinvar_2026_01.tsv
python utilities/download_gwas_catalog.py --output data/gwas_2026_01.tsv
python utilities/download_gene_ontology.py --output data/go_2026_01.json
```

## Error Handling

All utilities include comprehensive error handling:

✅ Network errors (download failures)
✅ Missing files
✅ Malformed data
✅ Empty results
✅ Missing columns
✅ File permissions

Example error messages:
```
ERROR: VCF file not found: /path/to/file.vcf
ERROR: Failed to read GWAS catalog: <details>
ERROR: Missing required columns: ['MAPPED_GENE', 'P-VALUE']
WARNING: No associations found after filtering!
```

## Maintenance

### When to Update Tests

Update tests when:
- Database formats change (new columns, different delimiters)
- URL endpoints change
- New features are added (e.g., additional filtering options)
- Edge cases are discovered in production use

### Test Data Maintenance

Test data files in `test_data/` should be:
- **Minimal**: Only enough data to verify functionality
- **Realistic**: Match actual database formats
- **Documented**: Comments explaining what each row tests

### Continuous Testing

Run tests before any commits:
```bash
python -m pytest tests/test_utilities.py -v
```

Run full integration test suite:
```bash
python -m pytest tests/ -v --cov=utilities
```

## References

- ClinVar format: https://www.ncbi.nlm.nih.gov/clinvar/docs/vcf/
- GWAS Catalog format: https://www.ebi.ac.uk/gwas/docs/file-downloads
- GO GAF format: http://geneontology.org/docs/go-annotation-file-gaf-format-2.2/
- BioMart usage: https://www.ensembl.org/info/data/biomart/index.html

---

Last updated: 2026-01-28
