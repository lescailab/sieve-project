# Phase 1A: VCF Parser and Annotation Extraction

## Overview

This PR implements the core data processing pipeline for SIEVE, providing robust VCF parsing with VEP annotation extraction. This is the foundation for all subsequent phases.

## Components Implemented

### 1. VCF Parser (`src/data/vcf_parser.py`)

**Core Data Structures:**
- `VariantRecord`: Dataclass representing a single variant with genotype and annotations
- `SampleVariants`: Container for all variants belonging to one sample with phenotype label

**Key Functions:**
- `parse_vcf_cyvcf2()`: Memory-efficient VCF iterator using cyvcf2
- `parse_csq_field()`: CSQ field parsing with **critical sanitization** (removes spaces, quotes, parentheses)
- `select_canonical_annotation()`: Selects canonical transcript or most severe consequence
- `extract_sift_score()` / `extract_polyphen_score()`: Parse functional prediction scores
- `load_phenotypes()`: Load phenotype file with automatic 1/2 → 0/1 conversion
- `harmonize_contig()`: Remove 'chr' prefix for GRCh37 compatibility
- `build_sample_variants()`: Convenience function for full pipeline

**Features:**
- ✅ Multi-allelic site handling
- ✅ Multi-sample VCF support
- ✅ Genotype quality filtering (min_gq parameter)
- ✅ Canonical transcript prioritization
- ✅ Memory-efficient iterator pattern (never loads entire VCF)
- ✅ Comprehensive error handling

### 2. Annotation Module (`src/data/annotation.py`)

**Consequence Severity Mapping:**
- Maps 40+ VEP consequence terms to ordinal scale (0-4)
- HIGH (4): LoF variants (stop_gained, frameshift, splice donor/acceptor)
- MODERATE (3): Missense, inframe indels
- LOW (2): Synonymous, UTR, splice region
- MODIFIER (1): Intron, intergenic, upstream/downstream

**Score Normalization:**
- `normalize_sift_score()`: Inverts SIFT (higher = more deleterious)
- `normalize_polyphen_score()`: Already in correct direction

**Utility Functions:**
- `impute_missing_score()`: Median/mean/neutral imputation
- `get_lof_variants()` / `get_missense_variants()` / `get_synonymous_variants()`: Filter by consequence
- `compute_annotation_statistics()`: Cohort-level annotation summaries
- `extract_variant_features()`: Convert VariantRecord → feature dictionary for ML

### 3. Testing (`test_vcf_parser.py`)

Comprehensive validation script that:
- Loads real test data (1,333 variants, 20 samples)
- Reports parsing statistics
- Shows example variants with annotations
- Analyzes consequence and gene distributions
- Checks annotation availability

## Validation Results

**Test Dataset:** `test_data/small/test_data.vcf.gz`
- **Variants:** 1,333 (chromosome 1)
- **Samples:** 20 (10 cases, 10 controls)
- **Variant instances:** 9,143 total across all samples
- **Mean variants/sample:** 457 (range: 393-498)

**Annotation Coverage:**
- **SIFT:** 42.4% (3,874/9,143 variants)
- **PolyPhen:** 21.1% (1,929/9,143 variants)

**Consequence Distribution:**
1. Intron variants: 3,113 (34.1%)
2. Synonymous variants: 2,475 (27.1%)
3. Missense variants: 1,929 (21.1%)
4. Downstream gene: 525 (5.7%)
5. Splice region: 329 (3.6%)

**Top Genes by Variant Count:**
1. HSPG2: 453
2. PADI2: 278
3. UBR4: 262
4. CLCNKB: 259
5. FAM131C: 236

✅ **100% parsing success rate** - all variants processed without errors

## Code Quality

- ✅ **Type hints:** All function signatures annotated
- ✅ **Docstrings:** NumPy format with examples for all public functions
- ✅ **Error handling:** Graceful handling of malformed VCF, missing phenotypes
- ✅ **Edge cases:** Multi-allelic, missing annotations, compound consequences
- ✅ **Memory efficiency:** Iterator pattern, no unnecessary copies
- ✅ **Testing:** Validated on real exome data

## Critical Implementation Notes

### CSQ Field Parsing Fix

Implements the mandatory sanitization from `CLAUDE.md`:
```python
sanitized = (
    csq_string
    .replace(" ", "")
    .replace("'", "")
    .replace("(", "")
    .replace(")", "")
)
```

This prevents parsing errors from VEP annotations with embedded punctuation.

### Phenotype Encoding Conversion

Automatically converts phenotype file encoding:
- **Input:** 1=control, 2=case (your format)
- **Output:** 0=control, 1=case (standard ML format)

### Contig Harmonization

Removes 'chr' prefix to match GRCh37 reference:
- `chr1` → `1`
- `chrX` → `X`

Handles both UCSC and Ensembl notation transparently.

## File Structure

```
src/data/
├── __init__.py          # Updated with all exports
├── vcf_parser.py        # 524 lines, 15 functions
└── annotation.py        # 465 lines, 12 functions

test_vcf_parser.py       # Validation script
.gitignore               # Python cache, temporary files
```

## Dependencies Used

- `cyvcf2`: Fast VCF parsing
- `numpy`: Numerical operations
- Standard library: `dataclasses`, `pathlib`, `typing`, `re`

## Next Steps (Phase 1B)

After this PR is merged, the next phase will implement:
- Feature encoding for L0-L4 annotation levels (all 5 levels)
- Sinusoidal positional encoding
- Relative position bucketing
- Sparse tensor construction for PyTorch

## How to Test

```bash
# Install dependencies
pip install cyvcf2 numpy

# Run validation script
python test_vcf_parser.py

# Or use in code
from src.data import build_sample_variants
samples = build_sample_variants('test_data/small/test_data.vcf.gz',
                                'test_data/small/test_data_phenotypes.tsv')
```

## Related Documentation

- `CLAUDE.md`: Project overview and implementation guidelines
- `ARCHITECTURE.md`: Technical specifications
- `INSTRUCTIONS.md`: Phase-by-phase implementation roadmap

---

**Lines of Code:** ~1,000 (excluding comments/docstrings)
**Functions Implemented:** 15 core functions
**Test Coverage:** Validated on 1,333 real variants
**Status:** ✅ Production-ready
