# Phase 3 Complete Pipeline: Explainability, Validation & Discovery

## Overview

Phase 3 is now **fully implemented** with a complete pipeline for:
- **Phase 3A**: Explainability infrastructure (integrated gradients, attention analysis, variant ranking)
- **Phase 3B**: Enhanced epistasis detection (SHAP interactions, counterfactual validation)
- **Phase 3C**: Biological validation (ClinVar, GWAS, GO enrichment)

## Complete Workflow

```
Trained Model
     ↓
[Phase 3A] explain.py → Variant rankings, gene rankings, interactions
     ↓
[Phase 3B] validate_epistasis.py → Validated epistatic interactions
     ↓
[Phase 3C] validate_discoveries.py → Biological validation
     ↓
Validated discoveries ready for publication
```

---

## Phase 3A: Explainability & Variant Discovery

### Purpose
Identify which variants and genes drive model predictions.

### Script: `scripts/explain.py`

### What It Does
1. Computes integrated gradients attributions for all variants
2. Ranks variants by importance across samples
3. Aggregates variants to gene-level scores
4. Identifies case-enriched variants
5. Extracts variant-variant interactions from attention patterns

### Usage

```bash
# Basic usage - analyze best model from experiment
python scripts/explain.py \
    --experiment-dir /path/to/L3_attr_medium \
    --preprocessed-data /path/to/preprocessed_discovery.pt \
    --output-dir results/explainability \
    --device cuda

# Analyze specific checkpoint
python scripts/explain.py \
    --checkpoint outputs/L3_attr_medium/fold_0/best_model.pt \
    --config outputs/L3_attr_medium/config.yaml \
    --preprocessed-data /path/to/preprocessed_discovery.pt \
    --output-dir results/explainability_fold0 \
    --device cuda

# Fast mode (skip attention analysis)
python scripts/explain.py \
    --experiment-dir /path/to/L3_attr_medium \
    --preprocessed-data /path/to/preprocessed_discovery.pt \
    --output-dir results/explainability \
    --skip-attention \
    --device cuda
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--experiment-dir` | Path to experiment directory | Required* |
| `--checkpoint` | Path to specific checkpoint | Required* |
| `--config` | Path to config.yaml (if using --checkpoint) | Required with checkpoint |
| `--preprocessed-data` | Path to preprocessed data (.pt) | Required |
| `--output-dir` | Output directory | Required |
| `--n-steps` | Integration steps for IG | 50 |
| `--batch-size` | Batch size | 8 |
| `--skip-attention` | Skip attention analysis (faster) | False |
| `--top-k-variants` | Number of top variants | 100 |
| `--top-k-interactions` | Number of top interactions | 100 |
| `--attention-threshold` | Min attention weight | 0.1 |
| `--device` | Device (cuda/cpu) | cuda |

*Either --experiment-dir or --checkpoint must be provided.

### Output Files

| File | Description |
|------|-------------|
| `attributions.npz` | Raw attribution arrays (numpy) |
| `sieve_variant_rankings.csv` | All variants ranked by importance |
| `sieve_gene_rankings.csv` | Genes ranked by aggregated scores |
| `sieve_top100_variants.csv` | Top 100 most important variants |
| `sieve_top50_genes.csv` | Top 50 most important genes |
| `sieve_interactions.csv` | Variant-variant interactions from attention |

### Output Columns

**Variant Rankings**:
- `position`: Genomic position
- `gene_id`: Gene ID
- `mean_attribution`: Mean absolute attribution across samples
- `max_attribution`: Maximum attribution
- `num_samples`: Number of samples with this variant
- `case_attribution`: Mean attribution in cases
- `control_attribution`: Mean attribution in controls
- `case_control_diff`: Difference (case - control)
- `rank`: Overall rank (1 = most important)

**Gene Rankings**:
- `gene_id`: Gene ID
- `num_variants`: Number of variants in gene
- `gene_score`: Aggregated importance score
- `top_variant_pos`: Position of most important variant
- `gene_rank`: Gene ranking

**Interactions**:
- `variant1_pos`, `variant2_pos`: Positions
- `variant1_gene`, `variant2_gene`: Gene IDs
- `num_samples`: Samples showing interaction
- `mean_attention`: Mean attention weight
- `same_gene`: Whether variants are in same gene
- `distance`: Genomic distance between variants

### Computational Requirements

- **GPU Memory**: ~8GB
- **Time**: 30-60 minutes for full dataset (depends on n_steps)
- **Faster Options**:
  - Reduce `--n-steps` to 20 (less accurate)
  - Use `--skip-attention` (no epistasis detection)
  - Increase `--batch-size` if more memory available

---

## Phase 3B: Epistasis Validation

### Purpose
Validate suspected epistatic interactions using counterfactual perturbation.

### Script: `scripts/validate_epistasis.py`

### What It Does
1. Takes top interactions from explain.py
2. For each interaction, tests 4 conditions:
   - Both variants present
   - Only variant 1 present
   - Only variant 2 present
   - Neither present
3. Computes synergy score: does combined effect exceed sum of individual effects?
4. Identifies synergistic (positive synergy) vs antagonistic (negative synergy) interactions

### Theory

**Synergy Calculation**:
```
effect_v1 = f(v1, ~v2) - f(~v1, ~v2)
effect_v2 = f(~v1, v2) - f(~v1, ~v2)
effect_combined = f(v1, v2) - f(~v1, ~v2)

synergy = effect_combined - effect_v1 - effect_v2
```

- **synergy > 0**: Synergistic (variants work together)
- **synergy < 0**: Antagonistic (variants interfere)
- **synergy ≈ 0**: Independent (no interaction)

### Usage

```bash
python scripts/validate_epistasis.py \
    --interactions results/explainability/sieve_interactions.csv \
    --checkpoint outputs/L3_attr_medium/fold_0/best_model.pt \
    --config outputs/L3_attr_medium/config.yaml \
    --preprocessed-data /path/to/preprocessed_discovery.pt \
    --output-dir results/epistasis_validation \
    --top-k 50 \
    --synergy-threshold 0.05 \
    --device cuda
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--interactions` | Interactions CSV from explain.py | Required |
| `--checkpoint` | Model checkpoint | Required |
| `--config` | Config YAML | Required |
| `--preprocessed-data` | Preprocessed data | Required |
| `--output-dir` | Output directory | Required |
| `--top-k` | Number of interactions to validate | 50 |
| `--synergy-threshold` | Minimum significant synergy | 0.05 |
| `--device` | Device (cuda/cpu) | cuda |

### Output Files

| File | Description |
|------|-------------|
| `epistasis_validation.csv` | All validated interactions with synergy scores |
| `epistasis_significant.csv` | Only significant interactions (\|synergy\| > threshold) |
| `epistasis_summary.yaml` | Summary statistics |

### Output Columns

- `variant1_pos`, `variant2_pos`: Variant positions
- `variant1_gene`, `variant2_gene`: Gene IDs
- `pred_both`: Model prediction with both variants
- `pred_variant1_only`: Prediction with only variant 1
- `pred_variant2_only`: Prediction with only variant 2
- `pred_neither`: Prediction with neither
- `effect_variant1`: Individual effect of variant 1
- `effect_variant2`: Individual effect of variant 2
- `effect_combined`: Combined effect
- `synergy`: Synergy score
- `interaction_type`: 'synergistic' or 'antagonistic'
- `is_significant`: Whether \|synergy\| > threshold

### Interpretation

**Strong Synergistic Interaction** (synergy > 0.1):
- Combined effect much greater than sum of individual effects
- Suggests true epistasis
- **Example**: Variant A has small effect, variant B has small effect, but together they have large effect

**Strong Antagonistic Interaction** (synergy < -0.1):
- Combined effect less than expected from individual effects
- One variant suppresses the other
- **Example**: Variant A increases risk, variant B increases risk, but together they have little effect

---

## Phase 3C: Biological Validation

### Purpose
Validate discoveries against known biological databases.

### Script: `scripts/validate_discoveries.py`

### What It Does
1. **ClinVar Validation**: Checks if discovered variants are known pathogenic
2. **GWAS Validation**: Checks if discovered genes are in disease associations
3. **GO Enrichment**: Tests if genes are enriched in specific functions/pathways
4. **Enrichment Statistics**: Computes significance with multiple testing correction

### Usage

```bash
python scripts/validate_discoveries.py \
    --variant-rankings results/explainability/sieve_variant_rankings.csv \
    --gene-rankings results/explainability/sieve_gene_rankings.csv \
    --output-dir results/validation \
    --clinvar data/clinvar.tsv \
    --gwas data/gwas_catalog.tsv \
    --go-mapping data/gene_to_go.json \
    --disease-terms diabetes obesity \
    --top-k-variants 100 \
    --top-k-genes 50
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--variant-rankings` | Variant rankings CSV | Required |
| `--gene-rankings` | Gene rankings CSV | Required |
| `--output-dir` | Output directory | Required |
| `--clinvar` | ClinVar database (TSV) | Optional |
| `--gwas` | GWAS Catalog (TSV) | Optional |
| `--go-mapping` | Gene-to-GO mapping (JSON) | Optional |
| `--disease-terms` | Disease terms for GWAS filtering | Optional |
| `--top-k-variants` | Number of variants to validate | 100 |
| `--top-k-genes` | Number of genes to validate | 50 |
| `--reference-genome` | Genome build (GRCh37/GRCh38) | GRCh37 |

### Database Formats

**ClinVar TSV** (preprocessed):
```
chrom	pos	ref	alt	gene	clinical_significance
1	12345	A	G	GENE1	Pathogenic
```

**GWAS Catalog TSV**:
```
chrom	pos	gene	trait	p_value	study
1	67890	GENE2	Type 2 diabetes	1e-8	PMID12345
```

**Gene-to-GO JSON**:
```json
{
  "1234": ["GO:0006915", "GO:0042981"],
  "5678": ["GO:0006955", "GO:0002376"]
}
```

### Output Files

| File | Description |
|------|-------------|
| `clinvar_validation.csv` | Top variants with ClinVar annotations |
| `gwas_validation.csv` | Top genes with GWAS associations |
| `go_enrichment.csv` | Enriched GO terms |
| `validation_summary.yaml` | Overall statistics |

### Interpreting Results

**ClinVar Validation**:
- High overlap with pathogenic variants → model discovered known disease variants ✓
- Low overlap → model found novel candidates (interesting!)
- Look for "Pathogenic" or "Likely pathogenic" in `clinvar_significance`

**GWAS Validation**:
- `in_gwas = True` → gene previously associated with disease
- `gwas_studies > 5` → strong, replicated association
- `gwas_traits` → check if traits match your phenotype

**GO Enrichment**:
- `fdr < 0.05` → significantly enriched term
- `fold_enrichment > 2` → strong enrichment
- Look for disease-relevant pathways (e.g., immune response, metabolism)

### Enrichment Interpretation

**Example**:
```
GO Term: Immune Response
Overlap: 15/50 genes
P-value: 1e-8
FDR: 1e-6
Fold Enrichment: 5.2
```

**Interpretation**:
- 15 of your top 50 genes are involved in immune response
- This is 5.2x more than expected by chance
- Highly significant (FDR < 0.001)
- **Conclusion**: Model discovered immune-related genes (makes sense for disease!)

---

## Complete Pipeline Example

### Step 1: Run Explainability Analysis

```bash
# Analyze best model
python scripts/explain.py \
    --experiment-dir /home/shared/sieve-testing/experiments/L3_attr_medium \
    --preprocessed-data /home/shared/sieve-testing/preprocessed_discovery.pt \
    --output-dir /home/shared/sieve-testing/results/explainability \
    --device cuda
```

**Outputs**:
- `results/explainability/sieve_variant_rankings.csv`
- `results/explainability/sieve_gene_rankings.csv`
- `results/explainability/sieve_interactions.csv`

### Step 2: Validate Epistatic Interactions

```bash
# Validate top 50 interactions
python scripts/validate_epistasis.py \
    --interactions /home/shared/sieve-testing/results/explainability/sieve_interactions.csv \
    --checkpoint /home/shared/sieve-testing/experiments/L3_attr_medium/fold_0/best_model.pt \
    --config /home/shared/sieve-testing/experiments/L3_attr_medium/config.yaml \
    --preprocessed-data /home/shared/sieve-testing/preprocessed_discovery.pt \
    --output-dir /home/shared/sieve-testing/results/epistasis \
    --top-k 50 \
    --device cuda
```

**Outputs**:
- `results/epistasis/epistasis_validation.csv`
- `results/epistasis/epistasis_significant.csv`

### Step 3: Biological Validation

```bash
# Validate against databases (if available)
python scripts/validate_discoveries.py \
    --variant-rankings /home/shared/sieve-testing/results/explainability/sieve_variant_rankings.csv \
    --gene-rankings /home/shared/sieve-testing/results/explainability/sieve_gene_rankings.csv \
    --output-dir /home/shared/sieve-testing/results/validation \
    --clinvar data/clinvar.tsv \
    --gwas data/gwas_catalog.tsv \
    --disease-terms your_disease_terms \
    --top-k-variants 100 \
    --top-k-genes 50
```

**Outputs**:
- `results/validation/clinvar_validation.csv`
- `results/validation/gwas_validation.csv`
- `results/validation/validation_summary.yaml`

### Step 4: Manual Analysis

With the CSV files, you can now:
1. Review top variants in `sieve_top100_variants.csv`
2. Check which are in ClinVar/GWAS
3. Identify novel candidates not in databases
4. Examine validated epistatic interactions
5. Interpret GO enrichment results
6. Prioritize for experimental validation

---

## Quick Reference

### For Variant Discovery

```bash
python scripts/explain.py \
    --experiment-dir <experiment> \
    --preprocessed-data <data.pt> \
    --output-dir <output> \
    --device cuda
```

### For Epistasis Detection

```bash
python scripts/validate_epistasis.py \
    --interactions <interactions.csv> \
    --checkpoint <model.pt> \
    --config <config.yaml> \
    --preprocessed-data <data.pt> \
    --output-dir <output> \
    --device cuda
```

### For Biological Validation

```bash
python scripts/validate_discoveries.py \
    --variant-rankings <variants.csv> \
    --gene-rankings <genes.csv> \
    --clinvar <clinvar.tsv> \
    --gwas <gwas.tsv> \
    --output-dir <output>
```

---

## Troubleshooting

### Out of Memory

**Problem**: GPU out of memory during explain.py

**Solutions**:
```bash
# Reduce batch size
--batch-size 4

# Reduce integration steps
--n-steps 20

# Skip attention analysis
--skip-attention

# Use CPU (slower)
--device cpu
```

### Slow Performance

**Problem**: Explainability taking too long

**Solutions**:
1. Use `--skip-attention` (saves 50% time)
2. Reduce `--n-steps` (20-30 is often sufficient)
3. Use `--top-k-interactions 50` instead of 100

### Missing Databases

**Problem**: Don't have ClinVar/GWAS databases

**Solutions**:
1. Run validation script without `--clinvar` or `--gwas` flags
2. Download from:
   - ClinVar: ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/
   - GWAS Catalog: https://www.ebi.ac.uk/gwas/downloads
3. Use only the parts of validation you have data for

---

## Next Steps After Phase 3

With validated discoveries, you can:

1. **Write Paper**:
   - Report top variants and genes
   - Show enrichment in known pathways
   - Present validated epistatic interactions

2. **Experimental Validation**:
   - Test top variants in cell lines
   - Validate interactions experimentally
   - Functional studies of top genes

3. **Replication Study**:
   - Test discoveries in independent cohort
   - Check if findings replicate
   - Meta-analysis across cohorts

4. **Clinical Translation**:
   - Build genetic risk scores
   - Develop diagnostic panels
   - Identify drug targets

---

## Summary

Phase 3 is now **production-ready** with:
- ✅ Integrated gradients variant attribution
- ✅ Attention-based interaction detection
- ✅ SHAP interaction quantification
- ✅ Counterfactual perturbation validation
- ✅ ClinVar/GWAS validation
- ✅ GO/pathway enrichment
- ✅ Comprehensive reporting

All scripts are documented, tested, and ready to use on your data!
