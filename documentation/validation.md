# Validation

This chapter covers the complete validation strategy for SIEVE gene sets in independent cohorts. Validation is a multi-step process that starts with generating gene lists from the discovery cohort, extracting burden counts from independent validation VCFs, and then testing whether the SIEVE gene sets carry disease signal using both scalar burden enrichment and non-linear classifier tests.

---

## Overview

SIEVE validation answers two complementary questions:

1. **Scalar burden enrichment** (Step 8c in the workflow): Do SIEVE genes carry more exonic variation in cases than controls? This is a set-level burden test comparing the observed case-control burden difference against a permutation null from random gene sets of equal size.

2. **Non-linear classifier validation** (this chapter's focus): Does the **pattern** of variation across SIEVE genes jointly discriminate cases from controls? This test preserves the multi-gene combinatorial structure that SIEVE is designed to capture.

```
┌──────────────────────────┐
│  Gene list generation    │  variant rankings → ranked gene list
└───────────┬──────────────┘
            ↓
┌──────────────────────────┐
│  Burden extraction       │  validation VCF → gene × sample matrix
└───────────┬──────────────┘
            ↓
     ┌──────┴──────┐
     ↓             ↓
┌─────────┐  ┌──────────────┐
│ Scalar  │  │ Non-linear   │  random forest on per-gene burden
│ burden  │  │ classifier   │  vector vs random gene-set null
│ test    │  │ validation   │
└─────────┘  └──────────────┘
```

The scalar burden test collapses the k-dimensional gene vector into a single sum. If SIEVE's signal is combinatorial (e.g. disease when gene A has variants **and** gene B has variants, but not either alone), a burden sum destroys this pattern. The non-linear classifier preserves it.

---

## Prerequisites

Before running non-linear classifier validation, you need:

1. **Gene-burden matrix** from `extract_validation_burden.py --compute-full-gene-matrix` (see [Complete Workflow, Step 8b](complete-workflow.md))
2. **SIEVE gene lists** from `generate_sieve_gene_list.py` (see [Complete Workflow, Step 8a](complete-workflow.md))
3. **Phenotype file** in standard SIEVE format (`sample_id<TAB>phenotype`, 1=control, 2=case)

---

## Non-Linear Classifier Validation

### Motivation

The scalar burden test showed that SIEVE gene sets do not necessarily carry more total exonic variation in cases than controls. But SIEVE's claim is not that its genes have more variants — it is that the **pattern** of variation across genes jointly discriminates cases from controls. A burden count destroys this pattern; a non-linear classifier preserves it.

The validation question is:

> *Can a non-linear classifier trained on per-gene burden counts across the SIEVE gene set discriminate cases from controls in an independent cohort better than the same classifier trained on random gene sets?*

- If **yes**: SIEVE's gene selection captures genuine multi-dimensional signal transferable across cohorts.
- If **no**: the discovery findings do not generalise to this cohort (which may reflect phenotype mismatch, population mismatch, or insufficient signal).

### Method

For each SIEVE gene set (defined by ablation level and top-k threshold):

1. Extract the per-gene burden sub-matrix for the top-k SIEVE genes
2. Train a random forest classifier using repeated stratified cross-validation
3. Record the observed mean AUC
4. Repeat for 1,000 random gene sets of the same size, using the **same CV fold assignments**
5. Compute an empirical p-value: the fraction of random sets with AUC >= observed

The fixed CV folds ensure the only variable is the gene set, not the data split.

### Quick Start

```bash
# Single level, single top-k
python scripts/validate_nonlinear_classifier.py \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --sieve-genes validation/sieve_gene_lists/sieve_genes_L1.tsv \
    --phenotypes /path/to/phenotypes.tsv \
    --output-dir validation/cohort_b/nonlinear_validation \
    --top-k 100 \
    --n-permutations 1000 \
    --seed 42 \
    --n-jobs 4
```

### Multi-Level Mode

To compare across ablation levels, point `--sieve-genes` at a directory containing the per-level gene list files (either `L{0,1,2,3}_sieve_genes.tsv` as produced by `generate_sieve_gene_list.py --ablation-level`, or `sieve_genes_L{0,1,2,3}.tsv`):

```bash
python scripts/validate_nonlinear_classifier.py \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --sieve-genes validation/sieve_gene_lists/ \
    --phenotypes /path/to/phenotypes.tsv \
    --output-dir validation/cohort_b/nonlinear_validation \
    --top-k 50 100 200 500 \
    --n-permutations 1000 \
    --cv-folds 5 \
    --seed 42 \
    --n-jobs 4
```

This automatically detects all level files and runs every level x top-k combination.

### Comparing Random Forest vs Logistic Regression

Use `--classifiers both` to run both a random forest and a logistic regression on every combination:

```bash
python scripts/validate_nonlinear_classifier.py \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --sieve-genes validation/sieve_gene_lists/ \
    --phenotypes /path/to/phenotypes.tsv \
    --output-dir validation/cohort_b/nonlinear_validation \
    --top-k 100 200 \
    --n-permutations 1000 \
    --classifiers both \
    --n-jobs 4
```

**Why include logistic regression?** As a linear baseline. If the random forest significantly outperforms logistic regression on SIEVE genes, that is evidence of non-linear signal — directly supporting SIEVE's core claim that multi-gene combinatorial patterns carry disease information. If logistic regression performs equally well, the signal is linear (which could have been captured by a PRS approach).

### Consequence-Stratified Testing

To test whether signal is driven by specific variant classes (e.g. missense, LoF), use the consequence-stratified burden matrices from `extract_validation_burden.py --consequence-stratify`:

```bash
python scripts/validate_nonlinear_classifier.py \
    --burden-matrix validation/cohort_b/gene_burden_matrix_missense.parquet \
    --sieve-genes validation/sieve_gene_lists/ \
    --phenotypes /path/to/phenotypes.tsv \
    --output-dir validation/cohort_b/nonlinear_validation_missense \
    --top-k 100 200 \
    --n-permutations 1000 \
    --consequence missense \
    --n-jobs 4
```

### Exporting Feature Matrices for External Analysis

Use `--also-export-csv` to export the SIEVE feature matrix as a CSV for analysis in R/tidymodels or other tools:

```bash
python scripts/validate_nonlinear_classifier.py \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --sieve-genes validation/sieve_gene_lists/sieve_genes_L1.tsv \
    --phenotypes /path/to/phenotypes.tsv \
    --output-dir validation/cohort_b/nonlinear_validation \
    --top-k 100 \
    --also-export-csv
```

The exported CSV has columns: `sample_id, GENE1, GENE2, ..., GENEk, phenotype` with genes in SIEVE rank order and phenotype as 0/1.

---

## Interpreting Non-Linear Classifier Results

### Per-Combination Results (YAML)

Each level x top-k combination produces a YAML file with the full result:

```yaml
parameters:
  ablation_level: L1
  top_k_requested: 100
  k_effective: 93          # genes matched in validation VCF
  missing_genes: [...]
  n_samples: 450
  n_cases: 220
  n_controls: 230
  classifier: random_forest
  cv_folds: 5
  cv_repeats: 3
  n_permutations: 1000

observed:
  mean_auc: 0.587
  std_auc: 0.042
  per_fold_aucs: [0.61, 0.55, 0.58, ...]

null_distribution:
  mean: 0.521
  std: 0.028
  median: 0.519
  p5: 0.478
  p95: 0.567

empirical_p: 0.0034
percentile_rank: 99.66

# When both classifiers are run (primary YAML only):
linear_baseline:
  mean_auc: 0.533
  std_auc: 0.038
  empirical_p: 0.089
  rf_minus_lr_auc: 0.054
```

### Key Metrics

| Metric | What it means |
|--------|--------------|
| `observed.mean_auc` | How well the SIEVE gene set discriminates cases from controls |
| `null_distribution.mean` | Expected AUC from a random gene set of the same size |
| `empirical_p` | Probability that a random gene set performs as well or better |
| `percentile_rank` | Where the observed AUC falls in the null distribution (higher = better) |
| `rf_minus_lr_auc` | AUC gap between random forest and logistic regression (positive = non-linear signal) |

### Interpretation Guide

| Empirical p | Interpretation |
|-------------|---------------|
| < 0.01 | Strong evidence: SIEVE gene set carries discriminative signal beyond random chance |
| 0.01 - 0.05 | Moderate evidence (check after Bonferroni correction across all tests) |
| > 0.05 | No significant evidence at this level/top-k combination |

### What to Look For

1. **Significant empirical p-value**: The SIEVE gene set outperforms random gene sets. This supports transfer of the discovery signal to the validation cohort.

2. **RF > LR gap**: If the random forest outperforms logistic regression on the SIEVE gene set, the signal has non-linear structure — combinations of gene burdens matter, not just their sum. This directly supports SIEVE's model design.

3. **Level consistency**: If multiple ablation levels show signal, the discovery is robust. If only L0 (genotype-only) shows signal, the discovery is annotation-free. If only L3 shows signal, it may depend on functional annotations.

4. **Top-k sensitivity**: Signal concentrated in top-50 genes suggests a small set of strong drivers. Signal appearing only at top-500 suggests a diffuse polygenic signal.

### Diagnostic Plots

Each combination produces a two-panel plot:

1. **Left panel**: Histogram of null AUC distribution with observed AUC marked as a red vertical line. The further right the line, the stronger the evidence.

2. **Right panel**: Box plot comparing observed per-fold AUCs against the null distribution, showing the spread of performance across CV splits.

### Summary Outputs

- **`nonlinear_validation_summary.tsv`**: One row per level x top-k x classifier combination with all key metrics and Bonferroni significance flags.
- **`nonlinear_validation_heatmap.png`**: Visual comparison of observed AUC across levels (rows) and top-k values (columns), with significance annotations (`*` for p < 0.05, `**` for Bonferroni-significant).
- **`nonlinear_validation_report.md`**: Human-readable summary of significant results, best combinations, and RF vs LR comparison.

---

## Output Structure

```
nonlinear_validation/
├── nonlinear_validation_L0_topK100.yaml
├── nonlinear_validation_L0_topK100_lr.yaml     # if --classifiers both
├── null_aucs_L0_topK100.npz
├── null_aucs_L0_topK100_lr.npz
├── validation_plot_L0_topK100.png
├── validation_plot_L0_topK100_lr.png
├── ...                                          # repeat per level x top-k
├── nonlinear_validation_summary.tsv
├── nonlinear_validation_heatmap.png
├── nonlinear_validation_report.md
└── csv/                                         # if --also-export-csv
    ├── feature_matrix_total_L0_top100.csv
    └── ...
```

---

## Computational Considerations

Rough estimates for one level x one top-k value:

| Permutations | Cores | Approximate time |
|-------------|-------|-----------------|
| 200 | 4 | ~25 min |
| 1000 | 4 | ~2 hours |
| 1000 | 8 | ~1 hour |

For 4 levels x 4 top-k values = 16 combinations with 1,000 permutations each on 4 cores, expect ~32 hours total.

!!! tip "Start small, then scale"
    Use `--n-permutations 200` for quick exploration. Once you identify the most promising level/top-k combinations, re-run those with `--n-permutations 1000` for publication-quality results.

---

## Design Decisions

### No Hyperparameter Tuning

The random forest uses a fixed, reasonable configuration (500 trees, `max_features='sqrt'`, `min_samples_leaf=5`, `class_weight='balanced'`). This is deliberate: the comparison between SIEVE and null gene sets must use identical classifier configurations. Tuning per gene set would conflate gene set quality with tuning luck.

### Parallelism at Permutation Level

The script parallelises across permutations (each on a single core) rather than within each random forest (`n_jobs=1` inside each classifier). This avoids nested parallelism issues and is more efficient for the 1,000-permutation workload.

### Fixed CV Folds

All evaluations (observed and every permutation) use the same cross-validation fold assignments. This ensures the only variable is the gene set, not the data split, making the comparison strictly fair.

---

## Complete Validation Example

Putting scalar burden and non-linear classifier validation together for one cohort:

```bash
# --- Step 1: Generate gene lists per ablation level ---
for level in L0 L1 L2 L3; do
    python scripts/generate_sieve_gene_list.py \
        --variant-rankings results/${level}_attribution_comparison/variant_rankings_corrected.csv \
        --output validation/sieve_gene_lists/sieve_genes.tsv \
        --ablation-level ${level} \
        --score-column z_attribution \
        --aggregation max
done

# --- Step 2: Extract burden matrix (once per cohort) ---
python scripts/extract_validation_burden.py \
    --vcf /path/to/validation_cohort.vcf.gz \
    --phenotypes /path/to/validation_phenotypes.tsv \
    --sieve-genes validation/sieve_gene_lists/L3_sieve_genes.tsv \
    --output-dir validation/cohort_b \
    --top-k 50 100 200 500 \
    --consequence-stratify \
    --compute-full-gene-matrix

# --- Step 3: Scalar burden enrichment (per level) ---
for level in L0 L1 L2 L3; do
    python scripts/test_burden_enrichment.py \
        --burden-dir validation/cohort_b \
        --sieve-genes validation/sieve_gene_lists/${level}_sieve_genes.tsv \
        --output-dir validation/cohort_b/enrichment_${level} \
        --n-permutations 10000 \
        --top-k 50 100 200 500 \
        --consequence-types total missense lof \
        --seed 42
done

# --- Step 4: Non-linear classifier validation (all levels at once) ---
python scripts/validate_nonlinear_classifier.py \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --sieve-genes validation/sieve_gene_lists/ \
    --phenotypes /path/to/validation_phenotypes.tsv \
    --output-dir validation/cohort_b/nonlinear_validation \
    --top-k 50 100 200 500 \
    --n-permutations 1000 \
    --classifiers both \
    --n-jobs 4 \
    --seed 42

# --- Step 5: Collect and plot scalar burden results ---
python scripts/plot_validation_burden.py \
    --input-dirs validation/cohort_b/enrichment_L0 \
                 validation/cohort_b/enrichment_L1 \
                 validation/cohort_b/enrichment_L2 \
                 validation/cohort_b/enrichment_L3 \
    --top-k 50 100 200 500 \
    --consequence-types total missense lof \
    --output-dir validation/cohort_b/burden_plots
```
