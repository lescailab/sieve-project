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

1. Load the corrected gene rankings for each annotation level
2. Select the observed top-k genes for that level from the requested score column
3. Train the requested classifier on the per-gene burden sub-matrix using fixed stratified CV folds
4. For each `(top_k, classifier)` pair, group annotation levels by their effective matched gene count (`k_effective`) after burden-matrix intersection
5. Draw one shared null distribution per `(top_k, classifier, k_effective)` group and reuse it across the levels in that group
6. Compute empirical p-values with the `(k + 1) / (N + 1)` convention
7. Apply Benjamini-Hochberg FDR across the full result grid

The fixed CV folds ensure the only variable is the gene set, not the data split. The shared null distribution ensures that `null_mean_auc` and `null_std_auc` are identical across levels within a given `(top_k, classifier, k_effective)` group.

### Quick Start

```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir results/ablation/rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --labels /path/to/phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_summary.tsv \
    --top-k 100,500,1000,2000 \
    --classifiers rf,lr \
    --levels L0,L1,L2,L3 \
    --n-permutations 1000 \
    --cv-folds 5 \
    --n-cores 8 \
    --seed 42
```

### Multi-Level Mode

Point `--real-rankings-dir` at a directory with one subdirectory per annotation level. Each level directory should contain `gene_rankings_with_significance.csv` (preferred), `corrected_gene_rankings_with_significance.csv`, or `corrected_gene_rankings.csv`.

```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir results/ablation/rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --labels /path/to/phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_summary.tsv \
    --top-k 50,100,200,500 \
    --classifiers rf,lr \
    --levels L0,L1,L2,L3 \
    --n-permutations 1000 \
    --cv-folds 5 \
    --n-cores 8 \
    --seed 42
```

This automatically detects all requested levels and runs every `level x top_k x classifier` combination.

### Comparing Random Forest vs Logistic Regression

Use `--classifiers rf,lr` to run both a random forest and a logistic regression on every combination:

```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir results/ablation/rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --labels /path/to/phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_summary.tsv \
    --top-k 100,200 \
    --n-permutations 1000 \
    --classifiers rf,lr \
    --n-cores 8
```

**Why include logistic regression?** As a linear baseline. If the random forest significantly outperforms logistic regression on SIEVE genes, that is evidence of non-linear signal — directly supporting SIEVE's core claim that multi-gene combinatorial patterns carry disease information. If logistic regression performs equally well, the signal is linear (which could have been captured by a PRS approach).

### Score Column Selection

Use `--score-column z_attribution` for the default corrected-score workflow. The script maps this onto the gene-level `gene_z_score` column inside `corrected_gene_rankings*.csv`.

Use `--score-column fdr_gene` when you want to rank genes by their gene-level null-contrast significance instead of corrected effect size. Lower values are treated as better for FDR-based ranking.

### FDR-Threshold Gene Selection

Instead of choosing a fixed number of top genes, you can let the gene set size be determined by an FDR cutoff:

```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir results/ablation/rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --labels /path/to/phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_fdr.tsv \
    --fdr-threshold 0.05 \
    --classifiers rf,lr \
    --levels L0,L1,L2,L3 \
    --n-permutations 1000 \
    --n-cores 8 \
    --seed 42
```

`--fdr-threshold` and `--top-k` are **mutually exclusive**. When `--fdr-threshold` is used:

- Each annotation level independently determines its gene set as the set of genes with `fdr_gene < threshold`.
- Different levels may produce different gene set sizes — this is scientifically meaningful, as it reflects how many genes are statistically significant at each annotation level.
- The summary TSV includes a `fdr_threshold` column to distinguish these results from fixed top-k runs.
- If no genes pass the threshold at a given level, that level is skipped with a warning.

**When to use which mode:**

| Mode | Best for |
|------|----------|
| `--top-k 50,100,200,500` | Exploratory analysis, comparing levels at matched gene-set sizes, sensitivity analysis across multiple thresholds |
| `--fdr-threshold 0.05` | Statistically motivated gene sets, validating only genes with null-contrast significance, manuscript-quality results |

The gene rankings files must contain `fdr_gene` for FDR-threshold mode to work. This column is available in `gene_rankings_with_significance.csv` (from `compare_attributions.py`) and in `corrected_gene_rankings.csv` (from `correct_chrx_bias.py`, which merges significance from the companion file).

---

## Interpreting Non-Linear Classifier Results

### Per-Combination Results (YAML)

Each level x top-k combination produces a YAML file with the full result:

```yaml
parameters:
  ablation_level: L1
  top_k: 100
  k_effective: 93
  missing_genes: [...]
  n_samples: 450
  n_cases: 220
  n_controls: 230
  classifier: random_forest
  cv_folds: 5
  n_permutations: 1000
  score_column: z_attribution

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
z_score: 2.36
fdr_bh: 0.0120

# When both classifiers are run (primary YAML only):
linear_baseline:
  mean_auc: 0.533
  std_auc: 0.038
  empirical_p: 0.089
  fdr_bh: 0.1740
  rf_minus_lr_auc: 0.054
```

### Key Metrics

| Metric | What it means |
|--------|--------------|
| `observed.mean_auc` | How well the SIEVE gene set discriminates cases from controls |
| `null_distribution.mean` | Expected AUC from the shared random-gene null of the same top-k |
| `empirical_p` | Probability that a random gene set performs as well or better |
| `z_score` | Observed AUC expressed as a z-score against the shared null |
| `fdr_bh` | BH-adjusted empirical p-value across the full result grid |
| `rf_minus_lr_auc` | AUC gap between random forest and logistic regression (positive = non-linear signal) |

### Interpretation Guide

| FDR | Interpretation |
|-----|----------------|
| `fdr_bh < 0.01` | Strong evidence after multiple-testing correction |
| `0.01 <= fdr_bh < 0.05` | Moderate evidence after multiple-testing correction |
| `fdr_bh >= 0.05` | No FDR-significant evidence at this level/top-k combination |

### What to Look For

1. **Significant FDR**: The SIEVE gene set outperforms the shared random-gene null after correction across the full grid. This supports transfer of the discovery signal to the validation cohort.

2. **RF > LR gap**: If the random forest outperforms logistic regression on the SIEVE gene set, the signal has non-linear structure — combinations of gene burdens matter, not just their sum. This directly supports SIEVE's model design.

3. **Level consistency**: If multiple ablation levels show signal, the discovery is robust. If only L0 (genotype-only) shows signal, the discovery is annotation-free. If only L3 shows signal, it may depend on functional annotations.

4. **Top-k sensitivity**: Signal concentrated in top-50 genes suggests a small set of strong drivers. Signal appearing only at top-500 suggests a diffuse polygenic signal.

### Diagnostic Plots

Each combination produces a two-panel plot:

1. **Left panel**: Histogram of the shared null AUC distribution with observed AUC marked as a red vertical line. The further right the line, the stronger the evidence.

2. **Right panel**: Box plot comparing observed per-fold AUCs against the shared null distribution, showing the spread of performance across CV splits.

### Summary Outputs

- **`nonlinear_validation_summary.tsv`**: One row per level x top-k x classifier combination with all key metrics and a single `fdr_bh` column.
- **`nonlinear_validation_heatmap.png`**: Visual comparison of observed AUC across levels (rows) and top-k values (columns), with `*` marking `fdr_bh < 0.05`.
- **`nonlinear_validation_report.md`**: Human-readable summary of significant results, best combinations, and RF vs LR comparison.

---

## Output Structure

```
nonlinear_validation/
├── nonlinear_validation_L0_topK100.yaml
├── nonlinear_validation_L0_topK100_lr.yaml     # if --classifiers rf,lr
├── null_aucs_L0_topK100.npz
├── null_aucs_L0_topK100_lr.npz
├── validation_plot_L0_topK100.png
├── validation_plot_L0_topK100_lr.png
├── ...                                          # repeat per level x top-k
├── nonlinear_validation_summary.tsv
├── nonlinear_validation_heatmap.png
└── nonlinear_validation_report.md
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
        --variant-rankings results/${level}_attribution_comparison/corrected/corrected_variant_rankings.csv \
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

# --- Step 4a: Non-linear classifier validation with fixed top-k ---
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir results/ablation/rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --labels /path/to/validation_phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_summary.tsv \
    --top-k 50,100,200,500 \
    --n-permutations 1000 \
    --classifiers rf,lr \
    --n-cores 8 \
    --seed 42

# --- Step 4b: Alternative — FDR-threshold gene selection ---
# Uses only genes with fdr_gene < 0.05 (gene set size determined per level)
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir results/ablation/rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --labels /path/to/validation_phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_fdr.tsv \
    --fdr-threshold 0.05 \
    --n-permutations 1000 \
    --classifiers rf,lr \
    --n-cores 8 \
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
