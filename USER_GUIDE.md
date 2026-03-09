# SIEVE User Guide

**Version**: 1.3
**Last Updated**: 2026-03-09
**For**: SIEVE v0.1.0+

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Complete Workflow](#complete-workflow)
5. [Detailed Usage](#detailed-usage)
6. [Command Reference](#command-reference)
7. [Interpreting Results](#interpreting-results)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)
10. [Appendices](#appendices)
    - [Appendix A: Model Architecture Details](#appendix-a-model-architecture-details)
    - [Appendix B: Experimental Protocol](#appendix-b-experimental-protocol)
    - [Appendix C: Method References](#appendix-c-method-references)

---

## Introduction

### What is SIEVE?

**SIEVE** (Sparse Interpretable Exome Variant Explainer) is a deep learning framework for discovering disease-associated genetic variants from exome sequencing data in case-control studies.

### What Makes SIEVE Different?

Unlike existing methods:
- **Direct VCF Processing**: No conversion to PLINK or custom formats required
- **Annotation-Free Discovery**: Tests whether ML can discover variants without prior knowledge
- **Position-Aware**: Learns spatial relationships between variants (e.g., compound heterozygosity)
- **Built-in Interpretability**: Attribution sparsity incorporated into training
- **Statistical Validation**: Null baseline analysis establishes significance thresholds

### Scientific Questions SIEVE Addresses

1. **Can deep learning discover variants that annotations miss?** → Annotation ablation experiments (L0-L4)
2. **Do spatial relationships between variants matter?** → Position-aware sparse attention
3. **Can we make models interpretable by design?** → Attribution-regularised training
4. **Are discoveries statistically significant?** → Null baseline analysis

### Key Capabilities

- **Train** models at multiple annotation levels (genotype-only to full annotations)
- **Explain** predictions with integrated gradients attribution
- **Discover** novel variant associations with statistical validation
- **Detect** epistatic interactions via attention patterns
- **Validate** discoveries against ClinVar, GWAS, and GO databases

---

## Quick Start

### For the Impatient

```bash
# 1. Install
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project
pip install -e .

# 2. (Optional, recommended) Infer genetic sex for ploidy-aware encoding
python scripts/infer_sex.py \
    --vcf your_data.vcf.gz \
    --output-dir results/sex_inference \
    --genome-build GRCh37

# 3. Preprocess (once)
python scripts/preprocess.py \
    --vcf your_data.vcf.gz \
    --phenotypes phenotypes.tsv \
    --output preprocessed.pt \
    --sex-map results/sex_inference/sample_sex.tsv \
    --genome-build GRCh37

# 4. Train
python scripts/train.py \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --experiment-name my_model \
    --output-dir experiments \
    --device cuda

# 5. Explain
python scripts/explain.py \
    --experiment-dir experiments/my_model \
    --preprocessed-data preprocessed.pt \
    --output-dir results/explainability

# 6. Validate (null baseline)
# Set required environment variables first
export INPUT_DATA="preprocessed.pt"
export REAL_EXPERIMENT="experiments/my_model"            # or experiments/my_model/fold_0
export REAL_RESULTS="results/explainability"             # directory with sieve_variant_rankings.csv
export OUTPUT_BASE="results/null_baseline_run"           # where null outputs will be written
bash scripts/run_null_baseline_analysis.sh

# 7. (Optional) Correct chrX ploidy bias in rankings
python scripts/correct_chrx_bias.py \
    --rankings results/explainability/sieve_variant_rankings.csv \
    --null-rankings results/null_baseline_run/results/null_attributions/sieve_variant_rankings.csv \
    --output-dir results/explainability_corrected \
    --exclude-sex-chroms
```

### 5-Minute Test Run

```bash
# Use included test data
python scripts/train.py \
    --vcf test_data/small/test_data.vcf.gz \
    --phenotypes test_data/small/test_data_phenotypes.tsv \
    --level L3 \
    --epochs 20 \
    --batch-size 8 \
    --output-dir test_run
```

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **GPU**: CUDA-capable GPU recommended (8GB+ VRAM for real datasets)
- **Storage**: ~100MB for software, ~10GB for large preprocessed datasets
- **RAM**: 16GB minimum, 32GB+ recommended for large cohorts

### Step 1: Clone Repository

```bash
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project
```

### Step 2: Create Environment

**Option A: Using conda** (recommended)
```bash
conda create -n sieve python=3.10
conda activate sieve
pip install -e .
```

**Option B: Using venv**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install -e .
```

### Step 3: Verify Installation

```bash
# Run test suite
python test_vcf_parser.py
python test_encoding_pipeline.py
python test_model_architecture.py
python test_training_pipeline.py
```

All tests should complete without errors. You can also run `pytest` for a more detailed test report.

### Dependencies Installed

Core packages:
- **PyTorch** 2.0+ (deep learning)
- **NumPy**, **Pandas**, **SciPy** (data processing)
- **cyvcf2/pysam** (VCF parsing)
- **captum** (integrated gradients)
- **scikit-learn** (metrics, preprocessing)
- **matplotlib** (visualisation)
- **PyYAML** (configuration)

See `pyproject.toml` for complete list.

### Conda Package Workflow

If you install SIEVE as a conda package (instead of editable source install), use the
`sieve-*` commands exposed by the package. A complete command-based walkthrough is in:

- `conda/USAGE.md`

---

## Complete Workflow

### Overview

```
┌─────────────────────┐
│  1. Data Prep       │  VCF + Phenotypes → preprocessed.pt
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  2. Train Model     │  Learn genotype-phenotype relationships
└─────────┬───────────┘  (repeat for each annotation level L0-L3)
          ↓
┌─────────────────────┐
│  3. Explainability  │  Compute variant attributions
└─────────┬───────────┘  (repeat for each annotation level)
          ↓
┌─────────────────────┐
│  4. Null Baseline   │  Establish statistical significance
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  5. Ablation        │  Compare rankings and performance
│     Comparison      │  across annotation levels
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  6. Validation      │  Cross-reference with databases
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  7. Biological      │  Experimental validation
│     Follow-up       │
└─────────────────────┘
```

### Workflow Steps

#### Step 1: Data Preparation

**Purpose**: Convert VCF to SIEVE-compatible format (optionally sex-aware)

**Theory**: SIEVE requires multi-sample VCF files annotated with VEP (Variant Effect Predictor). VEP adds functional annotations (SIFT, PolyPhen, consequence types) that enable multi-level analysis. For sex chromosomes, SIEVE can apply ploidy-aware dosage encoding to avoid chrX hemizygosity bias in male samples.

**Requirements**:
- Multi-sample VCF file (bgzipped and indexed)
- VEP-annotated (CSQ field with SIFT, PolyPhen, Consequence, SYMBOL)
- Reference genome build specified as GRCh37 or GRCh38 (`--genome-build`)
- Contig labels with or without `chr` prefix are accepted (normalised internally)
- Phenotype file (TSV: sample_id, phenotype)

**Command**:
```bash
python scripts/preprocess.py \
    --vcf cohort.vcf.gz \
    --phenotypes phenotypes.tsv \
    --output preprocessed.pt \
    --genome-build GRCh37 \
    --sex-map results/sex_inference/sample_sex.tsv
```

**Optional sex inference** (recommended for chrX/chrY analyses):
```bash
python scripts/infer_sex.py \
    --vcf cohort.vcf.gz \
    --output-dir results/sex_inference \
    --genome-build GRCh37
```

**Why sex-aware preprocessing?**
- Male chrX non-PAR variants are hemizygous; dosage 1 should be treated as 2
- Avoids spurious chrX attribution inflation
- Ensures downstream rankings are comparable across chromosomes

**Output**: Single `.pt` file containing all parsed variant data (~1-5 MB per 1000 samples)

**Why Preprocess?**
- VCF parsing: 30 mins to 5+ hours (one time)
- Loading preprocessed: 5-10 seconds (every run)
- **100-3600× speedup** for repeated experiments!

---

#### Step 2: Model Training

**Purpose**: Learn which variants predict case/control status

**Theory**: SIEVE uses position-aware sparse attention to learn relationships between variants. Training includes:
- Classification loss: Binary cross-entropy on case/control prediction
- Attribution regularisation (optional): Encourages model to rely on fewer variants

**Annotation Levels**:
- **L0**: Genotype dosage only (0, 1, 2) - tests annotation-free discovery
- **L1**: L0 + genomic position
- **L2**: L1 + consequence class (missense/synonymous/LoF)
- **L3**: L2 + SIFT + PolyPhen ← **recommended starting point**
- **L4**: L3 + additional annotations (extensible)

**Command**:
```bash
python scripts/train.py \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --val-split 0.2 \
    --lr 0.00001 \
    --lambda-attr 0.1 \
    --epochs 100 \
    --batch-size 16 \
    --chunk-size 3000 \
    --aggregation-method mean \
    --output-dir experiments \
    --experiment-name my_model \
    --device cuda
```

**Outputs**:
- `best_model.pt` - Best model checkpoint
- `training_history.yaml` - Loss curves and metrics
- `config.yaml` - Full configuration for reproducibility
- `fold_*/config.yaml` - Fold-specific config (CV mode)
- `fold_*/fold_info.yaml` - Fold split metadata and training summary (CV mode)

**Expected Results**:
- Validation AUC > 0.6: Model is learning signal
- Validation AUC > 0.7: Good performance
- Validation AUC ≈ 0.5: No signal (check data/encoding)

---

#### Step 3: Explainability Analysis

**Purpose**: Identify which variants drive predictions

**Theory**: Uses integrated gradients to compute attribution scores for each variant. Integrated gradients approximates the contribution of each input feature by integrating gradients along a path from a baseline (all zeros) to the actual input.

**Method**:
1. For each sample, compute gradient of prediction w.r.t. each variant
2. Integrate gradients from baseline (no variants) to observed genotype
3. Aggregate attributions across samples
4. Rank variants by mean absolute attribution

**Command**:
```bash
python scripts/explain.py \
    --experiment-dir experiments/my_model \
    --preprocessed-data preprocessed.pt \
    --output-dir results/explainability \
    --n-steps 50 \
    --device cuda
```

**Outputs**:
- `sieve_variant_rankings.csv` - All variants ranked by attribution
- `sieve_gene_rankings.csv` - Gene-level aggregated scores
- `sieve_interactions.csv` - High-attention variant pairs

**Interpretation**:
- **High attribution**: Variant strongly influences model prediction
- **Consistent across samples**: Variant is important for many individuals
- **Case-enriched**: Variant has higher attribution in cases than controls

---

#### Step 4: Null Baseline Analysis ⭐ **CRITICAL**

**Purpose**: Establish statistical significance of discoveries

**Theory**: When we train a model, every variant receives some attribution score. But which attributions represent genuine biological signal vs. random noise? By training an identical model on **permuted labels** (shuffled case/control assignments), we break any real genotype-phenotype relationship. Attributions from this null model represent the "noise floor" of our pipeline.

**Why It Matters**:
- Without null baseline: Can't distinguish signal from noise
- With null baseline: Identify variants exceeding chance expectations
- Establishes p-value thresholds (p<0.05, 0.01, 0.001)
- Computes enrichment factors (e.g., "5× more discoveries than expected by chance")

**Quick Start**:
```bash
# Set environment variables
export INPUT_DATA=preprocessed.pt
export REAL_EXPERIMENT=experiments/my_model          # or fold dir for CV
export REAL_RESULTS=results/explainability           # contains sieve_variant_rankings.csv
export OUTPUT_BASE=results/null_baseline_run

# Run complete pipeline
bash scripts/run_null_baseline_analysis.sh
```

The wrapper reads hyperparameters directly from the real run `config.yaml` (including `--sex-map` when used) so the null model is trained under matched settings.

**Manual Steps**:
```bash
# 1. Create permuted dataset
python scripts/create_null_baseline.py \
    --input preprocessed.pt \
    --output preprocessed_NULL.pt \
    --seed 42

# 2. Train null model (SAME params as real!)
python scripts/train.py \
    --preprocessed-data preprocessed_NULL.pt \
    --level L3 \
    --experiment-name null_baseline \
    [... exact same parameters as real model ...]

# 3. Run explainability on null
python scripts/explain.py \
    --experiment-dir experiments/null_baseline \
    --preprocessed-data preprocessed_NULL.pt \
    --output-dir results/null_attributions \
    --is-null-baseline

# 4. Compare real vs null
python scripts/compare_attributions.py \
    --real results/explainability/sieve_variant_rankings.csv \
    --null results/null_attributions/sieve_variant_rankings.csv \
    --output-dir results/comparison
```

**Outputs**:
- `comparison_summary.yaml` - Statistical tests and thresholds
- `significant_variants_p01.csv` - Variants exceeding p<0.01
- `variant_rankings_with_significance.csv` - All variants annotated
- `real_vs_null_comparison.png` - Distribution comparison plot

**Expected Results**:
- Null model AUC ≈ 0.50 (chance level - confirms permutation worked)
- Real distributions differ from null (KS test p < 0.001)
- Enrichment at p<0.01:
  - **< 1.5×**: Weak signal, be cautious
  - **1.5-2×**: Moderate signal, validate carefully
  - **> 2×**: Strong signal, proceed with confidence

**Interpretation Guide**:
```
Enrichment = Observed / Expected

Example:
- Real data: 50 variants exceed null p<0.01 threshold
- Expected by chance: 10 variants (1% of 1000 total)
- Enrichment: 50 / 10 = 5×
- Interpretation: 5× more discoveries than expected by chance
```

---

#### Step 5: Ablation Comparison

**Purpose**: Compare variant rankings and model performance across annotation levels to assess whether deep learning can discover disease-associated variants without relying on functional annotations.

**Theory**: The annotation ablation is the core experiment of SIEVE. By training models at levels L0 (genotype only) through L3 (full annotations), you can determine:
- Whether positional or functional information is needed for discovery
- Which variants are found regardless of annotation level (robust discoveries)
- Which variants are only found at specific levels (annotation-dependent)

**Prerequisites**: Train and run explain.py at each annotation level (Steps 2-3 repeated for L0, L1, L2, L3).

**Step 5a: Compare model performance across levels**:
```bash
python scripts/ablation_compare.py \
    --results-dir experiments \
    --out-summary-tsv results/ablation/ablation_summary.tsv \
    --out-summary-yaml results/ablation/ablation_summary.yaml
```

**Step 5b: Compare variant attribution rankings across levels**:
```bash
# Collect ranking files into one directory with level prefixes
mkdir -p results/ablation/rankings
cp results/L0_explainability/sieve_variant_rankings.csv results/ablation/rankings/L0_sieve_variant_rankings.csv
cp results/L1_explainability/sieve_variant_rankings.csv results/ablation/rankings/L1_sieve_variant_rankings.csv
cp results/L2_explainability/sieve_variant_rankings.csv results/ablation/rankings/L2_sieve_variant_rankings.csv
cp results/L3_explainability/sieve_variant_rankings.csv results/ablation/rankings/L3_sieve_variant_rankings.csv

# Run comparison
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rankings \
    --top-k 50,100,200,500 \
    --high-rank-threshold 100 \
    --low-rank-threshold 500 \
    --out-comparison results/ablation/ablation_ranking_comparison.yaml \
    --out-jaccard results/ablation/ablation_jaccard_matrix.tsv \
    --out-level-specific results/ablation/level_specific_variants.tsv
```

**Step 5c: Visualise the comparison**:
```bash
python scripts/plot_ablation_comparison.py \
    --jaccard-tsv results/ablation/ablation_jaccard_matrix.tsv \
    --level-specific-tsv results/ablation/level_specific_variants.tsv \
    --summary-yaml results/ablation/ablation_summary.yaml \
    --output results/ablation/ablation_comparison.png
```

**Outputs**:
- `ablation_summary.tsv` / `.yaml` — AUC, accuracy, loss per level, best level
- `ablation_jaccard_matrix.tsv` — pairwise Jaccard similarity at each top-k
- `level_specific_variants.tsv` — variants uniquely important at one level
- `ablation_ranking_comparison.yaml` — structured comparison summary
- `ablation_comparison.png` / `.pdf` — multi-panel publication figure

**Interpretation**:
- **High Jaccard (>0.7)** between L0 and L3 → annotations are redundant, model discovers the same variants from genotype alone
- **Low Jaccard (<0.3)** between L0 and L3 → annotations substantially change which variants are prioritised
- **Many L0-specific variants** → genotype-only model finds signals annotations miss (novel discoveries)
- **Many L3-specific variants** → those discoveries depend on annotation information (potentially circular)

---

#### Step 6: Epistasis Detection (Optional)

**Purpose**: Identify variant pairs with non-additive effects

**Theory**: Epistasis occurs when the combined effect of two variants differs from the sum of their individual effects. SIEVE detects epistasis through:
1. High attention weights between variant pairs (model looks at them together)
2. Counterfactual validation (test all four combinations)

**Command**:
```bash
python scripts/validate_epistasis.py \
    --interactions results/explainability/sieve_interactions.csv \
    --checkpoint experiments/my_model/best_model.pt \
    --config experiments/my_model/config.yaml \
    --preprocessed-data preprocessed.pt \
    --output-dir results/epistasis \
    --top-k 50 \
    --device cuda
```

**Synergy Calculation**:
```
effect_v1 = f(v1=1, v2=0) - f(v1=0, v2=0)
effect_v2 = f(v1=0, v2=1) - f(v1=0, v2=0)
effect_combined = f(v1=1, v2=1) - f(v1=0, v2=0)

synergy = effect_combined - effect_v1 - effect_v2

synergy > 0.05  → Synergistic (work together)
synergy < -0.05 → Antagonistic (interfere)
synergy ≈ 0     → Independent
```

---

#### Step 7: Biological Validation (Optional)

**Purpose**: Cross-reference discoveries with known databases

**Command**:
```bash
python scripts/validate_discoveries.py \
    --variant-rankings results/explainability/sieve_variant_rankings.csv \
    --gene-rankings results/explainability/sieve_gene_rankings.csv \
    --output-dir results/validation \
    --top-k-variants 100 \
    --top-k-genes 50
```

**Checks**:
- **ClinVar**: Are variants known pathogenic?
- **GWAS Catalog**: Are genes in disease associations?
- **GO Enrichment**: Are genes enriched in specific pathways?

---

## Detailed Usage

### Preparing Your VCF File

#### VCF Requirements

Your VCF must be:
1. **Multi-sample** (at least 50 samples recommended)
2. **VEP-annotated** with CSQ field containing:
   - Consequence (e.g., missense_variant)
   - SYMBOL (gene name)
   - SIFT (score)
   - PolyPhen (score)
3. **Reference build declared** as GRCh37 or GRCh38 via `--genome-build`
4. **Contig naming may be either style** (e.g., `1` or `chr1`; harmonised internally)
5. **Bgzipped and indexed** (`.vcf.gz` + `.vcf.gz.tbi`)

#### Running VEP

If your VCF is not annotated:

```bash
vep --input_file variants.vcf \
    --output_file variants_annotated.vcf \
    --vcf \
    --symbol \
    --sift b \
    --polyphen b \
    --assembly GRCh37 \
    --offline \
    --cache /path/to/vep_cache
```

#### Phenotype File Format

Tab-separated file with header:
```
sample_id	phenotype
SAMPLE001	1
SAMPLE002	0
SAMPLE003	1
SAMPLE004	0
```

- Column 1: `sample_id` (must match VCF exactly)
- Column 2: `phenotype` (0 = control, 1 = case)

**Note**: Sample order doesn't matter, but names must match VCF.

---

### Sex-Aware Preprocessing (Recommended for chrX/chrY Analyses)

SIEVE now supports a sex-aware preprocessing path to prevent chrX ploidy bias in downstream attributions. The pipeline uses the X-chromosome inbreeding coefficient (F-statistic) with pseudoautosomal region (PAR) exclusion to infer genetic sex, then applies ploidy-aware dosage encoding during VCF parsing.

#### 1) Infer genetic sex (X-chromosome F-statistic)

```bash
python scripts/infer_sex.py \
    --vcf cohort.vcf.gz \
    --output-dir results/sex_inference \
    --genome-build GRCh37 \
    --min-gq 20 \
    --min-maf 0.05 \
    --f-male 0.8 \
    --f-female 0.2
```

**Outputs**:
- `sample_sex.tsv`: sample_id → inferred sex (`M`, `F`, or ambiguous labels)
- `sex_inference_diagnostic.png`: histogram of F-statistics
- `sex_inference_summary.yaml`: summary counts and thresholds

**Interpretation**:
- High F-statistic (≈1): low heterozygosity → genetic male
- Low F-statistic (≈0): high heterozygosity → genetic female
- Ambiguous/discordant samples are kept but excluded from ploidy correction

#### 2) Check sex balance across cases/controls (recommended)

```bash
python scripts/check_sex_balance.py \
    --phenotypes phenotypes.tsv \
    --sex-map results/sex_inference/sample_sex.tsv \
    --output-dir results/sex_balance
```

If a significant imbalance is detected, consider sex-stratified analysis or adding sex as a covariate in downstream modeling.

#### 3) Preprocess with ploidy-aware encoding

```bash
python scripts/preprocess.py \
    --vcf cohort.vcf.gz \
    --phenotypes phenotypes.tsv \
    --output preprocessed.pt \
    --sex-map results/sex_inference/sample_sex.tsv \
    --genome-build GRCh37
```

**Encoding rules**:
- Male chrX non-PAR: hemizygous alt is doubled (dosage 2)
- Female chrY: variants are skipped (data quality safeguard)
- Unknown/ambiguous sex: no correction (conservative default)

#### 4) Train with sex covariate (recommended if imbalance exists)

```bash
python scripts/train.py \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --sex-map results/sex_inference/sample_sex.tsv \
    --experiment-name my_model_sex_adjusted
```

`--sex-map` has two effects:
- During VCF-based training, it enables ploidy-aware dosage encoding and adds sex covariate to the classifier.
- During `--preprocessed-data` training, dosages are unchanged (already baked into `.pt`), and sex is used as a classifier covariate.

---

### Choosing Annotation Levels

#### Scientific Rationale

The annotation ablation protocol tests whether deep learning can discover variants independently of prior knowledge:

- **L0 (Genotype only)**: Can patterns in 0/1/2 dosages alone predict disease?
- **L1 (+ Position)**: Does knowing where variants are located help?
- **L2 (+ Consequence)**: Does basic VEP info (missense/LoF) matter?
- **L3 (+ SIFT/PolyPhen)**: Do deleteriousness scores improve discovery?
- **L4 (Full annotations)**: Maximum information

#### Decision Guide

**Start with L3** for most analyses because:
- Includes standard functional annotations
- Good balance of information and interpretability
- Comparable to existing methods

**Use L0** to test annotation-free discovery:
- If L0 performs well (AUC > 0.6), genotype patterns alone carry signal
- Variants unique to L0 may represent novel mechanisms

**Compare L0 vs L3 vs L4** for ablation studies:
- Identifies which annotations are actually helpful
- Reveals annotation-dependent vs independent discoveries

---

### Training Strategies

#### Single Train/Val Split

Fast, good for initial exploration:
```bash
python scripts/train.py \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --val-split 0.2 \
    --epochs 100 \
    --experiment-name quick_test
```

#### Cross-Validation

More robust performance estimation:
```bash
python scripts/train.py \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --cv 5 \
    --epochs 100 \
    --experiment-name robust_eval
```

Creates 5 models (one per fold), reports mean ± std performance.

#### Memory-Efficient Training (Large Datasets)

For datasets with >1000 samples and 5000+ variants per sample:

```bash
python scripts/train.py \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --chunk-size 2000 \
    --epochs 100 \
    --experiment-name large_cohort
```

**Explanation**:
- `--batch-size 2`: Process 2 samples at a time (low memory)
- `--gradient-accumulation-steps 16`: Simulate batch_size=32 (no quality loss)
- `--chunk-size 2000`: Cap variants per forward pass (prevents OOM)

**Memory Usage**:
| Configuration | GPU Memory | Works On |
|--------------|------------|----------|
| batch=32, chunk=5000 | ~40 GB | A100 80GB |
| batch=8, chunk=3000 | ~12 GB | A100 40GB |
| batch=2, chunk=3000 | ~7 GB | T4/RTX5000 |
| batch=2, chunk=2000 | ~5 GB | Most GPUs |

---

### Attribution-Regularized Training

#### Theory

Standard training:
```
Loss = Classification_Loss
```

Attribution-regularised training:
```
Loss = Classification_Loss + λ × Attribution_Sparsity_Loss
```

The sparsity term encourages the model to:
- Rely on fewer variants (better interpretability)
- Produce more stable attributions across CV folds
- Potentially improve generalisation

#### Usage

```bash
# No regularisation (default)
python scripts/train.py --lambda-attr 0.0 ...

# Light regularisation
python scripts/train.py --lambda-attr 0.01 ...

# Medium regularisation (recommended)
python scripts/train.py --lambda-attr 0.1 ...

# Strong regularisation
python scripts/train.py --lambda-attr 0.5 ...
```

#### When to Use

- **λ = 0**: Standard training, maximum flexibility
- **λ = 0.01-0.1**: Mild sparsity, improves interpretability
- **λ = 0.5+**: Strong sparsity, may hurt performance

**Recommendation**: Start with λ=0, then try λ=0.1 if attributions are noisy.

---

### Multiple Null Permutations

For more robust null baseline estimation:

```bash
# Create 5 null permutations (stored under results/null_permutations)
python scripts/create_null_baseline.py \
    --input preprocessed.pt \
    --output-dir results/null_permutations \
    --n-permutations 5

# Train each (can parallelise)
for i in {0..4}; do
    python scripts/train.py \
        --preprocessed-data results/null_permutations/preprocessed_NULL_perm${i}.pt \
        --level L3 \
        --experiment-name null_perm${i} \
        --output-dir experiments

    python scripts/explain.py \
        --experiment-dir experiments/null_perm${i} \
        --preprocessed-data results/null_permutations/preprocessed_NULL_perm${i}.pt \
        --output-dir results/null_permutations/perm${i} \
        --is-null-baseline
done

# Compare using all permutations (null_dir restricted to null outputs only)
python scripts/compare_attributions.py \
    --real results/explainability/sieve_variant_rankings.csv \
    --null-dir results/null_permutations \
    --output-dir results/comparison_robust
```

**Benefits**:
- More stable null thresholds
- Better confidence in significance calls
- Recommended for publication-quality analyses

---

### Running Ablation Experiments

The annotation ablation is the central experiment in SIEVE. It trains models at multiple annotation levels and compares both their predictive performance and the variant discoveries they produce.

#### Full Ablation Pipeline

```bash
# Preprocess once
python scripts/preprocess.py \
    --vcf cohort.vcf.gz \
    --phenotypes phenotypes.tsv \
    --output preprocessed.pt \
    --genome-build GRCh37

# Train at each annotation level
for LEVEL in L0 L1 L2 L3; do
    python scripts/train.py \
        --preprocessed-data preprocessed.pt \
        --level ${LEVEL} \
        --cv 5 \
        --epochs 100 \
        --output-dir experiments \
        --experiment-name ablation_${LEVEL} \
        --device cuda
done

# Run explainability at each level
for LEVEL in L0 L1 L2 L3; do
    python scripts/explain.py \
        --experiment-dir experiments/ablation_${LEVEL} \
        --preprocessed-data preprocessed.pt \
        --output-dir results/${LEVEL}_explainability \
        --device cuda
done

# Compare model performance
python scripts/ablation_compare.py \
    --results-dir experiments \
    --out-summary-tsv results/ablation/ablation_summary.tsv \
    --out-summary-yaml results/ablation/ablation_summary.yaml

# Compare attribution rankings
mkdir -p results/ablation/rankings
for LEVEL in L0 L1 L2 L3; do
    cp results/${LEVEL}_explainability/sieve_variant_rankings.csv \
       results/ablation/rankings/${LEVEL}_sieve_variant_rankings.csv
done

python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rankings \
    --out-comparison results/ablation/ablation_ranking_comparison.yaml \
    --out-jaccard results/ablation/ablation_jaccard_matrix.tsv \
    --out-level-specific results/ablation/level_specific_variants.tsv

# Visualise everything
python scripts/plot_ablation_comparison.py \
    --jaccard-tsv results/ablation/ablation_jaccard_matrix.tsv \
    --level-specific-tsv results/ablation/level_specific_variants.tsv \
    --summary-yaml results/ablation/ablation_summary.yaml \
    --output results/ablation/ablation_comparison.png
```

#### Using Explicit Ranking Paths

If your ranking files are not in a single directory with level prefixes, you can specify them individually:

```bash
python scripts/compare_ablation_rankings.py \
    --rankings L0:results/L0_explainability/sieve_variant_rankings.csv \
               L1:results/L1_explainability/sieve_variant_rankings.csv \
               L2:results/L2_explainability/sieve_variant_rankings.csv \
               L3:results/L3_explainability/sieve_variant_rankings.csv \
    --out-comparison results/ablation/ablation_ranking_comparison.yaml \
    --out-jaccard results/ablation/ablation_jaccard_matrix.tsv \
    --out-level-specific results/ablation/level_specific_variants.tsv
```

#### Adjusting Comparison Thresholds

The level-specific variant detection uses two thresholds:

- `--high-rank-threshold` (default: 100): a variant must be in the top-N at one level
- `--low-rank-threshold` (default: 500): the variant must be outside the top-N at all other levels

Tighter thresholds (e.g., `--high-rank-threshold 50 --low-rank-threshold 200`) produce a more selective list; looser thresholds capture more candidates.

---

## Command Reference

### preprocess.py

```bash
python scripts/preprocess.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--vcf` | path | required | VCF file path (.vcf.gz) |
| `--phenotypes` | path | required | Phenotype TSV file |
| `--output` | path | required | Output .pt file |
| `--max-variants-per-sample` | int | None | Maximum variants per sample (for debugging/testing) |
| `--min-gq` | int | 20 | Minimum genotype quality threshold |
| `--genome-build` | str | GRCh37 | Reference genome build (GRCh37 or GRCh38) |
| `--sex-map` | path | None | Path to sample_sex.tsv for ploidy-aware encoding |

**Example**:
```bash
python scripts/preprocess.py \
    --vcf cohort.vcf.gz \
    --phenotypes pheno.tsv \
    --output preprocessed.pt \
    --sex-map results/sex_inference/sample_sex.tsv \
    --genome-build GRCh37
```

---

### infer_sex.py

```bash
python scripts/infer_sex.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--vcf` | path | required | Multi-sample VCF file |
| `--output-dir` | path | required | Output directory |
| `--genome-build` | str | GRCh37 | Reference genome build |
| `--min-gq` | int | 20 | Minimum genotype quality |
| `--min-maf` | float | 0.05 | Minimum minor allele frequency |
| `--max-missing` | float | 0.10 | Maximum missingness per variant |
| `--f-male` | float | 0.8 | F-statistic threshold for males |
| `--f-female` | float | 0.2 | F-statistic threshold for females |
| `--known-sex` | path | None | Optional known sex file for concordance |

**Example**:
```bash
python scripts/infer_sex.py \
    --vcf cohort.vcf.gz \
    --output-dir results/sex_inference \
    --genome-build GRCh37
```

---

### check_sex_balance.py

```bash
python scripts/check_sex_balance.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--phenotypes` | path | required | Phenotypes TSV |
| `--sex-map` | path | required | sample_sex.tsv from infer_sex.py |
| `--genome-build` | str | GRCh37 | Reference genome build |
| `--output-dir` | path | required | Output directory |

**Example**:
```bash
python scripts/check_sex_balance.py \
    --phenotypes phenotypes.tsv \
    --sex-map results/sex_inference/sample_sex.tsv \
    --output-dir results/sex_balance
```

---

### train.py

```bash
python scripts/train.py [OPTIONS]
```

#### Data Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--vcf` | path | - | VCF file (if not using preprocessed) |
| `--phenotypes` | path | - | Phenotype file (if not using preprocessed) |
| `--preprocessed-data` | path | - | Preprocessed .pt file |
| `--level` | str | required | Annotation level [L0, L1, L2, L3, L4] |

#### Training Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--batch-size` | int | 32 | Batch size |
| `--chunk-size` | int | 3000 | Max variants per chunk |
| `--chunk-overlap` | int | 0 | Overlap between consecutive chunks |
| `--gradient-accumulation-steps` | int | 1 | Gradient accumulation |
| `--epochs` | int | 100 | Maximum epochs |
| `--lr` | float | 0.001 | Learning rate |
| `--lambda-attr` | float | 0.0 | Attribution regularisation |
| `--early-stopping` | int | 10 | Early stopping patience |
| `--gradient-clip` | float | None | Gradient clipping value |

#### Model Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--latent-dim` | int | 64 | Embedding dimension |
| `--hidden-dim` | int | 128 | Hidden layer dimension |
| `--num-heads` | int | 4 | Number of attention heads |
| `--num-attention-layers` | int | 2 | Number of attention layers |
| `--aggregation-method` | str | mean | Chunk aggregation method [mean, max, attention, logit_mean] |

#### Cross-Validation Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cv` | int | - | Number of CV folds (if not using val-split) |
| `--val-split` | float | 0.2 | Validation split ratio |

#### Output Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output-dir` | path | outputs | Output directory |
| `--experiment-name` | str | `{level}_run` | Experiment name |
| `--device` | str | `cuda` if available, else `cpu` | Device [cuda, cpu] |
| `--num-workers` | int | 0 | DataLoader workers |
| `--genome-build` | str | GRCh37 | Reference genome build |
| `--sex-map` | path | None | Adds sex covariate in training (and ploidy-aware encoding when training from VCF) |
| `--seed` | int | 42 | Random seed |

Note: `--aggregation-method attention` is currently exposed but not implemented in `ChunkedSIEVEModel`; use `mean`, `max`, or `logit_mean`.

**Example**:
```bash
python scripts/train.py \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --val-split 0.2 \
    --lr 0.00001 \
    --lambda-attr 0.1 \
    --epochs 100 \
    --batch-size 16 \
    --chunk-size 3000 \
    --output-dir experiments \
    --experiment-name my_experiment \
    --device cuda
```

---

### explain.py

```bash
python scripts/explain.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--experiment-dir` | path | - | Experiment directory |
| `--checkpoint` | path | - | Specific checkpoint (alternative to experiment-dir) |
| `--config` | path | - | Config YAML (required with checkpoint) |
| `--preprocessed-data` | path | required | Preprocessed data file |
| `--output-dir` | path | required | Output directory |
| `--n-steps` | int | 50 | Integration steps for IG |
| `--max-variants` | int | 2000 | Max variants per sample for IG |
| `--batch-size` | int | 4 | Batch size |
| `--skip-attention` | flag | False | Skip attention analysis (faster) |
| `--skip-ig` | flag | False | Skip integrated gradients |
| `--top-k-variants` | int | 100 | Number of top variants |
| `--top-k-interactions` | int | 100 | Number of top interactions |
| `--attention-threshold` | float | 0.1 | Min attention weight |
| `--is-null-baseline` | flag | False | Flag for null baseline analysis |
| `--device` | str | cuda | Device [cuda, cpu] |
| `--genome-build` | str | GRCh37 | Reference genome build |

**Example**:
```bash
python scripts/explain.py \
    --experiment-dir experiments/my_model \
    --preprocessed-data preprocessed.pt \
    --output-dir results/explainability \
    --n-steps 50 \
    --device cuda
```

---

### create_null_baseline.py

```bash
python scripts/create_null_baseline.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | path | required | Input preprocessed file |
| `--output` | path | - | Output path (single permutation) |
| `--output-dir` | path | - | Output directory (multiple permutations) |
| `--n-permutations` | int | 5 | Number of permutations |
| `--seed` | int | 42 | Random seed (or base seed) |

**Example (single)**:
```bash
python scripts/create_null_baseline.py \
    --input preprocessed.pt \
    --output preprocessed_NULL.pt \
    --seed 42
```

**Example (multiple)**:
```bash
python scripts/create_null_baseline.py \
    --input preprocessed.pt \
    --output-dir null_permutations \
    --n-permutations 5 \
    --seed 42
```

---

### run_null_baseline_analysis.sh

```bash
bash scripts/run_null_baseline_analysis.sh
```

This wrapper is configured through environment variables:

| Variable | Required | Description |
|---------|----------|-------------|
| `INPUT_DATA` | Yes | Path to real preprocessed `.pt` file |
| `REAL_EXPERIMENT` | Yes | Real experiment directory (single run) or fold directory (CV) |
| `OUTPUT_BASE` | Yes | Base directory where null outputs are written |
| `REAL_RESULTS` | No (recommended) | Directory containing real `sieve_variant_rankings.csv` |
| `DEVICE` | No | `cuda` or `cpu` (default: `cuda`) |
| `PYTHON` | No | Python interpreter path override |

Behaviour:
- Reads model/training hyperparameters from real `config.yaml` (in `REAL_EXPERIMENT` or parent).
- Carries over `sex_map` automatically when the real model used sex covariates.
- Resolves script paths relative to wrapper location, so it can be run from any working directory.

**Example**:
```bash
export INPUT_DATA=data/preprocessed.pt
export REAL_EXPERIMENT=experiments/my_model
export REAL_RESULTS=results/explainability
export OUTPUT_BASE=results/null_baseline_run
export DEVICE=cuda

bash scripts/run_null_baseline_analysis.sh
```

---

### compare_attributions.py

```bash
python scripts/compare_attributions.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--real` | path | required | Real variant rankings CSV |
| `--null` | path | - | Single null rankings CSV |
| `--null-dir` | path | - | Directory with multiple null results |
| `--output-dir` | path | required | Output directory |
| `--top-k` | int | 100 | Number of top variants to output |
| `--genome-build` | str | GRCh37 | Reference genome build |
| `--exclude-sex-chroms` | flag | False | Exclude chrX/chrY before thresholding/comparison |

**Example**:
```bash
python scripts/compare_attributions.py \
    --real results/explainability/sieve_variant_rankings.csv \
    --null results/null/sieve_variant_rankings.csv \
    --output-dir results/comparison \
    --top-k 100
```

---

### correct_chrx_bias.py

```bash
python scripts/correct_chrx_bias.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--rankings` | path | required | Variant rankings CSV |
| `--output-dir` | path | required | Output directory |
| `--null-rankings` | path | None | Null rankings CSV for recalculating enrichment |
| `--exclude-sex-chroms` | flag | True | Exclude chrX/chrY from final rankings (default) |
| `--include-sex-chroms` | flag | False | Include chrX/chrY (flagged) in rankings |
| `--genome-build` | str | GRCh37 | Reference genome build |
| `--top-k` | int | 100 | Top variants to annotate in plot |

**Example**:
```bash
python scripts/correct_chrx_bias.py \
    --rankings results/explainability/sieve_variant_rankings.csv \
    --null-rankings results/null_baseline_run/results/null_attributions/sieve_variant_rankings.csv \
    --output-dir results/explainability_corrected \
    --exclude-sex-chroms
```

---

### compare_ablation_rankings.py

```bash
python scripts/compare_ablation_rankings.py [OPTIONS]
```

Compares variant attribution rankings across annotation levels. Computes pairwise Jaccard similarity at multiple top-k thresholds and identifies level-specific variant discoveries.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--ranking-dir` | path | - | Directory with `L{0..3}_sieve_variant_rankings.csv` files (mutually exclusive with `--rankings`) |
| `--rankings` | LEVEL:PATH | - | Explicit per-level paths, e.g. `L0:path/to/rankings.csv` (repeatable, mutually exclusive with `--ranking-dir`) |
| `--top-k` | str | `50,100,200,500` | Comma-separated top-k values for Jaccard computation |
| `--high-rank-threshold` | int | 100 | A variant must be in the top-N at one level to be level-specific |
| `--low-rank-threshold` | int | 500 | A variant must be outside the top-N at all other levels |
| `--out-comparison` | path | `ablation_ranking_comparison.yaml` | Output YAML summary |
| `--out-jaccard` | path | `ablation_jaccard_matrix.tsv` | Output Jaccard matrix TSV |
| `--out-level-specific` | path | `level_specific_variants.tsv` | Output level-specific variants TSV |

**Example**:
```bash
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rankings \
    --top-k 50,100,200,500 \
    --out-comparison results/ablation/ablation_ranking_comparison.yaml \
    --out-jaccard results/ablation/ablation_jaccard_matrix.tsv \
    --out-level-specific results/ablation/level_specific_variants.tsv
```

---

### ablation_compare.py

```bash
python scripts/ablation_compare.py [OPTIONS]
```

Compares model performance (AUC, accuracy, loss) across annotation levels. Reads `results.yaml` or `cv_results.yaml` from each run directory and ranks levels by predictive performance.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--run-dir` | path | - | Run directory containing results/config YAML (repeatable, mutually exclusive with `--results-dir`) |
| `--results-dir` | path | - | Parent directory with `ablation_L{0..3}/` sub-directories (mutually exclusive with `--run-dir`) |
| `--out-summary-tsv` | path | `ablation_summary.tsv` | Output TSV |
| `--out-summary-yaml` | path | `ablation_summary.yaml` | Output YAML |

**Example**:
```bash
python scripts/ablation_compare.py \
    --results-dir experiments \
    --out-summary-tsv results/ablation/ablation_summary.tsv \
    --out-summary-yaml results/ablation/ablation_summary.yaml
```

---

### plot_ablation_comparison.py

```bash
python scripts/plot_ablation_comparison.py [OPTIONS]
```

Creates a multi-panel publication figure from the outputs of `compare_ablation_rankings.py` and `ablation_compare.py`. Panels include a Jaccard heatmap, overlap-by-top-k line plot, level-specific variant counts, and (optionally) an AUC comparison bar chart.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--jaccard-tsv` | path | required | Jaccard matrix TSV from `compare_ablation_rankings.py` |
| `--level-specific-tsv` | path | required | Level-specific variants TSV from `compare_ablation_rankings.py` |
| `--summary-yaml` | path | None | Ablation summary YAML from `ablation_compare.py` (optional; adds AUC panel) |
| `--heatmap-top-k` | int | 100 | Top-k value for the heatmap panel |
| `--output` | path | `ablation_comparison.png` | Output figure path (PNG or PDF) |

**Example**:
```bash
python scripts/plot_ablation_comparison.py \
    --jaccard-tsv results/ablation/ablation_jaccard_matrix.tsv \
    --level-specific-tsv results/ablation/level_specific_variants.tsv \
    --summary-yaml results/ablation/ablation_summary.yaml \
    --output results/ablation/ablation_comparison.png
```

---

### validate_epistasis.py

```bash
python scripts/validate_epistasis.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--interactions` | path | required | Interactions CSV from explain.py |
| `--checkpoint` | path | required | Model checkpoint |
| `--config` | path | required | Config YAML |
| `--preprocessed-data` | path | required | Preprocessed data |
| `--output-dir` | path | required | Output directory |
| `--top-k` | int | 50 | Number of interactions to validate |
| `--synergy-threshold` | float | 0.05 | Minimum significant synergy |
| `--device` | str | cuda | Device [cuda, cpu] |
| `--genome-build` | str | GRCh37 | Reference genome build |

---

### validate_discoveries.py

```bash
python scripts/validate_discoveries.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--variant-rankings` | path | required | Variant rankings CSV |
| `--gene-rankings` | path | required | Gene rankings CSV |
| `--output-dir` | path | required | Output directory |
| `--clinvar-db` / `--clinvar` | path | - | ClinVar TSV database |
| `--gwas-db` / `--gwas` | path | - | GWAS Catalog TSV |
| `--go-annotations` / `--go-mapping` | path | - | Gene-to-GO mapping JSON |
| `--top-k-variants` | int | 100 | Number of top variants to validate |
| `--top-k-genes` | int | 50 | Number of top genes to validate |
| `--disease-terms` | str list | - | Optional GWAS trait filter terms |
| `--genome-build` | str | GRCh37 | Reference genome build |

---

## Interpreting Results

### Training Outputs

#### Single Run Results (`results.yaml`)

```yaml
auc: 0.75              # Area under ROC curve
accuracy: 0.70         # Classification accuracy
loss: 0.42             # Final validation loss
classification_loss: 0.42
attribution_loss: 0.0  # If lambda_attr > 0
```

**Interpretation**:
- **AUC = 0.5**: Random (no learning) → Check data/encoding
- **AUC = 0.6-0.7**: Weak signal → May need more data or better features
- **AUC = 0.7-0.8**: Good performance → Model learning meaningful patterns
- **AUC > 0.8**: Strong performance → Reliable predictions

#### Cross-Validation Results (`cv_results.yaml`)

```yaml
mean_auc: 0.75
std_auc: 0.05          # Lower is better (more stable)
mean_accuracy: 0.70
std_accuracy: 0.03
fold_results:
  - auc: 0.78
    accuracy: 0.72
    best_epoch: 14
    epochs_trained: 22
    training_time_seconds: 480.1
  ...
```

**Interpretation**:
- **Low std (<0.05)**: Stable performance across folds
- **High std (>0.10)**: Performance varies with data split → May indicate:
  - Small dataset
  - Label imbalance
  - Overfitting

---

### Explainability Outputs

#### Variant Rankings

Columns in `sieve_variant_rankings.csv`:

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `position` | Genomic position (chr:pos) | Variant location |
| `chromosome` | Chromosome | - |
| `gene_id` | Gene name | Which gene contains variant |
| `mean_attribution` | Mean absolute attribution | **Main importance score** |
| `max_attribution` | Max attribution | Peak importance |
| `num_samples` | Number of samples | Frequency in cohort |
| `case_attribution` | Mean in cases | Case-specific importance |
| `control_attribution` | Mean in controls | Control-specific importance |
| `case_control_diff` | Case - control | **Case enrichment** |
| `rank` | Overall rank | 1 = most important |

**Key Metrics**:

1. **mean_attribution**: Primary importance metric
   - High value → Variant influences predictions strongly
   - Sort by this to find top discoveries

2. **case_control_diff**: Disease specificity
   - Positive → More important in cases (disease-associated)
   - Negative → More important in controls (protective?)
   - Near zero → Affects both equally

3. **num_samples**: Confidence
   - High count → Consistent across individuals
   - Low count → May be population-specific

**Example Interpretation**:
```
variant: 17:41245466
gene: BRCA1
mean_attribution: 0.45
case_control_diff: 0.38
num_samples: 42

Interpretation: This BRCA1 variant has high attribution (0.45),
is strongly enriched in cases (diff=0.38), and appears in 42
samples. Likely a genuine disease-associated variant.
```

#### chrX Ploidy Bias Correction (Optional)

If you used sex-aware preprocessing or observe chrX inflation in rankings, run `correct_chrx_bias.py` to standardise mean attributions per chromosome. The script adds:

- `z_attribution`: per-chromosome z-scored attribution
- `corrected_rank`: rank based on `z_attribution`
- `is_sex_chrom`: flags chrX/chrY variants

By default, the corrected rankings exclude sex chromosomes. Use `--include-sex-chroms` if you want to keep them in the output (they remain flagged).

#### Gene Rankings

Columns in `sieve_gene_rankings.csv`:

| Column | Description |
|--------|-------------|
| `gene_id` | Gene name |
| `num_variants` | Number of variants in gene |
| `gene_score` | Aggregated importance (max or mean) |
| `top_variant_pos` | Position of most important variant |
| `gene_rank` | Gene ranking |

**Use Gene Rankings When**:
- Interested in gene-level associations (not specific variants)
- Comparing with gene-based GWAS
- Doing pathway enrichment analysis

---

### Null Baseline Comparison

#### Summary YAML (`comparison_summary.yaml`)

```yaml
thresholds:
  p_0.05: 0.152    # 95th percentile of null
  p_0.01: 0.238    # 99th percentile of null
  p_0.001: 0.394   # 99.9th percentile of null

distribution_comparison:
  real_mean: 0.089
  null_mean: 0.045
  ks_pvalue: 2.3e-145    # Distributions differ
  mannwhitney_pvalue: 1.1e-78

significance_counts:
  p_0.01:
    threshold: 0.238
    observed: 46        # Variants in real exceeding threshold
    expected: 10.0      # Expected by chance (1% of 1000)
    enrichment: 4.6     # 4.6× more than expected

interpretation:
  distributions_differ: true
  real_higher_than_null: true
  enrichment_at_p01: 4.6
  n_significant_p01: 46
```

**How to Interpret**:

1. **Check distributions differ** (KS p-value < 0.001):
   - ✓ Yes → Real model found signal
   - ✗ No → May need more data or different approach

2. **Check enrichment at p<0.01**:
   - **< 1.5×**: Weak, be cautious
   - **1.5-2×**: Moderate, validate carefully
   - **2-5×**: Strong, good confidence
   - **> 5×**: Very strong, high confidence

3. **Identify significant variants**:
   - Review `significant_variants_p01.csv`
   - These variants have attributions exceeding 99th percentile of null
   - Use these for biological validation

**Decision Framework**:

| Enrichment | KS p-value | Interpretation | Action |
|-----------|-----------|----------------|--------|
| > 2× | < 0.001 | Strong signal | Proceed to validation |
| 1.5-2× | < 0.01 | Moderate signal | Validate top hits carefully |
| < 1.5× | > 0.05 | Weak/no signal | Check data quality, increase sample size |

---

### Ablation Comparison Results

#### Performance Summary (`ablation_summary.yaml`)

```yaml
best_level: L2
best_run_id: ablation_L2
ranking_metric_priority: [auc, accuracy, loss]
levels:
  - level: L0
    run_id: ablation_L0
    auc: 0.68
    std_auc: 0.04
    accuracy: 0.65
    loss: 0.58
  - level: L1
    run_id: ablation_L1
    auc: 0.72
    std_auc: 0.03
    accuracy: 0.69
    loss: 0.51
  - level: L2
    run_id: ablation_L2
    auc: 0.76
    std_auc: 0.03
    accuracy: 0.72
    loss: 0.46
  - level: L3
    run_id: ablation_L3
    auc: 0.75
    std_auc: 0.04
    accuracy: 0.71
    loss: 0.47
```

**Interpretation**:
- **L0 AUC > 0.6**: Genotype patterns alone carry disease signal (annotation-free discovery is feasible)
- **L2 ≈ L3**: Consequence class is sufficient; SIFT/PolyPhen add little beyond consequence type
- **L3 > L0 by >0.1 AUC**: Annotations provide substantial additional signal
- **L3 ≈ L0**: Annotations do not help, model discovers signal from genotype structure alone

#### Jaccard Matrix (`ablation_jaccard_matrix.tsv`)

Each row represents a pairwise comparison at a given top-k:

| Column | Description |
|--------|-------------|
| `top_k` | Number of top variants compared |
| `level_a`, `level_b` | The two levels being compared |
| `jaccard` | Jaccard index (0-1; higher = more overlap) |
| `overlap` | Number of shared variants |
| `size_a`, `size_b` | Number of variants in each set |
| `union` | Size of the union |

**How to read it**:
- **Jaccard > 0.7**: Very similar rankings — the two levels discover largely the same variants
- **Jaccard 0.3-0.7**: Moderate overlap — some shared discoveries, some unique to each level
- **Jaccard < 0.3**: Different rankings — annotation level fundamentally changes which variants are prioritised

**Scientific significance**:
- High L0-vs-L3 Jaccard indicates the model can discover the same variants without annotations (supports annotation-free discovery)
- Low L0-vs-L3 Jaccard suggests annotations drive different discoveries (may indicate circular logic if annotations encode known associations)

#### Level-Specific Variants (`level_specific_variants.tsv`)

Variants ranked in the top-100 at one level but outside the top-500 at all other levels:

| Column | Description |
|--------|-------------|
| `variant_id` | Unique variant identifier (chrom:pos_gene_id) |
| `gene` | Gene name |
| `specific_to_level` | The annotation level where this variant is highly ranked |
| `rank_at_specific_level` | Rank at the specific level |
| `rank_at_L0` ... `rank_at_L3` | Rank at each level (for cross-reference) |
| `mean_attribution_at_specific_level` | Attribution score at the specific level |

**How to use these**:
- **L0-specific variants**: Discovered from genotype patterns alone — potentially novel mechanisms invisible to annotation-based methods. Priority candidates for experimental follow-up.
- **L3-specific variants**: Only discovered when SIFT/PolyPhen are provided — may reflect annotation-dependent signal (known pathogenicity) rather than novel discovery.
- **L1-specific variants**: Position carries information not captured by genotype alone — may indicate positional clustering or regulatory elements.

#### Multi-Panel Figure (`ablation_comparison.png`)

The figure produced by `plot_ablation_comparison.py` contains four panels:

1. **Jaccard Heatmap** (top-left): Pairwise overlap at a selected top-k. Warm colours indicate low overlap (different discoveries), cool colours indicate high overlap (similar discoveries).

2. **Jaccard by Top-k** (top-right): Line plot showing how overlap evolves as you consider more variants. If lines rise steeply, the top-ranked variants differ but broader rankings converge.

3. **Level-Specific Counts** (bottom-left): Bar chart of how many uniquely important variants each level discovers. Large L0 bars support annotation-free discovery.

4. **AUC Comparison** (bottom-right): Model performance per level with error bars. The best level is highlighted. The red dashed line marks random performance (AUC=0.5).

---

### Epistasis Results

#### Validation Output (`epistasis_validation.csv`)

Columns:

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `variant1_pos`, `variant2_pos` | Variant positions | - |
| `variant1_gene`, `variant2_gene` | Gene names | Same gene or trans? |
| `pred_both` | Prediction with both variants | Combined effect |
| `pred_variant1_only` | Prediction with only v1 | Individual effect |
| `pred_variant2_only` | Prediction with only v2 | Individual effect |
| `pred_neither` | Prediction with neither | Baseline |
| `effect_variant1` | v1 individual effect | - |
| `effect_variant2` | v2 individual effect | - |
| `effect_combined` | Combined effect | - |
| `synergy` | Combined - v1 - v2 | **Key metric** |
| `interaction_type` | synergistic/antagonistic | - |
| `is_significant` | \|synergy\| > threshold | - |

**Synergy Interpretation**:

- **synergy > 0.1**: Strong synergistic
  - Example: v1 effect=0.05, v2 effect=0.05, combined=0.25
  - Together they amplify each other's effects

- **synergy ≈ 0**: Independent (additive)
  - Example: v1=0.10, v2=0.15, combined=0.25
  - No interaction, just sum of effects

- **synergy < -0.1**: Strong antagonistic
  - Example: v1=0.20, v2=0.20, combined=0.10
  - They interfere with each other

**Biological Interpretation**:

1. **Synergistic in same gene**: Potential compound heterozygosity
2. **Synergistic across genes**: Gene-gene interaction
3. **Antagonistic**: Compensatory mechanism or regulatory feedback

---

## Troubleshooting

### Installation Issues

#### ImportError: No module named 'cyvcf2'

**Solution**:
```bash
pip install cyvcf2
# or if that fails:
conda install -c bioconda cyvcf2
```

#### CUDA out of memory

**Solution**: Reduce memory usage
```bash
python scripts/train.py \
    --batch-size 2 \
    --chunk-size 2000 \
    --gradient-accumulation-steps 16 \
    ...
```

---

### Data Preparation Issues

#### VCF parsing fails

**Symptom**: `KeyError: 'CSQ'` or `ValueError: No VEP annotations found`

**Solution**: Your VCF is not VEP-annotated. Run VEP:
```bash
vep --input_file your.vcf \
    --output_file annotated.vcf \
    --vcf --symbol --sift b --polyphen b \
    --assembly GRCh37
```

#### Sample name mismatch

**Symptom**: `KeyError: SAMPLE001` or `ValueError: Sample not found in VCF`

**Solution**: Check that phenotype file sample IDs exactly match VCF:
```bash
# Get VCF samples
bcftools query -l your.vcf.gz

# Check phenotype file
cut -f1 phenotypes.tsv
```

Sample names must match character-for-character (case-sensitive).

#### Chromosome naming issue

**Symptom**: `KeyError: 'chr1'` or no variants loaded

**Solution**: SIEVE normalises both styles (`1` and `chr1`) internally. If you still see this error, check:
- `--genome-build` matches your data (`GRCh37` or `GRCh38`)
- Contigs are standard autosomes/sex chromosomes (1-22, X, Y), or can be mapped cleanly
- Phenotype sample IDs match VCF sample IDs

If your VCF uses non-standard contig labels, rename contigs:
```bash
bcftools annotate --rename-chrs chr_name_conv.txt input.vcf.gz -O z -o output.vcf.gz

# Where chr_name_conv.txt contains:
chr1 1
chr2 2
...
```

---

### Training Issues

#### Model not learning (AUC ≈ 0.5)

**Possible Causes & Solutions**:

1. **Insufficient data**
   - Need: >100 cases and >100 controls minimum
   - Solution: Acquire more samples or use data augmentation

2. **Label imbalance**
   - Check: How many cases vs controls?
   - Solution: If extreme (<10% minority), consider class weights

3. **Encoding issues**
   - Check: Run `python test_encoding_pipeline.py`
   - Solution: Verify features have non-zero variance

4. **Wrong learning rate**
   - Try: `--lr 0.000001` (lower) or `--lr 0.0001` (higher)

5. **Model too complex for data size**
   - Try: `--latent-dim 16 --hidden-dim 32 --num-attention-layers 1`

6. **Data leakage or preprocessing error**
   - Verify: Cases and controls are truly different cohorts

#### Training very slow

**Solutions**:

1. **Use GPU**: `--device cuda`
2. **Increase batch size**: `--batch-size 32` (if memory allows)
3. **Use preprocessed data**: Much faster than parsing VCF each time
4. **Reduce integration steps**: `--n-steps 25` in explain.py

#### Out of memory during training

**Solution**: Use memory-efficient settings:
```bash
--batch-size 2 \
--gradient-accumulation-steps 16 \
--chunk-size 2000
```

See "Memory-Efficient Training" section above.

---

### Explainability Issues

#### Integrated gradients very slow

**Solutions**:

1. Reduce integration steps: `--n-steps 25` (less accurate but faster)
2. Limit variants per sample: `--max-variants 1500`
3. Use larger batch size: `--batch-size 8` (if memory allows)
4. Skip attention analysis: `--skip-attention` (if only need variant rankings)

#### AttributeError: 'NoneType' object has no attribute

**Cause**: Model checkpoint not found or corrupted

**Solution**: Verify checkpoint exists:
```bash
ls -lh experiments/my_model/best_model.pt
```

If using `--experiment-dir`, check that `best_model.pt` or `fold_*/best_model.pt` exists.

---

### Null Baseline Issues

#### Null model AUC ≠ 0.5

**Expected**: Null model AUC should be ≈0.50 ± 0.05

**If AUC > 0.6**:
- **Problem**: Permutation didn't properly break genotype-phenotype relationship
- **Check**: Did you use the same preprocessed file for null training?
- **Solution**: Verify null baseline file has `_null_baseline_metadata` field

**If AUC < 0.4**:
- This is actually fine - model is consistently wrong, which is equivalent to chance
- Attributions are still valid for null distribution

#### No significant variants (enrichment < 1)

**Possible Causes**:

1. **Real model didn't learn**: Check real model AUC first
2. **Null and real similar**: May indicate no genuine signal in data
3. **Sample size too small**: Need larger cohort for robust signal
4. **Wrong parameters**: Ensure null trained with exact same params as real

**Solution**: Review real model performance first, then consider increasing sample size.

---

### Interpretation Issues

#### All top variants in the same gene

**Is this a problem?**
- Depends! If studying a Mendelian disease, this is expected
- For complex diseases, expect multiple genes
- Check: Is the gene biologically relevant to your phenotype?

**Possible issue**: Overfitting to one gene
- Solution: Check cross-validation stability
- Look at fold-specific rankings - is it consistent?

#### chrX dominates top rankings

**Possible causes**:
- Ploidy differences (hemizygosity) inflate chrX attributions
- Sex imbalance across case/control groups

**Solutions**:
1. Run sex-aware preprocessing (`infer_sex.py` → `preprocess.py --sex-map`)
2. Check sex balance (`check_sex_balance.py`)
3. Apply post-hoc correction (`correct_chrx_bias.py`)

**Note**: `correct_chrx_bias.py` excludes sex chromosomes by default; use `--include-sex-chroms` if you need chrX/chrY retained.

#### Very low attributions overall

**Possible causes**:
- Model has low confidence (AUC close to 0.5)
- Attribution regularisation too strong (reduce `--lambda-attr`)
- Integration steps too low (increase `--n-steps`)

**Solution**:
1. Check model performance first
2. If AUC is good but attributions low, increase `--n-steps` to 100

#### Case-control differences all near zero

**Meaning**: Variants affect cases and controls similarly

**Interpretation**:
- May indicate population stratification (batch effects)
- Or: Model learned overall variant burden, not disease-specific patterns

**Solution**:
- Check for population structure (PCA analysis)
- Consider adjusting for covariates in future version

---

## FAQ

### General Questions

**Q: How many samples do I need?**
A: Minimum 50 cases + 50 controls for initial testing. Recommended 250+ cases and 250+ controls for robust results. For epistasis detection, 5000+ samples ideal.

**Q: Can I use WGS data instead of exome?**
A: Yes, but be aware:
- Much larger file sizes (slower preprocessing)
- More variants per sample (higher memory usage)
- May need to filter to exonic regions for meaningful results

**Q: What reference genome does SIEVE use?**
A: Both GRCh37 (hg19) and GRCh38 (hg38) are supported via `--genome-build`. Contigs with or without `chr` prefix are normalised automatically.

**Q: Can I use SIEVE for quantitative traits?**
A: Not currently. SIEVE is designed for binary case-control studies. Adaptation for quantitative traits would require modifying the loss function and output layer.

**Q: How long does a typical analysis take?**
A:
- Preprocessing: 30 mins - 5 hours (once)
- Training: 1-3 hours per model (on GPU)
- Explainability: 30-60 mins
- Null baseline: Same as training + explainability
- Total: 4-12 hours for complete analysis

### Technical Questions

**Q: What is "chunked processing"?**
A: SIEVE processes variants in chunks (default 3000) to fit in GPU memory. This allows handling whole-genome data without running out of memory. The chunk size is automatically managed but can be tuned with `--chunk-size`.

**Q: What happens if a sample has more variants than chunk_size?**
A: The sample is processed in multiple chunks, then results are aggregated. This is handled automatically.

**Q: Why use gradient accumulation?**
A: It simulates larger batch sizes without using more memory. For example, `--batch-size 2 --gradient-accumulation-steps 16` gives the training dynamics of `--batch-size 32` while only using memory for 2 samples at a time.

**Q: What's the difference between --batch-size and --chunk-size?**
A:
- `--batch-size`: Number of samples processed together
- `--chunk-size`: Maximum variants processed per forward pass (per sample)
- Both affect memory usage but in different ways

**Q: Can I use multiple GPUs?**
A: Not currently supported. SIEVE uses a single GPU. If you have multiple GPUs, you can run multiple experiments in parallel on different GPUs.

### Scientific Questions

**Q: What if L0 (genotype-only) performs as well as L3?**
A: This is scientifically interesting! It suggests:
- Genotype patterns alone carry disease signal
- Annotations may not add much information for this phenotype
- Potential for discovering novel variants missed by annotation-based methods

**Q: What enrichment factor is "good enough"?**
A: Guidelines:
- < 1.5×: Weak signal, be very cautious
- 1.5-2×: Moderate, validate top 10-20 variants
- 2-5×: Strong, proceed with confidence
- \> 5×: Very strong, high confidence in discoveries

**Q: Should I always run null baseline?**
A: **Yes, for publication-quality results.** It's the only way to establish statistical significance of your discoveries. For initial exploration, you can skip it, but include it before claiming discoveries.

**Q: How do I know if a variant is truly causal?**
A: You don't, from computational analysis alone. SIEVE identifies statistical associations. Causality requires:
1. High attribution score
2. Exceeds null baseline threshold
3. Biological plausibility (gene function, prior evidence)
4. **Experimental validation** (functional studies, replication cohort)

**Q: What's the difference between attention patterns and integrated gradients?**
A:
- **Integrated gradients**: Measures how much a variant contributes to the final prediction (variant importance)
- **Attention patterns**: Measures which variant pairs the model looks at together (variant interactions)
- Both are complementary - use both for full picture

### Troubleshooting Questions

**Q: Training works but explainability crashes with OOM**
A: Integrated gradients requires more memory than training. Solutions:
- Reduce `--n-steps` (try 25 instead of 50)
- Reduce `--max-variants` (try 1500 instead of 2000)
- Reduce `--batch-size` (try 2 instead of 4)

**Q: Null model has better AUC than real model?**
A: This occasionally happens by chance (especially with small datasets). Solutions:
- Run multiple null permutations (5-10) and use the most conservative threshold
- Increase sample size
- Check for data quality issues in real data

**Q: Cross-validation folds have very different AUC values?**
A: High variance across folds suggests:
- Small sample size → Increase if possible
- Label imbalance → Check case/control ratio
- Overfitting → Try simpler model or more regularisation
- Population stratification → Check for batch effects

---

## Citation

If you use SIEVE in your research, please cite:

```bibtex
@software{sieve2026,
  title = {SIEVE: Sparse Interpretable Exome Variant Explainer},
  author = {Francesco Lescai},
  year = {2026},
  url = {https://github.com/lescailab/sieve-project}
}
```

---

## Support

- **GitHub Issues**: https://github.com/lescailab/sieve-project/issues
- **Documentation**: This guide + README.md + code docstrings
- **Updates**: Check GitHub releases for new versions

---

## License

MIT License - See LICENSE file for details.

---

**Last Updated**: 2026-02-06
**Document Version**: 1.1
**SIEVE Version**: 0.1.0+


---

# Appendices

## Appendix A: Model Architecture Details

### Overview

SIEVE (Sparse Interpretable Exome Variant Explainer) is designed around three core innovations:
1. Position-aware sparse attention that preserves spatial relationships between variants
2. Attribution-regularised training that builds interpretability into the objective
3. Annotation-ablation protocol that distinguishes genuine discovery from annotation recovery

This appendix provides the mathematical foundations and implementation details.

### Data Representation

#### Input: Annotated Multi-sample VCF

The input is a VCF file with N samples and V variant sites. Each variant site v has:
- Chromosome and position (chrom_v, pos_v)
- Reference and alternate alleles
- Per-sample genotypes g_{v,n} ∈ {0, 1, 2} (reference homozygote, heterozygote, alternate homozygote)
- VEP annotations including gene assignment, consequence, and functional scores

#### Variant Feature Vector

Each variant v is represented by a feature vector x_v whose composition depends on the annotation level:

**Level L0** (genotype only):
```
x_v = [g_v]  # Just genotype dosage, dimension 1
```

**Level L1** (genotype + position):
```
x_v = [g_v, PE(pos_v)]  # Genotype + positional encoding, dimension 1 + d_pos
```

**Level L2** (L1 + consequence):
```
x_v = [g_v, PE(pos_v), one_hot(consequence_v)]  # Add consequence class
```

**Level L3** (L2 + SIFT/PolyPhen):
```
x_v = [g_v, PE(pos_v), one_hot(consequence_v), sift_v, polyphen_v]
```

**Level L4** (full annotations):
```
x_v = [g_v, PE(pos_v), one_hot(consequence_v), sift_v, polyphen_v, lof_v, ...]
```

#### Positional Encoding

We use sinusoidal positional encodings adapted for genomic positions:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_pos))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_pos))
```

where d_pos is the positional embedding dimension (default 64).

This encoding allows the model to learn functions of relative position (important for detecting compound heterozygosity or clustered variants).

#### Per-Sample Sparse Representation

For sample n, we construct a sparse representation S_n containing only their non-reference variants:

```
S_n = {(v, x_v, g_{v,n}) : g_{v,n} > 0}
```

This is the key to handling sparsity: we never materialise the full V-dimensional tensor, only the positions where the individual has variants.

### Model Architecture Components

#### Component 1: Variant Encoder

A small MLP that projects variant features into a latent space:

```python
class VariantEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        # x: (batch, num_variants, input_dim)
        return self.mlp(x)  # (batch, num_variants, latent_dim)
```

#### Component 2: Position-Aware Sparse Attention

This is the core architectural innovation. Standard self-attention is O(n²) in sequence length, infeasible for millions of genomic positions. Standard sparse attention (like in BigBird) uses fixed sparsity patterns. Our approach is naturally sparse because we only attend among variant-present positions.

The key insight: we want to preserve positional information without requiring dense encoding. We achieve this by:
1. Operating only on positions with variants (natural sparsity)
2. Encoding relative distances in attention computation

**Why this matters**: The relative position bias allows the model to learn that variants close together (potential compound heterozygosity) or at specific distances (potential haplotype patterns) are informative. This is impossible with permutation-invariant deep sets.

#### Component 3: Gene Aggregation

After attention layers process variant relationships, we aggregate to gene level using permutation-invariant pooling (max or mean).

#### Component 4: Phenotype Classifier

Simple classification head on gene representations with dropout for regularisation.

### Attribution-Regularised Training

#### Standard Loss

Binary cross-entropy for case-control classification:

```python
bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
```

#### Attribution Sparsity Loss

We want the model to produce sparse attributions—most of its prediction should depend on a small number of variants. This is implemented through a differentiable approximation.

During training, we compute a simplified attribution score (gradient × input) and penalise its entropy:

**Why this matters**: Without this regularisation, a model might achieve good classification by using many weak signals spread across the genome. Such a model is hard to interpret—which variants really matter? The sparsity loss encourages the model to "commit" to a small set of variants, making explainability more meaningful.

#### Hyperparameter: λ_attr

The attribution regularisation weight λ_attr controls the trade-off:
- λ_attr = 0: Standard training, no sparsity constraint
- λ_attr small (0.01-0.1): Mild encouragement toward sparsity
- λ_attr large (>0.5): Strong sparsity, may hurt classification performance

We recommend starting with λ_attr = 0.1 and tuning based on validation performance.

### Explainability Methods

#### 1. Integrated Gradients (Variant-Level Attribution)

For a trained model, compute integrated gradients from a zero baseline to the actual input using the Captum library. The baseline represents "no variants" and the integration path interpolates from this baseline to the actual genotype.

#### 2. Attention Weight Analysis (Positional Patterns)

The attention weights reveal which variant pairs the model considers together. High mutual attention between variant pairs can indicate:
- Compound heterozygosity (nearby variants in same gene)
- Haplotype structure (variants inherited together)
- Potential epistatic interactions

#### 3. Epistasis Detection via Counterfactual Perturbation

To validate that detected interactions are truly epistatic (non-additive):

```
effect_i = f(with variant i) - f(without variant i)
effect_j = f(with variant j) - f(without variant j)
effect_ij = f(with both) - f(without both)
epistasis_score = effect_ij - (effect_i + effect_j)
```

Non-zero epistasis score indicates non-additive interaction.

### Model Complexity and Scalability

#### Parameter Count

For a typical configuration (20,000 genes, latent_dim=128, 2 attention layers, 4 heads):
- Total parameters: ~5M (much smaller than foundation models)

#### Computational Complexity

The key computational step is attention, which is O(V²) where V is the number of variants per sample. But V is typically 10,000-50,000 (not millions), making this tractable.

- Per-sample forward pass: ~0.1-1 second on GPU
- Training 1,000 samples for 100 epochs: ~3-10 hours on single GPU

#### Memory Requirements

For batch=32, V=30,000, heads=4, latent_dim=128:
- Attention: ~14 GB (may need gradient checkpointing)
- Embeddings: ~0.5 GB

**Recommendation**: Use gradient checkpointing for attention layers, batch size 16-32.

---

## Appendix B: Experimental Protocol

### Overview

This appendix describes the rigorous experimental protocol for evaluating SIEVE. The experiments are designed to answer specific scientific questions rather than just demonstrate technical capability.

### Scientific Questions

#### Question 1: Can deep learning discover variants that annotation-based methods miss?

**Hypothesis**: Models trained with minimal annotations will identify some disease-associated variants that models using full annotations rank lower, because the annotation-heavy models may over-rely on prior knowledge.

**Experiment**: Annotation ablation study comparing variant rankings across annotation levels L0-L4.

#### Question 2: Do spatial relationships between variants carry disease signal?

**Hypothesis**: Position-aware models will outperform position-agnostic models (pure deep sets) on classification, and attention weights will show meaningful positional patterns (e.g., clustering of important variants).

**Experiment**: Compare SIEVE (with position-aware attention) against a DeepRVAT-style deep set baseline.

#### Question 3: Does attribution-regularised training improve discovery?

**Hypothesis**: Models trained with attribution sparsity loss will produce more stable and biologically meaningful variant rankings than models trained with classification loss alone.

**Experiment**: Compare variant rankings between models with λ_attr = 0 vs λ_attr > 0.

#### Question 4: Can we detect and validate epistatic interactions?

**Hypothesis**: Attention patterns will identify variant pairs with non-additive effects, validated through counterfactual perturbation.

**Experiment**: Identify high-attention variant pairs, test for epistasis via counterfactual analysis, compare with known gene-gene interactions if available.

### Experimental Design

#### Data Requirements

**Input data**:
- Multi-sample VCF file, annotated with VEP (CSQ field)
- Phenotype file: sample IDs with binary case/control labels
- Reference genome: GRCh37 or GRCh38

**Minimum dataset size**:
- At least 500 samples (250 cases, 250 controls) for meaningful cross-validation
- Literature suggests >5,000 samples for robust epistasis detection

**Quality control** (applied before experiments):
- Remove samples with >5% missing genotypes
- Remove variants with >5% missing genotypes
- Remove variants with HWE p-value < 1e-6 in controls
- Optionally filter by MAF (but track this for annotation level effects)

#### Cross-Validation Strategy

Use nested cross-validation to prevent overfitting during hyperparameter selection:

**Outer loop**: 5-fold CV for final performance estimation
**Inner loop**: 3-fold CV for hyperparameter tuning within each outer fold

```
For each outer fold (5 iterations):
    training_data = 80% of samples
    test_data = 20% of samples (held out)
    
    For each hyperparameter configuration:
        For each inner fold (3 iterations):
            inner_train = 67% of training_data
            inner_val = 33% of training_data
            Train model on inner_train
            Evaluate on inner_val
        Average inner validation performance
    
    Select best hyperparameters based on inner CV
    Train final model on full training_data with best hyperparameters
    Evaluate on test_data
    Store predictions and variant rankings

Report mean ± std of outer fold test performance
```

#### Evaluation Metrics

**Classification performance**:
- AUC-ROC (primary metric)
- AUC-PR (for imbalanced data)
- Accuracy, sensitivity, specificity at optimal threshold

**Variant discovery**:
- Overlap with known GWAS hits (if available)
- Gene-set enrichment analysis (KEGG, Reactome)
- Stability of top variants across CV folds (Jaccard similarity)

**Epistasis**:
- Number of significant epistatic pairs (p < 0.05 after Bonferroni)
- Proportion of pairs showing non-additive effects
- Replication in held-out data

### Experiment 1: Annotation Ablation Study

#### Purpose

Determine whether models with minimal annotations can discover variants that annotation-heavy models miss, testing the hypothesis that deep learning can find patterns beyond what prior knowledge encodes.

#### Protocol

1. **Train 5 models at each annotation level** (L0 through L4) using identical architecture and hyperparameters except for input dimension

2. **For each model**, compute integrated gradients to obtain variant-level attribution scores

3. **Compare variant rankings** across annotation levels:
   - Top 100 variants at each level
   - Overlap analysis (Jaccard similarity)
   - Identify "L0-specific" variants: high rank at L0, low rank at L4
   - Identify "L4-specific" variants: high rank at L4, low rank at L0

4. **Biological interpretation**:
   - Are L0-specific variants in genes not annotated as pathogenic?
   - Are they enriched for regulatory regions or novel mechanisms?
   - Do L4-specific variants simply have high CADD/SIFT scores?

#### Expected Outcomes

**If hypothesis is supported**:
- L0 model achieves reasonable (>0.6) AUC, showing genotype patterns alone carry signal
- Some L0-specific variants are not captured by standard annotation methods
- These variants may point to novel disease mechanisms

**If hypothesis is refuted**:
- L0 model fails to learn (AUC ~0.5), suggesting annotations are necessary
- All high-ranking variants at L0 are subset of L4 rankings
- This would still be informative: it means annotation-free discovery is not feasible for this phenotype

### Experiment 2: Position-Aware vs Position-Agnostic

#### Purpose

Test whether spatial relationships between variants carry disease-relevant information by comparing position-aware sparse attention against permutation-invariant deep sets.

#### Protocol

1. **Implement two model variants**:
   - SIEVE (position-aware): Full model with positional encodings and relative position bias
   - DeepSet baseline: Same architecture but without positional information

2. **Train both models** on identical data with identical hyperparameters

3. **Compare classification performance**: AUC, sensitivity, specificity

4. **Analyse attention patterns** (SIEVE only):
   - Distribution of distances between high-attention variant pairs
   - Are nearby variants (potential compound heterozygosity) attended together?

#### Expected Outcomes

**If position matters**:
- SIEVE outperforms DeepSet baseline by >2% AUC
- Attention weights show non-uniform distance distribution
- High-attention pairs are enriched for same-exon or functional domain

### Experiment 3: Attribution Regularisation Study

#### Purpose

Determine whether training with attribution sparsity loss improves the stability and biological meaningfulness of discovered variants.

#### Protocol

1. **Train models with varying λ_attr**: 0, 0.01, 0.05, 0.1, 0.2, 0.5

2. **For each λ_attr**, evaluate:
   - Classification performance (AUC)
   - Attribution sparsity: entropy of normalised attributions
   - Ranking stability: Jaccard similarity of top 100 variants across CV folds
   - Biological enrichment: KEGG/Reactome pathway p-values

3. **Select optimal λ_attr** that balances classification performance with interpretability

#### Expected Outcomes

**If regularisation helps**:
- Models with moderate λ_attr (0.05-0.1) have similar AUC but higher ranking stability
- Top variants are more concentrated (lower entropy)
- Pathway enrichment p-values are lower (more meaningful discoveries)

### Experiment 4: Epistasis Detection and Validation

#### Purpose

Test whether the model captures genuine epistatic (non-additive) interactions between variants.

#### Protocol

1. **Identify candidate epistatic pairs** from attention weights:
   - Extract pairs with mean attention weight > threshold
   - Filter to pairs where both variants have non-zero attribution
   - Rank by combined attention × attribution score

2. **Validate epistasis** through counterfactual perturbation:
   - For each candidate pair (v_i, v_j):
     - Compute effect of removing v_i alone: Δ_i
     - Compute effect of removing v_j alone: Δ_j
     - Compute effect of removing both: Δ_{ij}
     - Epistasis score: |Δ_{ij} - (Δ_i + Δ_j)|

3. **Statistical testing**:
   - Null hypothesis: effects are additive (epistasis score = 0)
   - Permutation test: shuffle phenotype labels, recompute epistasis scores
   - Report pairs with p < 0.05 after Bonferroni correction

4. **Biological validation**:
   - Are epistatic pairs in same pathway?
   - Are they known to physically interact (protein-protein)?
   - Are they in linkage disequilibrium (LD)? (If so, may be LD artefact rather than true epistasis)

#### Distinguishing True Epistasis from Artefacts

Several artefacts can mimic epistasis:

**Linkage disequilibrium**: Variants in LD are inherited together, so their "interaction" may just reflect a single haplotype effect.
- **Control**: Check r² between variant pairs. Exclude pairs with r² > 0.2.

**Main effect masking**: Strong main effects can create apparent interactions.
- **Control**: Include main effects in baseline model; test interaction as additional term.

**Population stratification**: Different populations may have different allele frequencies and disease rates.
- **Control**: Include principal components as covariates; stratify analysis by ancestry.

### Baseline Comparisons

#### External Baselines

1. **Standard GWAS**: Run single-variant association using PLINK or equivalent
2. **Burden test**: Gene-level rare variant burden test (SKAT-O or equivalent)
3. **Existing DL methods** (if feasible): DeepRVAT, GenNet

#### Internal Baselines

1. **Logistic regression on gene burdens**: Simple, interpretable baseline
2. **Random forest on variant presence**: Non-linear baseline without deep learning

### Reporting Standards

#### For Each Experiment

Report:
- Sample sizes (n_cases, n_controls, n_variants)
- Cross-validation scheme and number of folds
- Hyperparameters and how they were selected
- Mean ± std of all metrics across outer CV folds
- Statistical tests and p-values with correction method
- Compute time and hardware used

#### Figures to Generate

1. **Annotation ablation**: Heatmap of Jaccard similarities between levels
2. **Position-aware comparison**: ROC curves for SIEVE vs DeepSet
3. **Attribution regularisation**: Pareto plot of AUC vs stability for different λ_attr
4. **Epistasis**: Network diagram of significant epistatic pairs
5. **Biological validation**: Pathway enrichment bar plot

---

## Appendix C: Method References

This appendix lists key methodological references that motivate recent pipeline updates.

### Sex inference and ploidy-aware encoding

- **X-chromosome inbreeding coefficient (F-statistic)** for genetic sex inference:  
  Purcell S, et al. (2007). *PLINK: a tool set for whole-genome association and population-based linkage analyses.* **American Journal of Human Genetics**, 81(3):559–575.
- **X-chromosome association and pseudoautosomal regions (PAR)**:  
  Clayton DG. (2008). *Testing for association on the X chromosome.* **Biostatistics**, 9(4):593–600.

### Attribution and interpretability

- **Integrated gradients** for feature attribution in deep networks:  
  Sundararajan M, Taly A, Yan Q. (2017). *Axiomatic Attribution for Deep Networks.* **ICML**.

### Attention mechanisms

- **Scaled dot-product attention** for modeling interactions:  
  Vaswani A, et al. (2017). *Attention Is All You Need.* **NeurIPS**.

---
