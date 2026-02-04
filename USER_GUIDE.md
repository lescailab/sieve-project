# SIEVE User Guide

**Version**: 1.0
**Last Updated**: 2026-02-04
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

# 2. Preprocess (once)
python scripts/preprocess.py \
    --vcf your_data.vcf.gz \
    --phenotypes phenotypes.tsv \
    --output preprocessed.pt

# 3. Train
python scripts/train.py \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --experiment-name my_model \
    --device cuda

# 4. Explain
python scripts/explain.py \
    --experiment-dir experiments/my_model \
    --preprocessed-data preprocessed.pt \
    --output-dir results/explainability

# 5. Validate (null baseline)
bash scripts/run_null_baseline_analysis.sh
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

All tests should pass with ✓ symbols.

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
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  3. Explainability  │  Compute variant attributions
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  4. Null Baseline   │  Establish statistical significance
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  5. Validation      │  Cross-reference with databases
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  6. Biological      │  Experimental validation
│     Follow-up       │
└─────────────────────┘
```

### Workflow Steps

#### Step 1: Data Preparation

**Purpose**: Convert VCF to SIEVE-compatible format

**Theory**: SIEVE requires multi-sample VCF files annotated with VEP (Variant Effect Predictor). VEP adds functional annotations (SIFT, PolyPhen, consequence types) that enable multi-level analysis.

**Requirements**:
- Multi-sample VCF file (bgzipped and indexed)
- VEP-annotated (CSQ field with SIFT, PolyPhen, Consequence, SYMBOL)
- GRCh37 reference (contigs without 'chr' prefix)
- Phenotype file (TSV: sample_id, phenotype)

**Command**:
```bash
python scripts/preprocess.py \
    --vcf cohort.vcf.gz \
    --phenotypes phenotypes.tsv \
    --output preprocessed.pt
```

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
export REAL_EXPERIMENT=experiments/my_model
export OUTPUT_BASE=experiments

# Run complete pipeline
bash scripts/run_null_baseline_analysis.sh
```

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

#### Step 5: Epistasis Detection (Optional)

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

#### Step 6: Biological Validation (Optional)

**Purpose**: Cross-reference discoveries with known databases

**Command**:
```bash
python scripts/validate_discoveries.py \
    --variant-rankings results/explainability/sieve_variant_rankings.csv \
    --gene-rankings results/explainability/sieve_gene_rankings.csv \
    --output-dir results/validation \
    --top-k 100
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
3. **GRCh37 reference** (contigs: 1, 2, 3... not chr1, chr2, chr3...)
4. **Bgzipped and indexed** (`.vcf.gz` + `.vcf.gz.tbi`)

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
# Create 5 null permutations
python scripts/create_null_baseline.py \
    --input preprocessed.pt \
    --output-dir null_permutations \
    --n-permutations 5

# Train each (can parallelise)
for i in {0..4}; do
    python scripts/train.py \
        --preprocessed-data null_permutations/preprocessed_NULL_perm${i}.pt \
        --level L3 \
        --experiment-name null_perm${i}

    python scripts/explain.py \
        --experiment-dir experiments/null_perm${i} \
        --preprocessed-data null_permutations/preprocessed_NULL_perm${i}.pt \
        --output-dir results/null_perm${i} \
        --is-null-baseline
done

# Compare using all permutations
python scripts/compare_attributions.py \
    --real results/explainability/sieve_variant_rankings.csv \
    --null-dir results \
    --output-dir results/comparison_robust
```

**Benefits**:
- More stable null thresholds
- Better confidence in significance calls
- Recommended for publication-quality analyses

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

**Example**:
```bash
python scripts/preprocess.py \
    --vcf cohort.vcf.gz \
    --phenotypes pheno.tsv \
    --output preprocessed.pt
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
| `--batch-size` | int | 16 | Batch size |
| `--chunk-size` | int | 3000 | Max variants per chunk |
| `--gradient-accumulation-steps` | int | 4 | Gradient accumulation |
| `--epochs` | int | 100 | Maximum epochs |
| `--lr` | float | 0.00001 | Learning rate |
| `--lambda-attr` | float | 0.1 | Attribution regularisation |
| `--early-stopping` | int | 15 | Early stopping patience |
| `--gradient-clip` | float | 1.0 | Gradient clipping value |

#### Model Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--latent-dim` | int | 32 | Embedding dimension |
| `--hidden-dim` | int | 64 | Hidden layer dimension |
| `--num-attention-layers` | int | 1 | Number of attention layers |
| `--aggregation-method` | str | mean | Gene aggregation [mean, max] |

#### Cross-Validation Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cv` | int | - | Number of CV folds (if not using val-split) |
| `--val-split` | float | 0.2 | Validation split ratio |

#### Output Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output-dir` | path | experiments | Output directory |
| `--experiment-name` | str | - | Experiment name |
| `--device` | str | cuda | Device [cuda, cpu] |
| `--seed` | int | 42 | Random seed |

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

**Example**:
```bash
python scripts/compare_attributions.py \
    --real results/explainability/sieve_variant_rankings.csv \
    --null results/null/sieve_variant_rankings.csv \
    --output-dir results/comparison \
    --top-k 100
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
| `--top-k` | int | 100 | Number of top variants/genes |
| `--clinvar-vcf` | path | - | ClinVar VCF (optional) |
| `--gwas-catalog` | path | - | GWAS catalog (optional) |

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
  - fold: 0
    auc: 0.78
    accuracy: 0.72
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
variant: chr17:41245466
gene: BRCA1
mean_attribution: 0.45
case_control_diff: 0.38
num_samples: 42

Interpretation: This BRCA1 variant has high attribution (0.45),
is strongly enriched in cases (diff=0.38), and appears in 42
samples. Likely a genuine disease-associated variant.
```

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

**Solution**: SIEVE expects contigs without 'chr' prefix (1, 2, 3... not chr1, chr2, chr3...)

If your VCF has 'chr' prefix:
```bash
# Option 1: Rename contigs
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
A: GRCh37 (hg19). Contigs should be named 1, 2, 3... (not chr1, chr2, chr3...).

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
  author = {Lescai Lab},
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

**Last Updated**: 2026-02-04
**Document Version**: 1.0
**SIEVE Version**: 0.1.0+

