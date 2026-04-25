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
- **Built-in Interpretability**: Embedding sparsity regularisation incorporated into training
- **Statistical Validation**: Null baseline analysis establishes significance thresholds

### Scientific Questions SIEVE Addresses

1. **Can deep learning discover variants that annotations miss?** → Annotation ablation experiments (L0-L3, with L4 reserved as a compatibility placeholder)
2. **Do spatial relationships between variants matter?** → Position-aware sparse attention
3. **Can we make models interpretable by design?** → Embedding-sparsity-regularised training
4. **Are discoveries statistically significant?** → Null baseline analysis

### Key Capabilities

- **Train** models at multiple annotation levels (genotype-only to current functional-score annotations)
- **Explain** predictions with integrated gradients attribution
- **Discover** novel variant associations with statistical validation
- **Detect** epistatic interactions via attention patterns and post-hoc attribution/co-occurrence analysis
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

# 7. (Optional) Correct chrX ploidy bias for ranking/visualisation
#    Run AFTER step 6 on the significance-annotated file so that
#    empirical_p_variant and fdr_variant columns are preserved.
python scripts/correct_chrx_bias.py \
    --rankings results/null_baseline_run/results/attribution_comparison/variant_rankings_with_significance.csv \
    --output-dir results/null_baseline_run/results/attribution_comparison/corrected \
    --include-sex-chroms \
    --genome-build GRCh37
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
- Phenotype file (tab-delimited, no header: `sample_id<TAB>phenotype`, with 1=control and 2=case)

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
- Embedding sparsity regularisation (optional): Encourages model to concentrate signal in fewer variant or gene embeddings

**Annotation Levels**:
- **L0**: Genotype dosage only (0, 1, 2) - tests annotation-free discovery
- **L1**: L0 + genomic position
- **L2**: L1 + consequence class (missense/synonymous/LoF)
- **L3**: L2 + SIFT + PolyPhen ← **recommended starting point**
- **L4**: currently identical to L3; reserved for future annotation features

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
- Without null baseline: rankings have no empirical significance information
- With null baseline: both variant and gene rankings gain empirical p-values and BH-FDR
- The null comparison operates on raw `mean_attribution` — chrX inflation cancels because both models saw the same input data
- ChrX correction is applied separately to real rankings for visualisation purposes
- Manuscript-level per-gene claims should use `fdr_gene`, not raw ranking order

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

# 4. Compare raw real vs raw null attributions
python scripts/compare_attributions.py \
    --real results/explainability/sieve_variant_rankings.csv \
    --null results/null_attributions/sieve_variant_rankings.csv \
    --output-dir results/attribution_comparison \
    --genome-build GRCh37

# 5. (Separate) Apply chrX correction to the significance-annotated file
#    Run AFTER step 4 so empirical_p_variant and fdr_variant are preserved.
python scripts/correct_chrx_bias.py \
    --rankings results/attribution_comparison/variant_rankings_with_significance.csv \
    --output-dir results/attribution_comparison/corrected \
    --include-sex-chroms \
    --genome-build GRCh37
```

**Outputs**:
- `variant_rankings_with_significance.csv` - raw real rankings plus `empirical_p_variant` and `fdr_variant`
- `gene_rankings_with_significance.csv` - gene-level rankings plus `empirical_p_gene` and `fdr_gene`
- `significance_summary.yaml` - counts of variants and genes passing FDR thresholds 0.05, 0.01, 0.001

**Expected Results**:
- Null model AUC ≈ 0.50 (chance level - confirms permutation worked)
- Non-zero numbers of variants or genes may now pass `fdr_variant < 0.05` or `fdr_gene < 0.05`
- The minimum achievable empirical p-value is `1 / (N + 1)` where `N` is the null size

**Order of operations**:

The null comparison operates on raw `mean_attribution` because both models (real and null) saw the same input data with the same chrX inflation — the only difference is the labels. The raw attribution magnitude IS the signal. Applying per-chromosome z-scoring to both sides before comparison destroys the absolute signal difference because each chromosome is independently centred at zero — reducing the comparison to a within-chromosome shape test that is too weak for polygenic traits. ChrX correction is a ranking adjustment applied to the real model's output only, for cross-chromosome comparability in visualisation and ablation comparison, and should be run AFTER the null comparison.

---

#### Step 5: Ablation Comparison

**Purpose**: Compare variant rankings and model performance across annotation levels to assess whether deep learning can discover disease-associated variants without relying on functional annotations.

**Theory**: The annotation ablation is the core experiment of SIEVE. By training models at levels L0 (genotype only) through L3 (the current functional-score level), you can determine:
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

**Step 5b: Compare null-contrasted variant attribution rankings across levels**:
```bash
# Collect chrX-corrected significance files into one directory with level prefixes.
# Use corrected_variant_rankings.csv — it contains significance + chrX-corrected z-scores.
# PROJECT_DIR should match what was used in Step 5a.
PROJECT_DIR=/path/to/project
mkdir -p results/ablation/rankings
for LEVEL in L0 L1 L2 L3; do
    cp "${PROJECT_DIR}/real_experiments/${LEVEL}/attributions/corrected/corrected_variant_rankings.csv" \
       results/ablation/rankings/${LEVEL}_sieve_variant_rankings.csv
done

# Run comparison
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rankings \
    --score-column z_attribution \
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

#### Step 6: Epistasis Analysis (Optional)

**Purpose**: Characterise interactions from both the model's intrinsic attention patterns and its intrinsic attribution outputs.

##### Step 6A: Attention-based interaction discovery

This is the original SIEVE epistasis workflow:
1. `scripts/explain.py` extracts high-attention variant pairs from the trained model.
2. `scripts/validate_epistasis.py` tests whether those candidate pairs show non-additive effects by counterfactual perturbation.

This path is especially interesting because the candidate interactions come from the model's own attention mechanism rather than an external interaction scorer. Its main current limitation is that the search is restricted to pairs that appear within the same chunk. An empty `sieve_interactions.csv` therefore means that no pair crossed the discovery heuristic under the current chunking and threshold settings; it does not by itself prove that the cohort lacks interaction structure.

**Commands**:
```bash
python scripts/explain.py \
    --experiment-dir experiments/my_model \
    --preprocessed-data preprocessed.pt \
    --output-dir results/explainability \
    --attention-threshold-mode percentile \
    --attention-percentile 99.9 \
    --device cuda
```

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

##### Step 6B: Post-hoc attribution and co-occurrence interaction analysis

This complementary workflow uses attribution signals that are intrinsic to the trained SIEVE model together with observed variant co-occurrence. It is post-hoc in execution, but it is not based on an unrelated external explainer or a weight-only proxy.

Use it to answer three questions that the attention path alone cannot resolve:
1. Do candidate pairs or genes co-occur often enough to be testable?
2. Is the cohort powered to detect interaction effects of plausible magnitude?
3. Can multiple variant-level signals be pooled into gene-gene interaction hypotheses?

**Commands**:
```bash
python scripts/audit_cooccurrence.py \
    --preprocessed-data preprocessed.pt \
    --output-dir results/epistasis_audit
```

```bash
python scripts/aggregate_gene_interactions.py \
    --preprocessed-data preprocessed.pt \
    --variant-rankings results/attribution_comparison/corrected_variant_rankings.csv \
    --gene-rankings results/attribution_comparison/corrected_gene_rankings.csv \
    --null-rankings results/null_attributions/sieve_variant_rankings.csv \
    --cooccurrence results/epistasis_audit/cooccurrence_per_pair.csv \
    --output-dir results/gene_interactions
```

```bash
python scripts/epistasis_power_analysis.py \
    --cooccurrence results/epistasis_audit/cooccurrence_per_pair.csv \
    --cooccurrence-summary results/epistasis_audit/cooccurrence_by_maf_bin.csv \
    --real-attributions-npz results/explainability/attributions.npz \
    --null-attributions-npz results/null_attributions/attributions.npz \
    --output-dir results/epistasis_power
```

**Interpretation**:
- `cooccurrence_summary.yaml` tells you whether joint carriage exists across MAF bins, but not whether the model can see a pair in the same chunk.
- In `cooccurrence_summary.yaml`, `gte5` means "greater than or equal to 5".
- `n_pairs_gte5_cooccur` counts pairs with at least 5 joint carriers (`n11 >= 5`), which only tells you that both variants can appear together.
- `n_pairs_all_cells_gte5` is more important for interaction analysis: it counts pairs where all four cells of the `2x2` carrier table have at least 5 samples (`n11`, `n10`, `n01`, `n00`).
- This is relevant because estimating a non-additive interaction effect requires comparison across all four carrier states. If one cell is empty, the interaction contrast is not estimable in this framework; if one cell is very small, the estimate is unstable.
- The `>= 5` rule is a practical minimum-support heuristic, not a mathematical law.
- `power_analysis_summary.yaml` uses the full 2x2 carrier table for each pair, so near-ubiquitous common-common pairs no longer look artificially well-powered.
- `gene_pair_interactions.csv` ranks gene-gene hypotheses by combining attribution support and observed co-occurrence, which is often more stable than exact variant-pair recurrence in sparse cohorts.

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

#### How to Annotate Your VCF with Ensembl VEP

SIEVE requires VCF files annotated with [Ensembl VEP](https://www.ensembl.org/vep)
so that variant consequences, gene symbols, and functional scores are available
in the `CSQ` INFO field. **If your VCF is not VEP-annotated, preprocessing will
fail with a clear error message.**

##### Installing VEP (bioconda)

```bash
# Install VEP from bioconda
conda install -c bioconda ensembl-vep

# Download the VEP cache for your genome build (required for --offline mode)
# GRCh37:
vep_install -a cf -s homo_sapiens -y GRCh37 -c /path/to/vep_cache
# GRCh38:
vep_install -a cf -s homo_sapiens -y GRCh38 -c /path/to/vep_cache
```

The cache download may take a while (~15 GB for human). You only need to do this
once.

##### Running VEP

```bash
vep \
    --input_file variants.vcf.gz \
    --output_file variants_vep.vcf.gz \
    --vcf \
    --compress_output bgzip \
    --symbol \
    --canonical \
    --sift b \
    --polyphen b \
    --assembly GRCh37 \
    --offline \
    --cache \
    --dir_cache /path/to/vep_cache \
    --fork 4 \
    --no_stats
```

After annotation, create a tabix index:

```bash
tabix -p vcf variants_vep.vcf.gz
```

##### Required VEP Flags Explained

SIEVE relies on specific CSQ sub-fields at **hardcoded positions** in VEP's
default field order. **Do not use a custom `--fields` argument** — the default
VEP output order is expected.

| Flag | CSQ index | Why SIEVE needs it |
|------|-----------|-------------------|
| `--vcf` | — | Output must remain VCF format with CSQ in the INFO field |
| `--compress_output bgzip` | — | SIEVE expects `.vcf.gz` input; tabix index also required |
| `--symbol` | 3 | Gene symbol — used for gene-level aggregation |
| `--canonical` | 24 | Marks canonical transcript — used to select the representative annotation per variant |
| `--sift b` | 36 | SIFT prediction + score (e.g. `deleterious(0.01)`) — required for L3/L4 annotation levels |
| `--polyphen b` | 37 | PolyPhen prediction + score (e.g. `probably_damaging(0.999)`) — required for L3/L4 annotation levels |
| `--assembly` | — | Must match your reference build (GRCh37 or GRCh38) |
| `--offline --cache` | — | Use local cache; no internet required at runtime |
| `--fork N` | — | Optional; parallelise for speed |
| `--no_stats` | — | Optional; skip HTML stats report for faster runs |

The `b` option for `--sift` and `--polyphen` outputs both the prediction label
and the numeric score in `prediction(score)` format, which SIEVE's parser
extracts.

##### What Happens Without VEP Annotation

If you pass an unannotated VCF to `sieve-preprocess`, the parser will detect the
missing `CSQ` header and raise an error:

```
ValueError: VCF file 'variants.vcf.gz' does not contain VEP CSQ annotations.
SIEVE requires VCF files annotated with Ensembl VEP.
...
```

##### Verifying Your VEP Annotation

You can verify the CSQ field is present and correctly formatted:

```bash
# Check the header for CSQ definition
bcftools view -h variants_vep.vcf.gz | grep '##INFO=<ID=CSQ'

# Inspect a few CSQ values
bcftools query -f '%CHROM\t%POS\t%ALT\t%INFO/CSQ\n' variants_vep.vcf.gz | head -3
```

#### Phenotype File Format

Tab-delimited, **no header**, two columns:
```
SAMPLE001	1
SAMPLE002	2
SAMPLE003	1
SAMPLE004	2
```

- Column 1: `sample_id` — must match VCF sample names exactly
- Column 2: phenotype code — **1 = control, 2 = case** (standard PLINK convention)
- Lines beginning with `#` are treated as comments and ignored

**Note**: Sample order doesn't matter, but names must match VCF. Do not include a header row.

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
- **L4**: currently identical to L3; reserved for future annotation features

#### Decision Guide

**Start with L3** for most analyses because:
- Includes standard functional annotations
- Good balance of information and interpretability
- Comparable to existing methods

**Use L0** to test annotation-free discovery:
- If L0 performs well (AUC > 0.6), genotype patterns alone carry signal
- Variants unique to L0 may represent novel mechanisms

**Compare L0 vs L2 vs L3** for ablation studies:
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

### Embedding-Sparsity-Regularized Training

#### Theory

Standard training:
$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{BCE}}
$$

Embedding-sparsity-regularised training:
$$
\mathcal{L}_{\mathrm{total}}
= \mathcal{L}_{\mathrm{BCE}}
+ \lambda_{\mathrm{attr}}\mathcal{L}_{\mathrm{sparse}}
$$

The implemented sparsity term penalises L2 norms of variant embeddings in
non-chunked training, or gene embeddings in chunked training. It encourages the
model to:
- Concentrate signal in fewer variant or gene embeddings
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

# Compare raw real run against one raw null run
# (multi-permutation null support is not implemented in compare_attributions.py)
python scripts/compare_attributions.py \
    --real results/explainability/sieve_variant_rankings.csv \
    --null results/null_permutations/perm0/sieve_variant_rankings.csv \
    --output-dir results/comparison_robust \
    --genome-build GRCh37
```

**Benefits**:
- More stable null thresholds
- Better confidence in significance calls
- Recommended for publication-quality analyses

---

### Running Ablation Experiments

The annotation ablation is the central experiment in SIEVE. It trains models at multiple annotation levels and compares both their predictive performance and the variant discoveries they produce.

**The null baseline is a required step of the ablation workflow. Running `run_null_baseline_analysis.sh` for every level produces the null-contrasted rankings that all downstream analyses and manuscript claims depend on. Skipping this step leaves the rankings without significance information and invalidates any per-gene claim.**

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

# Run the required null baseline at each level (preferred: cohort-centric layout)
# Set PROJECT_DIR once; reuse it in the loop.
PROJECT_DIR=/path/to/project
for LEVEL in L0 L1 L2 L3; do
    PROJECT_DIR=$PROJECT_DIR \
    LEVEL=$LEVEL \
    bash scripts/run_null_baseline_analysis.sh
done

# Compare null-contrasted significance rankings (use chrX-corrected files)
mkdir -p results/ablation/significance_rankings
for LEVEL in L0 L1 L2 L3; do
    cp "${PROJECT_DIR}/real_experiments/${LEVEL}/attributions/corrected/corrected_variant_rankings.csv" \
       results/ablation/significance_rankings/${LEVEL}_sieve_variant_rankings.csv
done

python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/significance_rankings \
    --score-column z_attribution \
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
PROJECT_DIR=/path/to/project
python scripts/compare_ablation_rankings.py \
    --rankings L0:"${PROJECT_DIR}/real_experiments/L0/attributions/corrected/corrected_variant_rankings.csv" \
               L1:"${PROJECT_DIR}/real_experiments/L1/attributions/corrected/corrected_variant_rankings.csv" \
               L2:"${PROJECT_DIR}/real_experiments/L2/attributions/corrected/corrected_variant_rankings.csv" \
               L3:"${PROJECT_DIR}/real_experiments/L3/attributions/corrected/corrected_variant_rankings.csv" \
    --score-column z_attribution \
    --out-comparison results/ablation/ablation_ranking_comparison.yaml \
    --out-jaccard results/ablation/ablation_jaccard_matrix.tsv \
    --out-level-specific results/ablation/level_specific_variants.tsv
```

#### Adjusting Comparison Thresholds

The level-specific variant detection uses two thresholds:

- `--high-rank-threshold` (default: 100): a variant must be in the top-N at one level
- `--low-rank-threshold` (default: 500): the variant must be outside the top-N at all other levels

Tighter thresholds (e.g., `--high-rank-threshold 50 --low-rank-threshold 200`) produce a more selective list; looser thresholds capture more candidates.

#### Using Chromosome-Normalised Rankings

When chrX ploidy bias inflates attributions on sex chromosomes, run `correct_chrx_bias.py` on the significance-annotated file from each level.  Running it on `variant_rankings_with_significance.csv` (output of `compare_attributions.py`) preserves `empirical_p_variant` and `fdr_variant` alongside the new chrX-corrected z-scores:

```bash
# Set PROJECT_DIR once; reuse it throughout.
PROJECT_DIR=/path/to/project

# 1. Apply chrX correction to each level's significance-annotated file
for LEVEL in L0 L1 L2 L3; do
    python scripts/correct_chrx_bias.py \
        --rankings "${PROJECT_DIR}/real_experiments/${LEVEL}/attributions/variant_rankings_with_significance.csv" \
        --project-dir "$PROJECT_DIR" \
        --include-sex-chroms \
        --genome-build GRCh37
done
# Output: ${PROJECT_DIR}/real_experiments/{LEVEL}/attributions/corrected/corrected_variant_rankings.csv

# 2. Copy corrected files into a comparison directory
mkdir -p results/ablation/significance_rankings
for LEVEL in L0 L1 L2 L3; do
    cp "${PROJECT_DIR}/real_experiments/${LEVEL}/attributions/corrected/corrected_variant_rankings.csv" \
       results/ablation/significance_rankings/${LEVEL}_sieve_variant_rankings.csv
done

# 3. Compare using per-chromosome z-attribution ranking (recommended; see KNOWN_LIMITATIONS.md)
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/significance_rankings \
    --score-column z_attribution \
    --top-k 100,500,1000,2000 \
    --out-comparison results/ablation/significance_ablation_ranking_comparison.yaml \
    --out-jaccard results/ablation/significance_ablation_jaccard_matrix.tsv \
    --out-level-specific results/ablation/significance_level_specific_variants.tsv

# 4. Plot (reads from the significance-based TSV outputs)
python scripts/plot_ablation_comparison.py \
    --jaccard-tsv results/ablation/significance_ablation_jaccard_matrix.tsv \
    --level-specific-tsv results/ablation/significance_level_specific_variants.tsv \
    --summary-yaml results/ablation/ablation_summary.yaml \
    --heatmap-top-k 1000 \
    --output results/ablation/significance_ablation_comparison.png
```

Using `--include-sex-chroms` retains chrX/chrY variants in the output (flagged via `is_sex_chrom`) but normalises their scores relative to other variants on the same chromosome. This removes systematic inflation while preserving genuinely important sex-chromosome variants.

#### Bootstrap-Calibrated Ablation Comparison

After the chrX-corrected workflow, `bootstrap_null_calibration.py` adds a complementary `delta_rank` column to each level-specific rankings file. The ablation script can then be run in two complementary modes:

- `--score-column z_attribution`: the existing chrX-corrected continuity view, which preserves comparability with earlier numbers
- `--score-column delta_rank`: the new bootstrap-informed view, where positive `delta_rank` means the real model promotes a variant relative to the bootstrap-null ensemble

The gene-stats CSV also carries a `gene_delta_rank` column computed as `max(delta_rank)` per gene by default, mirroring the `gene_z_score = max(z_attribution)` convention; this is configurable via `--gene-delta-rank-aggregation mean` when running `bootstrap_null_calibration.py`.

Concordance between the two Jaccard matrices strengthens the level-specific-discovery claim. Divergence is also interpretable and worth investigating, because it shows where the chrX-corrected ordering and the bootstrap-null contrast disagree.

```bash
# Generate rank-calibrated files for each level first
python scripts/bootstrap_null_calibration.py \
    --real-rankings results/<cohort>/real_experiments/L1/attributions/variant_rankings_with_significance.csv \
    --null-attributions results/<cohort>/null_baselines/L1/attributions/attributions.npz \
    --output results/<cohort>/real_experiments/L1/attributions/variant_rankings_rank_calibrated.csv \
    --n-bootstrap 1000 \
    --seed 42

# View 1: chrX-corrected continuity
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rank_calibrated_rankings \
    --score-column z_attribution \
    --top-k 100,500,1000,2000 \
    --out-comparison results/ablation/z_ablation_comparison.yaml \
    --out-jaccard results/ablation/z_ablation_jaccard.tsv \
    --out-level-specific results/ablation/z_level_specific_variants.tsv

# View 2: bootstrap-informed
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rank_calibrated_rankings \
    --score-column delta_rank \
    --top-k 100,500,1000,2000 \
    --out-comparison results/ablation/delta_ablation_comparison.yaml \
    --out-jaccard results/ablation/delta_ablation_jaccard.tsv \
    --out-level-specific results/ablation/delta_level_specific_variants.tsv
```

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
| `--lambda-attr` | float | 0.0 | Embedding sparsity regularisation |
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
| `--attention-threshold-mode` | str | absolute | Interaction threshold mode [`absolute`, `percentile`] |
| `--attention-percentile` | float | 99.9 | Percentile cutoff when using percentile thresholding |
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
    --attention-threshold-mode percentile \
    --attention-percentile 99.9 \
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

This wrapper is configured through environment variables.

**Preferred interface (cohort-centric layout)**:

| Variable | Required | Description |
|---------|----------|-------------|
| `PROJECT_DIR` | Yes | Cohort project root directory (e.g. `/path/to/project`) |
| `LEVEL` | Yes | Annotation level to run (e.g. `L3`) |
| `NULL_DATA` | No | Pre-existing permuted `.pt` file — skips Step 1 if set |
| `DEVICE` | No | `cuda` or `cpu` (default: `cuda`) |
| `PYTHON` | No | Python interpreter path override |
| `EXCLUDE_SEX_CHROMS` | No | Set to `1` to exclude sex chromosomes from significance computation |

**Legacy variables (override derived paths when set)**:

| Variable | Description |
|---------|-------------|
| `INPUT_DATA` | Path to real preprocessed `.pt` file |
| `REAL_EXPERIMENT` | Real experiment training directory |
| `REAL_RESULTS` | Directory containing real `sieve_variant_rankings.csv` |
| `OUTPUT_BASE` | Base directory for null outputs |

Behaviour:
- Reads model/training hyperparameters from real `config.yaml` (in `REAL_EXPERIMENT` or parent).
- Carries over `sex_map` automatically when the real model used sex covariates.
- Resolves script paths relative to wrapper location, so it can be run from any working directory.

**Preferred example**:
```bash
PROJECT_DIR=/path/to/project \
LEVEL=L3 \
DEVICE=cuda \
bash scripts/run_null_baseline_analysis.sh
```

**Legacy example**:
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
| `--real` | path | required | Raw real variant rankings CSV (`sieve_variant_rankings.csv`) |
| `--null` | path | required | Raw null variant rankings CSV (`sieve_variant_rankings.csv`) |
| `--output-dir` | path | required* | Output directory. Mutually exclusive with `--project-dir` |
| `--project-dir` | path | required* | Cohort project root; routes output to `{project-dir}/real_experiments/{LEVEL}/attributions/`. Mutually exclusive with `--output-dir` |
| `--genome-build` | str | GRCh37 | Reference genome build |
| `--exclude-sex-chroms` | flag | False | Exclude chrX/chrY before empirical p-value and FDR computation |

\* Exactly one of `--output-dir` or `--project-dir` is required.

**Example**:
```bash
python scripts/compare_attributions.py \
    --real results/explainability/sieve_variant_rankings.csv \
    --null results/null_attributions/sieve_variant_rankings.csv \
    --output-dir results/attribution_comparison \
    --genome-build GRCh37
```

---

### bootstrap_null_calibration.py

```bash
python scripts/bootstrap_null_calibration.py [OPTIONS]
```

Generates a bootstrap ensemble of null rankings from `attributions.npz`, adds
rank-based empirical p-values and BH-FDR to a real rankings CSV, emits
gene-level Mann-Whitney statistics, and writes a YAML summary of top-k overlap
and KS diagnostics.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--real-rankings` | path | required | Real rankings CSV. Must contain `chromosome`, `position`, `gene_name`, and `mean_attribution` |
| `--null-attributions` | path | required | Null `attributions.npz` produced by `scripts/explain.py` |
| `--output` | path | required | Output CSV path for the rank-calibrated variant rankings |
| `--output-gene-stats` | path | `<output>_gene_stats.csv` | Optional output CSV path for per-gene Wilcoxon statistics |
| `--output-summary` | path | `<output>_summary.yaml` | Optional output YAML path for bootstrap overlap and KS summaries |
| `--n-bootstrap` | int | `1000` | Number of bootstrap replicates |
| `--seed` | int | `42` | Random seed for bootstrap sample draws |
| `--top-k` | str | `50,100,200,500,1000` | Comma-separated top-k thresholds for overlap and KS summaries |
| `--exclude-sex-chroms` | flag | `False` | Remove chrX/chrY from both the real and null inputs before ranking |
| `--min-variants-per-gene` | int | `10` | Minimum number of variants required to test a gene |
| `--n-jobs` | int | `-1` | Parallel workers for bootstrap replicates (`joblib`) |
| `--memmap-dir` | path | None | Optional fast-disk directory for the memmap-backed bootstrap rank matrix |
| `--genome-build` | str | nearby metadata or `GRCh37` | Genome build used for chromosome normalisation and sex-chromosome handling |
| `--verbose` | flag | `False` | Enable DEBUG logging |

**Example**:
```bash
python scripts/bootstrap_null_calibration.py \
    --real-rankings results/<cohort>/real_experiments/L1/attributions/variant_rankings_with_significance.csv \
    --null-attributions results/<cohort>/null_baselines/L1/attributions/attributions.npz \
    --output results/<cohort>/real_experiments/L1/attributions/variant_rankings_rank_calibrated.csv \
    --n-bootstrap 1000 \
    --seed 42
```

---

### correct_chrx_bias.py

```bash
python scripts/correct_chrx_bias.py [OPTIONS]
```

Run on `variant_rankings_with_significance.csv` (output of `compare_attributions.py`) to preserve `empirical_p_variant` and `fdr_variant` alongside the chrX-corrected z-scores. Do not run on null rankings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--rankings` | path | required | Variant rankings CSV (typically `variant_rankings_with_significance.csv`) |
| `--output-dir` | path | required* | Output directory. Mutually exclusive with `--project-dir` |
| `--project-dir` | path | required* | Cohort project root; routes output to `{project-dir}/real_experiments/{LEVEL}/attributions/corrected/`. Mutually exclusive with `--output-dir` |
| `--exclude-sex-chroms` | flag | True | Exclude chrX/chrY from final rankings (default) |
| `--include-sex-chroms` | flag | False | Include chrX/chrY (flagged) in rankings |
| `--genome-build` | str | GRCh37 | Reference genome build |
| `--top-k` | int | 100 | Top variants to annotate in plot |

\* Exactly one of `--output-dir` or `--project-dir` is required.

**Example**:
```bash
python scripts/correct_chrx_bias.py \
    --rankings results/attribution_comparison/variant_rankings_with_significance.csv \
    --output-dir results/attribution_comparison/corrected \
    --include-sex-chroms \
    --genome-build GRCh37
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
| `--score-column` | str | `z_attribution` | Column to rank variants by. Recommended choices are `z_attribution` for the chrX-corrected continuity view and `delta_rank` for the bootstrap-informed null-calibrated view. P/FDR-like columns and true rank columns are ranked ascending automatically; `delta_rank` is ranked descending. |

**Example**:
```bash
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rankings \
    --score-column z_attribution \
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

`validate_epistasis.py` validates candidate pairs from `sieve_interactions.csv`. Those candidates are limited to pairs visible within the same chunk during `explain.py`.

---

### audit_cooccurrence.py

```bash
python scripts/audit_cooccurrence.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--preprocessed-data` | path | required | Preprocessed data file |
| `--output-dir` | path | required | Output directory |
| `--maf-bins` | str | `0.001,0.01,0.05,0.1,0.5` | Carrier-frequency bin edges |
| `--max-pairs` | int | 100000 | Maximum number of evaluated pairs |
| `--top-k-variants` | int | 500 | Evaluate all pairs among the top-K carrier variants before adding low-frequency samples |
| `--seed` | int | 42 | Random seed |

Outputs include `cooccurrence_per_pair.csv`, which now carries the full `2x2` carrier contingency table for each evaluated pair.

---

### aggregate_gene_interactions.py

```bash
python scripts/aggregate_gene_interactions.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--preprocessed-data` | path | required | Preprocessed data file |
| `--variant-rankings` | path | required | Variant rankings CSV (raw or chrX-corrected) |
| `--gene-rankings` | path | required | Gene rankings CSV (raw or corrected) |
| `--null-rankings` | path | - | Null baseline variant rankings for significance-aware filtering |
| `--cooccurrence` | path | - | Per-pair co-occurrence CSV for variant-level enrichment |
| `--output-dir` | path | required | Output directory |
| `--min-cooccur-samples` | int | 5 | Minimum gene-pair co-occurrence |
| `--top-k-genes` | int | 50 | Top genes to include |
| `--min-gene-score` | float | 0.0 | Minimum gene score |
| `--significance-threshold` | str | `p_0.05` | Null-derived significance threshold to enforce when available |
| `--min-significant-variants` | int | 1 | Minimum number of significant variants required for a gene |
| `--allow-nonsignificant-genes` | flag | False | Allow genes with no null-significant variants |

This script is the preferred gene-level interaction workflow when you have null-comparison and chrX-corrected attribution outputs available.

---

### epistasis_power_analysis.py

```bash
python scripts/epistasis_power_analysis.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cooccurrence` | path | required | Per-pair co-occurrence CSV from `audit_cooccurrence.py` |
| `--cooccurrence-summary` | path | required | MAF-bin summary CSV from `audit_cooccurrence.py` |
| `--real-attributions-npz` | path | - | Real-model attributions archive |
| `--null-attributions-npz` | path | - | Null-model attributions archive |
| `--epistasis-results` | path | - | `epistasis_validation.csv` if available |
| `--output-dir` | path | required | Output directory |
| `--alpha` | float | 0.05 | Family-wise significance level |
| `--correction` | str | bonferroni | Multiple-testing correction [`bonferroni`, `fdr`] |

Power is computed from the full `2x2` carrier table for each pair, not just the joint-carrier count. This avoids overstating power for near-ubiquitous common-common pairs.

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

If you used sex-aware preprocessing or observe chrX inflation in rankings, run `correct_chrx_bias.py` on `variant_rankings_with_significance.csv` (the output of `compare_attributions.py`) to standardise mean attributions per chromosome while preserving the significance columns. The script adds:

- `z_attribution`: per-chromosome z-scored attribution
- `corrected_rank`: rank based on `z_attribution`
- `is_sex_chrom`: flags chrX/chrY variants

All existing columns — including `empirical_p_variant` and `fdr_variant` — are preserved unchanged. By default, the corrected rankings exclude sex chromosomes. Use `--include-sex-chroms` if you want to keep them in the output (they remain flagged).

For ablation comparison, use the chrX-corrected files `corrected_variant_rankings.csv` from the `corrected/` subdirectory and rank variants with `--score-column z_attribution` (the default). This column is recommended because it is comparable across chromosomes and is not subject to the empirical p-value resolution floor (see KNOWN_LIMITATIONS.md).

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

#### Significance Summary (`significance_summary.yaml`)

```yaml
genome_build: GRCh37
exclude_sex_chroms: false
n_real_variants_tested: 102341
n_null_variants: 101998
n_real_genes_tested: 18244
n_null_genes: 18190
min_achievable_empirical_p: 9.804e-06
variant_significance:
  fdr_0.05: 134
  fdr_0.01: 82
  fdr_0.001: 19
gene_significance:
  fdr_0.05: 27
  fdr_0.01: 14
  fdr_0.001: 3
```

**How to Interpret**:

1. **Read the variant-level file** `variant_rankings_with_significance.csv`:
   - `empirical_p_variant` is the empirical p-value against the null `mean_attribution` distribution
   - `fdr_variant` is the BH-adjusted value across all tested variants

2. **Read the gene-level file** `gene_rankings_with_significance.csv`:
   - `gene_score` is the maximum `mean_attribution` per gene
   - `empirical_p_gene` and `fdr_gene` are the gene-level significance metrics

3. **Use FDR for decisions**:
   - `fdr_gene < 0.05`: suitable for manuscript-level per-gene claims
   - `fdr_variant < 0.05`: variant-level follow-up candidates
   - `min_achievable_empirical_p = 1 / (N + 1)`: lower bound imposed by the null size

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
| `score_at_specific_level` | Attribution score at the specific level |

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

SIEVE now provides two complementary interaction views:

1. **Attention-based discovery**: high-attention variant pairs from `sieve_interactions.csv`, optionally followed by counterfactual validation in `epistasis_validation.csv`.
2. **Post-hoc attribution interaction analysis**: co-occurrence, power, and gene-gene aggregation from `audit_cooccurrence.py`, `epistasis_power_analysis.py`, and `aggregate_gene_interactions.py`.

#### Attention Discovery Output (`sieve_interactions.csv`)

This file contains variant pairs that exceeded the attention discovery threshold in `explain.py`. They are best treated as candidate interactions for follow-up, not as a complete interaction catalogue.

Key points:

- These pairs are discovered from the model's intrinsic attention mechanism, which is a distinctive feature of SIEVE.
- Discovery is currently restricted to pairs that occur within the same chunk.
- `--attention-threshold-mode percentile` is often more informative than a fixed absolute threshold when attention is diffuse across many variants.
- An empty `sieve_interactions.csv` means no pair crossed the current heuristic. It does not by itself prove an absence of interaction structure in the cohort.

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

#### Post-hoc Interaction Outputs

Use these when you need to understand whether the cohort is structurally able to support interaction detection even when the attention-based discovery file is sparse.

`cooccurrence_summary.yaml`
- Tells you how often evaluated pairs co-occur across MAF bins.
- Useful for diagnosing whether the rare-variant tail is too sparse.
- Does not solve the within-chunk visibility limit of the attention workflow.

`power_analysis_summary.yaml`
- Uses null-informed attribution noise plus the full `2x2` carrier table for each pair.
- The critical quantity is the effective interaction sample size, not just `n_cooccur`.
- Near-ubiquitous common-common pairs can have high co-occurrence but still low incremental interaction information.

`gene_pair_interactions.csv`
- Aggregates variant-level attribution support and co-occurrence at the gene-pair level.
- Useful when exact variant-pair recurrence is sparse but multiple variants implicate the same genes.
- Still grounded in the model's intrinsic attribution outputs rather than an external weight-only interaction score.

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

#### Rank-based null calibration

After the magnitude-based null comparison, `bootstrap_null_calibration.py` runs a complementary rank-based null calibration that generates an ensemble of `B = 1000` null rankings by bootstrap-resampling the null's per-sample attributions. This provides per-variant empirical p-values and BH-FDR with resolution `1 / (B + 1) ≈ 10^-3`, a per-gene Wilcoxon rank-sum test, and a global Kolmogorov-Smirnov statistic at top-k for `k ∈ {50, 100, 200, 500, 1000}`. The output also carries a `delta_rank` column (positive = variant promoted by real signal relative to the bootstrap-null ensemble), which is used as a complementary ranking view in the ablation analysis. Both the magnitude-based and rank-based tests are complementary; concordant variants across both tests form the highest-confidence shortlist.

```bash
python scripts/bootstrap_null_calibration.py \
    --real-rankings results/<cohort>/real_experiments/L1/attributions/variant_rankings_with_significance.csv \
    --null-attributions results/<cohort>/null_baselines/L1/attributions/attributions.npz \
    --output results/<cohort>/real_experiments/L1/attributions/variant_rankings_rank_calibrated.csv \
    --n-bootstrap 1000 \
    --seed 42
```

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
4. Re-run ablation comparison on the null-contrasted significance rankings:
   ```bash
   python scripts/compare_ablation_rankings.py \
       --ranking-dir results/ablation/significance_rankings \
       --score-column z_attribution \
       --out-comparison significance_ablation_ranking_comparison.yaml
   ```

#### Very low attributions overall

**Possible causes**:
- Model has low confidence (AUC close to 0.5)
- Embedding sparsity regularisation too strong (reduce `--lambda-attr`)
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
2. Embedding-sparsity-regularised training that builds interpretability into the objective
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
$$
x_v^{L0} = [d_v]
$$

**Level L1** (genotype + position):
$$
x_v^{L1} = [d_v, \mathrm{PE}(\mathrm{pos}_v)]
$$

**Level L2** (L1 + consequence):
$$
x_v^{L2} = [d_v, \mathrm{PE}(\mathrm{pos}_v), c_v]
$$

**Level L3** (L2 + SIFT/PolyPhen):
$$
x_v^{L3} =
[d_v, \mathrm{PE}(\mathrm{pos}_v), c_v,
\mathrm{sift}^{\mathrm{norm}}_v,
\mathrm{polyphen}^{\mathrm{norm}}_v]
$$

**Level L4** (compatibility placeholder):
$$
x_v^{L4} = x_v^{L3}
$$
L4 is currently identical to L3 in the implementation and is reserved for
future annotation features.

#### Positional Encoding

We use sinusoidal positional encodings adapted for genomic positions:

$$
\begin{aligned}
\mathrm{PE}(\mathrm{pos}, 2i)
&= \sin\!\left(\mathrm{pos}\cdot
   \exp\!\left(-\log(10000)\frac{2i}{d}\right)\right) \\
\mathrm{PE}(\mathrm{pos}, 2i+1)
&= \cos\!\left(\mathrm{pos}\cdot
   \exp\!\left(-\log(10000)\frac{2i}{d}\right)\right)
\end{aligned}
$$

where the positional embedding dimension is $d=64$.

This encoding allows the model to learn functions of relative position (important for detecting compound heterozygosity or clustered variants).
The implementation also builds chromosome indices. During attention, chromosome
embeddings can be added to variant embeddings, and cross-chromosome pairs use a
dedicated learned bias bucket instead of a meaningless coordinate difference.

#### Per-Sample Sparse Representation

For sample n, we construct a sparse representation S_n containing only their non-reference variants:

$$
S_n = \{(x_{nv}, \mathrm{pos}_v, \mathrm{chrom}_v, g(v)) : d_{nv} > 0\}
$$

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
2. Encoding within-chromosome relative distances in attention computation
3. Routing cross-chromosome pairs to a dedicated learned bias bucket

**Why this matters**: The relative position bias allows the model to learn that variants close together (potential compound heterozygosity) or at specific distances (potential haplotype patterns) are informative. This is impossible with permutation-invariant deep sets.

#### Component 3: Gene Aggregation

After attention layers process variant relationships, we aggregate to gene level using permutation-invariant pooling (max or mean).

#### Component 4: Phenotype Classifier

Simple classification head on gene representations with dropout for regularisation.

### Embedding-Sparsity-Regularised Training

#### Standard Loss

Binary cross-entropy for case-control classification:

$$
\mathcal{L}_{\mathrm{BCE}}(z,y)
= -y\log\sigma(z) - (1-y)\log(1-\sigma(z))
$$

#### Embedding Sparsity Loss

We want the model to concentrate predictive signal in a smaller number of loci.
During training this is implemented by penalising embedding magnitudes, not by
computing attribution gradients or Integrated Gradients.

For non-chunked training, the regulariser is the mean normalised sum of variant
embedding L2 norms. For chunked training, the same idea is applied to aggregated
gene embeddings:

$$
\mathcal{L}_{\mathrm{total}}
= \mathcal{L}_{\mathrm{BCE}}
+ \lambda_{\mathrm{attr}}\mathcal{L}_{\mathrm{sparse}}
$$

**Why this matters**: Without this regularisation, a model might achieve good classification by using many weak signals spread across the genome. Such a model is hard to interpret—which variants really matter? The sparsity loss encourages the model to "commit" to a small set of variants, making explainability more meaningful.

#### Hyperparameter: λ_attr

The embedding sparsity regularisation weight λ_attr controls the trade-off:
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

$$
\Delta_{ij} = p_{11} - p_{10} - p_{01} + p_{00}
$$

A non-zero $\Delta_{ij}$ indicates a non-additive model response under the
counterfactual perturbation.

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

**Hypothesis**: Models trained with minimal annotations will identify some disease-associated variants that models using current functional annotations rank lower, because annotation-informed models may over-rely on prior knowledge.

**Experiment**: Annotation ablation study comparing variant rankings across annotation levels L0-L3. L4 is currently identical to L3 and can be run only as a compatibility check.

#### Question 2: Do spatial relationships between variants carry disease signal?

**Hypothesis**: Position-aware models will outperform position-agnostic models (pure deep sets) on classification, and attention weights will show meaningful positional patterns (e.g., clustering of important variants).

**Experiment**: Compare SIEVE (with position-aware attention) against a DeepRVAT-style deep set baseline.

#### Question 3: Does embedding-sparsity-regularised training improve discovery?

**Hypothesis**: Models trained with embedding sparsity regularisation will produce more stable and biologically meaningful variant rankings than models trained with classification loss alone.

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

1. **Train replicate models at each implemented annotation level** (L0 through L3 as the primary comparison; L4 is currently identical to L3 and retained as a compatibility placeholder) using identical architecture and hyperparameters except for input dimension

2. **For each model**, compute integrated gradients to obtain variant-level attribution scores

3. **Compare variant rankings** across annotation levels:
   - Top 100 variants at each level
   - Overlap analysis (Jaccard similarity)
   - Identify "L0-specific" variants: high rank at L0, low rank at L3
   - Identify "L3-specific" variants: high rank at L3, low rank at L0

4. **Biological interpretation**:
   - Are L0-specific variants in genes not annotated as pathogenic?
   - Are they enriched for regulatory regions or novel mechanisms?
   - Do L3-specific variants simply have high SIFT or PolyPhen scores?

#### Expected Outcomes

**If hypothesis is supported**:
- L0 model achieves reasonable (>0.6) AUC, showing genotype patterns alone carry signal
- Some L0-specific variants are not captured by standard annotation methods
- These variants may point to novel disease mechanisms

**If hypothesis is refuted**:
- L0 model fails to learn (AUC ~0.5), suggesting annotations are necessary
- All high-ranking variants at L0 are a subset of L3 rankings
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

### Experiment 3: Embedding Sparsity Regularisation Study

#### Purpose

Determine whether training with embedding sparsity regularisation improves the stability and biological meaningfulness of discovered variants.

#### Protocol

1. **Train models with varying λ_attr**: 0, 0.01, 0.05, 0.1, 0.2, 0.5

2. **For each λ_attr**, evaluate:
   - Classification performance (AUC)
   - Embedding concentration during training and attribution concentration after explanation
   - Ranking stability: Jaccard similarity of top 100 variants across CV folds
   - Biological enrichment: KEGG/Reactome pathway p-values

3. **Select optimal λ_attr** that balances classification performance with interpretability

#### Expected Outcomes

**If regularisation helps**:
- Models with moderate λ_attr (0.05-0.1) have similar AUC but higher ranking stability
- Top variants are more concentrated and rankings are more stable
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
3. **Embedding sparsity regularisation**: Pareto plot of AUC vs stability for different λ_attr
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
