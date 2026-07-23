<!--
This file is GENERATED. Do not edit it by hand.

Source of truth: the documentation/ directory (rendered at
https://lescailab.github.io/sieve-project/).

Edit the relevant page under documentation/, then run:

    python scripts/assemble_user_guide.py

Direct edits to this file are detected by the docs-drift CI job and will
fail the build.
-->

# SIEVE User Guide

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Complete Workflow](#complete-workflow)
- [Detailed Usage](#detailed-usage)
- [Command Reference](#command-reference)
- [Validation](#validation)
- [Interpreting Results](#interpreting-results)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Citation](#citation)
- [Support](#support)
- [License](#license)
- [Appendix A: Model Architecture Details](#appendix-a-model-architecture-details)
- [Appendix B: Experimental Protocol](#appendix-b-experimental-protocol)
- [Appendix C: Method References](#appendix-c-method-references)

**Version**: 1.2.0
**Last Updated**: 2026-04-29
**For**: SIEVE v1.2.0+


## Introduction

### What is SIEVE?

**SIEVE** (Sparse Interpretable Exome Variant Explainer) is a deep learning framework for discovering disease-associated genetic variants from exome sequencing data in case-control studies.

### What Makes SIEVE Different?

Unlike existing methods:
- **Direct VCF Processing**: No conversion to PLINK or custom formats required
- **Annotation-Ablation Protocol**: Quantifies how much of the ranking is carried by genome structure and how much by supplied annotation
- **Position-Aware**: Learns spatial relationships between variants (e.g., compound heterozygosity)
- **Built-in Interpretability**: Embedding sparsity regularisation incorporated into training
- **Statistical Validation**: Null baseline analysis establishes significance thresholds

### Scientific Questions SIEVE Addresses

1. **How much of a variant ranking is carried by annotation?** → Annotation ablation experiments (L0-L3, with L4 reserved as a compatibility placeholder)
2. **Do spatial relationships between variants matter?** → Position-aware dense self-attention over the sparse variant set
3. **Can we make models interpretable by design?** → Embedding-sparsity-regularised training
4. **Are discoveries statistically significant?** → Null baseline analysis

### Key Capabilities

- **Train** models at multiple annotation levels (genotype-only to current functional-score annotations)
- **Explain** predictions with integrated gradients attribution
- **Discover** novel variant associations with statistical validation
- **Test** candidate epistatic interactions using counterfactual perturbation and post-hoc attribution and co-occurrence analysis, with an explicit power analysis reported alongside the result
- **Validate** discoveries against ClinVar, GWAS, and GO databases

---

## Quick Start

#### For the Impatient

```bash
# 1. Install
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project
pip install -e .

# 2. Annotate your VCF with Ensembl VEP (if not already done)
#    Install VEP and download cache (once):
conda install -c bioconda ensembl-vep
vep_install -a cf -s homo_sapiens -y GRCh37 -c /path/to/vep_cache

#    Run annotation:
vep \
    --input_file your_data.vcf.gz \
    --output_file your_data_vep.vcf.gz \
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
    --fork 4
tabix -p vcf your_data_vep.vcf.gz

# 3. (Optional, recommended) Infer genetic sex for ploidy-aware encoding
python scripts/infer_sex.py \
    --vcf your_data_vep.vcf.gz \
    --output-dir results/sex_inference \
    --genome-build GRCh37

# 4. Preprocess (once)
python scripts/preprocess.py \
    --vcf your_data_vep.vcf.gz \
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
#    empirical_p_variant and fdr_variant columns are preserved in the output.
python scripts/correct_chrx_bias.py \
    --rankings results/null_baseline_run/results/attribution_comparison/variant_rankings_with_significance.csv \
    --output-dir results/null_baseline_run/results/attribution_comparison/corrected \
    --include-sex-chroms
```

#### 5-Minute Test Run

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

#### Prerequisites

- **Python**: 3.10 or higher
- **GPU**: CUDA-capable GPU recommended (8GB+ VRAM for real datasets)
- **Storage**: ~100MB for software, ~15GB for VEP cache, ~10GB for large preprocessed datasets
- **RAM**: 16GB minimum, 32GB+ recommended for large cohorts
- **Ensembl VEP**: Required to annotate your VCF before preprocessing (see below)

---

#### Route 1: conda package (recommended)

The easiest way to install SIEVE is via the pre-built conda package on the
[lescailab](https://anaconda.org/lescailab) Anaconda channel.

##### Step 1: Create a conda environment

```bash
conda create -n sieve python=3.10
conda activate sieve
```

##### Step 2: Install SIEVE

```bash
conda install -c lescailab -c pytorch -c nvidia -c bioconda -c conda-forge sieve
```

The channel order matters: `lescailab` must appear first so that the SIEVE package
takes precedence, followed by `pytorch`, `nvidia`, `bioconda`, and `conda-forge` for
all dependencies.

##### Step 3: Verify installation

```bash
sieve --help
```

All `sieve-*` commands exposed by the package should be available immediately.

---

#### Route 2: source install (for developers)

Use this route if you want to modify the source code or work with an unreleased version.

##### Step 1: Clone the repository

```bash
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project
```

##### Step 2: Create an environment and install

**Option A: Using conda**
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

##### Step 3: Verify installation

```bash
# Run test suite
python test_vcf_parser.py
python test_encoding_pipeline.py
python test_model_architecture.py
python test_training_pipeline.py
```

All tests should complete without errors. You can also run `pytest` for a more detailed test report.

---

#### Dependencies

Core packages installed by either route:
- **PyTorch** 2.0+ (deep learning)
- **NumPy**, **Pandas**, **SciPy** (data processing)
- **cyvcf2/pysam** (VCF parsing)
- **captum** (integrated gradients)
- **scikit-learn** (metrics, preprocessing)
- **matplotlib** (visualisation)
- **PyYAML** (configuration)

See `pyproject.toml` for the complete list.

#### Step 4: Install Ensembl VEP

SIEVE requires VCF files annotated with Ensembl VEP. If your VCF is not already
annotated, install VEP and download the cache:

```bash
# Install VEP from bioconda
conda install -c bioconda ensembl-vep

# Download the cache for your genome build (one-time, ~15 GB)
vep_install -a cf -s homo_sapiens -y GRCh37 -c /path/to/vep_cache
```

See [Detailed Usage — How to Annotate Your VCF](detailed-usage.md#how-to-annotate-your-vcf-with-ensembl-vep)
for the full VEP command and required flags.

#### Conda Package Workflow

If you installed SIEVE via conda (Route 1), use the `sieve-*` commands exposed by the
package. A complete command-based walkthrough is in:

- `conda/USAGE.md`

---

## Complete Workflow

#### Overview

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
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  8. Cross-Cohort    │  Burden enrichment + non-linear
│     Validation      │  classifier test in independent cohorts
└─────────────────────┘
```

#### Workflow Steps

##### Step 1: Data Preparation

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

##### Step 2: Model Training

**Purpose**: Learn which variants predict case/control status

**Theory**: SIEVE uses position-aware dense self-attention over the sparse variant set to learn relationships between variants. Training includes:
- Classification loss: Binary cross-entropy on case/control prediction
- Embedding sparsity regularisation (optional): Encourages model to concentrate signal in fewer variant or gene embeddings

**Annotation Levels**:
- **L0**: Genotype dosage only (0, 1, 2) - the ablation floor
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

##### Step 3: Explainability Analysis

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

##### Step 4: Null Baseline Analysis

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

**Manual Steps** (for reference — the wrapper above covers all of these automatically):
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
#    Run this AFTER compare_attributions.py so significance columns are preserved
python scripts/correct_chrx_bias.py \
    --rankings results/attribution_comparison/variant_rankings_with_significance.csv \
    --output-dir results/attribution_comparison/corrected \
    --include-sex-chroms \
    --genome-build GRCh37
```

##### Order of operations

**The null comparison must operate on raw `mean_attribution` values — not on chrX-corrected z-scores.** Both models (real and null) saw the same input data with the same chrX inflation; the only difference is the labels. The raw attribution magnitude IS the signal. Applying per-chromosome z-scoring to both sides before comparison destroys the absolute signal difference because each chromosome is independently centred at zero — reducing the comparison to a within-chromosome shape test that is too weak for polygenic traits with individually small effects.

ChrX correction (`correct_chrx_bias.py`) is a separate ranking adjustment applied to the **real** model's output only, for cross-chromosome comparability in visualisation and ablation comparison. It should be run AFTER the null comparison.

**Why this is safe for chrX**: The chrX inflation affects both real and null equally (it comes from the input data, not the labels). A chrX variant will only get a low empirical p-value if its real attribution is genuinely higher than what the null model produces — the inflation cancels out in the comparison.

**Outputs from `compare_attributions.py`**:
- `variant_rankings_with_significance.csv` — raw real rankings plus `empirical_p_variant` and `fdr_variant` columns
- `gene_rankings_with_significance.csv` — gene-level rankings plus `empirical_p_gene` and `fdr_gene` columns
- `significance_summary.yaml` — counts of variants and genes passing FDR thresholds 0.05, 0.01, 0.001

**Expected Results**:
- Null model AUC ≈ 0.50 (chance level — confirms permutation worked)
- Variants with `fdr_variant < 0.05`: number depends on signal strength; expect non-zero for well-powered cohorts
- Genes with `fdr_gene < 0.05`: candidates for manuscript-level biological claims

---

##### Step 5: Ablation Comparison

**Purpose**: Compare variant rankings and model performance across annotation levels to assess whether deep learning can discover disease-associated variants without relying on functional annotations.

**Theory**: The annotation ablation is the core experiment of SIEVE. By training models at levels L0 (genotype only) through L3 (the current functional-score level), you can determine:
- Whether positional or functional information is needed for discovery
- Which variants are found regardless of annotation level (robust discoveries)
- Which variants are only found at specific levels (annotation-dependent)

**The null baseline is a required step of the ablation workflow. Running `run_null_baseline_analysis.sh` for every level produces the null-contrasted rankings that all downstream analyses and manuscript claims depend on. Skipping this step leaves the rankings without significance information and invalidates any per-gene claim.**

**Prerequisites**: Train and run explain.py at each annotation level (Steps 2-3 repeated for L0, L1, L2, L3), then run the null baseline wrapper once per level.

**Step 5a: Compare model performance across levels**:
```bash
python scripts/ablation_compare.py \
    --results-dir experiments \
    --out-summary-tsv results/ablation/ablation_summary.tsv \
    --out-summary-yaml results/ablation/ablation_summary.yaml
```

**Step 5b: Run the required null baseline at each level**:
```bash
# Preferred: cohort-centric layout (PROJECT_DIR must contain data/ and real_experiments/)
for LEVEL in L0 L1 L2 L3; do
    PROJECT_DIR=/path/to/project \
    LEVEL=$LEVEL \
    bash scripts/run_null_baseline_analysis.sh
done
```

**Step 5c: Compare null-contrasted variant attribution rankings across levels**:
```bash
# Collect chrX-corrected significance files into one directory with level prefixes.
# Use corrected_variant_rankings.csv — it contains significance + chrX-corrected z-scores.
mkdir -p results/ablation/rankings
for LEVEL in L0 L1 L2 L3; do
    cp /path/to/project/real_experiments/${LEVEL}/attributions/corrected/corrected_variant_rankings.csv \
       results/ablation/rankings/${LEVEL}_sieve_variant_rankings.csv
done

# Run comparison
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rankings \
    --score-column empirical_p_variant \
    --top-k 50,100,200,500 \
    --high-rank-threshold 100 \
    --low-rank-threshold 500 \
    --out-comparison results/ablation/ablation_ranking_comparison.yaml \
    --out-jaccard results/ablation/ablation_jaccard_matrix.tsv \
    --out-level-specific results/ablation/level_specific_variants.tsv
```

**Step 5d: Visualise the comparison**:
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

##### Step 6: Epistasis Analysis (Optional)

**Purpose**: Characterise interactions from both the model's intrinsic attention patterns and its intrinsic attribution outputs.

###### Step 6A: Attention-based interaction discovery

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

###### Step 6B: Post-hoc attribution and co-occurrence interaction analysis

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

##### Step 7: Biological Validation (Optional)

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

##### Step 8: Cross-Cohort Gene-Set Burden Validation

**Purpose**: Test whether the *specific genes* SIEVE identified in the discovery cohort carry an excess of exonic variation in cases vs controls in independent validation cohorts.

**Why this approach?**
Running SIEVE on the validation cohorts would validate that the pipeline works, not that the discovery results are meaningful. A PRS-style weighted score would linearise SIEVE's non-linear signal, contradicting the model's premise. Instead, this pipeline uses a **set-level burden enrichment test**: it asks whether SIEVE-highlighted genes are enriched for case-control variation, agnostic to how variants contribute. Analogous to confirming a telescope's star cluster discovery by pointing a different instrument at the same coordinates.

**Prerequisites**:
- Completed Steps 1-5 on the discovery cohort (variant rankings with null comparison and chrX correction)
- One or more **independent validation VCFs**: VEP-annotated, multi-sample, same genome build (GRCh37/GRCh38), with phenotype files
- Validation cohorts should study a related phenotype (e.g. a different disease cohort with overlapping genetic architecture)

**Input files from the SIEVE discovery pipeline**:

| File | Source step | Description |
|------|-----------|-------------|
| `corrected/corrected_variant_rankings.csv` | Step 4 → `correct_chrx_bias.py` | ChrX-corrected, null-compared variant rankings |
| `sieve_gene_rankings.csv` | Step 3 → `explain.py` | Gene-level attribution rankings |
| `corrected/corrected_variant_rankings.csv` (per level) | Steps 2-5 (per level) | Per-ablation-level corrected rankings (optional) |

**This step has three sub-steps**: generate the gene list, extract burden counts from the validation VCF, and test for enrichment against a permutation null.

---

###### Step 8a: Generate SIEVE Gene List

**Purpose**: Aggregate variant-level rankings to a standardised gene-level TSV suitable for burden testing.

**Command**:
```bash
python scripts/generate_sieve_gene_list.py \
    --variant-rankings results/attribution_comparison/variant_rankings_rank_calibrated.csv \
    --output validation/sieve_gene_lists/sieve_genes.tsv \
    --score-column delta_rank \
    --exclude-sex-chroms \
    --aggregation max
```

This takes the rank-calibrated variant rankings and produces a ranked gene list where each gene's score is the maximum `delta_rank` across its variants. `delta_rank` is the primary ranking metric. Sex chromosome genes are excluded by default.

**To generate per-ablation-level gene lists** (for testing whether different annotation levels replicate differently):
```bash
for level in L0 L1 L2 L3; do
    python scripts/generate_sieve_gene_list.py \
        --variant-rankings results/${level}_attribution_comparison/variant_rankings_rank_calibrated.csv \
        --output validation/sieve_gene_lists/sieve_genes.tsv \
        --ablation-level ${level} \
        --score-column delta_rank \
        --aggregation max
done
```

This produces `L0_sieve_genes.tsv`, `L1_sieve_genes.tsv`, etc.

**Optional: filter to null-significant genes only**:
```bash
python scripts/generate_sieve_gene_list.py \
    --variant-rankings results/attribution_comparison/corrected/corrected_variant_rankings.csv \
    --output validation/sieve_gene_lists/sieve_genes_sig.tsv \
    --min-null-threshold p01 \
    --aggregation max
```

This retains only genes containing at least one variant exceeding the null model's 99th percentile.

**Optional: filter by FDR threshold** (gene set size determined dynamically):
```bash
python scripts/generate_sieve_gene_list.py \
    --variant-rankings results/attribution_comparison/variant_rankings_rank_calibrated.csv \
    --output validation/sieve_gene_lists/sieve_genes_fdr05.tsv \
    --score-column delta_rank \
    --fdr-threshold 0.05 \
    --aggregation max
```

This includes only genes whose `fdr_gene` is below 0.05. The gene significance is auto-discovered from `gene_rankings_with_significance.csv` in the same directory as the variant rankings, or from `corrected_gene_rankings.csv` if it contains `fdr_gene` (see `correct_chrx_bias.py`). Use `--gene-significance` to override the auto-discovery path.

**`z_attribution` visualisation gene list**:

The same script accepts `--score-column z_attribution` when given the
chrX-corrected variant rankings produced by `correct_chrx_bias.py`. This is
the per-chromosome visualisation view, retained for continuity with Manhattan
plots and earlier runs; per-chromosome z-scoring flattens genome-wide signal,
so do not use it as the primary ranking.

```bash
for level in L0 L1 L2 L3; do
    python scripts/generate_sieve_gene_list.py \
        --variant-rankings results/${level}_attribution_comparison/corrected/corrected_variant_rankings.csv \
        --output validation/sieve_gene_lists/sieve_genes.tsv \
        --ablation-level ${level} \
        --score-column z_attribution \
        --aggregation max
done
```

This produces a parallel set of `L0_sieve_genes.tsv` ... `L3_sieve_genes.tsv`
gene lists ranked by `z_attribution`. Feed these into
`extract_validation_burden.py` and `test_burden_enrichment.py` exactly as the
`delta_rank` lists are used. Run the burden enrichment twice, once on each
gene-list family, and report the `delta_rank` family as primary and the
`z_attribution` family as the visualisation view, applying BH-FDR
independently within each family across the `{level × top-k ×
consequence-class}` grid. Do not pool the two families into a single FDR
correction.

Note: `--fdr-threshold` and `--min-null-threshold` continue to rely on the
bootstrap empirical p-value, which is floored at `1/(B+1)`. For `delta_rank`
gene lists, prefer fixed top-k cutoffs over `--fdr-threshold`.

**Output format** (`sieve_genes.tsv`):
```
gene_name    gene_rank    gene_score    n_variants    chromosome
CUL3         1            3.45          5             2
AP2A1        2            3.21          3             19
NEXN         3            2.98          2             1
...
```

---

###### Step 8b: Extract Burden Counts from Validation VCF

**Purpose**: Parse the validation cohort VCF and count non-reference alleles per sample within the SIEVE gene sets. This is a single-pass VCF scan that produces per-sample burden counts and optionally a full gene-level burden matrix for fast permutation testing.

**Command** (recommended — with full matrix for permutation testing):
```bash
python scripts/extract_validation_burden.py \
    --vcf /path/to/validation_cohort.vcf.gz \
    --phenotypes /path/to/validation_phenotypes.tsv \
    --sieve-genes validation/sieve_gene_lists/sieve_genes.tsv \
    --output-dir validation/cohort_b \
    --genome-build GRCh37 \
    --min-gq 20 \
    --top-k 50 100 200 \
    --consequence-stratify \
    --compute-full-gene-matrix
```

**What this does**:
1. Loads phenotypes (1=control, 2=case PLINK convention, same as SIEVE)
2. Selects the top 50, 100, and 200 genes from the SIEVE gene list
3. Iterates through the validation VCF using `cyvcf2`, reusing the same CSQ parsing, canonical transcript selection, and contig harmonisation as the main SIEVE pipeline
4. For each variant in a target gene, sums genotype dosages (0/1/2) per sample — a homozygous alt counts as 2
5. With `--consequence-stratify`: separately counts missense, LoF, synonymous, and other variants
6. With `--compute-full-gene-matrix`: builds a complete (samples × all genes) burden matrix stored as Parquet, enabling fast permutation testing in Step 8c without re-parsing the VCF

**Key flags**:

| Flag | When to use |
|------|-------------|
| `--consequence-stratify` | Always recommended — enables testing whether enrichment is driven by functional variants |
| `--compute-full-gene-matrix` | Required for Step 8c — builds the matrix that makes 10,000 permutations feasible |
| `--include-sex-chroms` | Only if your gene list includes sex chromosome genes |
| `--from-variant-rankings` | If passing the raw `corrected_variant_rankings.csv` instead of a pre-generated gene list |

> **Tip — multi-level validation in a single VCF pass**: The full gene matrix records
> burden for *every* gene in the VCF, regardless of which `--sieve-genes` file you
> provide. When comparing ablation levels (L0–L3), you only need to parse the VCF
> **once** with `--compute-full-gene-matrix`, then run `test_burden_enrichment.py`
> separately per level with the corresponding gene list — each enrichment run reads
> the parquet matrix without touching the VCF:
>
> ```bash
> # Parse VCF once (use any gene list — the matrix is gene-list agnostic)
> python scripts/extract_validation_burden.py \
>     --vcf /path/to/validation_cohort.vcf.gz \
>     --phenotypes /path/to/validation_phenotypes.tsv \
>     --sieve-genes validation/sieve_gene_lists/L3_sieve_genes.tsv \
>     --output-dir validation/cohort_b \
>     --top-k 50 100 200 \
>     --consequence-stratify \
>     --compute-full-gene-matrix
>
> # Test enrichment per level (fast — reads parquet, no VCF)
> for level in L0 L1 L2 L3; do
>     python scripts/test_burden_enrichment.py \
>         --burden-dir validation/cohort_b \
>         --sieve-genes validation/sieve_gene_lists/${level}_sieve_genes.tsv \
>         --output-dir validation/cohort_b/enrichment_${level} \
>         --top-k 50 100 200 \
>         --n-permutations 10000
> done
> ```

**Outputs**:
```
validation/cohort_b/
├── burden_topK50.tsv              # Per-sample burden (columns: sample_id, phenotype, total_burden, ...)
├── burden_topK100.tsv
├── burden_topK200.tsv
├── burden_topK50_summary.yaml     # Diagnostics: genes found/missing, mean burden by group
├── burden_topK100_summary.yaml
├── burden_topK200_summary.yaml
├── gene_burden_matrix.parquet     # Full (samples × genes) burden matrix
├── gene_burden_matrix_metadata.yaml
├── gene_burden_matrix_missense.parquet    # Consequence-stratified matrices
├── gene_burden_matrix_lof.parquet
├── gene_burden_matrix_synonymous.parquet
└── gene_burden_matrix_other.parquet
```

**Check the summary YAML** before proceeding to Step 8c:
- `n_sieve_genes_found_in_vcf` should be close to the total — if many genes are missing, the validation VCF may use different gene symbol conventions or have limited exome coverage
- `missing_genes` lists the specific SIEVE genes not found, which helps diagnose gene name mismatches between VEP versions
- `mean_burden_cases` vs `mean_burden_controls` gives a quick preview of whether there is a difference (but this is not yet tested for significance)

**Repeat for each validation cohort**:
```bash
python scripts/extract_validation_burden.py \
    --vcf /path/to/cohort_c.vcf.gz \
    --phenotypes /path/to/cohort_c_phenotypes.tsv \
    --sieve-genes validation/sieve_gene_lists/sieve_genes.tsv \
    --output-dir validation/cohort_c \
    --top-k 50 100 200 \
    --consequence-stratify \
    --compute-full-gene-matrix
```

---

###### Step 8c: Test Burden Enrichment

**Purpose**: Test whether the SIEVE gene set shows significantly stronger case-control burden difference than random gene sets of the same size, using a permutation null distribution.

**How it works**:
1. Loads the pre-computed gene burden matrix from Step 8b
2. Computes a logistic regression z-statistic (phenotype ~ burden) for the SIEVE gene set — this is the **observed test statistic**
3. Draws 10,000 random gene sets of size *k* from all genes in the validation exome
4. Computes the same z-statistic for each random set — this is the **null distribution**
5. Reports an **empirical p-value**: the fraction of random sets with a z-statistic at least as extreme as the observed one

Because the full gene matrix was pre-computed in Step 8b, each permutation is a fast column-slice + sum operation — the VCF is never re-parsed.

**Command**:
```bash
python scripts/test_burden_enrichment.py \
    --burden-dir validation/cohort_b \
    --sieve-genes validation/sieve_gene_lists/sieve_genes.tsv \
    --output-dir validation/cohort_b/enrichment \
    --n-permutations 10000 \
    --top-k 50 100 200 \
    --seed 42
```

**To test consequence-specific enrichment** (requires `--consequence-stratify` in Step 8b):
```bash
python scripts/test_burden_enrichment.py \
    --burden-dir validation/cohort_b \
    --sieve-genes validation/sieve_gene_lists/sieve_genes.tsv \
    --output-dir validation/cohort_b/enrichment \
    --n-permutations 10000 \
    --top-k 50 100 200 \
    --consequence-types total missense lof \
    --seed 42
```

**To include covariates** (e.g. sex, principal components):
```bash
python scripts/test_burden_enrichment.py \
    --burden-dir validation/cohort_b \
    --sieve-genes validation/sieve_gene_lists/sieve_genes.tsv \
    --output-dir validation/cohort_b/enrichment \
    --covariates /path/to/covariates.tsv \
    --n-permutations 10000 \
    --top-k 50 100 200 \
    --seed 42
```

The covariates TSV should have `sample_id` as the first column (or index) and one column per covariate.

**Outputs**:
```
validation/cohort_b/enrichment/
├── enrichment_topK50.yaml              # Full results: observed stats + permutation p-value
├── enrichment_topK100.yaml
├── enrichment_topK200.yaml
├── null_distribution_topK50.npz        # Saved null z-statistics (for custom plotting)
├── null_distribution_topK100.npz
├── null_distribution_topK200.npz
├── enrichment_plot_topK50.png          # Histogram: null distribution + observed value
├── enrichment_plot_topK100.png
├── enrichment_plot_topK200.png
├── cross_cohort_validation_summary.yaml  # Summary with Bonferroni correction
└── validation_report.md                  # Human-readable report
```

**Interpreting the results**:

The key metric in each `enrichment_topK{k}.yaml` is the **empirical p-value** under `permutation.empirical_p`. This tells you the probability that a random gene set of the same size would produce an equally strong or stronger case-control burden difference.

| Empirical p | Interpretation |
|-------------|---------------|
| < 0.01 | Strong evidence: SIEVE genes are enriched for case-control variation |
| 0.01 - 0.05 | Moderate evidence (check after Bonferroni correction) |
| > 0.05 | No significant enrichment at this top-k threshold |

The `cross_cohort_validation_summary.yaml` applies Bonferroni correction across all tests (multiple top-k thresholds and consequence types). A result that survives Bonferroni correction is robust.

**Additional diagnostics**:
- If enrichment is significant for **missense/LoF** but not **synonymous**, this suggests SIEVE genes harbour functional exonic variation — not just more variants by chance of gene length
- If enrichment is significant at **top-50** but not **top-200**, the signal is concentrated in the highest-ranked genes
- The `enrichment_plot_topK{k}.png` shows the null distribution with the observed value marked — the further right the red line, the stronger the evidence

**Per-ablation-level testing** (tests whether different annotation levels replicate differently):
```bash
for level in L0 L1 L2 L3; do
    python scripts/test_burden_enrichment.py \
        --burden-dir validation/cohort_b \
        --sieve-genes validation/sieve_gene_lists/${level}_sieve_genes.tsv \
        --output-dir validation/cohort_b/enrichment_${level} \
        --n-permutations 10000 \
        --top-k 50 100 200 \
        --seed 42
done
```

If L1-specific genes replicate in the cohort_b cohort but L0-specific ones do not (or vice versa), that directly strengthens the annotation-ablation narrative.

---

###### Step 8d: Non-Linear Classifier Validation

**Purpose**: Test whether the SIEVE gene set carries *non-linear* discriminative signal — combinatorial patterns across genes that a scalar burden sum would destroy.

**Why this step?** The scalar burden test (Step 8c) asks whether SIEVE genes have more total exonic variation in cases. But SIEVE's core claim is that the **pattern** of variation across genes matters, not just the total count. A random forest trained on per-gene burden counts preserves this multi-gene structure.

> **Pipeline branching note**: The gene list produced by Step 8a (`generate_sieve_gene_list.py`) feeds into the burden extraction path (Steps 8b–8c) only. This step reads the gene ranking CSV files directly from each level's results directory and performs its own gene selection internally — the TSV gene lists from Step 8a are not an input here.

**Setting up the rankings directory**: `--real-rankings-dir` expects one subdirectory per annotation level, each containing a gene rankings CSV. If your results follow the standard layout (`real_experiments/${LEVEL}/attributions/`), use symlinks:

```bash
mkdir -p ablation/significance_rankings
for LEVEL in L0 L1 L2 L3; do
    ln -sf "$(pwd)/real_experiments/${LEVEL}/attributions" \
           "ablation/significance_rankings/${LEVEL}"
done
```

The script auto-detects `gene_rankings_with_significance.csv` in each level subdirectory.

**Command — fixed top-k** (all levels at once):
```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir ablation/significance_rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --phenotypes /path/to/validation_phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_summary.tsv \
    --top-k 50,100,200,500 \
    --n-permutations 1000 \
    --classifiers rf,lr \
    --n-cores 8 \
    --seed 42
```

**Command — FDR-threshold** (gene set size determined per level):
```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir ablation/significance_rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --phenotypes /path/to/validation_phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_fdr.tsv \
    --fdr-threshold 0.05 \
    --n-permutations 1000 \
    --classifiers rf,lr \
    --n-cores 8 \
    --seed 42
```

`--top-k` and `--fdr-threshold` are mutually exclusive. Use `--top-k` for exploratory analysis across multiple gene-set sizes, and `--fdr-threshold` for statistically motivated gene sets where the number of genes is determined by the null-contrast significance.

Add `--also-export-csv` to write the actual feature matrix used by each classifier (samples × matched genes, plus a `phenotype` column) as a CSV file under `csv/` in the output directory. Useful for debugging and external analysis.

**How it works**:
1. For each ablation level and top-k threshold (or FDR-passing gene set), extracts the per-gene burden sub-matrix for the corrected SIEVE gene set
2. Trains the requested classifier using fixed stratified CV folds
3. Groups levels by effective matched gene count and generates one shared null distribution per `(top_k, classifier, k_effective)` group
4. Reports an empirical p-value, a null-relative z-score, and a single BH-FDR column across the full result grid

**Outputs**:
```
validation/cohort_b/nonlinear_validation/
├── nonlinear_validation_L{0..3}_topK{k}.yaml   # Full results per combination
├── null_aucs_L{0..3}_topK{k}.npz               # Null distributions
├── validation_plot_L{0..3}_topK{k}.png          # Diagnostic plots (null histogram + per-fold AUC)
├── nonlinear_validation_summary.tsv             # Summary table with fdr_bh
├── nonlinear_validation_heatmap.png             # AUC heatmap across levels x top-k
├── nonlinear_validation_report.md               # Human-readable report of significant results
└── csv/                                         # Only present when --also-export-csv is set
    └── feature_matrix_L{0..3}_topK{k}.csv      # Feature matrix per (level, top_k); one file per combination, phenotype column uses 0=control 1=case
```

**Interpreting the results**: see the [Validation](validation.md) chapter for detailed guidance.

!!! tip "Start with quick exploration"
    Use `--n-permutations 200` for a fast initial run. Once you identify the most promising level/top-k combinations, re-run with `--n-permutations 1000` for publication-quality results.

---

###### Step 8e: Summarise Classifier Comparison

**Purpose**: Produce per-combination comparison figures and a collated PDF comparing RF and LR results from Step 8d.

**Command**:
```bash
python scripts/summarize_classifier_comparison.py \
    --results-dir validation/cohort_b/nonlinear_validation/ \
    --output-dir validation/cohort_b/nonlinear_validation/summary_plots/
```

This script scans the YAML outputs from Step 8d, pairs RF and LR results per `(level, top_k)` combination, and produces overlapping null-distribution density curves with observed AUC markers alongside per-fold AUC comparisons. All figures are also collected into a single A4-landscape PDF.

> **Note**: This step is only meaningful when `--classifiers rf,lr` was used in Step 8d.

---

###### Step 8f: Visualise Scalar Burden Results

**Purpose**: Collect scalar burden enrichment results across annotation levels into summary plots.

**Command**:
```bash
python scripts/plot_validation_burden.py \
    --input-dirs validation/cohort_b/enrichment_L0 \
                 validation/cohort_b/enrichment_L1 \
                 validation/cohort_b/enrichment_L2 \
                 validation/cohort_b/enrichment_L3 \
    --top-k 50 100 200 500 \
    --consequence-types total missense lof \
    --output-dir validation/cohort_b/burden_plots
```

**Outputs**:
- Summary TSV with all results across levels, consequence types, and top-k values
- Multi-panel line plot of -log10(empirical p) vs top-k
- Heatmap of logistic regression z-statistics

---

###### Complete Step 8 Example

Putting it all together for two validation cohorts:

```bash
# --- Gene list from discovery cohort ---
python scripts/generate_sieve_gene_list.py \
    --variant-rankings results/attribution_comparison/variant_rankings_rank_calibrated.csv \
    --output validation/sieve_gene_lists/sieve_genes.tsv \
    --score-column delta_rank \
    --aggregation max

# --- Cohort B ---
python scripts/extract_validation_burden.py \
    --vcf /path/to/validation_cohort_b.vcf.gz \
    --phenotypes /path/to/validation_cohort_b_phenotypes.tsv \
    --sieve-genes validation/sieve_gene_lists/sieve_genes.tsv \
    --output-dir validation/cohort_b \
    --top-k 50 100 200 \
    --consequence-stratify \
    --compute-full-gene-matrix

python scripts/test_burden_enrichment.py \
    --burden-dir validation/cohort_b \
    --sieve-genes validation/sieve_gene_lists/sieve_genes.tsv \
    --output-dir validation/cohort_b/enrichment \
    --n-permutations 10000 \
    --top-k 50 100 200 \
    --consequence-types total missense lof \
    --seed 42

# --- Cohort C ---
python scripts/extract_validation_burden.py \
    --vcf /path/to/validation_cohort_c.vcf.gz \
    --phenotypes /path/to/validation_cohort_c_phenotypes.tsv \
    --sieve-genes validation/sieve_gene_lists/sieve_genes.tsv \
    --output-dir validation/cohort_c \
    --top-k 50 100 200 \
    --consequence-stratify \
    --compute-full-gene-matrix

python scripts/test_burden_enrichment.py \
    --burden-dir validation/cohort_c \
    --sieve-genes validation/sieve_gene_lists/sieve_genes.tsv \
    --output-dir validation/cohort_c/enrichment \
    --n-permutations 10000 \
    --top-k 50 100 200 \
    --consequence-types total missense lof \
    --seed 42

# --- Non-linear classifier validation (Cohort B) ---
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir ablation/significance_rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --phenotypes /path/to/cohort_b_phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_summary.tsv \
    --top-k 50,100,200 \
    --n-permutations 1000 \
    --classifiers rf,lr \
    --n-cores 8

python scripts/summarize_classifier_comparison.py \
    --results-dir validation/cohort_b/nonlinear_validation/ \
    --output-dir validation/cohort_b/nonlinear_validation/summary_plots/

# --- Non-linear classifier validation (Cohort C) ---
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir ablation/significance_rankings \
    --burden-matrix validation/cohort_c/gene_burden_matrix.parquet \
    --phenotypes /path/to/cohort_c_phenotypes.tsv \
    --output-tsv validation/cohort_c/nonlinear_validation/nonlinear_validation_summary.tsv \
    --top-k 50,100,200 \
    --n-permutations 1000 \
    --classifiers rf,lr \
    --n-cores 8

python scripts/summarize_classifier_comparison.py \
    --results-dir validation/cohort_c/nonlinear_validation/ \
    --output-dir validation/cohort_c/nonlinear_validation/summary_plots/

# --- Collect and plot scalar burden results ---
python scripts/plot_validation_burden.py \
    --input-dirs validation/cohort_b/enrichment_L0 \
                 validation/cohort_b/enrichment_L1 \
                 validation/cohort_b/enrichment_L2 \
                 validation/cohort_b/enrichment_L3 \
    --top-k 50 100 200 \
    --consequence-types total missense lof \
    --output-dir validation/cohort_b/burden_plots
```

**Expected output tree**:
```
validation/
├── sieve_gene_lists/
│   └── sieve_genes.tsv
├── cohort_b/
│   ├── gene_burden_matrix.parquet
│   ├── gene_burden_matrix_metadata.yaml
│   ├── gene_burden_matrix_missense.parquet
│   ├── gene_burden_matrix_lof.parquet
│   ├── gene_burden_matrix_synonymous.parquet
│   ├── gene_burden_matrix_other.parquet
│   ├── burden_topK{50,100,200}.tsv
│   ├── burden_topK{50,100,200}_summary.yaml
│   └── enrichment/
│       ├── enrichment_topK{50,100,200}.yaml
│       ├── enrichment_topK{50,100,200}_missense.yaml
│       ├── enrichment_topK{50,100,200}_lof.yaml
│       ├── null_distribution_topK{50,100,200}.npz
│       ├── enrichment_plot_topK{50,100,200}.png
│       ├── cross_cohort_validation_summary.yaml
│       └── validation_report.md
│   ├── nonlinear_validation/
│   │   ├── nonlinear_validation_L{0..3}_topK{k}.yaml
│   │   ├── null_aucs_L{0..3}_topK{k}.npz
│   │   ├── validation_plot_L{0..3}_topK{k}.png
│   │   ├── nonlinear_validation_summary.tsv
│   │   ├── nonlinear_validation_heatmap.png
│   │   └── nonlinear_validation_report.md
│   └── burden_plots/
│       ├── validation_burden_summary.tsv
│       ├── validation_burden_pvalue_lines.png
│       └── validation_burden_zscore_heatmap.png
└── cohort_c/
    └── ... (same structure)
```

---

## Detailed Usage

#### Preparing Your VCF File

##### VCF Requirements

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

##### How to Annotate Your VCF with Ensembl VEP

SIEVE requires VCF files annotated with [Ensembl VEP](https://www.ensembl.org/vep)
so that variant consequences, gene symbols, and functional scores are available
in the `CSQ` INFO field. **If your VCF is not VEP-annotated, preprocessing will
fail with a clear error message.**

###### Installing VEP (bioconda)

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

###### Running VEP

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

###### Required VEP Flags Explained

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

###### What Happens Without VEP Annotation

If you pass an unannotated VCF to `sieve-preprocess`, the parser will detect the
missing `CSQ` header and raise an error:

```
ValueError: VCF file 'variants.vcf.gz' does not contain VEP CSQ annotations.
SIEVE requires VCF files annotated with Ensembl VEP.
...
```

###### Verifying Your VEP Annotation

You can verify the CSQ field is present and correctly formatted:

```bash
# Check the header for CSQ definition
bcftools view -h variants_vep.vcf.gz | grep '##INFO=<ID=CSQ'

# Inspect a few CSQ values
bcftools query -f '%CHROM\t%POS\t%ALT\t%INFO/CSQ\n' variants_vep.vcf.gz | head -3
```

##### Phenotype File Format

Tab-delimited, **no header**, two columns:
```
SAMPLE001	1
SAMPLE002	2
SAMPLE003	1
SAMPLE004	2
```

- Column 1: `sample_id` (must match VCF exactly)
- Column 2: `phenotype` (**1 = control, 2 = case**)

**Note**: Sample order doesn't matter, but names must match VCF. Do not include a header row.

---

#### Sex-Aware Preprocessing (Recommended for chrX/chrY Analyses)

SIEVE now supports a sex-aware preprocessing path to prevent chrX ploidy bias in downstream attributions. The pipeline uses the X-chromosome inbreeding coefficient (F-statistic) with pseudoautosomal region (PAR) exclusion to infer genetic sex, then applies ploidy-aware dosage encoding during VCF parsing.

##### 1) Infer genetic sex (X-chromosome F-statistic)

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

##### 2) Check sex balance across cases/controls (recommended)

```bash
python scripts/check_sex_balance.py \
    --phenotypes phenotypes.tsv \
    --sex-map results/sex_inference/sample_sex.tsv \
    --output-dir results/sex_balance
```

If a significant imbalance is detected, consider sex-stratified analysis or adding sex as a covariate in downstream modeling.

##### 3) Preprocess with ploidy-aware encoding

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

##### 4) Train with sex covariate (recommended if imbalance exists)

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

#### Choosing Annotation Levels

##### Scientific Rationale

The annotation ablation protocol tests whether deep learning can discover variants independently of prior knowledge:

- **L0 (Genotype only)**: Can patterns in 0/1/2 dosages alone predict disease?
- **L1 (+ Position)**: Does knowing where variants are located help?
- **L2 (+ Consequence)**: Does basic VEP info (missense/LoF) matter?
- **L3 (+ SIFT/PolyPhen)**: Do deleteriousness scores improve discovery?
- **L4**: currently identical to L3; reserved for future annotation features

##### Decision Guide

**Start with L3** for most analyses because:
- Includes standard functional annotations
- Good balance of information and interpretability
- Comparable to existing methods

**Use L0** as the ablation floor of the protocol:
- If L0 performs well (AUC > 0.6), genotype patterns alone carry signal
- Variants unique to L0 may represent novel mechanisms

**Compare L0 vs L2 vs L3** for ablation studies:
- Identifies which annotations are actually helpful
- Reveals annotation-dependent vs independent discoveries

---

#### Training Strategies

##### Single Train/Val Split

Fast, good for initial exploration:
```bash
python scripts/train.py \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --val-split 0.2 \
    --epochs 100 \
    --experiment-name quick_test
```

##### Cross-Validation

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

##### Memory-Efficient Training (Large Datasets)

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

Peak GPU memory is set by `--chunk-size` and `--batch-size`, not by how many
samples the cohort contains. Holding those two flags fixed, peak memory held at
a plateau of roughly 18 GB across three cohorts spanning 1,968 to 3,420 samples,
because chunking bounds the resident working set by `chunk_size` rather than by
the number of variants a sample carries.

So size the run with those two flags: lower either to fit a smaller card, raise
either to use a larger one. Measure once on your own hardware at the settings
you intend to use, since the 18 GB figure is specific to the configuration it
was measured at.

---

#### Embedding-Sparsity-Regularized Training

##### Theory

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

##### Usage

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

##### When to Use

- **λ = 0**: Standard training, maximum flexibility
- **λ = 0.01-0.1**: Mild sparsity, improves interpretability
- **λ = 0.5+**: Strong sparsity, may hurt performance

**Recommendation**: Start with λ=0, then try λ=0.1 if attributions are noisy.

---

#### Multiple Null Permutations

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

#### Running Ablation Experiments

The annotation ablation is the central experiment in SIEVE. It trains models at multiple annotation levels and compares both their predictive performance and the variant discoveries they produce.

##### Full Ablation Pipeline

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

# Compare attribution rankings, ranked by the primary metric delta_rank.
# delta_rank lives in the rank-calibrated CSVs from bootstrap_null_calibration.py,
# not in the chrX-corrected files, so collect those.
mkdir -p results/ablation/rank_calibrated_rankings
for LEVEL in L0 L1 L2 L3; do
    cp results/null_baseline_${LEVEL}/results/attribution_comparison/variant_rankings_rank_calibrated.csv \
       results/ablation/rank_calibrated_rankings/${LEVEL}_sieve_variant_rankings.csv
done

python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rank_calibrated_rankings \
    --score-column delta_rank \
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

##### Using Explicit Ranking Paths

If your ranking files are not in a single directory with level prefixes, you can specify them individually:

```bash
# Rank by delta_rank, the primary ranking metric, using the rank-calibrated CSVs
python scripts/compare_ablation_rankings.py \
    --rankings L0:results/null_baseline_L0/results/attribution_comparison/variant_rankings_rank_calibrated.csv \
               L1:results/null_baseline_L1/results/attribution_comparison/variant_rankings_rank_calibrated.csv \
               L2:results/null_baseline_L2/results/attribution_comparison/variant_rankings_rank_calibrated.csv \
               L3:results/null_baseline_L3/results/attribution_comparison/variant_rankings_rank_calibrated.csv \
    --score-column delta_rank \
    --out-comparison results/ablation/ablation_ranking_comparison.yaml \
    --out-jaccard results/ablation/ablation_jaccard_matrix.tsv \
    --out-level-specific results/ablation/level_specific_variants.tsv
```

##### Adjusting Comparison Thresholds

The level-specific variant detection uses two thresholds:

- `--high-rank-threshold` (default: 100): a variant must be in the top-N at one level
- `--low-rank-threshold` (default: 500): the variant must be outside the top-N at all other levels

Tighter thresholds (e.g., `--high-rank-threshold 50 --low-rank-threshold 200`) produce a more selective list; looser thresholds capture more candidates.

##### Using Null-Contrasted Significance Rankings

Rank the ablation comparison by `delta_rank`, the primary ranking metric, using the
rank-calibrated files from `bootstrap_null_calibration.py`. The chrX-corrected files
(`corrected_variant_rankings.csv`, produced by `correct_chrx_bias.py`) carry `z_attribution`,
which is a per-chromosome visualisation score. The `variant_rankings_with_significance.csv`
files from `run_null_baseline_analysis.sh` carry neither and would have to be ranked by
`empirical_p_variant`, which is bounded below by `1/(N_null + 1)` and pins most real variants
at that floor when the model is informative, making top-K selection a draw from a tied set.

```bash
# 1. Copy chrX-corrected significance files into a comparison directory
mkdir -p results/ablation/significance_rankings
for LEVEL in L0 L1 L2 L3; do
    cp results/null_baseline_${LEVEL}/results/attribution_comparison/corrected/corrected_variant_rankings.csv \
       results/ablation/significance_rankings/${LEVEL}_sieve_variant_rankings.csv
done

# 2. Compare using per-chromosome z-attribution ranking (recommended)
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/significance_rankings \
    --score-column z_attribution \
    --top-k 100,500,1000,2000 \
    --out-comparison results/ablation/significance_ablation_ranking_comparison.yaml \
    --out-jaccard results/ablation/significance_ablation_jaccard_matrix.tsv \
    --out-level-specific results/ablation/significance_level_specific_variants.tsv

# 3. Plot (reads from the significance-based TSV outputs)
python scripts/plot_ablation_comparison.py \
    --jaccard-tsv results/ablation/significance_ablation_jaccard_matrix.tsv \
    --level-specific-tsv results/ablation/significance_level_specific_variants.tsv \
    --summary-yaml results/ablation/ablation_summary.yaml \
    --heatmap-top-k 1000 \
    --output results/ablation/significance_ablation_comparison.png
```

Using `--include-sex-chroms` retains chrX/chrY variants in the output (flagged via `is_sex_chrom`) but normalises their scores relative to other variants on the same chromosome. This removes systematic inflation while preserving genuinely important sex-chromosome variants.

##### Rank-Based Null Calibration

After the magnitude-based null comparison, `bootstrap_null_calibration.py` runs a complementary rank-based null calibration that generates an ensemble of `B = 1000` null rankings by bootstrap-resampling the null's per-sample attributions. This adds per-variant empirical p-values and BH-FDR with resolution `1 / (B + 1)`, a per-gene Wilcoxon rank-sum test, top-k overlap and KS diagnostics, and a `delta_rank` column where positive values mean the real model promotes that variant relative to the bootstrap-null ensemble. The gene-stats CSV also carries a `gene_delta_rank` column computed as `max(delta_rank)` per gene by default (mirroring `gene_z_score = max(z_attribution)`), configurable via `--gene-delta-rank-aggregation`.

```bash
python scripts/bootstrap_null_calibration.py \
    --real-rankings results/<cohort>/real_experiments/L1/attributions/variant_rankings_with_significance.csv \
    --null-attributions results/<cohort>/null_baselines/L1/attributions/attributions.npz \
    --output results/<cohort>/real_experiments/L1/attributions/variant_rankings_rank_calibrated.csv \
    --n-bootstrap 1000 \
    --seed 42
```

##### Bootstrap-Calibrated Ablation Workflow

The bootstrap-calibrated file carries both the `delta_rank` ranking metric and the `z_attribution` visualisation score. Run `compare_ablation_rankings.py` twice:

- `--score-column delta_rank`: the primary ranking view, scale-free and stable across annotation levels
- `--score-column z_attribution`: the per-chromosome visualisation view, retained for continuity with Manhattan plots and earlier runs

Concordance between the two Jaccard matrices strengthens the level-specific-discovery claim. Divergence is also informative: it tells you which discoveries depend mostly on the real-signal ordering versus the bootstrap-null contrast.

```bash
# View 1: primary ranking
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rank_calibrated_rankings \
    --score-column delta_rank \
    --top-k 100,500,1000,2000 \
    --out-comparison results/ablation/delta_ablation_comparison.yaml \
    --out-jaccard results/ablation/delta_ablation_jaccard.tsv \
    --out-level-specific results/ablation/delta_level_specific_variants.tsv

# View 2: per-chromosome visualisation
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rank_calibrated_rankings \
    --score-column z_attribution \
    --top-k 100,500,1000,2000 \
    --out-comparison results/ablation/z_ablation_comparison.yaml \
    --out-jaccard results/ablation/z_ablation_jaccard.tsv \
    --out-level-specific results/ablation/z_level_specific_variants.tsv
```

##### Non-Linear Classifier Robustness Run Pattern

The non-linear classifier validation (`validate_nonlinear_classifier.py`) also supports `--score-column delta_rank`, which resolves automatically to the `gene_delta_rank` column in the gene-stats CSV. The recommended workflow is to run the validation twice with separate output TSVs — one primary run using `--score-column delta_rank` and one visualisation-view run using `--score-column z_attribution` — and apply Benjamini-Hochberg FDR independently within each invocation across the full 16-cell grid. Do not pool the two sets of p-values into a single FDR correction, as that would halve statistical power and obscure whether the robustness finding survives on its own.

##### Top-K Stability Sweep, Per-Pair FDR, and Direction of the Fisher Test

The recommended invocation for the gene-pair analysis is:

```bash
python scripts/aggregate_gene_interactions.py \
    --preprocessed-data preprocessed_<cohort>.pt \
    --variant-rankings <rank-calibrated variant rankings CSV> \
    --gene-rankings <calibrated gene stats CSV> \
    --null-rankings <null variant rankings CSV> \
    --cooccurrence <cooccurrence_per_pair.csv> \
    --output-dir results/<cohort>/gene_interactions \
    --score-column delta_rank \
    --allow-nonsignificant-genes \
    --top-k-genes 100 \
    --correction fdr_bh \
    --alpha 0.05 \
    --alternative greater
```

This produces a single network at K=100 (the recommended primary value for
interpretable hub analysis) with per-pair Benjamini-Hochberg FDR control on
a one-sided Fisher exact test of carrier-state independence in the direction
of excess co-occurrence. The `padj` and `reject` columns in
`gene_pair_interactions.csv` flag pairs that survive the chosen correction.
The CLI default is now K=100; pass `--top-k-genes 50` explicitly to reproduce
older default runs.

###### Choosing `--alternative`

The Fisher exact test on the 2x2 carrier table can be run in three
directions, each corresponding to a distinct scientific hypothesis. The
choice is consequential and must be made *before* looking at the data -
post-hoc switching of direction inflates Type I error and invalidates
the FDR control.

**`--alternative greater` (default).** Tests for *excess* co-occurrence
relative to independence: pairs where carriers of both genes cluster
together more often than carriers of either gene alone would predict.
This is the appropriate direction for:

- Classical synthetic-lethal or synergistic epistasis hypotheses, where
  having both hits is jointly required (or jointly worsens) the phenotype.
- Adult-onset cohort studies where ascertainment is not expected to select
  against a specific double-carrier combination.
- Most case-control studies of complex phenotype_x where the working
  hypothesis is that risk variants compound rather than cancel.

This is the right default for nearly all applications.

**`--alternative less`.** Tests for *deficit* of co-occurrence: pairs
where the double-carrier state appears less often than independence
predicts. This is the appropriate direction for:

- Early-onset or developmental phenotype_x studies where carriers of both
  genes may not survive to recruitment, producing survivorship bias in the
  observed cohort. The deficit is a secondary signal of an interaction
  that is actively selected against in the population.
- Compound heterozygous lethality screens.
- Hypotheses about epistatic suppression - where one variant masks the
  effect of another - under specific cohort-design conditions.

This is not generally the right test for adult-onset phenotype_x cohorts,
because deficit-of-co-occurrence in a recruited adult cohort is usually an
artefact of recruitment criteria rather than a biological signal.

**`--alternative two-sided`.** Tests for departure in either direction
without prior commitment. Use when:

- Genuine prior uncertainty exists about direction.
- Reporting a methodological survey rather than a hypothesis test.
- A reviewer or methods paper specifically requests a direction-agnostic
  version.

The cost is reduced power: at the same alpha, the two-sided test
is approximately half as powerful as a correctly-specified one-sided
test in each direction. Do not switch from one-sided to two-sided
after seeing no rejections under one-sided - that is post-hoc selection
of the test and inflates Type I error.

###### Documenting the choice in the manuscript

The `--alternative` value is recorded in the `fisher_alternative` field
of `gene_interaction_summary.yaml`. When reporting results, state the
direction explicitly: e.g. *"per-pair excess co-occurrence was tested
with a one-sided Fisher exact test (`alternative='greater'`) under
Benjamini-Hochberg FDR control at q < 0.05."* A reviewer cannot evaluate
the FDR claim without knowing which direction was tested.

###### Top-K stability sweep

To run a stability sweep that aligns with the ablation analysis convention
(K in {100, 2000}):

```bash
python scripts/aggregate_gene_interactions.py \
    ... \
    --top-k-genes 100 2000 \
    --correction fdr_bh \
    --alternative greater
```

This produces per-K outputs (`gene_pair_interactions_topK{100,2000}.csv`,
matching network and summary files) plus a top-level
`gene_interaction_summary_index.yaml` summarising both runs. K=2000 is
quadratic - about 2 million pairs - and the resulting network is too dense
for hub interpretation in isolation; its primary value is as a sensitivity
check that the K=100 hub structure is robust to broader gene inclusion,
not as a standalone analysis.

**Note on the interaction score formula.** The score uses rank-quantile-
normalised gene scores so that the ranking is invariant to `--score-column`
choice. The raw `gene_score_a`/`_b` columns are retained alongside the new
`gene_score_quantile_a`/`_b` columns for transparency. The score remains a
heuristic for candidate ranking; use the `padj` and `obs_exp_ratio` columns
for departure-from-independence inference.

##### Power-Analysis Correction

The `--correction` argument for `epistasis_power_analysis.py` now accepts
`bonferroni` or `fdr_bh`; the default is `fdr_bh`. The previous `fdr` value
was implemented identically to Bonferroni (a known bug) and now errors out
with a message directing the user to `fdr_bh`. For power-analysis MDE
planning under FDR control, the per-test threshold is taken as `alpha`
itself rather than the rank-1 BH threshold (which equals Bonferroni); this
is a conventional planning choice and avoids reporting MDEs that are
conservative by a factor of the test count.

---

## Command Reference

#### preprocess.py

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

#### infer_sex.py

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

#### check_sex_balance.py

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

#### train.py

```bash
python scripts/train.py [OPTIONS]
```

##### Data Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--vcf` | path | - | VCF file (if not using preprocessed) |
| `--phenotypes` | path | - | Phenotype file (if not using preprocessed) |
| `--preprocessed-data` | path | - | Preprocessed .pt file |
| `--level` | str | required | Annotation level [L0, L1, L2, L3, L4] |

##### Training Options

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

##### Model Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--latent-dim` | int | 64 | Embedding dimension |
| `--hidden-dim` | int | 128 | Hidden layer dimension |
| `--num-heads` | int | 4 | Number of attention heads |
| `--num-attention-layers` | int | 2 | Number of attention layers |
| `--aggregation-method` | str | mean | Chunk aggregation method [mean, max, attention, logit_mean] |

##### Cross-Validation Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--cv` | int | - | Number of CV folds (if not using val-split) |
| `--val-split` | float | 0.2 | Validation split ratio |

##### Output Options

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

#### explain.py

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

#### create_null_baseline.py

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

#### run_null_baseline_analysis.sh

```bash
bash scripts/run_null_baseline_analysis.sh
```

This wrapper is configured through environment variables.

##### Preferred interface (cohort-centric layout)

| Variable | Required | Description |
|---------|----------|-------------|
| `PROJECT_DIR` | Yes | Cohort project root directory (e.g. `/path/to/project`) |
| `LEVEL` | Yes | Annotation level to run (e.g. `L3`) |
| `NULL_DATA` | No | Pre-existing permuted `.pt` file — skips Step 1 if set |
| `DEVICE` | No | `cuda` or `cpu` (default: `cuda`) |
| `PYTHON` | No | Python interpreter path override |
| `EXCLUDE_SEX_CHROMS` | No | Set to `1` to pass `--exclude-sex-chroms` to the comparison step |

When `PROJECT_DIR` and `LEVEL` are set, all paths are derived automatically:
- Input data: first `preprocessed*.pt` (not `*_NULL*`) found in `${PROJECT_DIR}/data/`
- Real experiment: `${PROJECT_DIR}/real_experiments/${LEVEL}/training`
- Real results: `${PROJECT_DIR}/real_experiments/${LEVEL}/attributions`
- Null outputs: `${PROJECT_DIR}/null_baselines/${LEVEL}/`
- Significance output: `${PROJECT_DIR}/real_experiments/${LEVEL}/attributions/`

**Preferred example**:
```bash
PROJECT_DIR=/path/to/project \
LEVEL=L3 \
DEVICE=cuda \
bash scripts/run_null_baseline_analysis.sh
```

##### Legacy interface (still supported)

Explicit variable overrides take precedence over derived paths when set alongside `PROJECT_DIR`/`LEVEL`, or can replace them entirely.

| Variable | Description |
|---------|-------------|
| `INPUT_DATA` | Path to real preprocessed `.pt` file |
| `REAL_EXPERIMENT` | Real experiment training directory |
| `REAL_RESULTS` | Directory containing real `sieve_variant_rankings.csv` |
| `OUTPUT_BASE` | Base directory where null outputs are written |

**Legacy example**:
```bash
export INPUT_DATA=data/preprocessed.pt
export REAL_EXPERIMENT=experiments/my_model
export REAL_RESULTS=results/explainability
export OUTPUT_BASE=results/null_baseline_run
export DEVICE=cuda

bash scripts/run_null_baseline_analysis.sh
```

Behaviour:
- Reads model/training hyperparameters from real `config.yaml` (in `REAL_EXPERIMENT` or parent).
- Carries over `sex_map` automatically when the real model used sex covariates.
- Resolves script paths relative to wrapper location, so it can be run from any working directory.

---

#### compare_attributions.py

```bash
python scripts/compare_attributions.py [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--real` | path | required | Raw real variant rankings CSV (`sieve_variant_rankings.csv`) |
| `--null` | path | required | Raw null variant rankings CSV (`sieve_variant_rankings.csv`) |
| `--output-dir` | path | required* | Output directory. Mutually exclusive with `--project-dir` |
| `--project-dir` | path | required* | Cohort project root. Output routed to `{project-dir}/real_experiments/{LEVEL}/attributions/` (level inferred from `--real` path). Mutually exclusive with `--output-dir` |
| `--genome-build` | str | GRCh37 | Reference genome build |
| `--exclude-sex-chroms` | flag | False | Exclude chrX/chrY before empirical p-value and FDR computation |

\* Exactly one of `--output-dir` or `--project-dir` is required.

**Example (explicit output)**:
```bash
python scripts/compare_attributions.py \
    --real results/explainability/sieve_variant_rankings.csv \
    --null results/null_attributions/sieve_variant_rankings.csv \
    --output-dir results/attribution_comparison \
    --genome-build GRCh37
```

**Example (project-dir routing)**:
```bash
python scripts/compare_attributions.py \
    --real /path/to/project/real_experiments/L3/attributions/sieve_variant_rankings.csv \
    --null /path/to/project/null_baselines/L3/attributions/sieve_variant_rankings.csv \
    --project-dir /path/to/project \
    --genome-build GRCh37
# Output: /path/to/project/real_experiments/L3/attributions/
```

---

#### bootstrap_null_calibration.py

```bash
python scripts/bootstrap_null_calibration.py [OPTIONS]
```

Generates a bootstrap ensemble of null rankings from `attributions.npz`, adds
rank-based empirical p-values and BH-FDR to a real rankings CSV, emits
gene-level Mann-Whitney statistics, and writes a YAML summary of top-k overlap
and KS diagnostics. This is the rank-based complement to
`compare_attributions.py`.

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

See `USER_GUIDE.md` under **Rank-based null calibration** and
**Bootstrap-calibrated ablation comparison** for the recommended integrated
workflow.

---

#### correct_chrx_bias.py

```bash
python scripts/correct_chrx_bias.py [OPTIONS]
```

Run this script on the significance-annotated file
(`variant_rankings_with_significance.csv` from `compare_attributions.py`) to add
chrX-corrected rankings while preserving all existing columns including
`empirical_p_variant` and `fdr_variant`. Do not run on null rankings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--rankings` | path | required | Variant rankings CSV (typically `variant_rankings_with_significance.csv`) |
| `--output-dir` | path | required* | Output directory. Mutually exclusive with `--project-dir` |
| `--project-dir` | path | required* | Cohort project root. Output routed to `{project-dir}/real_experiments/{LEVEL}/attributions/corrected/` (level inferred from `--rankings` path). Mutually exclusive with `--output-dir` |
| `--exclude-sex-chroms` | flag | True | Exclude chrX/chrY from final rankings (default) |
| `--include-sex-chroms` | flag | False | Include chrX/chrY (flagged) in rankings |
| `--genome-build` | str | GRCh37 | Reference genome build |
| `--top-k` | int | 100 | Top variants to annotate in plot |

\* Exactly one of `--output-dir` or `--project-dir` is required.

**Example (explicit output)**:
```bash
python scripts/correct_chrx_bias.py \
    --rankings results/attribution_comparison/variant_rankings_with_significance.csv \
    --output-dir results/attribution_comparison/corrected \
    --include-sex-chroms \
    --genome-build GRCh37
```

**Example (project-dir routing)**:
```bash
python scripts/correct_chrx_bias.py \
    --rankings /path/to/project/real_experiments/L3/attributions/variant_rankings_with_significance.csv \
    --project-dir /path/to/project \
    --include-sex-chroms \
    --genome-build GRCh37
# Output: /path/to/project/real_experiments/L3/attributions/corrected/
```

---

#### compare_ablation_rankings.py

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
| `--score-column` | str | `z_attribution` | Column to rank variants by. **Use `delta_rank`**: it is the primary ranking metric, scale-free and stable across annotation levels. `z_attribution` is a per-chromosome z-score retained as a visualisation score for Manhattan plots and for continuity with earlier runs; per-chromosome z-scoring flattens genome-wide signal, so it is not suitable for cross-level ranking comparison. P/FDR-like columns and true rank columns are ranked ascending automatically; `delta_rank` is ranked descending. The default is unchanged for reproducibility of prior runs, so pass `--score-column delta_rank` explicitly. |

**Example**:
```bash
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rank_calibrated_rankings \
    --score-column delta_rank \
    --top-k 50,100,200,500 \
    --out-comparison results/ablation/ablation_ranking_comparison.yaml \
    --out-jaccard results/ablation/ablation_jaccard_matrix.tsv \
    --out-level-specific results/ablation/level_specific_variants.tsv
```

To additionally produce the `z_attribution` visualisation view, rerun with
`--score-column z_attribution` and separate output paths. For the two-run
workflow, see **Bootstrap-calibrated ablation comparison** under Detailed Usage.

---

#### ablation_compare.py

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

#### plot_ablation_comparison.py

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

#### validate_epistasis.py

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

#### audit_cooccurrence.py

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
The summary file `cooccurrence_summary.yaml` distinguishes:

- `n_pairs_gte5_cooccur`: pairs with at least 5 joint carriers (`n11 >= 5`)
- `n_pairs_all_cells_gte5`: pairs where all four carrier states have at least 5 samples (`n11`, `n10`, `n01`, `n00`)

The second metric is the more relevant one for interaction analysis, because estimating a non-additive interaction effect requires support across all four states. In these field names, `gte5` means `>= 5`.

---

#### aggregate_gene_interactions.py

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
| `--top-k-genes` | int list | `100` | Top-K gene set sizes. List supported (e.g. `100 2000`) for stability sweeps. Quadratic in K. The default is now 100; pass `50` explicitly to reproduce older default runs. |
| `--min-gene-score` | float | 0.0 | Minimum gene score |
| `--score-column` | str | `z_attribution` | Variant-level score column for ranking and gene scoring. Choices: `z_attribution`, `delta_rank`. **Use `delta_rank`** with a rank-calibrated input from `bootstrap_null_calibration.py`: it is the primary ranking metric. `z_attribution` is a per-chromosome visualisation score. The default is unchanged for reproducibility of prior runs. |
| `--significance-threshold` | str | `p_0.05` | Null-derived significance threshold to enforce when available |
| `--min-significant-variants` | int | 1 | Minimum number of significant variants required for a gene |
| `--allow-nonsignificant-genes` | flag | False | Allow genes with no null-significant variants |
| `--correction` | str | `fdr_bh` | Multiple-testing correction for per-pair Fisher exact p-values. Choices: `none`, `bonferroni`, `fdr_bh`. |
| `--alpha` | float | `0.05` | Significance level for the per-pair correction. |
| `--alternative` | str | `greater` | Direction of the per-pair Fisher exact test. Choices: `greater` (excess co-occurrence; classical synthetic-lethal-style epistasis), `less` (deficit; selection against the double-carrier state), `two-sided` (either direction). Must be chosen before looking at the data. See `detailed-usage.md`. |

This script is the preferred gene-level interaction workflow when you have null-comparison and chrX-corrected attribution outputs available.

---

#### epistasis_power_analysis.py

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
| `--correction` | str | `fdr_bh` | Multiple-testing correction method for MDE planning. Choices: `bonferroni`, `fdr_bh`. The deprecated `fdr` value now errors out with a clear message; users should switch to `fdr_bh`. |

Power is computed from the full `2x2` carrier table for each pair, not just the joint-carrier count. This avoids overstating power for near-ubiquitous common-common pairs.

---

#### validate_discoveries.py

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

#### generate_sieve_gene_list.py

```bash
python scripts/generate_sieve_gene_list.py [OPTIONS]
```

Aggregates variant-level SIEVE rankings to a gene-level TSV for cross-cohort burden validation.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--variant-rankings` | path | required | Variant rankings CSV that contains the column named by `--score-column`. For `delta_rank` this is the rank-calibrated CSV from `bootstrap_null_calibration.py`; for `z_attribution` it is `corrected_variant_rankings.csv` from `correct_chrx_bias.py`. |
| `--output` | path | required | Output gene list TSV |
| `--score-column` | str | `z_attribution` | Variant-level column to aggregate per gene into the output `gene_score`. **Use `delta_rank`**, the primary ranking metric, together with a rank-calibrated input from `bootstrap_null_calibration.py`. The column must be present in `--variant-rankings` or the script exits with `Score column not found`; there is no alias resolution here. `z_attribution` is a per-chromosome visualisation score retained for continuity with Manhattan plots and earlier runs. The default is unchanged for reproducibility of prior runs. |
| `--exclude-sex-chroms` | flag | True | Exclude sex chromosome genes |
| `--include-sex-chroms` | flag | False | Include sex chromosome genes (overrides --exclude-sex-chroms) |
| `--min-null-threshold` | str | None | Only include genes with variants exceeding this null threshold (`p05`, `p01`, `p001`) |
| `--aggregation` | str | `max` | How to aggregate variant scores per gene: `max` or `mean` |
| `--ablation-level` | str | None | Prefix output filename with level label (e.g. `L0`) |
| `--fdr-threshold` | float | None | Only include genes with `fdr_gene` below this value. Gene set size is determined dynamically. |
| `--gene-significance` | path | None | Path to gene significance CSV for FDR filtering. Auto-discovered if not specified. |

**Example** (fixed gene list):
```bash
python scripts/generate_sieve_gene_list.py \
    --variant-rankings results/attribution_comparison/variant_rankings_rank_calibrated.csv \
    --output validation/sieve_gene_lists/sieve_genes.tsv \
    --score-column delta_rank \
    --aggregation max
```

**Example** (FDR-threshold filtered):
```bash
python scripts/generate_sieve_gene_list.py \
    --variant-rankings results/attribution_comparison/variant_rankings_rank_calibrated.csv \
    --output validation/sieve_gene_lists/sieve_genes_fdr05.tsv \
    --score-column delta_rank \
    --fdr-threshold 0.05 \
    --aggregation max
```

---

#### extract_validation_burden.py

```bash
python scripts/extract_validation_burden.py [OPTIONS]
```

Parses a validation VCF and computes per-sample burden counts within SIEVE gene sets. Optionally builds a full gene-level burden matrix for permutation testing.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--vcf` | path | required | Validation VCF (bgzipped, tabix-indexed) |
| `--phenotypes` | path | required | Phenotype TSV (sample_id, phenotype: 1=ctrl, 2=case) |
| `--sieve-genes` | path | required | SIEVE gene list TSV |
| `--output-dir` | path | required | Output directory |
| `--genome-build` | str | `GRCh37` | Reference genome build |
| `--min-gq` | int | 20 | Minimum genotype quality |
| `--top-k` | int list | `50 100 200` | Gene set sizes to test |
| `--consequence-stratify` | flag | False | Compute burden stratified by consequence class |
| `--include-sex-chroms` | flag | False | Include sex chromosome variants |
| `--from-variant-rankings` | flag | False | Input is a variant rankings CSV (aggregate internally) |
| `--compute-full-gene-matrix` | flag | False | Build full gene-level burden matrix for permutation testing |

> **Tip — multi-level validation in a single VCF pass**: The full gene matrix records
> burden for *every* gene in the VCF, regardless of which `--sieve-genes` file you
> provide. When comparing ablation levels (L0–L3), run `extract_validation_burden.py`
> **once** with `--compute-full-gene-matrix` and any gene list, then call
> `test_burden_enrichment.py` separately per level with the level-specific gene list.
> This avoids re-parsing the VCF for each level:
>
> ```bash
> # Parse VCF once
> python scripts/extract_validation_burden.py \
>     --vcf /path/to/validation_cohort.vcf.gz \
>     --phenotypes /path/to/phenotypes.tsv \
>     --sieve-genes validation/sieve_gene_lists/L3_sieve_genes.tsv \
>     --output-dir validation/cohort_b \
>     --top-k 50 100 200 \
>     --consequence-stratify \
>     --compute-full-gene-matrix
>
> # Test enrichment per level (fast — reads parquet, no VCF)
> for level in L0 L1 L2 L3; do
>     python scripts/test_burden_enrichment.py \
>         --burden-dir validation/cohort_b \
>         --sieve-genes validation/sieve_gene_lists/${level}_sieve_genes.tsv \
>         --output-dir validation/cohort_b/enrichment_${level} \
>         --top-k 50 100 200 \
>         --n-permutations 10000
> done
> ```

**Example**:
```bash
python scripts/extract_validation_burden.py \
    --vcf /path/to/validation_cohort.vcf.gz \
    --phenotypes /path/to/phenotypes.tsv \
    --sieve-genes validation/sieve_gene_lists/sieve_genes.tsv \
    --output-dir validation/cohort_b \
    --top-k 50 100 200 \
    --consequence-stratify \
    --compute-full-gene-matrix
```

---

#### test_burden_enrichment.py

```bash
python scripts/test_burden_enrichment.py [OPTIONS]
```

Permutation-based enrichment test comparing SIEVE gene sets against random gene sets of the same size. Requires the gene burden matrix from `extract_validation_burden.py --compute-full-gene-matrix`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--burden-dir` | path | required | Directory with burden files from `extract_validation_burden.py` |
| `--gene-matrix` | path | None | Path to gene burden matrix Parquet (default: `<burden-dir>/gene_burden_matrix.parquet`) |
| `--background-genes` | path | None | Text file listing background gene symbols (default: all genes in matrix) |
| `--phenotypes` | path | None | Phenotype file (only if not derivable from burden files) |
| `--sieve-genes` | path | required | SIEVE gene list TSV |
| `--output-dir` | path | required | Output directory |
| `--n-permutations` | int | 10000 | Number of random gene set permutations |
| `--seed` | int | 42 | Random seed |
| `--top-k` | int list | `50 100 200` | Gene set sizes to test |
| `--covariates` | path | None | TSV with covariates for logistic regression |
| `--consequence-types` | str list | `total` | Burden types to test: `total`, `missense`, `lof`, `synonymous` |
| `--correction` | str | `fdr_bh` | Multiple-testing correction for the multi-top-K summary in the report. Choices: `bonferroni`, `fdr_bh`. |

**Example**:
```bash
python scripts/test_burden_enrichment.py \
    --burden-dir validation/cohort_b \
    --sieve-genes validation/sieve_gene_lists/sieve_genes.tsv \
    --output-dir validation/cohort_b/enrichment \
    --n-permutations 10000 \
    --top-k 50 100 200 \
    --consequence-types total missense lof \
    --seed 42
```

---

#### validate_nonlinear_classifier.py

```bash
python scripts/validate_nonlinear_classifier.py [OPTIONS]
```

Tests whether SIEVE gene sets carry non-linear discriminative information by training a random forest (and optionally logistic regression) on the per-gene burden vector and comparing performance against a permutation null from size-matched random gene sets. See [Validation](validation.md) for detailed usage and interpretation.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--real-rankings-dir` | path | required | Directory with one subdirectory per level containing corrected gene rankings |
| `--burden-matrix` | path | required | Gene-burden matrix parquet file |
| `--phenotypes` | path | required | Phenotype TSV (sample_id, phenotype: 1=ctrl, 2=case) |
| `--output-tsv` | path | required | Summary TSV path |
| `--top-k` | str | — | Comma-separated top-k values, e.g. `100,500,1000,2000`. Mutually exclusive with `--fdr-threshold`. |
| `--fdr-threshold` | float | — | FDR cutoff for gene selection (e.g. `0.05`). Gene set size determined per level. Mutually exclusive with `--top-k`. |
| `--classifiers` | str | required | Comma-separated classifier list from `rf,lr` |
| `--levels` | str | `L0,L1,L2,L3` | Comma-separated annotation levels |
| `--n-permutations` | int | `1000` | Number of random gene set permutations |
| `--cv-folds` | int | `5` | Number of stratified CV folds |
| `--seed` | int | `42` | Random seed |
| `--n-cores` | int | `-1` | Number of outer-loop cores for permutation evaluation |
| `--score-column` | str | `z_attribution` | Gene-ranking score to use. **Use `delta_rank`**, the primary ranking metric; this script resolves it to the `gene_delta_rank` column when present. `z_attribution` resolves to `gene_z_score` and is the per-chromosome visualisation view. The default is unchanged for reproducibility of prior runs. |
| `--also-export-csv` | flag | off | Export classifier input matrices as CSV under `csv/` in the output directory |

**Example** (fixed top-k, multi-level with both classifiers):
```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir ablation/significance_rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --phenotypes /path/to/phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_summary.tsv \
    --top-k 50,100,200,500 \
    --n-permutations 1000 \
    --classifiers rf,lr \
    --n-cores 4
```

**Example** (FDR-threshold gene selection):
```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir ablation/significance_rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --phenotypes /path/to/phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_fdr.tsv \
    --fdr-threshold 0.05 \
    --n-permutations 1000 \
    --classifiers rf,lr \
    --n-cores 4
```

---

#### summarize_classifier_comparison.py

```bash
python scripts/summarize_classifier_comparison.py [OPTIONS]
```

Scans YAML outputs from `validate_nonlinear_classifier.py`, pairs RF and LR results per `(level, top_k)` combination, and produces per-combination comparison figures and a collated A4-landscape PDF. Only meaningful when `validate_nonlinear_classifier.py` was run with `--classifiers rf,lr`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--results-dir` | path | required | Directory containing the YAML outputs from `validate_nonlinear_classifier.py` |
| `--output-dir` | path | `<results-dir>/classifier_comparison/` | Directory for output figures and collated PDF |

**Example**:
```bash
python scripts/summarize_classifier_comparison.py \
    --results-dir validation/cohort_b/nonlinear_validation/ \
    --output-dir validation/cohort_b/nonlinear_validation/summary_plots/
```

---

#### plot_validation_burden.py

```bash
python scripts/plot_validation_burden.py [OPTIONS]
```

Collects and visualises scalar burden enrichment results across annotation levels. Reads YAML outputs from `test_burden_enrichment.py` and produces summary tables and plots.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-dirs` | path list | required | Directories containing enrichment YAML files (one per level) |
| `--labels` | str list | None | Labels for each input directory (auto-detected from directory names if omitted) |
| `--output-dir` | path | `.` | Output directory |
| `--top-k` | int list | `100 500 1000 2000` | Top-k values to include |
| `--consequence-types` | str list | `total missense lof` | Consequence types to include |
| `--output-prefix` | str | `validation_burden` | Prefix for output filenames |
| `--format` | str | `png` | Output format: `png`, `pdf`, or `both` |

**Example**:
```bash
python scripts/plot_validation_burden.py \
    --input-dirs validation/cohort_b/enrichment_L0 \
                 validation/cohort_b/enrichment_L1 \
                 validation/cohort_b/enrichment_L2 \
                 validation/cohort_b/enrichment_L3 \
    --top-k 100 500 1000 2000 \
    --consequence-types total missense lof \
    --output-dir validation/cohort_b/burden_plots
```

---

## Validation

This chapter covers the complete validation strategy for SIEVE gene sets in independent cohorts. Validation is a multi-step process that starts with generating gene lists from the discovery cohort, extracting burden counts from independent validation VCFs, and then testing whether the SIEVE gene sets carry disease signal using both scalar burden enrichment and non-linear classifier tests.

---

### Overview

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

### Prerequisites

Before running non-linear classifier validation, you need:

1. **Gene-burden matrix** from `extract_validation_burden.py --compute-full-gene-matrix` (see [Complete Workflow, Step 8b](complete-workflow.md))
2. **SIEVE gene lists** from `generate_sieve_gene_list.py` (see [Complete Workflow, Step 8a](complete-workflow.md))
3. **Phenotype file** in standard SIEVE format (`sample_id<TAB>phenotype`, 1=control, 2=case)

---

### Non-Linear Classifier Validation

#### Motivation

The scalar burden test showed that SIEVE gene sets do not necessarily carry more total exonic variation in cases than controls. But SIEVE's claim is not that its genes have more variants — it is that the **pattern** of variation across genes jointly discriminates cases from controls. A burden count destroys this pattern; a non-linear classifier preserves it.

The validation question is:

> *Can a non-linear classifier trained on per-gene burden counts across the SIEVE gene set discriminate cases from controls in an independent cohort better than the same classifier trained on random gene sets?*

- If **yes**: SIEVE's gene selection captures genuine multi-dimensional signal transferable across cohorts.
- If **no**: the discovery findings do not generalise to this cohort (which may reflect phenotype mismatch, population mismatch, or insufficient signal).

#### Method

For each SIEVE gene set (defined by ablation level and top-k threshold):

1. Load the corrected gene rankings for each annotation level
2. Select the observed top-k genes for that level from the requested score column
3. Train the requested classifier on the per-gene burden sub-matrix using fixed stratified CV folds
4. For each `(top_k, classifier)` pair, group annotation levels by their effective matched gene count (`k_effective`) after burden-matrix intersection
5. Draw one shared null distribution per `(top_k, classifier, k_effective)` group and reuse it across the levels in that group
6. Compute empirical p-values with the `(k + 1) / (N + 1)` convention
7. Apply Benjamini-Hochberg FDR across the full result grid

The fixed CV folds ensure the only variable is the gene set, not the data split. The shared null distribution ensures that `null_mean_auc` and `null_std_auc` are identical across levels within a given `(top_k, classifier, k_effective)` group.

#### Quick Start

```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir results/ablation/rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --phenotypes /path/to/phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_summary.tsv \
    --top-k 100,500,1000,2000 \
    --classifiers rf,lr \
    --levels L0,L1,L2,L3 \
    --n-permutations 1000 \
    --cv-folds 5 \
    --n-cores 8 \
    --seed 42
```

#### Multi-Level Mode

Point `--real-rankings-dir` at a directory with one subdirectory per annotation level. Each level directory should contain `gene_rankings_with_significance.csv` (preferred), `corrected_gene_rankings_with_significance.csv`, or `corrected_gene_rankings.csv`.

```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir results/ablation/rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --phenotypes /path/to/phenotypes.tsv \
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

#### Comparing Random Forest vs Logistic Regression

Use `--classifiers rf,lr` to run both a random forest and a logistic regression on every combination:

```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir results/ablation/rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --phenotypes /path/to/phenotypes.tsv \
    --output-tsv validation/cohort_b/nonlinear_validation/nonlinear_validation_summary.tsv \
    --top-k 100,200 \
    --n-permutations 1000 \
    --classifiers rf,lr \
    --n-cores 8
```

**Why include logistic regression?** As a linear baseline. If the random forest significantly outperforms logistic regression on SIEVE genes, that is evidence of non-linear signal — directly supporting SIEVE's core claim that multi-gene combinatorial patterns carry disease information. If logistic regression performs equally well, the signal is linear (which could have been captured by a PRS approach).

#### Score Column Selection

Use `--score-column delta_rank`. This is the primary ranking metric: scale-free, stable across annotation levels, and mapped automatically onto the `gene_delta_rank` column in the gene-stats CSV produced by `bootstrap_null_calibration.py`, where each gene's score is `max(delta_rank)` across its variants by default. Higher values indicate stronger promotion of the gene's variants by the real model relative to the bootstrap-null ensemble.

Use `--score-column z_attribution` for the per-chromosome visualisation view, retained for continuity with Manhattan plots and earlier runs. The script maps this onto the gene-level `gene_z_score` column inside `corrected_gene_rankings*.csv`. Per-chromosome z-scoring flattens genome-wide signal, so do not use it as the primary ranking.

Use `--score-column fdr_gene` when you want to rank genes by their gene-level null-contrast significance instead of effect size. Lower values are treated as better for FDR-based ranking.

Run the validation twice with separate `--output-tsv` paths, one for `delta_rank` and one for `z_attribution`, and apply BH-FDR independently within each invocation across the full result grid. Pooling the two runs would halve statistical power and prevent a clean determination of whether each view holds independently. The same pattern applies upstream when generating the gene list itself (`generate_sieve_gene_list.py --score-column delta_rank`); see the Complete Workflow, Step 8a.

The `argparse` default is still `z_attribution` so that prior runs reproduce exactly. Pass `--score-column delta_rank` explicitly.

#### FDR-Threshold Gene Selection

Instead of choosing a fixed number of top genes, you can let the gene set size be determined by an FDR cutoff:

```bash
python scripts/validate_nonlinear_classifier.py \
    --real-rankings-dir results/ablation/rankings \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --phenotypes /path/to/phenotypes.tsv \
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

### Interpreting Non-Linear Classifier Results

#### Per-Combination Results (YAML)

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
  score_column: delta_rank

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

#### Key Metrics

| Metric | What it means |
|--------|--------------|
| `observed.mean_auc` | How well the SIEVE gene set discriminates cases from controls |
| `null_distribution.mean` | Expected AUC from the shared random-gene null of the same top-k |
| `empirical_p` | Probability that a random gene set performs as well or better |
| `z_score` | Observed AUC expressed as a z-score against the shared null |
| `fdr_bh` | BH-adjusted empirical p-value across the full result grid |
| `rf_minus_lr_auc` | AUC gap between random forest and logistic regression (positive = non-linear signal) |

#### Interpretation Guide

| FDR | Interpretation |
|-----|----------------|
| `fdr_bh < 0.01` | Strong evidence after multiple-testing correction |
| `0.01 <= fdr_bh < 0.05` | Moderate evidence after multiple-testing correction |
| `fdr_bh >= 0.05` | No FDR-significant evidence at this level/top-k combination |

#### What to Look For

1. **Significant FDR**: The SIEVE gene set outperforms the shared random-gene null after correction across the full grid. This supports transfer of the discovery signal to the validation cohort.

2. **RF > LR gap**: If the random forest outperforms logistic regression on the SIEVE gene set, the signal has non-linear structure — combinations of gene burdens matter, not just their sum. This directly supports SIEVE's model design.

3. **Level consistency**: If multiple ablation levels show signal, the discovery is robust. If only L0 (genotype-only) shows signal, the discovery survives at the ablation floor and is carried by genome structure alone. If only L3 shows signal, it may depend on functional annotations.

4. **Top-k sensitivity**: Signal concentrated in top-50 genes suggests a small set of strong drivers. Signal appearing only at top-500 suggests a diffuse polygenic signal.

#### Diagnostic Plots

Each combination produces a two-panel plot:

1. **Left panel**: Histogram of the shared null AUC distribution with observed AUC marked as a red vertical line. The further right the line, the stronger the evidence.

2. **Right panel**: Box plot comparing observed per-fold AUCs against the shared null distribution, showing the spread of performance across CV splits.

#### Summary Outputs

- **`nonlinear_validation_summary.tsv`**: One row per level x top-k x classifier combination with all key metrics and a single `fdr_bh` column.
- **`nonlinear_validation_heatmap.png`**: Visual comparison of observed AUC across levels (rows) and top-k values (columns), with `*` marking `fdr_bh < 0.05`.
- **`nonlinear_validation_report.md`**: Human-readable summary of significant results, best combinations, and RF vs LR comparison.

---

### Output Structure

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

### Computational Considerations

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

### Design Decisions

#### No Hyperparameter Tuning

The random forest uses a fixed, reasonable configuration (500 trees, `max_features='sqrt'`, `min_samples_leaf=5`, `class_weight='balanced'`). This is deliberate: the comparison between SIEVE and null gene sets must use identical classifier configurations. Tuning per gene set would conflate gene set quality with tuning luck.

#### Parallelism at Permutation Level

The script parallelises across permutations (each on a single core) rather than within each random forest (`n_jobs=1` inside each classifier). This avoids nested parallelism issues and is more efficient for the 1,000-permutation workload.

#### Fixed CV Folds

All evaluations (observed and every permutation) use the same cross-validation fold assignments. This ensures the only variable is the gene set, not the data split, making the comparison strictly fair.

---

### Complete Validation Example

Putting scalar burden and non-linear classifier validation together for one cohort:

```bash
# --- Step 1: Generate gene lists per ablation level ---
for level in L0 L1 L2 L3; do
    python scripts/generate_sieve_gene_list.py \
        --variant-rankings results/${level}_attribution_comparison/variant_rankings_rank_calibrated.csv \
        --output validation/sieve_gene_lists/sieve_genes.tsv \
        --ablation-level ${level} \
        --score-column delta_rank \
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
    --phenotypes /path/to/validation_phenotypes.tsv \
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
    --phenotypes /path/to/validation_phenotypes.tsv \
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

## Interpreting Results

#### Training Outputs

##### Single Run Results (`results.yaml`)

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

##### Cross-Validation Results (`cv_results.yaml`)

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

#### Explainability Outputs

##### Variant Rankings

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

##### chrX Ploidy Bias Correction (Optional)

If you used sex-aware preprocessing or observe chrX inflation in rankings, run `correct_chrx_bias.py` to standardise mean attributions per chromosome. Run this script on `variant_rankings_with_significance.csv` (the output of `compare_attributions.py`) so that the significance columns are preserved alongside the chrX-corrected z-scores:

```bash
python scripts/correct_chrx_bias.py \
    --rankings results/attribution_comparison/variant_rankings_with_significance.csv \
    --output-dir results/attribution_comparison/corrected \
    --include-sex-chroms
```

The script adds:

- `z_attribution`: per-chromosome z-scored attribution
- `corrected_rank`: rank based on `z_attribution`
- `is_sex_chrom`: flags chrX/chrY variants

All existing columns — including `empirical_p_variant` and `fdr_variant` — are preserved unchanged. By default, the corrected rankings exclude sex chromosomes. Use `--include-sex-chroms` if you want to keep them in the output (they remain flagged).

##### Choosing a ranking metric

Rank variants by `delta_rank`. It is defined as `median_null_rank - real_rank`, so it is scale-free, it is stable across annotation levels, and it holds the chromosome X share of the top-ranked set at 3 to 8 per cent. It is the primary ranking metric, and it is what you should pass to `--score-column` for cross-level comparison, gene-list generation and validation.

`z_attribution` is a per-chromosome z-score. Z-scoring within each chromosome removes the between-chromosome component of the signal, which flattens genome-wide differences; it is retained as a visualisation score for Manhattan plots and for continuity with earlier runs, not as a ranking metric.

Do not rank by a naive magnitude-based empirical p-value when comparing models. Real and null attributions sit on different scales: the real model learns signal and its attribution distribution shifts upward, while the null sits at an area under the curve near 0.50. A p-value computed by comparing raw magnitudes across that scale gap is therefore not a valid cross-model comparison.

The `argparse` defaults still name `z_attribution`, so that prior runs reproduce exactly. Pass `--score-column delta_rank` explicitly.

For ablation comparison, run `bootstrap_null_calibration.py` first, then rank the resulting rank-calibrated files with `--score-column delta_rank`. If you also want the `z_attribution` view for figures, rerun with separate output paths rather than pooling the two.

##### Gene Rankings

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

#### Null Baseline Comparison

##### Significance Summary (`significance_summary.yaml`)

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
   - ChrX correction is applied separately (via `correct_chrx_bias.py`) for ranking purposes

3. **Use FDR for decisions**:
   - `fdr_gene < 0.05`: suitable for manuscript-level per-gene claims
   - `fdr_variant < 0.05`: variant-level follow-up candidates
   - `min_achievable_empirical_p = 1 / (N + 1)`: lower bound imposed by the null size

---

#### Ablation Comparison Results

##### Performance Summary (`ablation_summary.yaml`)

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
- **L0 AUC > 0.6**: Genotype patterns alone carry disease signal. L0 is the ablation floor of the protocol, so this tells you how much of the model's discrimination survives when every supplied annotation is removed.
- **L2 ≈ L3**: Consequence class is sufficient; SIFT/PolyPhen add little beyond consequence type
- **L3 > L0 by >0.1 AUC**: Annotations provide substantial additional signal
- **L3 ≈ L0**: Annotations do not help, model discovers signal from genotype structure alone

##### Jaccard Matrix (`ablation_jaccard_matrix.tsv`)

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
- Agreement between the L0 and L3 rankings measures how much of the ranking is stable under annotation ablation: a high L0-vs-L3 Jaccard means the ordering is carried largely by genome structure rather than by the supplied annotations
- Low L0-vs-L3 Jaccard suggests annotations drive different discoveries (may indicate circular logic if annotations encode known associations)

##### Level-Specific Variants (`level_specific_variants.tsv`)

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

If you rank the same ablation inputs by `delta_rank`, interpret positive `delta_rank` as bootstrap-null-corrected promotion: the real model ranks that variant better than the null ensemble does. Comparing the `level_specific_variants.tsv` list from the `z_attribution` run against the `delta_rank` run shows which level-specific discoveries are robust across both views and which ones only appear under one ranking scheme.

##### Multi-Panel Figure (`ablation_comparison.png`)

The figure produced by `plot_ablation_comparison.py` contains four panels:

1. **Jaccard Heatmap** (top-left): Pairwise overlap at a selected top-k. Warm colours indicate low overlap (different discoveries), cool colours indicate high overlap (similar discoveries).

2. **Jaccard by Top-k** (top-right): Line plot showing how overlap evolves as you consider more variants. If lines rise steeply, the top-ranked variants differ but broader rankings converge.

3. **Level-Specific Counts** (bottom-left): Bar chart of how many uniquely important variants each level discovers. Large L0 bars mean a substantial part of the ranking is set at the ablation floor, before any annotation is supplied.

4. **AUC Comparison** (bottom-right): Model performance per level with error bars. The best level is highlighted. The red dashed line marks random performance (AUC=0.5).

---

#### Non-Linear Classifier Validation Results

The non-linear classifier validation tests whether the **pattern** of variation across SIEVE genes jointly discriminates cases from controls, beyond what a scalar burden sum can capture. See the [Validation](validation.md) chapter for full usage details.

##### Summary Table (`nonlinear_validation_summary.tsv`)

| Column | Description |
|--------|-------------|
| `level` | Ablation level (L0-L3) |
| `top_k` | Number of top genes used |
| `k_effective` | Genes matched in validation VCF |
| `classifier` | `rf` (random forest) or `lr` (logistic regression) |
| `observed_auc` | Mean AUC across CV folds |
| `observed_std` | Standard deviation of per-fold AUCs |
| `null_mean_auc` | Mean AUC of the null distribution |
| `null_std_auc` | Standard deviation of the null distribution |
| `empirical_p` | Fraction of random gene sets with AUC >= observed |
| `z_score` | Observed AUC as a z-score against the shared null |
| `fdr_bh` | BH-adjusted empirical p-value across the full result grid |

##### Decision Framework

| Observed AUC vs null | RF vs LR | Interpretation |
|---------------------|----------|----------------|
| Significant (p < 0.05) | RF >> LR | Non-linear multi-gene signal transfers to validation cohort |
| Significant (p < 0.05) | RF ≈ LR | Linear signal transfers (could be captured by PRS) |
| Not significant | — | Signal does not transfer at this level/top-k |

##### Heatmap (`nonlinear_validation_heatmap.png`)

Rows are ablation levels, columns are top-k values. Cell values show observed AUC with significance annotations:

- `*` = `fdr_bh < 0.05`
- No marker = not significant

Look for patterns: does signal concentrate at specific levels or top-k values? Consistent signal across levels suggests a robust discovery; signal only at L3 may indicate annotation dependence.

---

#### Epistasis Results

SIEVE now provides two complementary interaction views:

1. **Attention-based discovery**: high-attention variant pairs from `sieve_interactions.csv`, optionally followed by counterfactual validation in `epistasis_validation.csv`.
2. **Post-hoc attribution interaction analysis**: co-occurrence, power, and gene-gene aggregation from `audit_cooccurrence.py`, `epistasis_power_analysis.py`, and `aggregate_gene_interactions.py`.

##### Attention Discovery Output (`sieve_interactions.csv`)

This file contains variant pairs that exceeded the attention discovery threshold in `explain.py`. They are best treated as candidate interactions for follow-up, not as a complete interaction catalogue.

Key points:

- These pairs are discovered from the model's intrinsic attention mechanism, which is a distinctive feature of SIEVE.
- Discovery is currently restricted to pairs that occur within the same chunk.
- `--attention-threshold-mode percentile` is often more informative than a fixed absolute threshold when attention is diffuse across many variants.
- An empty `sieve_interactions.csv` means no pair crossed the current heuristic. It does not by itself prove an absence of interaction structure in the cohort.

##### Validation Output (`epistasis_validation.csv`)

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

##### Post-hoc Interaction Outputs

Use these when you need to understand whether the cohort is structurally able to support interaction detection even when the attention-based discovery file is sparse.

`cooccurrence_summary.yaml`
- Tells you how often evaluated pairs co-occur across MAF bins.
- Useful for diagnosing whether the rare-variant tail is too sparse.
- Does not solve the within-chunk visibility limit of the attention workflow.
- The key distinction is between `n_pairs_gte5_cooccur` and `n_pairs_all_cells_gte5`.
- `n_pairs_gte5_cooccur` only asks whether at least 5 samples carry both variants (`n11 >= 5`).
- `n_pairs_all_cells_gte5` asks whether the full `2x2` carrier table has support in every cell: `n11` (both), `n10` (A only), `n01` (B only), `n00` (neither).
- This matters because interaction estimation needs contrast across all four carrier states. If one cell is empty, the simple interaction contrast is not estimable in this framework; if one cell is very small, the estimate becomes unstable.
- In these field names, `gte5` means "greater than or equal to 5". The threshold of 5 is a pragmatic stability rule, not a mathematical theorem.

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

#### Installation Issues

##### ImportError: No module named 'cyvcf2'

**Solution**:
```bash
pip install cyvcf2
# or if that fails:
conda install -c bioconda cyvcf2
```

##### CUDA out of memory

**Solution**: Reduce memory usage
```bash
python scripts/train.py \
    --batch-size 2 \
    --chunk-size 2000 \
    --gradient-accumulation-steps 16 \
    ...
```

---

#### Data Preparation Issues

##### "does not contain VEP CSQ annotations"

**Symptom**: `ValueError: VCF file '...' does not contain VEP CSQ annotations.`

**Cause**: Your VCF has not been annotated with Ensembl VEP, or uses a different
annotation format (e.g. SnpEff `ANN` field).

**Solution**: Run VEP before preprocessing. See the
[Detailed Usage](detailed-usage.md#how-to-annotate-your-vcf-with-ensembl-vep)
page for the full command and required flags. The minimal command is:

```bash
vep \
    --input_file your.vcf.gz \
    --output_file annotated.vcf.gz \
    --vcf \
    --compress_output bgzip \
    --symbol \
    --canonical \
    --sift b \
    --polyphen b \
    --assembly GRCh37 \
    --offline \
    --cache \
    --dir_cache /path/to/vep_cache
tabix -p vcf annotated.vcf.gz
```

!!! warning "Do not use `--fields`"
    SIEVE expects VEP's **default CSQ field order** (hardcoded indices).
    Passing a custom `--fields` argument will break parsing silently.

##### "zero variant-sample assignments"

**Symptom**: `ValueError: Preprocessing produced zero variant-sample assignments
from N VCF records.`

**Cause**: The VCF header declares a CSQ field, but no variant data was loaded.
The error message includes diagnostics; the most common causes are:

- **All CSQ values empty**: The VCF was re-header'd or filtered after VEP
  annotation, stripping the actual CSQ values while keeping the header line.
- **Allele mismatch**: VEP's allele representation in CSQ doesn't match the
  VCF ALT field (can happen with post-VEP normalisation tools).
- **All genotypes filtered**: Every genotype fell below the GQ threshold
  (default 20). Try `--min-gq 0` to test.

**Verify** your CSQ values exist:
```bash
bcftools query -f '%INFO/CSQ\n' your.vcf.gz | head -3
```

If this prints empty lines or `.`, the CSQ values are missing despite the
header declaring the field.

##### Sample name mismatch

**Symptom**: `KeyError: SAMPLE001` or `ValueError: Sample not found in VCF`

**Solution**: Check that phenotype file sample IDs exactly match VCF:
```bash
# Get VCF samples
bcftools query -l your.vcf.gz

# Check phenotype file
cut -f1 phenotypes.tsv
```

Sample names must match character-for-character (case-sensitive).

##### Chromosome naming issue

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

#### Training Issues

##### Model not learning (AUC ≈ 0.5)

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

##### Training very slow

**Solutions**:

1. **Use GPU**: `--device cuda`
2. **Increase batch size**: `--batch-size 32` (if memory allows)
3. **Use preprocessed data**: Much faster than parsing VCF each time
4. **Reduce integration steps**: `--n-steps 25` in explain.py

##### Out of memory during training

**Solution**: Use memory-efficient settings:
```bash
--batch-size 2 \
--gradient-accumulation-steps 16 \
--chunk-size 2000
```

See "Memory-Efficient Training" section above.

---

#### Explainability Issues

##### Integrated gradients very slow

**Solutions**:

1. Reduce integration steps: `--n-steps 25` (less accurate but faster)
2. Limit variants per sample: `--max-variants 1500`
3. Use larger batch size: `--batch-size 8` (if memory allows)
4. Skip attention analysis: `--skip-attention` (if only need variant rankings)

##### AttributeError: 'NoneType' object has no attribute

**Cause**: Model checkpoint not found or corrupted

**Solution**: Verify checkpoint exists:
```bash
ls -lh experiments/my_model/best_model.pt
```

If using `--experiment-dir`, check that `best_model.pt` or `fold_*/best_model.pt` exists.

---

#### Null Baseline Issues

##### Null model AUC ≠ 0.5

**Expected**: Null model AUC should be ≈0.50 ± 0.05

**If AUC > 0.6**:
- **Problem**: Permutation didn't properly break genotype-phenotype relationship
- **Check**: Did you use the same preprocessed file for null training?
- **Solution**: Verify null baseline file has `_null_baseline_metadata` field

**If AUC < 0.4**:
- This is actually fine - model is consistently wrong, which is equivalent to chance
- Attributions are still valid for null distribution

##### No significant variants (enrichment < 1)

**Possible Causes**:

1. **Real model didn't learn**: Check real model AUC first
2. **Null and real similar**: May indicate no genuine signal in data
3. **Sample size too small**: Need larger cohort for robust signal
4. **Wrong parameters**: Ensure null trained with exact same params as real

**Solution**: Review real model performance first, then consider increasing sample size.

##### Bootstrap null calibration runs but produces no significant variants

Check these first:

1. The real model AUC. If it is below about `0.55`, there may be little signal to detect.
2. The `at_resolution_floor` flag in the rank-calibrated output. If zero variants hit the floor, the real ranking is not separating clearly from the null bootstrap ensemble.
3. The summary YAML `top_k_analysis` entries. Low KS statistics at `k = 100` usually mean the real and null rank distributions still overlap heavily near the top of the ranking.

##### Bootstrap runs out of memory

The bootstrap stores one null-rank value per tested variant per replicate. On large runs, `n_variants x n_bootstrap` can exceed RAM.

- Reduce `--n-bootstrap` from `1000` to `500` to halve storage.
- Use `--memmap-dir /path/to/fast/disk` to place the memmap-backed rank matrix on fast local storage. This flag controls where the backing file is created; it does not switch the matrix between in-memory and on-disk modes.

##### Bootstrap saturates only a few cores

If the wall-clock time scales poorly with `--n-jobs`, confirm the script is capping BLAS threads at startup (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS` all set to `1`). Without those caps, BLAS threads compete with `joblib` workers and reduce parallel efficiency.

---

#### Interpretation Issues

##### All top variants in the same gene

**Is this a problem?**
- Depends! If studying a Mendelian disease, this is expected
- For complex diseases, expect multiple genes
- Check: Is the gene biologically relevant to your phenotype?

**Possible issue**: Overfitting to one gene
- Solution: Check cross-validation stability
- Look at fold-specific rankings - is it consistent?

##### chrX dominates top rankings

**Possible causes**:
- Ploidy differences (hemizygosity) inflate chrX attributions
- Sex imbalance across case/control groups

**Solutions**:
1. Run sex-aware preprocessing (`infer_sex.py` → `preprocess.py --sex-map`)
2. Check sex balance (`check_sex_balance.py`)
3. Apply post-hoc correction (`correct_chrx_bias.py`)

**Note**: `correct_chrx_bias.py` excludes sex chromosomes by default; use `--include-sex-chroms` if you need chrX/chrY retained.
4. Re-run the ablation comparison ranked by `delta_rank`, the primary ranking metric, which holds the chromosome X share of the top-ranked set at 3 to 8 per cent:
   ```bash
   python scripts/compare_ablation_rankings.py \
       --ranking-dir results/ablation/rank_calibrated_rankings \
       --score-column delta_rank \
       --out-comparison delta_ablation_ranking_comparison.yaml
   ```

##### Ablation comparison with `--score-column delta_rank` gives Jaccard values of `1.0` across all level pairs

This usually means `delta_rank` is being sorted in the wrong direction, so the comparison is selecting the most demoted variants rather than the most promoted ones. Confirm you are on a version where `_score_column_is_ascending("delta_rank")` returns `False`, then rerun:

```bash
pytest tests/test_compare_ablation_rankings.py -v
```

##### Very low attributions overall

**Possible causes**:
- Model has low confidence (AUC close to 0.5)
- Embedding sparsity regularisation too strong (reduce `--lambda-attr`)
- Integration steps too low (increase `--n-steps`)

**Solution**:
1. Check model performance first
2. If AUC is good but attributions low, increase `--n-steps` to 100

##### Case-control differences all near zero

**Meaning**: Variants affect cases and controls similarly

**Interpretation**:
- May indicate population stratification (batch effects)
- Or: Model learned overall variant burden, not disease-specific patterns

**Solution**:
- Check for population structure (PCA analysis)
- Consider adjusting for covariates in future version

---

## FAQ

#### General Questions

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

#### Technical Questions

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

#### Scientific Questions

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

#### Troubleshooting Questions

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

## Appendices

## Appendix A: Model Architecture Details

This appendix explains the mathematical structure implemented by SIEVE. The
order follows the workflow of the model: sparse variant encoding, positional
attention, gene aggregation, phenotype prediction, training, explanation, and
interaction validation.

### 1. Sparse Variant Representation

For sample $n$, SIEVE keeps only non-reference variants:

$$
S_n = \{(x_{nv}, \mathrm{pos}_v, \mathrm{chrom}_v, g(v)) :
d_{nv} > 0\}
$$

where $x_{nv}$ is the encoded feature vector, $\mathrm{pos}_v$ is the
chromosome-local coordinate, $\mathrm{chrom}_v$ is the chromosome identifier,
$g(v)$ is the gene index, and $d_{nv}$ is genotype dosage. This sparse
representation avoids materialising a genome-wide dense tensor.

### 2. Annotation Levels

Each variant is encoded at one of five ablation levels:

$$
\begin{aligned}
x_v^{L0} &= [d_v] \\
x_v^{L1} &= [d_v, \mathrm{PE}(\mathrm{pos}_v)] \\
x_v^{L2} &= [d_v, \mathrm{PE}(\mathrm{pos}_v), c_v] \\
x_v^{L3} &= [d_v, \mathrm{PE}(\mathrm{pos}_v), c_v,
              \mathrm{sift}^{\mathrm{norm}}_v,
              \mathrm{polyphen}^{\mathrm{norm}}_v] \\
x_v^{L4} &= x_v^{L3}
\end{aligned}
$$

Current feature dimensions are:

| Level | Features | Dimension |
|-------|----------|-----------|
| L0 | genotype dosage | 1 |
| L1 | L0 + 64-dimensional position encoding | 65 |
| L2 | L1 + 4-dimensional consequence one-hot vector | 69 |
| L3 | L2 + SIFT + PolyPhen | 71 |
| L4 | currently identical to L3; reserved for future features | 71 |

SIFT is inverted so that larger values mean more deleterious. PolyPhen is kept
on the same direction:

$$
\mathrm{sift}^{\mathrm{norm}}_v = 1-\mathrm{sift}^{\mathrm{raw}}_v,
\qquad
\mathrm{polyphen}^{\mathrm{norm}}_v =
\mathrm{polyphen}^{\mathrm{raw}}_v
$$

Missing functional scores are imputed to the neutral value $0.5$.

### 3. Position Encoding

For L1-L4, chromosome-local coordinates are converted to sinusoidal features:

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

with $d=64$. The model also builds chromosome indices. During attention,
chromosome embeddings can be added to variant embeddings, and
cross-chromosome pairs are routed to a dedicated learned relative-position
bias bucket. Cross-chromosome attention is not masked; the separate bucket only
prevents coordinate differences between chromosomes from being treated as
within-chromosome distances.

### 4. Variant Encoder

The input feature vector is projected to the latent dimension by a two-layer
MLP:

$$
h_v =
\mathrm{Linear}_2\!\left(
  \mathrm{Dropout}\!\left(
    \mathrm{LayerNorm}\!\left(
      \mathrm{ReLU}(\mathrm{Linear}_1(x_v))
    \right)
  \right)
\right)
$$

If chromosome embeddings are enabled, the attention input is:

$$
\tilde{h}_v = h_v + e_{\mathrm{chrom}(v)}
$$

### 5. Position-Aware Self-Attention

Attention is computed among the variant-present positions within each sample and
is dense over that set. Cost is quadratic in the number of variants a sample
carries, not in the number of genomic positions, which is what makes the
computation tractable. Attention scores carry a learnable relative-position bias
indexed by T5-style logarithmic distance buckets: small within-chromosome
distances receive near-exact buckets, larger distances are logarithmically
compressed up to a maximum of 100,000 bases, and a dedicated bucket indexes
cross-chromosome pairs. Cross-chromosome attention is not masked; the separate
bucket prevents coordinate differences between chromosomes from being read as
within-chromosome distances.

The sparsity in SIEVE is therefore a property of the input representation, which
materialises only the alternate-allele sites each individual carries, and not a
property of the attention pattern: there is no fixed sparsity mask, no block
structure, and no local window.

For each attention layer:

$$
Q = \tilde{H}W_Q,\qquad K=\tilde{H}W_K,\qquad V=\tilde{H}W_V
$$

Within a chromosome, relative position is bucketed from
$r_{ij}=\mathrm{pos}_i-\mathrm{pos}_j$. Small distances use near-exact buckets;
larger distances use logarithmic buckets up to the configured maximum distance.
For variants on different chromosomes, a dedicated cross-chromosome bucket is
used.

The attention score is:

$$
s_{ijh} =
\frac{Q_{ih}\cdot K_{jh}}{\sqrt{d_{\mathrm{head}}}}
+ \beta_{\mathrm{bucket}(i,j),h}
$$

Padding is masked on the key side, then attention is:

$$
\alpha_{ijh}=\mathrm{softmax}_j(s_{ijh}),\qquad
a_{ih}=\sum_j \alpha_{ijh}V_{jh}
$$

Each layer applies output projection, residual connection, and layer
normalisation:

$$
H_{\mathrm{next}} =
\mathrm{LayerNorm}\!\left(H + \mathrm{Attention}(H)\right)
$$

### 6. Gene Aggregation

After attention, every variant embedding is pooled into its assigned gene.
Supported aggregations are:

$$
\begin{aligned}
E^{\max}_{gk} &= \max_{v:g(v)=g} H_{vk} \\
E^{\mathrm{mean}}_{gk}
&= \frac{1}{|\{v:g(v)=g\}|}\sum_{v:g(v)=g}H_{vk} \\
E^{\mathrm{sum}}_{gk} &= \sum_{v:g(v)=g}H_{vk}
\end{aligned}
$$

Genes without observed variants receive zero embeddings. In chunked training,
each chunk produces a gene embedding matrix. The default wrapper averages
non-zero gene embeddings across chunks:

$$
E_{ng} =
\frac{\sum_c E_{ncg}}
     {\#\{c:\|E_{ncg}\|_2 > 0\}}
$$

### 7. Phenotype Classifier

The classifier flattens the gene embedding matrix and optionally concatenates
sample-level covariates such as sex or ancestry principal components:

$$
u_n = [\mathrm{vec}(E_n), c_n]
$$

The logit and predicted case probability are:

$$
\begin{aligned}
z_n &= W_2\,\mathrm{Dropout}\!\left(
  \mathrm{ReLU}(W_1u_n+b_1)
\right)+b_2 \\
p_n &= \sigma(z_n)
\end{aligned}
$$

### 8. Training Objective

The classification term is binary cross-entropy with logits:

$$
\mathcal{L}_{\mathrm{BCE}}(z,y)
= -y\log\sigma(z) - (1-y)\log(1-\sigma(z))
$$

Optional class weighting uses the positive-class weight:

$$
w_+ = \frac{n_{\mathrm{total}}}{2n_+}
$$

When `--lambda-attr` is greater than zero, SIEVE adds embedding sparsity
regularisation:

$$
\mathcal{L}_{\mathrm{total}}
= \mathcal{L}_{\mathrm{BCE}}
+ \lambda_{\mathrm{attr}}\mathcal{L}_{\mathrm{sparse}}
$$

The public result key is still named `attribution_sparsity` for backward
compatibility, but the implemented regulariser is not gradient entropy and does
not compute Integrated Gradients during training. Integrated gradients are
computed only in the explain step, on the best-validation-area-under-the-curve
checkpoint, never during training.

For non-chunked batches, the sparsity term is the mean normalised sum of
variant embedding L2 norms:

$$
\begin{aligned}
m_{nv} &= \|H_{nv}\|_2 \\
\mathcal{L}^{(n)}_{\mathrm{sparse}}
&= \frac{\sum_v m_{nv}M_{nv}}
        {\max(1,\sum_v M_{nv})} \\
\mathcal{L}_{\mathrm{sparse}}
&= \frac{1}{N}\sum_n \mathcal{L}^{(n)}_{\mathrm{sparse}}
\end{aligned}
$$

For chunked training, the same idea is applied to aggregated gene embeddings:

$$
\begin{aligned}
m_{ng} &= \|E_{ng}\|_2 \\
\mathcal{L}^{(n)}_{\mathrm{sparse}}
&= \frac{\sum_g m_{ng}}
        {\max(1,\#\{g:m_{ng}>0\})} \\
\mathcal{L}_{\mathrm{sparse}}
&= \frac{1}{N}\sum_n \mathcal{L}^{(n)}_{\mathrm{sparse}}
\end{aligned}
$$

### 9. Explanation

After training, variant attributions are computed with Integrated Gradients
using a zero feature baseline:

$$
\mathrm{IG}_k(x)
= (x_k-x'_k)
\int_0^1
\frac{\partial F(x' + \alpha(x-x'))}{\partial x_k}
\,d\alpha,\qquad x'=0
$$

Feature-level attributions are collapsed to a variant score, usually by the L2
norm:

$$
\mathrm{score}_v=\|\mathrm{IG}_v\|_2
$$

Population-level rankings aggregate these scores across carrier samples, for
example by mean attribution:

$$
\bar{a}_v =
\frac{1}{N_v}\sum_{n:v\in S_n}\mathrm{score}_{nv}
$$

### 10. Epistasis Validation

Attention-derived pairs are hypotheses. Counterfactual validation evaluates
the model with both variants present, each variant removed alone, and both
variants removed:

$$
\Delta_{ij} = p_{11}-p_{10}-p_{01}+p_{00}
$$

where $p_{11}$ is the original prediction, $p_{10}$ removes variant $j$,
$p_{01}$ removes variant $i$, and $p_{00}$ removes both. A non-zero
$\Delta_{ij}$ means the joint model effect is not additive under this
counterfactual perturbation.

### 11. Model Complexity and Scalability

The selected configuration is:

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-5 |
| `lambda_attr` | 0.1 |
| `latent_dim` | 32 |
| `hidden_dim` | 64 |
| Attention layers | 1 |
| Attention heads | 4 |
| Classifier head | flatten |
| Flatten input dimension | 16,089 genes x 32 + 1 sex covariate = 514,849 |

Two points about the dimensions are worth stating explicitly, since both have
been misread.

`latent_dim` is the model-wide working dimension. It is the output width of the
variant encoder, and it carries through attention, gene aggregation and the
classifier input.

`hidden_dim` controls only the intermediate layer inside the variant-encoder
multilayer perceptron (the width between $\mathrm{Linear}_1$ and
$\mathrm{Linear}_2$ in section 4). It does not set the attention width.

#### Memory

Across three cohorts spanning 1,968 to 3,420 samples, and holding `chunk_size`
and `batch_size` fixed, peak GPU memory held at a plateau of roughly 18 GB under
the chunked configuration, which processes a selectable number of variants at a
time.

The plateau holds across cohort size, not across settings. Chunking bounds the
resident working set by `chunk_size` rather than by the number of variants a
sample carries, so adding samples does not raise the peak. Raising
`--chunk-size` or `--batch-size` does raise it, and lowering either lowers it,
so those two flags are what to change when fitting the run to a particular
card.

## Appendix B: Experimental Protocol

#### Overview

This appendix describes the rigorous experimental protocol for evaluating SIEVE. The experiments are designed to answer specific scientific questions rather than just demonstrate technical capability.

#### Scientific Questions

##### Question 1: Can deep learning discover variants that annotation-based methods miss?

**Hypothesis**: Models trained with minimal annotations will identify some disease-associated variants that models using current functional annotations rank lower, because annotation-informed models may over-rely on prior knowledge.

**Experiment**: Annotation ablation study comparing variant rankings across annotation levels L0-L3. L4 is currently identical to L3 and can be run only as a compatibility check.

##### Question 2: Do spatial relationships between variants carry disease signal?

**Hypothesis**: Position-aware models will outperform position-agnostic models (pure deep sets) on classification, and attention weights will show meaningful positional patterns (e.g., clustering of important variants).

**Experiment**: Compare SIEVE (with position-aware attention) against a DeepRVAT-style deep set baseline.

##### Question 3: Does embedding-sparsity-regularised training improve discovery?

**Hypothesis**: Models trained with embedding sparsity regularisation will produce more stable and biologically meaningful variant rankings than models trained with classification loss alone.

**Experiment**: Compare variant rankings between models with λ_attr = 0 vs λ_attr > 0.

##### Question 4: Can we detect and validate epistatic interactions?

**Hypothesis**: Attention patterns will identify variant pairs with non-additive effects, validated through counterfactual perturbation.

**Experiment**: Identify high-attention variant pairs, test for epistasis via counterfactual analysis, compare with known gene-gene interactions if available.

#### Experimental Design

##### Data Requirements

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

##### Cross-Validation Strategy

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

##### Evaluation Metrics

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

#### Experiment 1: Annotation Ablation Study

##### Purpose

Determine whether models with minimal annotations can discover variants that annotation-heavy models miss, testing the hypothesis that deep learning can find patterns beyond what prior knowledge encodes.

##### Protocol

1. **Train replicate models at each implemented annotation level** (L0 through L3 as the primary comparison; L4 is currently identical to L3 and is retained as a compatibility placeholder) using identical architecture and hyperparameters except for input dimension

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

##### Expected Outcomes

**If hypothesis is supported**:
- L0 model achieves reasonable (>0.6) AUC, showing genotype patterns alone carry signal
- Some L0-specific variants are not captured by standard annotation methods
- These variants may point to novel disease mechanisms

**If hypothesis is refuted**:
- L0 model fails to learn (AUC ~0.5), suggesting annotations are necessary
- All high-ranking variants at L0 are a subset of L3 rankings
- This would still be informative: it means the ranking for this phenotype is carried by the supplied annotations rather than by genome structure

#### Experiment 2: Position-Aware vs Position-Agnostic

##### Purpose

Test whether spatial relationships between variants carry disease-relevant information by comparing position-aware self-attention against permutation-invariant deep sets.

##### Protocol

1. **Implement two model variants**:
   - SIEVE (position-aware): Full model with positional encodings and relative position bias
   - DeepSet baseline: Same architecture but without positional information

2. **Train both models** on identical data with identical hyperparameters

3. **Compare classification performance**: AUC, sensitivity, specificity

4. **Analyse attention patterns** (SIEVE only):
   - Distribution of distances between high-attention variant pairs
   - Are nearby variants (potential compound heterozygosity) attended together?

##### Expected Outcomes

**If position matters**:
- SIEVE outperforms DeepSet baseline by >2% AUC
- Attention weights show non-uniform distance distribution
- High-attention pairs are enriched for same-exon or functional domain

#### Experiment 3: Embedding Sparsity Regularisation Study

##### Purpose

Determine whether training with embedding sparsity regularisation improves the stability and biological meaningfulness of discovered variants.

##### Protocol

1. **Train models with varying λ_attr**: 0, 0.01, 0.05, 0.1, 0.2, 0.5

2. **For each λ_attr**, evaluate:
   - Classification performance (AUC)
   - Embedding concentration during training and attribution concentration after explanation
   - Ranking stability: Jaccard similarity of top 100 variants across CV folds
   - Biological enrichment: KEGG/Reactome pathway p-values

3. **Select optimal λ_attr** that balances classification performance with interpretability

##### Expected Outcomes

**If regularisation helps**:
- Models with moderate λ_attr (0.05-0.1) have similar AUC but higher ranking stability
- Top variants are more concentrated and rankings are more stable
- Pathway enrichment p-values are lower (more meaningful discoveries)

#### Experiment 4: Epistasis Detection and Validation

##### Purpose

Test whether the model captures genuine epistatic (non-additive) interactions between variants.

##### Protocol

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

##### Distinguishing True Epistasis from Artefacts

Several artefacts can mimic epistasis:

**Linkage disequilibrium**: Variants in LD are inherited together, so their "interaction" may just reflect a single haplotype effect.
- **Control**: Check r² between variant pairs. Exclude pairs with r² > 0.2.

**Main effect masking**: Strong main effects can create apparent interactions.
- **Control**: Include main effects in baseline model; test interaction as additional term.

**Population stratification**: Different populations may have different allele frequencies and disease rates.
- **Control**: Include principal components as covariates; stratify analysis by ancestry.

#### Baseline Comparisons

##### External Baselines

1. **Standard GWAS**: Run single-variant association using PLINK or equivalent
2. **Burden test**: Gene-level rare variant burden test (SKAT-O or equivalent)
3. **Existing DL methods** (if feasible): DeepRVAT, GenNet

##### Internal Baselines

1. **Logistic regression on gene burdens**: Simple, interpretable baseline
2. **Random forest on variant presence**: Non-linear baseline without deep learning

#### Reporting Standards

##### For Each Experiment

Report:
- Sample sizes (n_cases, n_controls, n_variants)
- Cross-validation scheme and number of folds
- Hyperparameters and how they were selected
- Mean ± std of all metrics across outer CV folds
- Statistical tests and p-values with correction method
- Compute time and hardware used

##### Figures to Generate

1. **Annotation ablation**: Heatmap of Jaccard similarities between levels
2. **Position-aware comparison**: ROC curves for SIEVE vs DeepSet
3. **Embedding sparsity regularisation**: Pareto plot of AUC vs stability for different λ_attr
4. **Epistasis**: Network diagram of significant epistatic pairs
5. **Biological validation**: Pathway enrichment bar plot

---

## Appendix C: Method References

This appendix lists key methodological references that motivate recent pipeline updates.

#### Sex inference and ploidy-aware encoding

- **X-chromosome inbreeding coefficient (F-statistic)** for genetic sex inference:  
  Purcell S, et al. (2007). *PLINK: a tool set for whole-genome association and population-based linkage analyses.* **American Journal of Human Genetics**, 81(3):559–575.
- **X-chromosome association and pseudoautosomal regions (PAR)**:  
  Clayton DG. (2008). *Testing for association on the X chromosome.* **Biostatistics**, 9(4):593–600.

#### Attribution and interpretability

- **Integrated gradients** for feature attribution in deep networks:  
  Sundararajan M, Taly A, Yan Q. (2017). *Axiomatic Attribution for Deep Networks.* **ICML**.

#### Attention mechanisms

- **Scaled dot-product attention** for modeling interactions:
  Vaswani A, et al. (2017). *Attention Is All You Need.* **NeurIPS**.

#### Epistasis detection

- **EpiDetect/EpiCID** for network analysis of epistatic interactions:
  Mastropietro A, Markopoulos G, Evangelou E, Anagnostopoulos A. (2026). *A novel explainable deep-learning approach for network analysis of epistatic interactions.* **NAR Genomics and Bioinformatics**, 8(1):lqag004.
- **EpiCID interaction scoring** derives neural feature vectors from learned weights, uses cosine similarity to filter marginal-effect-dominated pairs, and reports first-layer interaction influence as the core explainability signal.
- **Scope and comparison to SIEVE**: EpiDetect was demonstrated on UK Biobank blood-pressure traits using pre-selected GWAS-significant common SNPs, whereas SIEVE works on common and rare variants directly from VCF-derived tensors and emphasizes sample-level attributions plus counterfactual validation rather than global weight-space similarity alone.

---
