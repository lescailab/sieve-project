# SIEVE Conda Package Usage

This page describes how to run the SIEVE pipeline when installed as a conda package (without calling `python scripts/...` directly).

## Install

### Option A: Install from a published channel

```bash
conda create -n sieve python=3.10
conda activate sieve
conda install -c <your-channel> sieve
```

### Option B: Build and install locally from this repository

```bash
conda create -n sieve-build -c conda-forge python=3.11 conda>=26 conda-build>=26
conda activate sieve-build
CONDA_SOLVER=libmamba conda build conda --no-anaconda-upload --croot /tmp/sieve-conda-bld
conda install -n sieve -c file:///tmp/sieve-conda-bld sieve
```

## Verify commands are available

```bash
sieve-preprocess --help
sieve-train --help
sieve-explain --help
```

## CLI commands exposed by the package

| Pipeline stage | Command |
|---|---|
| Sex inference (optional) | `sieve-infer-sex` |
| Sex balance QC (optional) | `sieve-check-sex-balance` |
| Data preprocessing | `sieve-preprocess` |
| Model training | `sieve-train` |
| Explainability | `sieve-explain` |
| Null data creation | `sieve-create-null-baseline` |
| Full null baseline wrapper | `sieve-run-null-baseline` |
| Real vs null comparison | `sieve-compare-attributions` |
| chrX post-hoc correction (optional) | `sieve-correct-chrx-bias` |
| Discovery validation (optional) | `sieve-validate-discoveries` |
| Epistasis validation (optional) | `sieve-validate-epistasis` |

## End-to-end example (conda-installed CLI mode)

### 1. Optional: infer sample sex

```bash
sieve-infer-sex \
    --vcf cohort.vcf.gz \
    --output-dir results/sex_inference \
    --genome-build GRCh37
```

### 2. Preprocess VCF

```bash
sieve-preprocess \
    --vcf cohort.vcf.gz \
    --phenotypes phenotypes.tsv \
    --output preprocessed.pt \
    --sex-map results/sex_inference/sample_sex.tsv \
    --genome-build GRCh37
```

### 3. Train model

```bash
sieve-train \
    --preprocessed-data preprocessed.pt \
    --level L3 \
    --experiment-name my_model \
    --output-dir experiments \
    --device cuda
```

### 4. Run explainability

```bash
sieve-explain \
    --experiment-dir experiments/my_model \
    --preprocessed-data preprocessed.pt \
    --output-dir results/explainability \
    --device cuda
```

### 5. Run null baseline

Use the full wrapper (preferred — cohort-centric layout):

```bash
PROJECT_DIR=/data/CohortName \
LEVEL=L3 \
DEVICE="cuda" \
sieve-run-null-baseline
```

Or use the legacy variable interface:

```bash
export INPUT_DATA="preprocessed.pt"
export REAL_EXPERIMENT="experiments/my_model"
export REAL_RESULTS="results/explainability"
export OUTPUT_BASE="results/null_baseline_run"
export DEVICE="cuda"

sieve-run-null-baseline
```

Or run steps manually:

```bash
sieve-create-null-baseline --input preprocessed.pt --output preprocessed_NULL.pt --seed 42

sieve-train \
    --preprocessed-data preprocessed_NULL.pt \
    --level L3 \
    --experiment-name null_baseline \
    --output-dir experiments \
    --device cuda

sieve-explain \
    --experiment-dir experiments/null_baseline \
    --preprocessed-data preprocessed_NULL.pt \
    --output-dir results/null_attributions \
    --is-null-baseline \
    --device cuda

sieve-compare-attributions \
    --real results/explainability/sieve_variant_rankings.csv \
    --null results/null_attributions/sieve_variant_rankings.csv \
    --output-dir results/attribution_comparison
```

### 6. Optional post-processing and biological checks

```bash
# Apply chrX correction to the significance-annotated file (run AFTER step 5)
sieve-correct-chrx-bias \
    --rankings results/attribution_comparison/variant_rankings_with_significance.csv \
    --output-dir results/attribution_comparison/corrected \
    --include-sex-chroms

sieve-validate-discoveries \
    --variant-rankings results/explainability/sieve_variant_rankings.csv \
    --gene-rankings results/explainability/sieve_gene_rankings.csv \
    --output-dir results/validation \
    --clinvar data/clinvar.tsv \
    --gwas data/gwas_catalog.tsv
```

## Notes

- In conda package mode, prefer `sieve-*` commands instead of `python scripts/...`.
- The package dependencies are based on `pyproject.toml`; in some cases conda-specific compatibility constraints may differ from pip constraints.
- For GPU usage, ensure your conda environment includes a CUDA-compatible PyTorch build.
