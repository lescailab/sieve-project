# Command Reference

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

