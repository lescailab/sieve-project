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
| `--score-column` | str | None (auto-detect) | Column to rank variants by. Use `z_attribution` for chromosome-normalised rankings from `correct_chrx_bias.py` |

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
The summary file `cooccurrence_summary.yaml` distinguishes:

- `n_pairs_gte5_cooccur`: pairs with at least 5 joint carriers (`n11 >= 5`)
- `n_pairs_all_cells_gte5`: pairs where all four carrier states have at least 5 samples (`n11`, `n10`, `n01`, `n00`)

The second metric is the more relevant one for interaction analysis, because estimating a non-additive interaction effect requires support across all four states. In these field names, `gte5` means `>= 5`.

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

### generate_sieve_gene_list.py

```bash
python scripts/generate_sieve_gene_list.py [OPTIONS]
```

Aggregates variant-level SIEVE rankings to a gene-level TSV for cross-cohort burden validation.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--variant-rankings` | path | required | Corrected variant rankings CSV |
| `--output` | path | required | Output gene list TSV |
| `--score-column` | str | `z_attribution` | Column to use for scoring |
| `--exclude-sex-chroms` | flag | True | Exclude sex chromosome genes |
| `--include-sex-chroms` | flag | False | Include sex chromosome genes (overrides --exclude-sex-chroms) |
| `--min-null-threshold` | str | None | Only include genes with variants exceeding this null threshold (`p05`, `p01`, `p001`) |
| `--aggregation` | str | `max` | How to aggregate variant scores per gene: `max` or `mean` |
| `--ablation-level` | str | None | Prefix output filename with level label (e.g. `L0`) |

**Example**:
```bash
python scripts/generate_sieve_gene_list.py \
    --variant-rankings results/attribution_comparison/variant_rankings_corrected.csv \
    --output validation/sieve_gene_lists/sieve_genes.tsv \
    --score-column z_attribution \
    --aggregation max
```

---

### extract_validation_burden.py

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

### test_burden_enrichment.py

```bash
python scripts/test_burden_enrichment.py [OPTIONS]
```

Permutation-based enrichment test comparing SIEVE gene sets against random gene sets of the same size. Requires the gene burden matrix from `extract_validation_burden.py --compute-full-gene-matrix`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--burden-dir` | path | required | Directory with burden files from `extract_validation_burden.py` |
| `--gene-matrix` | path | None | Path to gene burden matrix Parquet (default: `<burden-dir>/gene_burden_matrix.parquet`) |
| `--background-genes` | path | None | Text file listing background gene symbols (default: all genes in matrix) |
| `--validation-vcf` | path | None | Validation VCF (only if background genes not available) |
| `--phenotypes` | path | None | Phenotype file (only if not derivable from burden files) |
| `--sieve-genes` | path | required | SIEVE gene list TSV |
| `--output-dir` | path | required | Output directory |
| `--n-permutations` | int | 10000 | Number of random gene set permutations |
| `--genome-build` | str | `GRCh37` | Genome build |
| `--seed` | int | 42 | Random seed |
| `--top-k` | int list | `50 100 200` | Gene set sizes to test |
| `--ablation-levels` | str list | None | Run analysis per ablation level |
| `--covariates` | path | None | TSV with covariates for logistic regression |
| `--consequence-types` | str list | `total` | Burden types to test: `total`, `missense`, `lof`, `synonymous` |

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
