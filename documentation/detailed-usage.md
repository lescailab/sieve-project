# Detailed Usage

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

- Column 1: `sample_id` (must match VCF exactly)
- Column 2: `phenotype` (**1 = control, 2 = case**)

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

# Compare attribution rankings (use chrX-corrected files, which contain z_attribution)
mkdir -p results/ablation/rankings
for LEVEL in L0 L1 L2 L3; do
    cp results/null_baseline_${LEVEL}/results/attribution_comparison/corrected/corrected_variant_rankings.csv \
       results/ablation/rankings/${LEVEL}_sieve_variant_rankings.csv
done

python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rankings \
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
# Use chrX-corrected files, which contain z_attribution
python scripts/compare_ablation_rankings.py \
    --rankings L0:results/null_baseline_L0/results/attribution_comparison/corrected/corrected_variant_rankings.csv \
               L1:results/null_baseline_L1/results/attribution_comparison/corrected/corrected_variant_rankings.csv \
               L2:results/null_baseline_L2/results/attribution_comparison/corrected/corrected_variant_rankings.csv \
               L3:results/null_baseline_L3/results/attribution_comparison/corrected/corrected_variant_rankings.csv \
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

#### Using Null-Contrasted Significance Rankings

The recommended ablation comparison uses the chrX-corrected files (`corrected_variant_rankings.csv`,
produced by `correct_chrx_bias.py`), which contain `z_attribution`. The `variant_rankings_with_significance.csv`
files from `run_null_baseline_analysis.sh` do not contain `z_attribution` and should be ranked by
`empirical_p_variant` if used directly (see KNOWN_LIMITATIONS.md for the resolution-floor caveat).

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

#### Rank-Based Null Calibration

After the magnitude-based null comparison, `bootstrap_null_calibration.py` runs a complementary rank-based null calibration that generates an ensemble of `B = 1000` null rankings by bootstrap-resampling the null's per-sample attributions. This adds per-variant empirical p-values and BH-FDR with resolution `1 / (B + 1)`, a per-gene Wilcoxon rank-sum test, top-k overlap and KS diagnostics, and a `delta_rank` column where positive values mean the real model promotes that variant relative to the bootstrap-null ensemble. The gene-stats CSV also carries a `gene_delta_rank` column computed as `max(delta_rank)` per gene by default (mirroring `gene_z_score = max(z_attribution)`), configurable via `--gene-delta-rank-aggregation`.

```bash
python scripts/bootstrap_null_calibration.py \
    --real-rankings results/<cohort>/real_experiments/L1/attributions/variant_rankings_with_significance.csv \
    --null-attributions results/<cohort>/null_baselines/L1/attributions/attributions.npz \
    --output results/<cohort>/real_experiments/L1/attributions/variant_rankings_rank_calibrated.csv \
    --n-bootstrap 1000 \
    --seed 42
```

#### Bootstrap-Calibrated Ablation Workflow

The bootstrap-calibrated file carries both the chrX-corrected `z_attribution` view and the bootstrap-informed `delta_rank` view. Run `compare_ablation_rankings.py` twice:

- `--score-column z_attribution`: preserves continuity with the existing chrX-corrected workflow and manuscript numbers
- `--score-column delta_rank`: adds a scale-free view that incorporates the null contribution explicitly

Concordance between the two Jaccard matrices strengthens the level-specific-discovery claim. Divergence is also informative: it tells you which discoveries depend mostly on the real-signal ordering versus the bootstrap-null contrast.

```bash
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

#### Non-Linear Classifier Robustness Run Pattern

The non-linear classifier validation (`validate_nonlinear_classifier.py`) also supports `--score-column delta_rank`, which resolves automatically to the `gene_delta_rank` column in the gene-stats CSV. The recommended workflow is to run the validation twice with separate output TSVs — one primary run using `--score-column z_attribution` and one robustness run using `--score-column delta_rank` — and apply Benjamini-Hochberg FDR independently within each invocation across the full 16-cell grid. Do not pool the two sets of p-values into a single FDR correction, as that would halve statistical power and obscure whether the robustness finding survives on its own.

#### Epistasis Aggregation Robustness Run Pattern

`aggregate_gene_interactions.py` accepts `--score-column z_attribution` (default,
preserves continuity) and `--score-column delta_rank` (bootstrap-informed view).
Run the aggregation twice with separate output directories and compare the
resulting gene-pair networks for stability of hub genes and top pairs:

```bash
# Primary view (chrX-corrected z-scores)
python scripts/aggregate_gene_interactions.py \
    --preprocessed-data preprocessed_<cohort>.pt \
    --variant-rankings results/<cohort>/real_experiments/L1/attributions/variant_rankings_rank_calibrated.csv \
    --gene-rankings results/<cohort>/real_experiments/L1/attributions/gene_rankings_with_significance.csv \
    --null-rankings results/<cohort>/null_baselines/L1/attributions/variant_rankings_with_significance.csv \
    --cooccurrence results/<cohort>/epistasis_audit/cooccurrence_per_pair.csv \
    --output-dir results/<cohort>/gene_interactions_z \
    --score-column z_attribution \
    --top-k-genes 50

# Robustness view (bootstrap-informed)
python scripts/aggregate_gene_interactions.py \
    --preprocessed-data preprocessed_<cohort>.pt \
    --variant-rankings results/<cohort>/real_experiments/L1/attributions/variant_rankings_rank_calibrated.csv \
    --gene-rankings results/<cohort>/real_experiments/L1/attributions/gene_rankings_with_significance.csv \
    --null-rankings results/<cohort>/null_baselines/L1/attributions/variant_rankings_with_significance.csv \
    --cooccurrence results/<cohort>/epistasis_audit/cooccurrence_per_pair.csv \
    --output-dir results/<cohort>/gene_interactions_delta \
    --score-column delta_rank \
    --top-k-genes 50 \
    --allow-nonsignificant-genes
```

Both runs require a rank-calibrated variant rankings CSV (output of
`bootstrap_null_calibration.py`) and a gene-stats CSV that carries the
`gene_delta_rank` column. The `--allow-nonsignificant-genes` flag on the
`delta_rank` run keeps the per-variant `exceeds_null_*` annotations in the
output as descriptive metadata but disables the floored bootstrap-p filter so
that gene selection and pair ranking are driven entirely by `delta_rank`. The
`z_attribution` run keeps the floored-p filter enabled by default for
continuity with earlier numbers.

---
