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

# Compare attribution rankings
mkdir -p results/ablation/rankings
for LEVEL in L0 L1 L2 L3; do
    cp results/null_baseline_${LEVEL}/results/attribution_comparison/variant_rankings_with_significance.csv \
       results/ablation/rankings/${LEVEL}_sieve_variant_rankings.csv
done

python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rankings \
    --score-column empirical_p_variant \
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
    --rankings L0:results/null_baseline_L0/results/attribution_comparison/variant_rankings_with_significance.csv \
               L1:results/null_baseline_L1/results/attribution_comparison/variant_rankings_with_significance.csv \
               L2:results/null_baseline_L2/results/attribution_comparison/variant_rankings_with_significance.csv \
               L3:results/null_baseline_L3/results/attribution_comparison/variant_rankings_with_significance.csv \
    --score-column empirical_p_variant \
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

The ablation comparison should operate on the null-contrasted significance files produced by `run_null_baseline_analysis.sh`, not on the standalone corrected rankings:

```bash
# 1. Copy null-contrasted significance files into a comparison directory
mkdir -p results/ablation/significance_rankings
for LEVEL in L0 L1 L2 L3; do
    cp results/null_baseline_${LEVEL}/results/attribution_comparison/variant_rankings_with_significance.csv \
       results/ablation/significance_rankings/${LEVEL}_sieve_variant_rankings.csv
done

# 2. Compare using the null-contrast empirical p-value ranking
python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/significance_rankings \
    --score-column empirical_p_variant \
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

---
