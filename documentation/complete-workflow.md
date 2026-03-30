# Complete Workflow

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
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  8. Cross-Cohort    │  Burden enrichment + non-linear
│     Validation      │  classifier test in independent cohorts
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

#### Step 8: Cross-Cohort Gene-Set Burden Validation

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
| `variant_rankings_corrected.csv` | Step 4 → `correct_chrx_bias.py` | ChrX-corrected, null-compared variant rankings |
| `sieve_gene_rankings.csv` | Step 3 → `explain.py` | Gene-level attribution rankings |
| `L{0..3}_variant_rankings_corrected.csv` | Steps 2-5 (per level) | Per-ablation-level corrected rankings (optional) |

**This step has three sub-steps**: generate the gene list, extract burden counts from the validation VCF, and test for enrichment against a permutation null.

---

##### Step 8a: Generate SIEVE Gene List

**Purpose**: Aggregate variant-level rankings to a standardised gene-level TSV suitable for burden testing.

**Command**:
```bash
python scripts/generate_sieve_gene_list.py \
    --variant-rankings results/attribution_comparison/variant_rankings_corrected.csv \
    --output validation/sieve_gene_lists/sieve_genes.tsv \
    --score-column z_attribution \
    --exclude-sex-chroms \
    --aggregation max
```

This takes the chrX-corrected variant rankings and produces a ranked gene list where each gene's score is the maximum `z_attribution` across its variants. Sex chromosome genes are excluded by default (consistent with the corrected rankings).

**To generate per-ablation-level gene lists** (for testing whether different annotation levels replicate differently):
```bash
for level in L0 L1 L2 L3; do
    python scripts/generate_sieve_gene_list.py \
        --variant-rankings results/${level}_attribution_comparison/variant_rankings_corrected.csv \
        --output validation/sieve_gene_lists/sieve_genes.tsv \
        --ablation-level ${level} \
        --score-column z_attribution \
        --aggregation max
done
```

This produces `L0_sieve_genes.tsv`, `L1_sieve_genes.tsv`, etc.

**Optional: filter to null-significant genes only**:
```bash
python scripts/generate_sieve_gene_list.py \
    --variant-rankings results/attribution_comparison/variant_rankings_corrected.csv \
    --output validation/sieve_gene_lists/sieve_genes_sig.tsv \
    --min-null-threshold p01 \
    --aggregation max
```

This retains only genes containing at least one variant exceeding the null model's 99th percentile.

**Output format** (`sieve_genes.tsv`):
```
gene_name    gene_rank    gene_score    n_variants    chromosome
CUL3         1            3.45          5             2
AP2A1        2            3.21          3             19
NEXN         3            2.98          2             1
...
```

---

##### Step 8b: Extract Burden Counts from Validation VCF

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
| `--from-variant-rankings` | If passing the raw `variant_rankings_corrected.csv` instead of a pre-generated gene list |

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

##### Step 8c: Test Burden Enrichment

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

##### Step 8d: Non-Linear Classifier Validation

**Purpose**: Test whether the SIEVE gene set carries *non-linear* discriminative signal — combinatorial patterns across genes that a scalar burden sum would destroy.

**Why this step?** The scalar burden test (Step 8c) asks whether SIEVE genes have more total exonic variation in cases. But SIEVE's core claim is that the **pattern** of variation across genes matters, not just the total count. A random forest trained on per-gene burden counts preserves this multi-gene structure.

**Command** (all levels at once):
```bash
python scripts/validate_nonlinear_classifier.py \
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --sieve-genes validation/sieve_gene_lists/ \
    --phenotypes /path/to/validation_phenotypes.tsv \
    --output-dir validation/cohort_b/nonlinear_validation \
    --top-k 50 100 200 500 \
    --n-permutations 1000 \
    --classifiers both \
    --n-jobs 4 \
    --seed 42
```

**How it works**:
1. For each ablation level and top-k threshold, extracts the per-gene burden sub-matrix for the SIEVE gene set
2. Trains a random forest using repeated stratified CV with fixed fold assignments
3. Generates a null distribution by repeating the same procedure on 1,000 random gene sets of equal size
4. Reports an empirical p-value and, when `--classifiers both` is used, compares RF vs logistic regression to test for non-linear structure

**Outputs**:
```
validation/cohort_b/nonlinear_validation/
├── nonlinear_validation_L{0..3}_topK{k}.yaml   # Full results per combination
├── null_aucs_L{0..3}_topK{k}.npz               # Null distributions
├── validation_plot_L{0..3}_topK{k}.png          # Diagnostic plots
├── nonlinear_validation_summary.tsv             # Summary table
├── nonlinear_validation_heatmap.png             # AUC heatmap across levels x top-k
└── nonlinear_validation_report.md               # Human-readable report
```

**Interpreting the results**: see the [Validation](validation.md) chapter for detailed guidance.

!!! tip "Start with quick exploration"
    Use `--n-permutations 200` for a fast initial run. Once you identify the most promising level/top-k combinations, re-run with `--n-permutations 1000` for publication-quality results.

---

##### Step 8e: Visualise Scalar Burden Results

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

##### Complete Step 8 Example

Putting it all together for two validation cohorts:

```bash
# --- Gene list from discovery cohort ---
python scripts/generate_sieve_gene_list.py \
    --variant-rankings results/attribution_comparison/variant_rankings_corrected.csv \
    --output validation/sieve_gene_lists/sieve_genes.tsv \
    --score-column z_attribution \
    --aggregation max

# --- Cohort B ---
python scripts/extract_validation_burden.py \
    --vcf /data/cohort_b/cohort.vcf.gz \
    --phenotypes /data/cohort_b/phenotypes.tsv \
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
    --vcf /data/cohort_c/cohort.vcf.gz \
    --phenotypes /data/cohort_c/phenotypes.tsv \
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
    --burden-matrix validation/cohort_b/gene_burden_matrix.parquet \
    --sieve-genes validation/sieve_gene_lists/ \
    --phenotypes /path/to/cohort_b_phenotypes.tsv \
    --output-dir validation/cohort_b/nonlinear_validation \
    --top-k 50 100 200 \
    --n-permutations 1000 \
    --classifiers both \
    --n-jobs 4

# --- Non-linear classifier validation (Cohort C) ---
python scripts/validate_nonlinear_classifier.py \
    --burden-matrix validation/cohort_c/gene_burden_matrix.parquet \
    --sieve-genes validation/sieve_gene_lists/ \
    --phenotypes /path/to/cohort_c_phenotypes.tsv \
    --output-dir validation/cohort_c/nonlinear_validation \
    --top-k 50 100 200 \
    --n-permutations 1000 \
    --classifiers both \
    --n-jobs 4

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
