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
