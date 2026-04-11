# Interpreting Results

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

For ablation comparison, use the chrX-corrected files `corrected_variant_rankings.csv` from `corrected/` and rank variants with `--score-column empirical_p_variant`. Lower empirical p-values are treated as better ranks automatically, so the cross-level comparison operates on null-compared evidence while also having cross-chromosome-normalised z-scores available.

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
   - ChrX correction is applied separately (via `correct_chrx_bias.py`) for ranking purposes

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

### Non-Linear Classifier Validation Results

The non-linear classifier validation tests whether the **pattern** of variation across SIEVE genes jointly discriminates cases from controls, beyond what a scalar burden sum can capture. See the [Validation](validation.md) chapter for full usage details.

#### Summary Table (`nonlinear_validation_summary.tsv`)

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

#### Decision Framework

| Observed AUC vs null | RF vs LR | Interpretation |
|---------------------|----------|----------------|
| Significant (p < 0.05) | RF >> LR | Non-linear multi-gene signal transfers to validation cohort |
| Significant (p < 0.05) | RF ≈ LR | Linear signal transfers (could be captured by PRS) |
| Not significant | — | Signal does not transfer at this level/top-k |

#### Heatmap (`nonlinear_validation_heatmap.png`)

Rows are ablation levels, columns are top-k values. Cell values show observed AUC with significance annotations:

- `*` = `fdr_bh < 0.05`
- No marker = not significant

Look for patterns: does signal concentrate at specific levels or top-k values? Consistent signal across levels suggests a robust discovery; signal only at L3 may indicate annotation dependence.

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
