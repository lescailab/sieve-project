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

If you used sex-aware preprocessing or observe chrX inflation in rankings, run `correct_chrx_bias.py` to standardise mean attributions per chromosome. The script adds:

- `z_attribution`: per-chromosome z-scored attribution
- `corrected_rank`: rank based on `z_attribution`
- `is_sex_chrom`: flags chrX/chrY variants

By default, the corrected rankings exclude sex chromosomes. Use `--include-sex-chroms` if you want to keep them in the output (they remain flagged).

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

#### Summary YAML (`comparison_summary.yaml`)

```yaml
thresholds:
  p_0.05: 0.152    # 95th percentile of null
  p_0.01: 0.238    # 99th percentile of null
  p_0.001: 0.394   # 99.9th percentile of null

distribution_comparison:
  real_mean: 0.089
  null_mean: 0.045
  ks_pvalue: 2.3e-145    # Distributions differ
  mannwhitney_pvalue: 1.1e-78

significance_counts:
  p_0.01:
    threshold: 0.238
    observed: 46        # Variants in real exceeding threshold
    expected: 10.0      # Expected by chance (1% of 1000)
    enrichment: 4.6     # 4.6× more than expected

interpretation:
  distributions_differ: true
  real_higher_than_null: true
  enrichment_at_p01: 4.6
  n_significant_p01: 46
```

**How to Interpret**:

1. **Check distributions differ** (KS p-value < 0.001):
   - ✓ Yes → Real model found signal
   - ✗ No → May need more data or different approach

2. **Check enrichment at p<0.01**:
   - **< 1.5×**: Weak, be cautious
   - **1.5-2×**: Moderate, validate carefully
   - **2-5×**: Strong, good confidence
   - **> 5×**: Very strong, high confidence

3. **Identify significant variants**:
   - Review `significant_variants_p01.csv`
   - These variants have attributions exceeding 99th percentile of null
   - Use these for biological validation

**Decision Framework**:

| Enrichment | KS p-value | Interpretation | Action |
|-----------|-----------|----------------|--------|
| > 2× | < 0.001 | Strong signal | Proceed to validation |
| 1.5-2× | < 0.01 | Moderate signal | Validate top hits carefully |
| < 1.5× | > 0.05 | Weak/no signal | Check data quality, increase sample size |

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
| `mean_attribution_at_specific_level` | Attribution score at the specific level |

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

---

