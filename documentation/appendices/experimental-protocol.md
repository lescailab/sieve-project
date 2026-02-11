# Appendix B: Experimental Protocol

### Overview

This appendix describes the rigorous experimental protocol for evaluating SIEVE. The experiments are designed to answer specific scientific questions rather than just demonstrate technical capability.

### Scientific Questions

#### Question 1: Can deep learning discover variants that annotation-based methods miss?

**Hypothesis**: Models trained with minimal annotations will identify some disease-associated variants that models using full annotations rank lower, because the annotation-heavy models may over-rely on prior knowledge.

**Experiment**: Annotation ablation study comparing variant rankings across annotation levels L0-L4.

#### Question 2: Do spatial relationships between variants carry disease signal?

**Hypothesis**: Position-aware models will outperform position-agnostic models (pure deep sets) on classification, and attention weights will show meaningful positional patterns (e.g., clustering of important variants).

**Experiment**: Compare SIEVE (with position-aware attention) against a DeepRVAT-style deep set baseline.

#### Question 3: Does attribution-regularised training improve discovery?

**Hypothesis**: Models trained with attribution sparsity loss will produce more stable and biologically meaningful variant rankings than models trained with classification loss alone.

**Experiment**: Compare variant rankings between models with λ_attr = 0 vs λ_attr > 0.

#### Question 4: Can we detect and validate epistatic interactions?

**Hypothesis**: Attention patterns will identify variant pairs with non-additive effects, validated through counterfactual perturbation.

**Experiment**: Identify high-attention variant pairs, test for epistasis via counterfactual analysis, compare with known gene-gene interactions if available.

### Experimental Design

#### Data Requirements

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

#### Cross-Validation Strategy

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

#### Evaluation Metrics

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

### Experiment 1: Annotation Ablation Study

#### Purpose

Determine whether models with minimal annotations can discover variants that annotation-heavy models miss, testing the hypothesis that deep learning can find patterns beyond what prior knowledge encodes.

#### Protocol

1. **Train 5 models at each annotation level** (L0 through L4) using identical architecture and hyperparameters except for input dimension

2. **For each model**, compute integrated gradients to obtain variant-level attribution scores

3. **Compare variant rankings** across annotation levels:
   - Top 100 variants at each level
   - Overlap analysis (Jaccard similarity)
   - Identify "L0-specific" variants: high rank at L0, low rank at L4
   - Identify "L4-specific" variants: high rank at L4, low rank at L0

4. **Biological interpretation**:
   - Are L0-specific variants in genes not annotated as pathogenic?
   - Are they enriched for regulatory regions or novel mechanisms?
   - Do L4-specific variants simply have high CADD/SIFT scores?

#### Expected Outcomes

**If hypothesis is supported**:
- L0 model achieves reasonable (>0.6) AUC, showing genotype patterns alone carry signal
- Some L0-specific variants are not captured by standard annotation methods
- These variants may point to novel disease mechanisms

**If hypothesis is refuted**:
- L0 model fails to learn (AUC ~0.5), suggesting annotations are necessary
- All high-ranking variants at L0 are subset of L4 rankings
- This would still be informative: it means annotation-free discovery is not feasible for this phenotype

### Experiment 2: Position-Aware vs Position-Agnostic

#### Purpose

Test whether spatial relationships between variants carry disease-relevant information by comparing position-aware sparse attention against permutation-invariant deep sets.

#### Protocol

1. **Implement two model variants**:
   - SIEVE (position-aware): Full model with positional encodings and relative position bias
   - DeepSet baseline: Same architecture but without positional information

2. **Train both models** on identical data with identical hyperparameters

3. **Compare classification performance**: AUC, sensitivity, specificity

4. **Analyse attention patterns** (SIEVE only):
   - Distribution of distances between high-attention variant pairs
   - Are nearby variants (potential compound heterozygosity) attended together?

#### Expected Outcomes

**If position matters**:
- SIEVE outperforms DeepSet baseline by >2% AUC
- Attention weights show non-uniform distance distribution
- High-attention pairs are enriched for same-exon or functional domain

### Experiment 3: Attribution Regularisation Study

#### Purpose

Determine whether training with attribution sparsity loss improves the stability and biological meaningfulness of discovered variants.

#### Protocol

1. **Train models with varying λ_attr**: 0, 0.01, 0.05, 0.1, 0.2, 0.5

2. **For each λ_attr**, evaluate:
   - Classification performance (AUC)
   - Attribution sparsity: entropy of normalised attributions
   - Ranking stability: Jaccard similarity of top 100 variants across CV folds
   - Biological enrichment: KEGG/Reactome pathway p-values

3. **Select optimal λ_attr** that balances classification performance with interpretability

#### Expected Outcomes

**If regularisation helps**:
- Models with moderate λ_attr (0.05-0.1) have similar AUC but higher ranking stability
- Top variants are more concentrated (lower entropy)
- Pathway enrichment p-values are lower (more meaningful discoveries)

### Experiment 4: Epistasis Detection and Validation

#### Purpose

Test whether the model captures genuine epistatic (non-additive) interactions between variants.

#### Protocol

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

#### Distinguishing True Epistasis from Artefacts

Several artefacts can mimic epistasis:

**Linkage disequilibrium**: Variants in LD are inherited together, so their "interaction" may just reflect a single haplotype effect.
- **Control**: Check r² between variant pairs. Exclude pairs with r² > 0.2.

**Main effect masking**: Strong main effects can create apparent interactions.
- **Control**: Include main effects in baseline model; test interaction as additional term.

**Population stratification**: Different populations may have different allele frequencies and disease rates.
- **Control**: Include principal components as covariates; stratify analysis by ancestry.

### Baseline Comparisons

#### External Baselines

1. **Standard GWAS**: Run single-variant association using PLINK or equivalent
2. **Burden test**: Gene-level rare variant burden test (SKAT-O or equivalent)
3. **Existing DL methods** (if feasible): DeepRVAT, GenNet

#### Internal Baselines

1. **Logistic regression on gene burdens**: Simple, interpretable baseline
2. **Random forest on variant presence**: Non-linear baseline without deep learning

### Reporting Standards

#### For Each Experiment

Report:
- Sample sizes (n_cases, n_controls, n_variants)
- Cross-validation scheme and number of folds
- Hyperparameters and how they were selected
- Mean ± std of all metrics across outer CV folds
- Statistical tests and p-values with correction method
- Compute time and hardware used

#### Figures to Generate

1. **Annotation ablation**: Heatmap of Jaccard similarities between levels
2. **Position-aware comparison**: ROC curves for SIEVE vs DeepSet
3. **Attribution regularisation**: Pareto plot of AUC vs stability for different λ_attr
4. **Epistasis**: Network diagram of significant epistatic pairs
5. **Biological validation**: Pathway enrichment bar plot

---

