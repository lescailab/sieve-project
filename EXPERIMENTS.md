# EXPERIMENTS.md - Experimental Protocol

## Overview

This document describes the experimental protocol for evaluating SIEVE. The experiments are designed to answer specific scientific questions rather than just demonstrate technical capability.

## Scientific Questions

### Question 1: Can deep learning discover variants that annotation-based methods miss?

**Hypothesis**: Models trained with minimal annotations will identify some disease-associated variants that models using full annotations rank lower, because the annotation-heavy models may over-rely on prior knowledge.

**Experiment**: Annotation ablation study comparing variant rankings across annotation levels L0-L4.

### Question 2: Do spatial relationships between variants carry disease signal?

**Hypothesis**: Position-aware models will outperform position-agnostic models (pure deep sets) on classification, and attention weights will show meaningful positional patterns (e.g., clustering of important variants).

**Experiment**: Compare SIEVE (with position-aware attention) against a DeepRVAT-style deep set baseline.

### Question 3: Does attribution-regularized training improve discovery?

**Hypothesis**: Models trained with attribution sparsity loss will produce more stable and biologically meaningful variant rankings than models trained with classification loss alone.

**Experiment**: Compare variant rankings between models with λ_attr = 0 vs λ_attr > 0.

### Question 4: Can we detect and validate epistatic interactions?

**Hypothesis**: Attention patterns will identify variant pairs with non-additive effects, validated through counterfactual perturbation.

**Experiment**: Identify high-attention variant pairs, test for epistasis via counterfactual analysis, compare with known gene-gene interactions if available.

## Experimental Design

### Data Requirements

**Input data**:
- Multi-sample VCF file, annotated with VEP (CSQ field)
- Phenotype file: sample IDs with binary case/control labels
- Reference genome: GRCh37

**Minimum dataset size**:
- At least 500 samples (250 cases, 250 controls) for meaningful cross-validation
- Literature suggests >5,000 samples for robust epistasis detection

**Quality control** (applied before experiments):
- Remove samples with >5% missing genotypes
- Remove variants with >5% missing genotypes
- Remove variants with HWE p-value < 1e-6 in controls
- Optionally filter by MAF (but track this for annotation level effects)

### Cross-Validation Strategy

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

### Evaluation Metrics

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

## Experiment 1: Annotation Ablation Study

### Purpose

Determine whether models with minimal annotations can discover variants that annotation-heavy models miss, testing the hypothesis that deep learning can find patterns beyond what prior knowledge encodes.

### Protocol

1. **Train 5 models at each annotation level** (L0 through L4) using identical architecture and hyperparameters except for input dimension:
   - L0: Genotype only
   - L1: Genotype + position
   - L2: Genotype + position + consequence
   - L3: L2 + SIFT + PolyPhen
   - L4: Full annotations

2. **For each model**, compute integrated gradients to obtain variant-level attribution scores.

3. **Compare variant rankings** across annotation levels:
   - Top 100 variants at each level
   - Overlap analysis (Jaccard similarity)
   - Identify "L0-specific" variants: high rank at L0, low rank at L4
   - Identify "L4-specific" variants: high rank at L4, low rank at L0

4. **Biological interpretation**:
   - Are L0-specific variants in genes not annotated as pathogenic?
   - Are they enriched for regulatory regions or novel mechanisms?
   - Do L4-specific variants simply have high CADD/SIFT scores?

### Expected Outcomes

**If hypothesis is supported**:
- L0 model achieves reasonable (>0.6) AUC, showing genotype patterns alone carry signal
- Some L0-specific variants are not captured by standard annotation methods
- These variants may point to novel disease mechanisms

**If hypothesis is refuted**:
- L0 model fails to learn (AUC ~0.5), suggesting annotations are necessary
- All high-ranking variants at L0 are subset of L4 rankings
- This would still be informative: it means annotation-free discovery is not feasible for this phenotype

### Analysis Code

```python
def annotation_ablation_analysis(models_by_level, test_data):
    """
    Compare variant rankings across annotation levels.
    """
    rankings = {}
    
    for level, model in models_by_level.items():
        # Compute attributions
        attributions = compute_variant_attributions(model, test_data)
        
        # Rank variants by absolute attribution
        ranked_variants = attributions.abs().mean(dim=0).argsort(descending=True)
        rankings[level] = ranked_variants[:1000]  # Top 1000
    
    # Compute pairwise Jaccard similarity
    similarity_matrix = {}
    for l1 in rankings:
        for l2 in rankings:
            if l1 < l2:
                intersection = set(rankings[l1][:100].tolist()) & set(rankings[l2][:100].tolist())
                union = set(rankings[l1][:100].tolist()) | set(rankings[l2][:100].tolist())
                similarity_matrix[(l1, l2)] = len(intersection) / len(union)
    
    # Identify level-specific variants
    l0_specific = set(rankings['L0'][:100].tolist()) - set(rankings['L4'][:500].tolist())
    l4_specific = set(rankings['L4'][:100].tolist()) - set(rankings['L0'][:500].tolist())
    
    return {
        'rankings': rankings,
        'similarity': similarity_matrix,
        'l0_specific': l0_specific,
        'l4_specific': l4_specific
    }
```

## Experiment 2: Position-Aware vs Position-Agnostic

### Purpose

Test whether spatial relationships between variants carry disease-relevant information by comparing position-aware sparse attention against permutation-invariant deep sets.

### Protocol

1. **Implement two model variants**:
   - SIEVE (position-aware): Full model with positional encodings and relative position bias
   - DeepSet baseline: Same architecture but without positional information (no PE, no relative bias)

2. **Train both models** on identical data with identical hyperparameters.

3. **Compare classification performance**: AUC, sensitivity, specificity.

4. **Analyze attention patterns** (SIEVE only):
   - Distribution of distances between high-attention variant pairs
   - Are nearby variants (potential compound heterozygosity) attended together?
   - Are there consistent long-range patterns?

5. **Biological interpretation**:
   - Do high-attention pairs fall within same exon/domain?
   - Are they in known regulatory relationships?

### Expected Outcomes

**If position matters**:
- SIEVE outperforms DeepSet baseline by >2% AUC
- Attention weights show non-uniform distance distribution
- High-attention pairs are enriched for same-exon or functional domain

**If position doesn't matter**:
- Similar performance between SIEVE and DeepSet
- This suggests permutation-invariant aggregation is sufficient
- Position-aware attention adds complexity without benefit

## Experiment 3: Attribution Regularization Study

### Purpose

Determine whether training with attribution sparsity loss improves the stability and biological meaningfulness of discovered variants.

### Protocol

1. **Train models with varying λ_attr**: 0, 0.01, 0.05, 0.1, 0.2, 0.5

2. **For each λ_attr**, evaluate:
   - Classification performance (AUC)
   - Attribution sparsity: entropy of normalized attributions
   - Ranking stability: Jaccard similarity of top 100 variants across CV folds
   - Biological enrichment: KEGG/Reactome pathway p-values

3. **Select optimal λ_attr** that balances classification performance with interpretability.

### Analysis Metrics

```python
def evaluate_attribution_regularization(model, data, cv_folds):
    """
    Evaluate attribution quality metrics.
    """
    attributions_per_fold = []
    top_variants_per_fold = []
    
    for fold_data in cv_folds:
        attr = compute_variant_attributions(model, fold_data)
        attributions_per_fold.append(attr)
        top_variants_per_fold.append(attr.abs().mean(dim=0).argsort(descending=True)[:100])
    
    # Sparsity: average entropy of attributions
    sparsity = []
    for attr in attributions_per_fold:
        attr_normalized = attr.abs() / attr.abs().sum(dim=-1, keepdim=True)
        entropy = -(attr_normalized * torch.log(attr_normalized + 1e-10)).sum(dim=-1)
        sparsity.append(entropy.mean().item())
    
    # Stability: Jaccard similarity between folds
    stability = []
    for i in range(len(top_variants_per_fold)):
        for j in range(i+1, len(top_variants_per_fold)):
            intersection = set(top_variants_per_fold[i].tolist()) & set(top_variants_per_fold[j].tolist())
            union = set(top_variants_per_fold[i].tolist()) | set(top_variants_per_fold[j].tolist())
            stability.append(len(intersection) / len(union))
    
    return {
        'mean_sparsity': np.mean(sparsity),
        'mean_stability': np.mean(stability),
        'std_stability': np.std(stability)
    }
```

### Expected Outcomes

**If regularization helps**:
- Models with moderate λ_attr (0.05-0.1) have similar AUC but higher ranking stability
- Top variants are more concentrated (lower entropy)
- Pathway enrichment p-values are lower (more meaningful discoveries)

**If regularization hurts**:
- AUC drops significantly with any λ_attr > 0
- This suggests the phenotype requires distributed signal across many variants
- May indicate weak effect sizes or high polygenicity

## Experiment 4: Epistasis Detection and Validation

### Purpose

Test whether the model captures genuine epistatic (non-additive) interactions between variants.

### Protocol

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
   - Are they in linkage disequilibrium (LD)? (If so, may be LD artifact rather than true epistasis)

### Distinguishing True Epistasis from Artifacts

Several artifacts can mimic epistasis:

**Linkage disequilibrium**: Variants in LD are inherited together, so their "interaction" may just reflect a single haplotype effect.
- **Control**: Check r² between variant pairs. Exclude pairs with r² > 0.2.

**Main effect masking**: Strong main effects can create apparent interactions.
- **Control**: Include main effects in baseline model; test interaction as additional term.

**Population stratification**: Different populations may have different allele frequencies and disease rates.
- **Control**: Include principal components as covariates; stratify analysis by ancestry.

### Expected Outcomes

**If true epistasis is detected**:
- Significant epistatic pairs after multiple testing correction
- Pairs not in strong LD (r² < 0.2)
- Biological plausibility (same pathway, physical interaction)
- Replication in held-out fold or independent dataset

**If epistasis is not detected**:
- No pairs survive multiple testing correction
- This may indicate: (a) epistasis is rare for this phenotype, (b) sample size is insufficient, or (c) the model doesn't capture interactions well

## Baseline Comparisons

### External Baselines

1. **Standard GWAS**: Run single-variant association using PLINK or equivalent.
   - Compare: Do SIEVE top variants overlap with GWAS hits?
   - Identify: SIEVE-specific discoveries not reaching GWAS significance

2. **Burden test**: Gene-level rare variant burden test (SKAT-O or equivalent).
   - Compare: Do SIEVE gene rankings correlate with burden test p-values?

3. **Existing DL methods** (if feasible):
   - DeepRVAT: If their code is available and applicable
   - GenNet: If PLINK conversion is straightforward

### Internal Baselines

1. **Logistic regression on gene burdens**: Simple, interpretable baseline.
   - For each gene, count LOF + missense variants
   - Fit L1-regularized logistic regression
   - Compare AUC and top gene rankings

2. **Random forest on variant presence**: Non-linear baseline without deep learning.
   - Binary features: variant present/absent
   - Feature importance for variant ranking

## Reporting Standards

### For Each Experiment

Report:
- Sample sizes (n_cases, n_controls, n_variants)
- Cross-validation scheme and number of folds
- Hyperparameters and how they were selected
- Mean ± std of all metrics across outer CV folds
- Statistical tests and p-values with correction method
- Compute time and hardware used

### Figures to Generate

1. **Annotation ablation**: Heatmap of Jaccard similarities between levels
2. **Position-aware comparison**: ROC curves for SIEVE vs DeepSet
3. **Attribution regularization**: Pareto plot of AUC vs stability for different λ_attr
4. **Epistasis**: Network diagram of significant epistatic pairs
5. **Biological validation**: Pathway enrichment bar plot

### Code and Data Availability

- All code version-controlled in GitHub repository
- Processed intermediate data (tensors, model checkpoints) stored in standard locations
- Random seeds recorded for reproducibility
- Configuration files for each experiment

## Timeline

### Week 1-2: Data Pipeline and Baselines
- Implement VCF parser with quality control
- Create multi-level feature encodings
- Train and evaluate logistic regression baseline
- Verify data integrity and sample sizes

### Week 3-4: Core Model Implementation
- Implement SIEVE architecture components
- Unit tests for each component
- Initial training runs to verify learning

### Week 5-6: Experiment 1 and 2
- Annotation ablation study (all 5 levels)
- Position-aware vs position-agnostic comparison
- Generate figures and preliminary analysis

### Week 7-8: Experiment 3 and 4
- Attribution regularization sweep
- Epistasis detection and validation
- Statistical testing and correction

### Week 9-10: Analysis and Writing
- Biological interpretation of discoveries
- Comparison with known GWAS results
- Draft methods and results sections

## Risk Mitigation

### If Model Doesn't Learn

Possible causes and solutions:

1. **Encoding issues**: Visualize tensor distributions, check for constant features
2. **Label problems**: Verify case/control balance, check for mislabeling
3. **Insufficient signal**: Try larger sample size, pre-filter to candidate genes
4. **Architecture mismatch**: Simplify model, verify gradients flow

### If No Novel Discoveries

Possible interpretations:

1. **Annotations are comprehensive**: CADD/SIFT capture all relevant signal → still publishable as negative result
2. **Phenotype is highly polygenic**: No single variants detectable → focus on pathway analysis
3. **Sample size too small**: Power calculation, discuss limitations

### If Epistasis Validation Fails

Possible causes:

1. **Attention patterns reflect LD, not interaction**: Control for LD
2. **Sample size insufficient for interaction detection**: Need ~10x more samples
3. **True epistasis is rare**: Report null finding, which is still informative
