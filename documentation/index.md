# SIEVE User Guide

**Version**: 1.2.0
**Last Updated**: 2026-04-29
**For**: SIEVE v1.2.0+


## Introduction

### What is SIEVE?

**SIEVE** (Sparse Interpretable Exome Variant Explainer) is a deep learning framework for discovering disease-associated genetic variants from exome sequencing data in case-control studies.

### What Makes SIEVE Different?

Unlike existing methods:
- **Direct VCF Processing**: No conversion to PLINK or custom formats required
- **Annotation-Ablation Protocol**: Quantifies how much of the ranking is carried by genome structure and how much by supplied annotation
- **Position-Aware**: Learns spatial relationships between variants (e.g., compound heterozygosity)
- **Built-in Interpretability**: Embedding sparsity regularisation incorporated into training
- **Statistical Validation**: Null baseline analysis establishes significance thresholds

### Scientific Questions SIEVE Addresses

1. **How much of a variant ranking is carried by annotation?** → Annotation ablation experiments (L0-L3, with L4 reserved as a compatibility placeholder)
2. **Do spatial relationships between variants matter?** → Position-aware dense self-attention over the sparse variant set
3. **Can we make models interpretable by design?** → Embedding-sparsity-regularised training
4. **Are discoveries statistically significant?** → Null baseline analysis

### Key Capabilities

- **Train** models at multiple annotation levels (genotype-only to current functional-score annotations)
- **Explain** predictions with integrated gradients attribution
- **Discover** novel variant associations with statistical validation
- **Test** candidate epistatic interactions using counterfactual perturbation and post-hoc attribution and co-occurrence analysis, with an explicit power analysis reported alongside the result
- **Validate** discoveries against ClinVar, GWAS, and GO databases

---
