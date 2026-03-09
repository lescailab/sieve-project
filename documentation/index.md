# SIEVE User Guide

**Version**: 1.3
**Last Updated**: 2026-03-09
**For**: SIEVE v0.1.0+


## Introduction

### What is SIEVE?

**SIEVE** (Sparse Interpretable Exome Variant Explainer) is a deep learning framework for discovering disease-associated genetic variants from exome sequencing data in case-control studies.

### What Makes SIEVE Different?

Unlike existing methods:
- **Direct VCF Processing**: No conversion to PLINK or custom formats required
- **Annotation-Free Discovery**: Tests whether ML can discover variants without prior knowledge
- **Position-Aware**: Learns spatial relationships between variants (e.g., compound heterozygosity)
- **Built-in Interpretability**: Attribution sparsity incorporated into training
- **Statistical Validation**: Null baseline analysis establishes significance thresholds

### Scientific Questions SIEVE Addresses

1. **Can deep learning discover variants that annotations miss?** → Annotation ablation experiments (L0-L4)
2. **Do spatial relationships between variants matter?** → Position-aware sparse attention
3. **Can we make models interpretable by design?** → Attribution-regularised training
4. **Are discoveries statistically significant?** → Null baseline analysis

### Key Capabilities

- **Train** models at multiple annotation levels (genotype-only to full annotations)
- **Explain** predictions with integrated gradients attribution
- **Discover** novel variant associations with statistical validation
- **Detect** epistatic interactions via attention patterns
- **Validate** discoveries against ClinVar, GWAS, and GO databases

---

