# Appendix C: Method References

This appendix lists key methodological references that motivate recent pipeline updates.

### Sex inference and ploidy-aware encoding

- **X-chromosome inbreeding coefficient (F-statistic)** for genetic sex inference:  
  Purcell S, et al. (2007). *PLINK: a tool set for whole-genome association and population-based linkage analyses.* **American Journal of Human Genetics**, 81(3):559–575.
- **X-chromosome association and pseudoautosomal regions (PAR)**:  
  Clayton DG. (2008). *Testing for association on the X chromosome.* **Biostatistics**, 9(4):593–600.

### Attribution and interpretability

- **Integrated gradients** for feature attribution in deep networks:  
  Sundararajan M, Taly A, Yan Q. (2017). *Axiomatic Attribution for Deep Networks.* **ICML**.

### Attention mechanisms

- **Scaled dot-product attention** for modeling interactions:
  Vaswani A, et al. (2017). *Attention Is All You Need.* **NeurIPS**.

### Epistasis detection

- **EpiDetect/EpiCID** for network analysis of epistatic interactions:
  Mastropietro A, Markopoulos G, Evangelou E, Anagnostopoulos A. (2026). *A novel explainable deep-learning approach for network analysis of epistatic interactions.* **NAR Genomics and Bioinformatics**, 8(1):lqag004.
- **EpiCID interaction scoring** derives neural feature vectors from learned weights, uses cosine similarity to filter marginal-effect-dominated pairs, and reports first-layer interaction influence as the core explainability signal.
- **Scope and comparison to SIEVE**: EpiDetect was demonstrated on UK Biobank blood-pressure traits using pre-selected GWAS-significant common SNPs, whereas SIEVE works on common and rare variants directly from VCF-derived tensors and emphasizes sample-level attributions plus counterfactual validation rather than global weight-space similarity alone.

---
