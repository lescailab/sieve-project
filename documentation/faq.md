# FAQ

### General Questions

**Q: How many samples do I need?**
A: Minimum 50 cases + 50 controls for initial testing. Recommended 250+ cases and 250+ controls for robust results. For epistasis detection, 5000+ samples ideal.

**Q: Can I use WGS data instead of exome?**
A: Yes, but be aware:
- Much larger file sizes (slower preprocessing)
- More variants per sample (higher memory usage)
- May need to filter to exonic regions for meaningful results

**Q: What reference genome does SIEVE use?**
A: Both GRCh37 (hg19) and GRCh38 (hg38) are supported via `--genome-build`. Contigs with or without `chr` prefix are normalised automatically.

**Q: Can I use SIEVE for quantitative traits?**
A: Not currently. SIEVE is designed for binary case-control studies. Adaptation for quantitative traits would require modifying the loss function and output layer.

**Q: How long does a typical analysis take?**
A:
- Preprocessing: 30 mins - 5 hours (once)
- Training: 1-3 hours per model (on GPU)
- Explainability: 30-60 mins
- Null baseline: Same as training + explainability
- Total: 4-12 hours for complete analysis

### Technical Questions

**Q: What is "chunked processing"?**
A: SIEVE processes variants in chunks (default 3000) to fit in GPU memory. This allows handling whole-genome data without running out of memory. The chunk size is automatically managed but can be tuned with `--chunk-size`.

**Q: What happens if a sample has more variants than chunk_size?**
A: The sample is processed in multiple chunks, then results are aggregated. This is handled automatically.

**Q: Why use gradient accumulation?**
A: It simulates larger batch sizes without using more memory. For example, `--batch-size 2 --gradient-accumulation-steps 16` gives the training dynamics of `--batch-size 32` while only using memory for 2 samples at a time.

**Q: What's the difference between --batch-size and --chunk-size?**
A:
- `--batch-size`: Number of samples processed together
- `--chunk-size`: Maximum variants processed per forward pass (per sample)
- Both affect memory usage but in different ways

**Q: Can I use multiple GPUs?**
A: Not currently supported. SIEVE uses a single GPU. If you have multiple GPUs, you can run multiple experiments in parallel on different GPUs.

### Scientific Questions

**Q: What if L0 (genotype-only) performs as well as L3?**
A: This is scientifically interesting! It suggests:
- Genotype patterns alone carry disease signal
- Annotations may not add much information for this phenotype
- Potential for discovering novel variants missed by annotation-based methods

**Q: What enrichment factor is "good enough"?**
A: Guidelines:
- < 1.5×: Weak signal, be very cautious
- 1.5-2×: Moderate, validate top 10-20 variants
- 2-5×: Strong, proceed with confidence
- \> 5×: Very strong, high confidence in discoveries

**Q: Should I always run null baseline?**
A: **Yes, for publication-quality results.** It's the only way to establish statistical significance of your discoveries. For initial exploration, you can skip it, but include it before claiming discoveries.

**Q: How do I know if a variant is truly causal?**
A: You don't, from computational analysis alone. SIEVE identifies statistical associations. Causality requires:
1. High attribution score
2. Exceeds null baseline threshold
3. Biological plausibility (gene function, prior evidence)
4. **Experimental validation** (functional studies, replication cohort)

**Q: What's the difference between attention patterns and integrated gradients?**
A:
- **Integrated gradients**: Measures how much a variant contributes to the final prediction (variant importance)
- **Attention patterns**: Measures which variant pairs the model looks at together (variant interactions)
- Both are complementary - use both for full picture

### Troubleshooting Questions

**Q: Training works but explainability crashes with OOM**
A: Integrated gradients requires more memory than training. Solutions:
- Reduce `--n-steps` (try 25 instead of 50)
- Reduce `--max-variants` (try 1500 instead of 2000)
- Reduce `--batch-size` (try 2 instead of 4)

**Q: Null model has better AUC than real model?**
A: This occasionally happens by chance (especially with small datasets). Solutions:
- Run multiple null permutations (5-10) and use the most conservative threshold
- Increase sample size
- Check for data quality issues in real data

**Q: Cross-validation folds have very different AUC values?**
A: High variance across folds suggests:
- Small sample size → Increase if possible
- Label imbalance → Check case/control ratio
- Overfitting → Try simpler model or more regularisation
- Population stratification → Check for batch effects

---

