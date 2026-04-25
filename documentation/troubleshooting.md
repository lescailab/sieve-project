# Troubleshooting

### Installation Issues

#### ImportError: No module named 'cyvcf2'

**Solution**:
```bash
pip install cyvcf2
# or if that fails:
conda install -c bioconda cyvcf2
```

#### CUDA out of memory

**Solution**: Reduce memory usage
```bash
python scripts/train.py \
    --batch-size 2 \
    --chunk-size 2000 \
    --gradient-accumulation-steps 16 \
    ...
```

---

### Data Preparation Issues

#### "does not contain VEP CSQ annotations"

**Symptom**: `ValueError: VCF file '...' does not contain VEP CSQ annotations.`

**Cause**: Your VCF has not been annotated with Ensembl VEP, or uses a different
annotation format (e.g. SnpEff `ANN` field).

**Solution**: Run VEP before preprocessing. See the
[Detailed Usage](detailed-usage.md#how-to-annotate-your-vcf-with-ensembl-vep)
page for the full command and required flags. The minimal command is:

```bash
vep \
    --input_file your.vcf.gz \
    --output_file annotated.vcf.gz \
    --vcf \
    --compress_output bgzip \
    --symbol \
    --canonical \
    --sift b \
    --polyphen b \
    --assembly GRCh37 \
    --offline \
    --cache \
    --dir_cache /path/to/vep_cache
tabix -p vcf annotated.vcf.gz
```

!!! warning "Do not use `--fields`"
    SIEVE expects VEP's **default CSQ field order** (hardcoded indices).
    Passing a custom `--fields` argument will break parsing silently.

#### "zero variant-sample assignments"

**Symptom**: `ValueError: Preprocessing produced zero variant-sample assignments
from N VCF records.`

**Cause**: The VCF header declares a CSQ field, but no variant data was loaded.
The error message includes diagnostics; the most common causes are:

- **All CSQ values empty**: The VCF was re-header'd or filtered after VEP
  annotation, stripping the actual CSQ values while keeping the header line.
- **Allele mismatch**: VEP's allele representation in CSQ doesn't match the
  VCF ALT field (can happen with post-VEP normalisation tools).
- **All genotypes filtered**: Every genotype fell below the GQ threshold
  (default 20). Try `--min-gq 0` to test.

**Verify** your CSQ values exist:
```bash
bcftools query -f '%INFO/CSQ\n' your.vcf.gz | head -3
```

If this prints empty lines or `.`, the CSQ values are missing despite the
header declaring the field.

#### Sample name mismatch

**Symptom**: `KeyError: SAMPLE001` or `ValueError: Sample not found in VCF`

**Solution**: Check that phenotype file sample IDs exactly match VCF:
```bash
# Get VCF samples
bcftools query -l your.vcf.gz

# Check phenotype file
cut -f1 phenotypes.tsv
```

Sample names must match character-for-character (case-sensitive).

#### Chromosome naming issue

**Symptom**: `KeyError: 'chr1'` or no variants loaded

**Solution**: SIEVE normalises both styles (`1` and `chr1`) internally. If you still see this error, check:
- `--genome-build` matches your data (`GRCh37` or `GRCh38`)
- Contigs are standard autosomes/sex chromosomes (1-22, X, Y), or can be mapped cleanly
- Phenotype sample IDs match VCF sample IDs

If your VCF uses non-standard contig labels, rename contigs:
```bash
bcftools annotate --rename-chrs chr_name_conv.txt input.vcf.gz -O z -o output.vcf.gz

# Where chr_name_conv.txt contains:
chr1 1
chr2 2
...
```

---

### Training Issues

#### Model not learning (AUC ≈ 0.5)

**Possible Causes & Solutions**:

1. **Insufficient data**
   - Need: >100 cases and >100 controls minimum
   - Solution: Acquire more samples or use data augmentation

2. **Label imbalance**
   - Check: How many cases vs controls?
   - Solution: If extreme (<10% minority), consider class weights

3. **Encoding issues**
   - Check: Run `python test_encoding_pipeline.py`
   - Solution: Verify features have non-zero variance

4. **Wrong learning rate**
   - Try: `--lr 0.000001` (lower) or `--lr 0.0001` (higher)

5. **Model too complex for data size**
   - Try: `--latent-dim 16 --hidden-dim 32 --num-attention-layers 1`

6. **Data leakage or preprocessing error**
   - Verify: Cases and controls are truly different cohorts

#### Training very slow

**Solutions**:

1. **Use GPU**: `--device cuda`
2. **Increase batch size**: `--batch-size 32` (if memory allows)
3. **Use preprocessed data**: Much faster than parsing VCF each time
4. **Reduce integration steps**: `--n-steps 25` in explain.py

#### Out of memory during training

**Solution**: Use memory-efficient settings:
```bash
--batch-size 2 \
--gradient-accumulation-steps 16 \
--chunk-size 2000
```

See "Memory-Efficient Training" section above.

---

### Explainability Issues

#### Integrated gradients very slow

**Solutions**:

1. Reduce integration steps: `--n-steps 25` (less accurate but faster)
2. Limit variants per sample: `--max-variants 1500`
3. Use larger batch size: `--batch-size 8` (if memory allows)
4. Skip attention analysis: `--skip-attention` (if only need variant rankings)

#### AttributeError: 'NoneType' object has no attribute

**Cause**: Model checkpoint not found or corrupted

**Solution**: Verify checkpoint exists:
```bash
ls -lh experiments/my_model/best_model.pt
```

If using `--experiment-dir`, check that `best_model.pt` or `fold_*/best_model.pt` exists.

---

### Null Baseline Issues

#### Null model AUC ≠ 0.5

**Expected**: Null model AUC should be ≈0.50 ± 0.05

**If AUC > 0.6**:
- **Problem**: Permutation didn't properly break genotype-phenotype relationship
- **Check**: Did you use the same preprocessed file for null training?
- **Solution**: Verify null baseline file has `_null_baseline_metadata` field

**If AUC < 0.4**:
- This is actually fine - model is consistently wrong, which is equivalent to chance
- Attributions are still valid for null distribution

#### No significant variants (enrichment < 1)

**Possible Causes**:

1. **Real model didn't learn**: Check real model AUC first
2. **Null and real similar**: May indicate no genuine signal in data
3. **Sample size too small**: Need larger cohort for robust signal
4. **Wrong parameters**: Ensure null trained with exact same params as real

**Solution**: Review real model performance first, then consider increasing sample size.

#### Bootstrap null calibration runs but produces no significant variants

Check these first:

1. The real model AUC. If it is below about `0.55`, there may be little signal to detect.
2. The `at_resolution_floor` flag in the rank-calibrated output. If zero variants hit the floor, the real ranking is not separating clearly from the null bootstrap ensemble.
3. The summary YAML `top_k_analysis` entries. Low KS statistics at `k = 100` usually mean the real and null rank distributions still overlap heavily near the top of the ranking.

#### Bootstrap runs out of memory

The bootstrap stores one null-rank value per tested variant per replicate. On large runs, `n_variants x n_bootstrap` can exceed RAM.

- Reduce `--n-bootstrap` from `1000` to `500` to halve storage.
- Use `--memmap-dir /path/to/fast/disk` to place the memmap-backed rank matrix on fast local storage. This flag controls where the backing file is created; it does not switch the matrix between in-memory and on-disk modes.

#### Bootstrap saturates only a few cores

If the wall-clock time scales poorly with `--n-jobs`, confirm the script is capping BLAS threads at startup (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `BLIS_NUM_THREADS` all set to `1`). Without those caps, BLAS threads compete with `joblib` workers and reduce parallel efficiency.

---

### Interpretation Issues

#### All top variants in the same gene

**Is this a problem?**
- Depends! If studying a Mendelian disease, this is expected
- For complex diseases, expect multiple genes
- Check: Is the gene biologically relevant to your phenotype?

**Possible issue**: Overfitting to one gene
- Solution: Check cross-validation stability
- Look at fold-specific rankings - is it consistent?

#### chrX dominates top rankings

**Possible causes**:
- Ploidy differences (hemizygosity) inflate chrX attributions
- Sex imbalance across case/control groups

**Solutions**:
1. Run sex-aware preprocessing (`infer_sex.py` → `preprocess.py --sex-map`)
2. Check sex balance (`check_sex_balance.py`)
3. Apply post-hoc correction (`correct_chrx_bias.py`)

**Note**: `correct_chrx_bias.py` excludes sex chromosomes by default; use `--include-sex-chroms` if you need chrX/chrY retained.
4. Re-run ablation comparison on the null-contrasted significance rankings:
   ```bash
   python scripts/compare_ablation_rankings.py \
       --ranking-dir results/ablation/significance_rankings \
       --score-column empirical_p_variant \
       --out-comparison significance_ablation_ranking_comparison.yaml
   ```

#### Ablation comparison with `--score-column delta_rank` gives Jaccard values of `1.0` across all level pairs

This usually means `delta_rank` is being sorted in the wrong direction, so the comparison is selecting the most demoted variants rather than the most promoted ones. Confirm you are on a version where `_score_column_is_ascending("delta_rank")` returns `False`, then rerun:

```bash
pytest tests/test_compare_ablation_rankings.py -v
```

#### Very low attributions overall

**Possible causes**:
- Model has low confidence (AUC close to 0.5)
- Embedding sparsity regularisation too strong (reduce `--lambda-attr`)
- Integration steps too low (increase `--n-steps`)

**Solution**:
1. Check model performance first
2. If AUC is good but attributions low, increase `--n-steps` to 100

#### Case-control differences all near zero

**Meaning**: Variants affect cases and controls similarly

**Interpretation**:
- May indicate population stratification (batch effects)
- Or: Model learned overall variant burden, not disease-specific patterns

**Solution**:
- Check for population structure (PCA analysis)
- Consider adjusting for covariates in future version

---
