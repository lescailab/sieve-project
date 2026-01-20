# SIEVE Training Pipeline - Deployment Instructions

This document provides step-by-step instructions for deploying and running the SIEVE training pipeline (Phase 1: Foundation).

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional but recommended for larger datasets)
- VCF file annotated with VEP
- Phenotype file (TSV format with sample_id and phenotype columns)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -e .
```

This will install all required dependencies from `pyproject.toml`:
- PyTorch
- NumPy, Pandas, SciPy
- scikit-learn
- cyvcf2, pysam (VCF parsing)
- PyYAML
- captum, shap (explainability - for Phase 2)
- matplotlib, seaborn (visualization)

### 4. Verify Installation

Run the test suite to ensure everything is working:

```bash
# Test VCF parser (Phase 1A)
python test_vcf_parser.py

# Test encoding pipeline (Phase 1B)
python test_encoding_pipeline.py

# Test model architecture (Phase 1C)
python test_model_architecture.py

# Test training pipeline (Phase 1D)
python test_training_pipeline.py
```

All tests should pass with "✓" symbols.

## Quick Start - Small Test Dataset

The repository includes a small test dataset for validation:

```bash
python scripts/train.py \
    --vcf test_data/small/test_data.vcf.gz \
    --phenotypes test_data/small/test_data_phenotypes.tsv \
    --level L3 \
    --epochs 20 \
    --batch-size 8 \
    --output-dir outputs/test_run
```

This should complete in a few minutes on CPU and will create:
- `outputs/test_run/config.yaml` - Training configuration
- `outputs/test_run/best_model.pt` - Best model checkpoint
- `outputs/test_run/last_model.pt` - Last model checkpoint
- `outputs/test_run/results.yaml` - Final results

## Running on Your Data

### 1. Prepare Your Data

#### VCF File Requirements:
- Multi-sample VCF file (bgzipped and indexed recommended)
- Must be annotated with VEP (Variant Effect Predictor)
- GRCh37 reference genome (contig names without 'chr' prefix)
- Required VEP fields in CSQ: Consequence, SYMBOL, SIFT, PolyPhen

Example VEP command:
```bash
vep --input_file your_variants.vcf \
    --output_file your_variants_annotated.vcf \
    --vcf \
    --symbol \
    --sift b \
    --polyphen b \
    --assembly GRCh37 \
    --offline
```

#### Phenotype File Requirements:
- TSV format with header
- Column 1: `sample_id` (must match VCF sample names)
- Column 2: `phenotype` (1 = control, 2 = case)

Example `phenotypes.tsv`:
```
sample_id	phenotype
sample1	2
sample2	1
sample3	2
sample4	1
```

### 2. Choose Annotation Level

SIEVE supports 5 annotation levels (L0-L4) for ablation experiments:

- **L0**: Genotype dosage only (0, 1, 2) - completely annotation-free
- **L1**: L0 + genomic position (sinusoidal encoding)
- **L2**: L1 + VEP consequence type (missense, synonymous, LoF)
- **L3**: L2 + SIFT + PolyPhen scores (recommended starting point)
- **L4**: L3 + additional annotations (extensible)

**Recommendation**: Start with L3 for standard analysis, then compare with L0-L2 to assess annotation dependence.

### 3. Preprocessing Data (Recommended for Large Datasets)

**For datasets with >500 samples**, preprocessing significantly speeds up experimentation by caching parsed VCF data:

```bash
# Step 1: Preprocess once
python scripts/preprocess.py \
    --vcf /path/to/your/data.vcf.gz \
    --phenotypes /path/to/phenotypes.tsv \
    --output /path/to/preprocessed_data.pt
```

This creates a `.pt` file (~1-5 MB per 1000 samples) containing all parsed variant data.

**Benefits:**
- VCF parsing: **~30 minutes to 5+ hours** depending on dataset size (one time)
- Loading preprocessed: **~5-10 seconds** (every training run)
- **100-3000x speedup** for repeated experiments (5 hours → 5 seconds = 3600x!)

**Example workflow:**
```bash
# Preprocess once
python scripts/preprocess.py \
    --vcf ottawa_filtered_vepped.vcf.gz \
    --phenotypes phenotypes.tsv \
    --output preprocessed_data.pt

# Train multiple times (fast!)
python scripts/train.py --preprocessed-data preprocessed_data.pt --level L3 ...
python scripts/train.py --preprocessed-data preprocessed_data.pt --level L3 --cv 5 ...
python scripts/train.py --preprocessed-data preprocessed_data.pt --level L2 ...
```

Each training run loads in seconds instead of minutes!

### 4. Single Training Run

For a single train/validation split (80/20):

```bash
python scripts/train.py \
    --vcf /path/to/your/data.vcf.gz \
    --phenotypes /path/to/phenotypes.tsv \
    --level L3 \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --early-stopping 10 \
    --output-dir outputs/my_experiment \
    --experiment-name cohort_L3 \
    --device cuda  # or cpu
```

### 5. Cross-Validation

For robust performance estimation with 5-fold cross-validation:

```bash
# Using preprocessed data (recommended)
python scripts/train.py \
    --preprocessed-data /path/to/preprocessed_data.pt \
    --level L3 \
    --cv 5 \
    --epochs 100 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --max-variants-per-batch 3000 \
    --lr 0.001 \
    --early-stopping 10 \
    --output-dir outputs/my_experiment \
    --experiment-name cohort_L3_cv5 \
    --device cuda

# Or from VCF directly (slower)
python scripts/train.py \
    --vcf /path/to/your/data.vcf.gz \
    --phenotypes /path/to/phenotypes.tsv \
    --level L3 \
    --cv 5 \
    --epochs 100 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --max-variants-per-batch 3000 \
    --lr 0.001 \
    --early-stopping 10 \
    --output-dir outputs/my_experiment \
    --experiment-name cohort_L3_cv5 \
    --device cuda
```

This will create:
- `outputs/my_experiment/cohort_L3_cv5/config.yaml` - Configuration
- `outputs/my_experiment/cohort_L3_cv5/fold_0/` - Fold 0 checkpoints
- `outputs/my_experiment/cohort_L3_cv5/fold_1/` - Fold 1 checkpoints
- ... (one directory per fold)
- `outputs/my_experiment/cohort_L3_cv5/cv_results.yaml` - Summary statistics

### 5. Annotation Ablation Experiment

To test whether annotations help discovery:

```bash
# L0: No annotations (genotype only)
python scripts/train.py --vcf data.vcf.gz --phenotypes phenotypes.tsv \
    --level L0 --cv 5 --experiment-name ablation_L0 --output-dir outputs/ablation

# L1: Position only
python scripts/train.py --vcf data.vcf.gz --phenotypes phenotypes.tsv \
    --level L1 --cv 5 --experiment-name ablation_L1 --output-dir outputs/ablation

# L2: Consequence type
python scripts/train.py --vcf data.vcf.gz --phenotypes phenotypes.tsv \
    --level L2 --cv 5 --experiment-name ablation_L2 --output-dir outputs/ablation

# L3: SIFT + PolyPhen
python scripts/train.py --vcf data.vcf.gz --phenotypes phenotypes.tsv \
    --level L3 --cv 5 --experiment-name ablation_L3 --output-dir outputs/ablation
```

Then compare AUC across levels to assess annotation importance.

## Command-Line Arguments Reference

### Data Arguments
- `--vcf`: Path to VCF file (required if not using --preprocessed-data)
- `--phenotypes`: Path to phenotype TSV file (required if not using --preprocessed-data)
- `--preprocessed-data`: Path to preprocessed data file (.pt from preprocess.py)
- `--level`: Annotation level [L0, L1, L2, L3, L4] (required)

### Training Arguments
- `--batch-size`: Batch size (default: 32, **use 2-4 for large datasets**)
- `--gradient-accumulation-steps`: Simulate larger batches (default: 1, **use 8-16 for large datasets**)
- `--max-variants-per-batch`: Cap variants per batch (default: 3000, prevents OOM)
- `--epochs`: Maximum number of epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--lambda-attr`: Attribution regularization weight (default: 0.0)
- `--early-stopping`: Early stopping patience in epochs (default: 10)
- `--gradient-clip`: Gradient clipping value (default: None)

### Model Arguments
- `--latent-dim`: Embedding dimension (default: 64)
- `--num-heads`: Number of attention heads (default: 4)
- `--num-attention-layers`: Number of attention layers (default: 2)
- `--hidden-dim`: Hidden dimension in encoder (default: 128)

### Cross-Validation Arguments
- `--cv`: Number of CV folds (default: None = single split)
- `--val-split`: Validation split ratio if not using CV (default: 0.2)

### Output Arguments
- `--output-dir`: Output directory (default: outputs)
- `--experiment-name`: Experiment name (default: {level}_run)

### Device Arguments
- `--device`: Device [cuda, cpu] (default: cuda if available)
- `--num-workers`: Data loading workers (default: 0)

### Other Arguments
- `--seed`: Random seed for reproducibility (default: 42)

## Memory Optimization for Large Datasets

**Critical for datasets with >1000 samples and 16K+ genes!**

### The Problem

With real exome data:
- ~5000-7000 variants per sample
- Attention matrix: `[batch_size, heads, variants, variants]`
- batch=32, variants=5000 → **373 GB memory required!**

### The Solution

Use these three parameters together:

```bash
--batch-size 2                      # Small batches (2-4 samples)
--gradient-accumulation-steps 16    # Simulate batch=32 (2×16=32)
--max-variants-per-batch 3000       # Cap variants to prevent OOM
```

### Memory Usage by Configuration

| Configuration | GPU Memory | Notes |
|--------------|------------|-------|
| batch=32, no limit | 373 GB | **FAILS** |
| batch=32, max=3000 | ~40 GB | Still too large for most GPUs |
| batch=8, max=3000 | ~12 GB | Borderline for T4 (15GB) |
| **batch=2, max=3000** | **~7 GB** | ✓ **Works on T4/RTX5000** |
| batch=1, max=2000 | ~3 GB | Very safe, slower training |

### Recommended Settings by GPU

**T4 / RTX 5000 (15-16 GB):**
```bash
--batch-size 2 --gradient-accumulation-steps 16 --max-variants-per-batch 3000
```

**A100 40GB:**
```bash
--batch-size 4 --gradient-accumulation-steps 8 --max-variants-per-batch 4000
```

**A100 80GB:**
```bash
--batch-size 8 --gradient-accumulation-steps 4 --max-variants-per-batch 5000
```

### Gradient Accumulation Explained

Simulates larger batches without extra memory:
- Physical batch=2: Process 2 samples at a time
- Accumulation=16: Accumulate gradients over 16 mini-batches
- Effective batch=32: Same training dynamics as batch_size=32

**No loss in training quality!**

## Interpreting Results

### Single Run Results

Check `results.yaml`:

```yaml
auc: 0.75              # Area under ROC curve (0.5 = random, 1.0 = perfect)
accuracy: 0.70         # Classification accuracy
loss: 0.42             # Final validation loss
classification_loss: 0.42   # BCE loss component
attribution_loss: 0.0       # Sparsity loss (if lambda_attr > 0)
```

**Interpretation**:
- AUC > 0.5: Model learned some signal
- AUC > 0.7: Good performance
- AUC > 0.8: Strong performance
- AUC ≈ 0.5: Model is no better than random (check data/encoding)

### Cross-Validation Results

Check `cv_results.yaml`:

```yaml
mean_auc: 0.75
std_auc: 0.05
mean_accuracy: 0.70
std_accuracy: 0.03
fold_results:
  - auc: 0.78
    accuracy: 0.72
    loss: 0.40
  - auc: 0.72
    accuracy: 0.68
    loss: 0.44
  ...
```

**Interpretation**:
- Mean AUC: Average performance across folds
- Std AUC: Performance stability (lower is better)
- High std → Model is sensitive to data split
- Check individual fold results for outliers

## Troubleshooting

### Model Not Learning (AUC ≈ 0.5)

**Possible causes**:

1. **Insufficient data**: Try with more samples (>100 cases, >100 controls)
2. **Encoding issues**: Run `test_encoding_pipeline.py` to verify feature extraction
3. **Label imbalance**: Check case/control ratio in output - extreme imbalance may need class weights
4. **Learning rate**: Try `--lr 0.0001` (lower) or `--lr 0.01` (higher)
5. **Model capacity**: Reduce if very small dataset: `--latent-dim 32 --hidden-dim 64`

### Out of Memory

**Solutions**:

1. Reduce batch size: `--batch-size 16` or `--batch-size 8`
2. Reduce model size: `--latent-dim 32 --num-attention-layers 1`
3. Use gradient accumulation (not yet implemented - Phase 2)
4. Process chromosomes separately (not yet implemented - Phase 2)

### Very Slow Training

**Solutions**:

1. Use GPU: `--device cuda`
2. Increase batch size: `--batch-size 64` (if memory allows)
3. Reduce data: Subsample variants or use smaller chromosomes first
4. Increase workers: `--num-workers 4` (for larger datasets)

### VCF Parsing Errors

**Common issues**:

1. **Missing VEP annotation**: Ensure VCF has CSQ field
2. **Wrong genome build**: Convert to GRCh37 if using GRCh38
3. **Sample name mismatch**: Check VCF sample names match phenotype file
4. **Contig naming**: Code expects no 'chr' prefix (chr1 → 1)

Run `test_vcf_parser.py` to debug VCF issues.

## Performance Benchmarks

On the test dataset (20 samples, 1333 variants):

- **VCF parsing**: <1 second
- **Feature encoding (L3)**: <1 second
- **Model forward pass**: ~100ms per batch (CPU)
- **Training epoch**: ~1 second (CPU)
- **Complete training (20 epochs)**: ~30 seconds (CPU)

Expected scaling for real data:

- **1,000 samples, 100K variants**: ~10-30 minutes per epoch (GPU), ~2-5 hours (CPU)
- **5,000 samples, 500K variants**: ~1-2 hours per epoch (GPU), ~10-20 hours (CPU)

Use early stopping to avoid unnecessary epochs.

## Next Steps

After completing Phase 1D training:

1. **Phase 2 (Innovation)**: Implement full attribution-regularized training
2. **Phase 2 (Innovation)**: Test position-aware vs position-agnostic models
3. **Phase 3 (Validation)**: Implement explainability analysis (integrated gradients, SHAP)
4. **Phase 3 (Validation)**: Epistasis detection and validation
5. **Phase 3 (Validation)**: Comparison with existing GWAS signals

## Support

For issues or questions:

1. Check this deployment guide
2. Run relevant test scripts to isolate the problem
3. Review CLAUDE.md for project architecture details
4. Open an issue on GitHub: https://github.com/lescailab/sieve-project/issues

## Citation

If you use SIEVE in your research, please cite:

```
[Citation to be added upon publication]
```

## License

MIT License - See LICENSE file for details.
