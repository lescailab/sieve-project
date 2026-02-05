# SIEVE: Sparse Interpretable Exome Variant Explainer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SIEVE** is a deep learning framework for discovering disease-associated genetic variants from exome sequencing data. Unlike existing methods that rely heavily on pre-computed functional annotations, SIEVE tests whether deep learning can discover variants from genotype patterns alone—enabling identification of associations that prior knowledge may miss.

## Scientific Motivation

Genome-wide association studies (GWAS) and existing deep learning methods for variant discovery share a common limitation: they either ignore prior biological knowledge entirely (standard GWAS) or depend heavily on it (methods like DeepRVAT that use 34 functional annotations). Neither approach directly tests whether machine learning can discover genuinely novel associations—patterns in case-control data that existing annotations don't capture.

SIEVE addresses three scientific questions:

1. **Can deep learning discover variants that annotation-based methods miss?** We implement an annotation-ablation protocol, training models with minimal annotations (genotype only) through full annotations, to identify variants that emerge only from genotype patterns.

2. **Do spatial relationships between variants carry disease signal?** Unlike permutation-invariant deep set approaches, SIEVE uses position-aware sparse attention to test whether the relative positions of an individual's variants (e.g., potential compound heterozygosity) are informative.

3. **Can inherent interpretability improve discovery?** Rather than applying explainability post-hoc, SIEVE incorporates attribution sparsity into the training objective, encouraging the model to rely on a small set of variants and producing more stable, meaningful attributions.

## Key Innovations

### 1. Annotation-Ablation Discovery Protocol

SIEVE trains models at five annotation levels:

- **L0**: Genotype dosage only (0, 1, 2)
- **L1**: L0 + genomic position
- **L2**: L1 + consequence class (missense/synonymous/LoF)
- **L3**: L2 + SIFT + PolyPhen scores
- **L4**: Full functional annotations

By comparing variant rankings across levels, we identify:

- *L0-specific variants*: Associations found without any annotations—potential novel discoveries
- *L4-specific variants*: Associations that require annotation context—validating that annotations add value

### 2. Position-Aware Sparse Attention

Standard approaches treat variants as unordered sets (permutation-invariant). SIEVE preserves positional information through:

- Sinusoidal positional encodings for genomic coordinates
- Relative position bias in attention computation
- Sparse attention that operates only on variant-present positions

This enables learning that nearby variants (compound heterozygosity) or specific distance patterns matter, while avoiding the computational burden of dense tensors.

### 3. Attribution-Regularized Training

Instead of training purely for classification and explaining afterward, SIEVE incorporates interpretability into the loss function:

```text
Loss = Classification_Loss + λ × Attribution_Sparsity_Loss
```

This encourages the model to achieve good classification while relying on a small number of variants, producing more stable and meaningful attributions.

### 4. Null Baseline Attribution Analysis

To distinguish genuine signal from noise, SIEVE includes a null baseline analysis pipeline:

- Trains identical models on permuted case/control labels
- Establishes null distribution of attribution scores
- Computes significance thresholds (p < 0.05, 0.01, 0.001)
- Identifies variants with attributions exceeding chance expectations

## Installation

```bash
# Clone repository
git clone https://github.com/lescailab/sieve-project.git
cd sieve-project

# Create environment
conda create -n sieve python=3.10
conda activate sieve

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- cyvcf2 or pysam for VCF parsing
- CUDA-capable GPU recommended (8GB+ VRAM)
- scipy, pandas, pyyaml, matplotlib (for analysis)

## Quick Start

### 1. Prepare Input Data

SIEVE requires:

- Multi-sample VCF file annotated with VEP (`.vcf.gz` with index)
- Phenotype file mapping samples to case/control labels

```bash
# Example phenotype file format (TSV)
sample_id    phenotype
SAMPLE001    1
SAMPLE002    0
...
```

### 2. Preprocess VCF

```bash
python scripts/preprocess.py \
    --vcf data/cohort.vcf.gz \
    --phenotypes data/phenotypes.tsv \
    --output data/preprocessed.pt
```

### 3. Train Model

```bash
# Train at annotation level L3 (SIFT + PolyPhen)
python scripts/train.py \
    --preprocessed-data data/preprocessed.pt \
    --level L3 \
    --val-split 0.2 \
    --lr 0.00001 \
    --lambda-attr 0.1 \
    --epochs 100 \
    --batch-size 16 \
    --chunk-size 3000 \
    --output-dir experiments \
    --experiment-name my_model \
    --device cuda
```

### 4. Run Explainability Analysis

```bash
python scripts/explain.py \
    --experiment-dir experiments/my_model \
    --preprocessed-data data/preprocessed.pt \
    --output-dir results/explainability \
    --device cuda
```

This produces:

- `sieve_variant_rankings.csv` - All variants ranked by attribution
- `sieve_gene_rankings.csv` - Gene-level aggregated scores
- `sieve_interactions.csv` - High-attention variant pairs

### 5. Null Baseline Analysis (Statistical Validation)

To establish statistical significance:

```bash
# Option 1: Use the complete pipeline wrapper
export INPUT_DATA=data/preprocessed.pt
export REAL_EXPERIMENT=experiments/my_model
export OUTPUT_BASE=experiments
bash scripts/run_null_baseline_analysis.sh

# Option 2: Run steps manually
# Step 1: Create permuted dataset
python scripts/create_null_baseline.py \
    --input data/preprocessed.pt \
    --output data/preprocessed_NULL.pt \
    --seed 42

# Step 2: Train null model (same params as real)
python scripts/train.py \
    --preprocessed-data data/preprocessed_NULL.pt \
    --level L3 \
    --experiment-name null_baseline \
    [... same parameters as real model ...]

# Step 3: Run explainability on null
python scripts/explain.py \
    --experiment-dir experiments/null_baseline \
    --preprocessed-data data/preprocessed_NULL.pt \
    --output-dir results/null_attributions \
    --is-null-baseline

# Step 4: Compare real vs null
python scripts/compare_attributions.py \
    --real results/explainability/sieve_variant_rankings.csv \
    --null results/null_attributions/sieve_variant_rankings.csv \
    --output-dir results/comparison
```

The comparison produces:

- `comparison_summary.yaml` - Statistical tests and thresholds
- `significant_variants_p01.csv` - Variants exceeding p<0.01 threshold
- `variant_rankings_with_significance.csv` - All variants annotated with significance flags
- `real_vs_null_comparison.png` - Visualization comparing distributions

### 6. Annotation Ablation (Compare Levels)

```bash
# Train models at multiple annotation levels
for level in L0 L1 L2 L3 L4; do
    python scripts/train.py \
        --preprocessed-data data/preprocessed.pt \
        --level $level \
        --experiment-name ablation_$level \
        --output-dir experiments
done

# Compare discoveries across levels
python scripts/compare_levels.py \
    --experiments experiments/ablation_* \
    --output-dir results/ablation_comparison
```

## Complete Workflow

The recommended SIEVE workflow:

```text
1. Preprocess VCF → preprocessed.pt
2. Train real model → experiments/real_model/
3. Run explainability → results/explainability/
4. Create null baseline → preprocessed_NULL.pt
5. Train null model → experiments/null_baseline/
6. Run null explainability → results/null_attributions/
7. Compare real vs null → results/comparison/
   ↓ Review significant_variants_p01.csv
8. Biological validation (experiments)
```

## Output Files

### Core Outputs

**From Training:**

- `best_model.pt` - Model checkpoint
- `training_history.yaml` - Loss curves and metrics
- `config.yaml` - Full configuration for reproducibility

**From Explainability:**

- `sieve_variant_rankings.csv` - Variant-level attributions
  - Columns: position, chromosome, gene_id, mean_attribution, max_attribution, num_samples
- `sieve_gene_rankings.csv` - Gene-level scores
  - Columns: gene_id, num_variants, gene_score, top_variant_pos
- `sieve_interactions.csv` - Variant pairs with high attention
  - Columns: pos1, pos2, gene1, gene2, mean_attention, frequency

**From Null Comparison:**

- `comparison_summary.yaml` - Statistical summary
  - Thresholds at p<0.10, 0.05, 0.01, 0.001
  - KS and Mann-Whitney test results
  - Enrichment factors
- `significant_variants_p01.csv` - Statistically significant discoveries
- `variant_rankings_with_significance.csv` - All variants with significance flags

## Project Structure

```text
sieve-project/
├── CLAUDE.md              # Development context (for AI assistants)
├── INSTRUCTIONS.md        # Development workflow guide
├── USER_GUIDE.md          # Comprehensive user documentation
├── README.md              # This file
├── src/
│   ├── data/              # VCF parsing and dataset construction
│   ├── encoding/          # Multi-level feature encoding
│   ├── models/            # SIEVE architecture components
│   ├── training/          # Training with attribution regularization
│   ├── explain/           # Explainability methods (IG, attention, epistasis)
│   └── visualization/     # R scripts for figures
├── configs/               # Experiment configurations
├── scripts/               # Entry point scripts
│   ├── preprocess.py      # VCF preprocessing
│   ├── train.py           # Model training
│   ├── explain.py         # Attribution analysis
│   ├── create_null_baseline.py   # Permute labels
│   ├── compare_attributions.py   # Real vs null comparison
│   └── run_null_baseline_analysis.sh  # Full pipeline
└── tests/                 # Unit tests
```

## Comparison with Existing Methods

| Feature                       | SIEVE | DeepRVAT | GenNet  | GWAS_NN |
|-------------------------------|-------|----------|---------|---------|
| Direct VCF input              | ✓     | ✗        | ✗       | ✗       |
| Annotation-free discovery     | ✓     | ✗        | ✗       | ✗       |
| Position-aware                | ✓     | ✗        | ✗       | ✗       |
| Built-in interpretability     | ✓     | ✗        | Partial | ✗       |
| Epistasis validation          | ✓     | ✗        | ✗       | ✓       |
| Null baseline calibration     | ✓     | ✗        | ✗       | ✗       |
| Common + rare variants        | ✓     | Rare only| All     | Common  |

## Troubleshooting

### Memory Issues

If you encounter OOM errors:

- Reduce `--chunk-size` (e.g., 2000 → 1000)
- Reduce `--batch-size` (e.g., 16 → 8)
- Use `--gradient-accumulation-steps 8` for effective larger batch
- Process chromosomes separately with `--chromosomes 1,2,3`

### Slow Training

If training is too slow:

- Increase `--chunk-size` (more variants per batch)
- Increase `--batch-size` if GPU memory allows
- Reduce `--n-steps` for integrated gradients (e.g., 50 → 25)
- Use `--skip-attention` in explain.py if only need variant rankings

### Model Not Learning

If validation AUC stays near 0.5:

- Check that phenotypes are loaded correctly (cases/controls balanced?)
- Reduce learning rate: `--lr 0.000001`
- Increase `--lambda-attr` if model is too sparse
- Check for data leakage in preprocessing

## Citation

If you use SIEVE in your research, please cite:

```bibtex
@software{sieve2026,
  title = {SIEVE: Sparse Interpretable Exome Variant Explainer},
  author = {Lescai, Francesco},
  year = {2026},
  url = {https://github.com/lescailab/sieve-project}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions! Key areas for improvement:

- Additional annotation levels or feature encodings
- Alternative aggregation methods (beyond max pooling)
- Visualization improvements
- Performance optimization

Please open an issue to discuss major changes before submitting PRs.

## Acknowledgments

This project builds on insights from:

- DeepRVAT (Clarke et al., 2024, Nature Genetics)
- GenNet (van Hilten et al., 2021, Communications Biology)
- GWAS_NN (Cui et al., 2022, Communications Biology)
- DeepCOMBI (Mieth et al., 2021, NAR Genomics and Bioinformatics)

## Support

- **Issues**: [GitHub Issues](https://github.com/lescailab/sieve-project/issues)
- **Documentation**: See CLAUDE.md, USER_GUIDE.md, INSTRUCTIONS.md
- **Questions**: Open a GitHub discussion

---

**Note**: SIEVE is research software. Results should be validated experimentally before making clinical or biological claims.
