# CLAUDE.md - DL-EXOME-IMG Project

## Project Overview

This project develops **SIEVE** (Sparse Interpretable Exome Variant Explainer), a deep learning framework for discovering disease-associated genetic variants from exome sequencing data in case-control studies. The key innovation is combining VCF-native processing with annotation-ablation experiments, position-aware sparse attention, and attribution-regularized training to enable genuine variant discovery rather than recovery of known annotations.

### Scientific Goals

1. **Test whether deep learning can discover variants that annotation-based methods miss** by systematically comparing models trained with minimal versus full functional annotations
2. **Test whether spatial relationships between variants carry disease signal** through position-aware attention mechanisms
3. **Produce inherently interpretable models** by incorporating attribution sparsity into training objectives
4. **Detect and validate epistatic interactions** through attention patterns and counterfactual perturbation

### What Makes This Novel

Existing methods (DeepRVAT, GenNet, GWAS_NN) share limitations we address:
- They require format conversion from VCF to PLINK or custom matrices
- They rely heavily on functional annotations (34 scores in DeepRVAT), creating circular logic where models "discover" what annotations already encode
- They apply explainability post-hoc rather than building interpretability into training
- They treat variants as unordered sets, ignoring potentially informative spatial patterns

## Technical Architecture

### Data Flow

```
Multi-sample VCF (VEP-annotated)
    ↓
VCF Parser (cyvcf2/pysam)
    ↓
Variant Feature Extraction
    ↓
Gene-grouped Sparse Tensors
    ↓
Position-Aware Sparse Attention Network
    ↓
Attribution-Regularized Training
    ↓
Explainability Analysis → Variant Rankings
```

### Core Components

#### 1. VCF Processing Module (`src/data/`)

Handles parsing of multi-sample VCF files annotated with VEP. Critical implementation notes:

**CSQ Field Parsing Fix** (from previous work):
```python
# VEP CSQ field requires careful sanitization
annotations = [
    x for x in str(info_field)
        .replace(" ", "")
        .replace("'", "")
        .replace("(", "")
        .split(',')
    if x.startswith(alt + '|')
]
```

**Contig Harmonization** (mandatory):
- Remove 'chr' prefixes for consistency
- Handle both UCSC and Ensembl notation
- Reference genome is GRCh37

#### 2. Feature Encoding Module (`src/encoding/`)

Implements multiple annotation levels for ablation experiments:

| Level | Features | Purpose |
|-------|----------|---------|
| L0 | Genotype dosage only (0,1,2) | Annotation-free baseline |
| L1 | L0 + genomic position | Test positional signal |
| L2 | L1 + consequence class (missense/synonymous/LoF) | Minimal VEP |
| L3 | L2 + SIFT + PolyPhen | Standard functional |
| L4 | L3 + additional annotations | Full annotation |

**Key design rule**: Continuous scores remain continuous, never binarized.

#### 3. Model Architecture (`src/models/`)

**PositionAwareSparseAttention**: The core innovation. Unlike dense convolutions (which fail on sparse data) or pure deep sets (which ignore position), this:
- Encodes each variant with sinusoidal positional embedding
- Applies self-attention only among variant-present positions
- Uses relative position encodings in attention computation
- Produces interpretable attention weights showing which variant pairs matter

**GeneAggregator**: Groups variants by gene and computes gene-level representations using permutation-invariant pooling (element-wise max).

**PhenotypeClassifier**: Final classification head with sigmoid output for binary case-control.

#### 4. Training Module (`src/training/`)

**Attribution-Regularized Loss**:
```python
loss = classification_loss + lambda_attr * attribution_sparsity_loss
```

The attribution sparsity term encourages the model to rely on a small number of variants for its predictions, improving interpretability and potentially identifying true causal variants.

#### 5. Explainability Module (`src/explain/`)

- Integrated gradients for variant-level attribution
- Attention weight analysis for positional patterns
- SHAP interaction values for epistasis detection
- Counterfactual perturbation for epistasis validation

## File Structure

```
dl-exome-img/
├── CLAUDE.md                 # This file
├── ARCHITECTURE.md           # Detailed model architecture
├── EXPERIMENTS.md            # Experimental protocol
├── README.md                 # User-facing documentation
├── pyproject.toml            # Project dependencies
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── vcf_parser.py     # VCF loading and parsing
│   │   ├── annotation.py     # Feature extraction from VEP
│   │   ├── dataset.py        # PyTorch Dataset class
│   │   └── utils.py          # Contig harmonization, etc.
│   ├── encoding/
│   │   ├── __init__.py
│   │   ├── levels.py         # Annotation level definitions
│   │   ├── positional.py     # Positional encodings
│   │   └── sparse_tensor.py  # Sparse tensor construction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py      # Position-aware sparse attention
│   │   ├── aggregation.py    # Gene-level aggregation
│   │   ├── classifier.py     # Phenotype prediction head
│   │   └── sieve.py          # Main SIEVE model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── loss.py           # Attribution-regularized loss
│   │   ├── trainer.py        # Training loop
│   │   └── validation.py     # Cross-validation utilities
│   ├── explain/
│   │   ├── __init__.py
│   │   ├── gradients.py      # Integrated gradients
│   │   ├── attention.py      # Attention weight analysis
│   │   ├── shap_values.py    # SHAP computation
│   │   └── counterfactual.py # Epistasis validation
│   └── visualization/        # R scripts for figures
│       ├── manhattan.R
│       ├── attention_heatmap.R
│       └── ablation_comparison.R
├── configs/
│   ├── default.yaml          # Default hyperparameters
│   └── experiments/          # Experiment-specific configs
├── scripts/
│   ├── preprocess.py         # VCF preprocessing
│   ├── train.py              # Training entry point
│   ├── explain.py            # Explainability analysis
│   └── compare_levels.py     # Annotation ablation comparison
├── tests/
│   └── ...                   # Unit tests
└── notebooks/
    └── exploration.ipynb     # Data exploration
```

## Implementation Priorities

### Phase 1: Foundation (Current Focus)

1. **VCF Parser** - robust parsing with CSQ fix, contig harmonization
2. **Multi-level encoding** - implement L0-L4 annotation levels
3. **Baseline model** - gene-aggregation network without position awareness
4. **Basic training** - standard cross-entropy loss, proper CV

### Phase 2: Innovation

1. **Position-aware sparse attention** - the architectural innovation
2. **Attribution-regularized training** - interpretability in objective
3. **Annotation ablation experiments** - systematic comparison

### Phase 3: Validation

1. **Explainability analysis** - integrated gradients, SHAP
2. **Epistasis detection** - attention patterns, counterfactual tests
3. **Comparison with known signals** - validate against prior GWAS

## Coding Standards

### Python Style
- Python 3.10+
- Type hints on all function signatures
- Docstrings in NumPy format
- Black formatting, isort imports
- Pytest for testing

### Key Libraries
- `cyvcf2` or `pysam` for VCF parsing
- `torch` for deep learning
- `captum` for explainability (integrated gradients)
- `shap` for SHAP values
- `scikit-learn` for preprocessing and metrics
- `pandas` for data manipulation
- `numpy` for numerical operations

### R for Visualization
- `ggplot2` for publication figures
- `ComplexHeatmap` for attention visualization
- Scripts in `src/visualization/`

## Critical Implementation Notes

### The Sparse Tensor Problem

The fundamental challenge: exomes have millions of positions, but any individual has variants at only thousands. Dense tensors are infeasible.

**Solution approaches implemented here**:

1. **Variant-only encoding**: Only store positions where at least one sample has a non-reference allele. This is the approach used by DeepRVAT.

2. **Sparse attention**: Instead of dense convolutions over genomic positions, attend only to variant-present positions with distance-aware positional encodings.

3. **Gene-level aggregation**: Reduce dimensionality by aggregating variants within genes, producing fixed-size gene vectors.

### Memory Management

For large cohorts (thousands of samples, millions of variants):
- Use memory-mapped arrays or HDF5 for variant data
- Process chromosomes independently when possible
- Use gradient checkpointing for deep models
- Batch by sample, not by variant

### Reproducibility

- Set random seeds explicitly
- Log all hyperparameters
- Save model checkpoints
- Use deterministic algorithms where possible
- Document exact software versions

## Common Tasks

### Adding a new annotation level

1. Define features in `src/encoding/levels.py`
2. Implement extraction in `src/data/annotation.py`
3. Add config in `configs/experiments/`
4. Update `EXPERIMENTS.md`

### Running an experiment

```bash
# Preprocess VCF (once)
python scripts/preprocess.py --vcf data/cohort.vcf.gz --output data/processed/

# Train at annotation level L2
python scripts/train.py --config configs/experiments/level_L2.yaml

# Run explainability
python scripts/explain.py --checkpoint models/level_L2/best.pt --output results/
```

### Debugging model not learning

If the model doesn't learn (AUC near 0.5):

1. **Check encoding**: Visualize tensor distributions, ensure non-zero variance
2. **Check labels**: Verify case/control balance, no label leakage
3. **Simplify model**: Try logistic regression on gene burden counts
4. **Inspect gradients**: Check for vanishing/exploding gradients
5. **Increase learning rate**: Sometimes needed for sparse data

Remember: "If the model does not learn, the encoding is wrong." - from previous project documentation

## References

Key papers informing this work:
- DeepRVAT (Clarke et al., 2024, Nature Genetics): Deep set networks for rare variants
- GenNet (van Hilten et al., 2021, Comm Bio): Visible neural networks
- GWAS_NN (Cui et al., 2022, Comm Bio): Gene-gene interaction detection
- DeepCOMBI (Mieth et al., 2021, NAR Genomics): LRP for GWAS

## Contact

This project is part of the lescailab/dl-exome-img repository.
