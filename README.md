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

```
Loss = Classification_Loss + λ × Attribution_Sparsity_Loss
```

This encourages the model to achieve good classification while relying on a small number of variants, producing more stable and meaningful attributions.

## Installation

```bash
# Clone repository
git clone https://github.com/lescailab/dl-exome-img.git
cd dl-exome-img

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
    --output data/processed/
```

### 3. Train Model

```bash
# Train at annotation level L2 (minimal annotations)
python scripts/train.py \
    --config configs/experiments/level_L2.yaml \
    --data data/processed/ \
    --output models/level_L2/

# Run annotation ablation (all levels)
python scripts/train.py \
    --config configs/experiments/ablation.yaml \
    --data data/processed/ \
    --output models/ablation/
```

### 4. Run Explainability Analysis

```bash
python scripts/explain.py \
    --checkpoint models/level_L2/best.pt \
    --data data/processed/ \
    --output results/
```

### 5. Compare Annotation Levels

```bash
python scripts/compare_levels.py \
    --models models/ablation/ \
    --output results/ablation_comparison/
```

## Output

SIEVE produces:

1. **Variant rankings** at each annotation level with attribution scores
2. **Annotation ablation analysis** identifying level-specific discoveries
3. **Attention patterns** showing which variant pairs are considered together
4. **Epistasis candidates** with counterfactual validation scores
5. **Publication-ready figures** (via R scripts in `src/visualization/`)

## Project Structure

```
sieve/
├── CLAUDE.md              # Development context (for AI assistants)
├── ARCHITECTURE.md        # Technical model specification
├── EXPERIMENTS.md         # Experimental protocol
├── src/
│   ├── data/              # VCF parsing and dataset construction
│   ├── encoding/          # Multi-level feature encoding
│   ├── models/            # SIEVE architecture components
│   ├── training/          # Training with attribution regularization
│   ├── explain/           # Explainability methods
│   └── visualization/     # R scripts for figures
├── configs/               # Experiment configurations
├── scripts/               # Entry point scripts
└── tests/                 # Unit tests
```

## Comparison with Existing Methods

| Feature | SIEVE | DeepRVAT | GenNet | GWAS_NN |
|---------|-------|----------|--------|---------|
| Direct VCF input | ✓ | ✗ | ✗ | ✗ |
| Annotation-free discovery | ✓ | ✗ | ✗ | ✗ |
| Position-aware | ✓ | ✗ | ✗ | ✗ |
| Built-in interpretability | ✓ | ✗ | Partial | ✗ |
| Epistasis validation | ✓ | ✗ | ✗ | ✓ |
| Common + rare variants | ✓ | Rare only | All | Common |

## Citation

If you use SIEVE in your research, please cite:

```bibtex
@software{sieve2025,
  title = {SIEVE: Sparse Interpretable Exome Variant Explainer},
  author = {Lescai Lab},
  year = {2025},
  url = {https://github.com/lescailab/dl-exome-img}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This project builds on insights from:
- DeepRVAT (Clarke et al., 2024, Nature Genetics)
- GenNet (van Hilten et al., 2021, Communications Biology)
- GWAS_NN (Cui et al., 2022, Communications Biology)
- DeepCOMBI (Mieth et al., 2021, NAR Genomics and Bioinformatics)
