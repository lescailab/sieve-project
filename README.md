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

- Multi-sample VCF file **annotated with Ensembl VEP** (`.vcf.gz` with tabix index)
- Phenotype file mapping samples to case/control labels

#### Annotate your VCF with VEP

If your VCF is not already annotated, you **must** run Ensembl VEP first.
SIEVE will refuse to preprocess an unannotated VCF.

```bash
# Install VEP and download cache (once)
conda install -c bioconda ensembl-vep
vep_install -a cf -s homo_sapiens -y GRCh37 -c /path/to/vep_cache

# Annotate
vep \
    --input_file variants.vcf.gz \
    --output_file variants_vep.vcf.gz \
    --vcf \
    --compress_output bgzip \
    --symbol \
    --canonical \
    --sift b \
    --polyphen b \
    --assembly GRCh37 \
    --offline \
    --cache \
    --dir_cache /path/to/vep_cache \
    --fork 4

# Index
tabix -p vcf variants_vep.vcf.gz
```

> **Important**: Do not use a custom `--fields` argument. SIEVE expects VEP's
> default CSQ field order. See the [User Guide](USER_GUIDE.md) for details on
> which fields are used and why.

#### Phenotype file

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
# Option 1: Use the complete pipeline wrapper (preferred — cohort-centric layout)
PROJECT_DIR=/data/CohortName \
LEVEL=L3 \
bash scripts/run_null_baseline_analysis.sh

# Option 2: Legacy variable interface
export INPUT_DATA=data/preprocessed.pt
export REAL_EXPERIMENT=experiments/my_model
export REAL_RESULTS=results/explainability
export OUTPUT_BASE=results/null_baseline_run
bash scripts/run_null_baseline_analysis.sh

# Option 3: Run steps manually
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

# Step 4: Compare raw real vs raw null attributions
python scripts/compare_attributions.py \
    --real results/explainability/sieve_variant_rankings.csv \
    --null results/null_attributions/sieve_variant_rankings.csv \
    --output-dir results/attribution_comparison

# Step 5: (Separate) Apply chrX correction to significance-annotated file
python scripts/correct_chrx_bias.py \
    --rankings results/attribution_comparison/variant_rankings_with_significance.csv \
    --output-dir results/attribution_comparison/corrected \
    --include-sex-chroms
```

The comparison produces:

- `variant_rankings_with_significance.csv` - All variants annotated with `empirical_p_variant` and `fdr_variant`
- `gene_rankings_with_significance.csv` - Gene-level rankings with `empirical_p_gene` and `fdr_gene`
- `significance_summary.yaml` - Counts of variants and genes passing FDR thresholds

### 6. Annotation Ablation (Compare Levels)

```bash
# Train and explain at each annotation level
for LEVEL in L0 L1 L2 L3; do
    python scripts/train.py \
        --preprocessed-data data/preprocessed.pt \
        --level $LEVEL \
        --experiment-name ablation_$LEVEL \
        --output-dir experiments

    python scripts/explain.py \
        --experiment-dir experiments/ablation_$LEVEL \
        --preprocessed-data data/preprocessed.pt \
        --output-dir results/${LEVEL}_explainability \
        --device cuda
done

# Compare model performance across levels
python scripts/ablation_compare.py \
    --results-dir experiments \
    --out-summary-yaml results/ablation/ablation_summary.yaml

# Compare variant attribution rankings across levels
mkdir -p results/ablation/rankings
for LEVEL in L0 L1 L2 L3; do
    cp results/${LEVEL}_explainability/sieve_variant_rankings.csv \
       results/ablation/rankings/${LEVEL}_sieve_variant_rankings.csv
done

python scripts/compare_ablation_rankings.py \
    --ranking-dir results/ablation/rankings \
    --out-jaccard results/ablation/ablation_jaccard_matrix.tsv \
    --out-level-specific results/ablation/level_specific_variants.tsv

# Visualise the ablation comparison
python scripts/plot_ablation_comparison.py \
    --jaccard-tsv results/ablation/ablation_jaccard_matrix.tsv \
    --level-specific-tsv results/ablation/level_specific_variants.tsv \
    --summary-yaml results/ablation/ablation_summary.yaml \
    --output results/ablation/ablation_comparison.png
```

## Complete Workflow

The recommended SIEVE workflow:

```text
1. Preprocess VCF → preprocessed.pt
2. Train model(s) → experiments/ (repeat for each annotation level)
3. Run explainability → results/ (repeat for each annotation level)
4. Create null baseline → preprocessed_NULL.pt
5. Train null model → experiments/null_baseline/
6. Run null explainability → results/null_attributions/
7. Compare real vs null → results/comparison/
   ↓ Review significant_variants_p01.csv
8. Compare across annotation levels → results/ablation/
   ↓ Review ablation_comparison.png + level_specific_variants.tsv
9. Biological validation (experiments)
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
- `sieve_interactions.csv` - High-attention within-chunk variant pairs from the intrinsic attention analysis
  - Best interpreted as candidate interaction seeds for counterfactual validation, not as an exhaustive catalogue of all cohort interactions

**From Null Comparison:**

- `comparison_summary.yaml` - Statistical summary
  - Thresholds at p<0.10, 0.05, 0.01, 0.001
  - KS and Mann-Whitney test results
  - Enrichment factors
- `significant_variants_p01.csv` - Statistically significant discoveries
- `variant_rankings_with_significance.csv` - All variants with significance flags

**From Post-hoc Interaction Analysis:**

- `cooccurrence_summary.yaml` - Variant-pair co-occurrence audit stratified by allele frequency
- `power_analysis_summary.yaml` - Detectable interaction effect sizes using null-informed noise and pair contingency tables
- `gene_pair_interactions.csv` - Gene-gene interaction hypotheses combining co-occurrence and attribution support

**From Ablation Comparison:**

- `ablation_summary.yaml` - Performance (AUC, accuracy, loss) per annotation level
- `ablation_jaccard_matrix.tsv` - Pairwise ranking overlap at multiple top-k thresholds

### Epistasis Workflows

SIEVE now supports two complementary views of epistasis:

1. **Attention-based discovery**: `scripts/explain.py` extracts high-attention variant pairs and `scripts/validate_epistasis.py` tests them by counterfactual perturbation. This is the most direct and novel interaction signal because it comes from the model's intrinsic attention patterns. Its main current limitation is that interactions are only visible when both variants occur in the same chunk.
2. **Post-hoc attribution interaction analysis**: `scripts/audit_cooccurrence.py`, `scripts/aggregate_gene_interactions.py`, and `scripts/epistasis_power_analysis.py` analyse co-occurrence, effective interaction sample size, and gene-level interaction structure using the model's intrinsic attribution outputs. This is also not an external post-hoc explainer in the usual sense: it reuses attribution signals that are part of SIEVE's training and interpretation workflow.

An empty `sieve_interactions.csv` therefore means that no pair crossed the attention discovery heuristic under the current chunking and threshold settings. It does not, by itself, prove that the cohort lacks interaction structure.
Within the co-occurrence audit, `n_pairs_gte5_cooccur` only means at least 5 joint carriers, while `n_pairs_all_cells_gte5` means all four cells of the `2x2` carrier table have at least 5 samples. The latter is the relevant criterion for interaction estimation because non-additive effects require contrast across all four carrier states.
- `level_specific_variants.tsv` - Variants uniquely important at one annotation level
- `ablation_comparison.png` / `.pdf` - Multi-panel publication figure

## Project Structure

```text
sieve-project/
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
│   ├── create_null_baseline.py        # Permute labels
│   ├── compare_attributions.py        # Real vs null comparison
│   ├── run_null_baseline_analysis.sh  # Full null baseline pipeline
│   ├── ablation_compare.py            # Performance across levels
│   ├── compare_ablation_rankings.py   # Ranking overlap across levels
│   └── plot_ablation_comparison.py    # Multi-panel ablation figure
└── tests/                 # Unit tests
```

## Comparison with Existing Methods

| Feature                       | SIEVE | DeepRVAT | GenNet  | GWAS_NN | EpiDetect |
|-------------------------------|-------|----------|---------|---------|-----------|
| Direct VCF input              | ✓     | ✗        | ✗       | ✗       | ✗         |
| Annotation-free discovery     | ✓     | ✗        | ✗       | ✗       | ✗         |
| Position-aware                | ✓     | ✗        | ✗       | ✗       | ✗         |
| Built-in interpretability     | ✓     | ✗        | Partial | ✗       | ✗         |
| Epistasis detection           | ✓     | ✗        | ✗       | ✓       | ✓         |
| Null baseline calibration     | ✓     | ✗        | ✗       | ✗       | ✗         |
| Common + rare variants        | ✓     | Rare only| All     | Common  | Common    |
| Marginal effect filtering     | N/A   | N/A      | N/A     | ✗       | ✓         |
| Sample-level explanations     | ✓     | ✗        | ✗       | ✗       | ✗         |
| Weight-based global explain.  | ✗     | ✗        | ✓       | ✗       | ✓         |
| Network/centrality analysis   | ✗     | ✗        | ✗       | ✗       | ✓         |

EpiDetect is included here as the closest recent epistasis-focused comparator, but it operates in a different regime from SIEVE: a small pre-filtered set of GWAS-significant common SNPs, a shallow MLP for continuous-trait regression, very large cohorts (for example UK Biobank scale), and global weight-based interaction scoring rather than sample-level attribution with null-baseline calibration.

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

### Preprocessing Fails with "does not contain VEP CSQ annotations"

Your VCF has not been annotated with Ensembl VEP. Run `vep` as described in the
Quick Start section before calling `sieve-preprocess`. See the
[User Guide](USER_GUIDE.md#how-to-annotate-your-vcf-with-ensembl-vep) for the
full command and required flags.

### Preprocessing Fails with "zero variant-sample assignments"

The VCF header declares a CSQ field, but no variants were loaded. The error
message includes diagnostics; the most common causes are:

- **All CSQ values are empty**: The VCF was re-header'd or filtered after VEP
  annotation, stripping the actual CSQ values while keeping the header line.
- **Allele mismatch**: VEP's allele representation in CSQ doesn't match the
  VCF ALT field (can happen with post-VEP normalisation tools).
- **All genotypes filtered**: Every genotype fell below the GQ threshold
  (default 20). Try `--min-gq 0` to test.

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
- EpiDetect (Mastropietro et al., 2026, NAR Genomics and Bioinformatics)

## Support

- **Issues**: [GitHub Issues](https://github.com/lescailab/sieve-project/issues)
- **Documentation**: See USER_GUIDE.md
- **Questions**: Open a GitHub discussion

---

**Note**: SIEVE is research software. Results should be validated experimentally before making clinical or biological claims.
