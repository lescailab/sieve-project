# INSTRUCTIONS.md - Claude Code Session Guide

## Quick Start for Claude Code

When starting a Claude Code session on this project, read CLAUDE.md first for full context. This file provides specific task instructions and workflow guidance.

## Repository Information

- **Repository**: https://github.com/lescailab/sieve-project
- **Branch naming**: Use `claude/<descriptive-name>-<session-id>` for feature branches
- **Main branch**: Check gitStatus for current main branch name

## Current Project State (Updated 2026-02-04)

**Completed**:
- ✅ Phase 1A: Data pipeline (VCF parsing, feature extraction, datasets)
- ✅ Phase 1B: Model architecture (attention, aggregation, SIEVE model)
- ✅ Phase 1C: Training pipeline (loss functions, trainer, cross-validation)
- ✅ Phase 1D: Explainability (integrated gradients, attention analysis, variant ranking)
- ✅ Phase 2: Chunked processing for whole-genome coverage
- ✅ Phase 3: Null baseline attribution analysis pipeline

**In Progress**:
- Validation on real datasets
- Performance optimization
- Additional visualization tools

**Next Priority**:
- Biological validation of discoveries
- Annotation ablation experiments (L0-L4 comparison)
- Publication-ready figure generation

## Critical Workflow: Null Baseline Analysis

The null baseline is essential for validating discoveries. **Always** run null baseline analysis after training and explainability to establish statistical significance.

### Quick Reference

```bash
# 1. Permute labels
python scripts/create_null_baseline.py \
    --input data/preprocessed.pt \
    --output data/preprocessed_NULL.pt

# 2. Train null model (same params as real!)
python scripts/train.py \
    --preprocessed-data data/preprocessed_NULL.pt \
    [... exact same parameters as real model ...]

# 3. Run explainability on null
python scripts/explain.py \
    --experiment-dir experiments/null_baseline \
    --preprocessed-data data/preprocessed_NULL.pt \
    --output-dir results/null_attributions \
    --is-null-baseline

# 4. Compare
python scripts/compare_attributions.py \
    --real results/real/sieve_variant_rankings.csv \
    --null results/null/sieve_variant_rankings.csv \
    --output-dir results/comparison
```

### Or use the wrapper:

```bash
export INPUT_DATA=data/preprocessed.pt
export REAL_EXPERIMENT=experiments/real_model
bash scripts/run_null_baseline_analysis.sh
```

## Complete SIEVE Workflow

### 1. Preprocessing

```bash
python scripts/preprocess.py \
    --vcf data/cohort.vcf.gz \
    --phenotypes data/phenotypes.tsv \
    --output data/preprocessed.pt
```

**Validates**:
- VCF is VEP-annotated
- Phenotypes match VCF samples
- Chromosome notation is harmonized

### 2. Training

```bash
python scripts/train.py \
    --preprocessed-data data/preprocessed.pt \
    --level L3 \
    --val-split 0.2 \
    --lr 0.00001 \
    --lambda-attr 0.1 \
    --epochs 100 \
    --batch-size 16 \
    --chunk-size 3000 \
    --aggregation-method mean \
    --gradient-accumulation-steps 4 \
    --latent-dim 32 \
    --hidden-dim 64 \
    --num-attention-layers 1 \
    --output-dir experiments \
    --experiment-name my_experiment \
    --device cuda
```

**Monitors**:
- Training/validation loss curves
- AUC on validation set
- Early stopping triggers

**Expected**: Validation AUC > 0.6 indicates model is learning signal

### 3. Explainability Analysis

```bash
python scripts/explain.py \
    --experiment-dir experiments/my_experiment \
    --preprocessed-data data/preprocessed.pt \
    --output-dir results/explainability \
    --n-steps 50 \
    --device cuda
```

**Produces**:
- `sieve_variant_rankings.csv` - Variant attributions
- `sieve_gene_rankings.csv` - Gene-level scores
- `sieve_interactions.csv` - High-attention pairs
- `attributions.npz` - Raw attribution arrays

**Validates**: Check that chromosome distribution covers all chroms (not just chr1/chr2)

### 4. Null Baseline (CRITICAL)

See "Critical Workflow" section above.

**Expected Results**:
- Null model validation AUC ≈ 0.50 (chance level)
- Real attributions show enrichment > 1.5× at p<0.01
- KS test p-value < 0.001 indicates distributions differ

**Interpretation**:
- Enrichment < 1.5×: Weak signal, be cautious
- Enrichment 1.5-2×: Moderate signal, validate carefully
- Enrichment > 2×: Strong signal, proceed with confidence

### 5. Biological Validation

Review `significant_variants_p01.csv` for candidates:
- Cross-reference with GWAS catalog
- Check gene function (OMIM, GeneCards)
- Look for compound heterozygosity patterns
- Validate in independent cohort if possible

## Development Phases (Historical Reference)

### Phase 1A: Data Pipeline ✅ COMPLETE

**Created**:
- `src/data/vcf_parser.py` - Multi-sample VCF parsing with cyvcf2
- `src/data/annotation.py` - VEP annotation extraction
- `src/encoding/levels.py` - L0-L4 annotation level definitions
- `src/data/dataset.py` - PyTorch datasets with chunking

**Key Features**:
- CSQ field sanitization
- Contig harmonization (removes 'chr' prefix)
- Memory-efficient chromosome-by-chromosome processing

### Phase 1B: Model Implementation ✅ COMPLETE

**Created**:
- `src/encoding/positional.py` - Sinusoidal positional encodings
- `src/models/attention.py` - Position-aware sparse attention
- `src/models/aggregation.py` - Gene-level aggregation (mean/max)
- `src/models/sieve.py` - Full SIEVE model

**Key Features**:
- Sparse attention operates only on variant-present positions
- Relative position encoding in attention
- Permutation-invariant gene aggregation

### Phase 1C: Training Pipeline ✅ COMPLETE

**Created**:
- `src/training/loss.py` - Classification + attribution sparsity loss
- `src/training/trainer.py` - Training loop with early stopping
- `src/training/validation.py` - Cross-validation utilities
- `scripts/train.py` - Training entry point

**Key Features**:
- Attribution-regularized training (λ controls sparsity)
- Gradient accumulation for large effective batch sizes
- Automatic model checkpointing

### Phase 1D: Explainability ✅ COMPLETE

**Created**:
- `src/explain/gradients.py` - Integrated gradients with Captum
- `src/explain/attention.py` - Attention weight analysis
- `src/explain/variant_ranking.py` - Ranking and aggregation
- `src/explain/epistasis.py` - Non-additivity detection
- `scripts/explain.py` - Explainability entry point

**Key Features**:
- Per-variant attributions via integrated gradients
- Case vs control enrichment analysis
- Attention-based interaction detection
- Counterfactual epistasis validation

### Phase 2: Chunked Processing ✅ COMPLETE

**Fixed**:
- Whole-genome chunked variant processing
- Position collision bug (chromosome awareness)
- Memory-efficient batching for large cohorts

**Impact**: Now covers all chromosomes, not just chr1/chr2

### Phase 3: Null Baseline Analysis ✅ COMPLETE

**Created**:
- `scripts/create_null_baseline.py` - Label permutation
- `scripts/compare_attributions.py` - Statistical comparison
- `scripts/run_null_baseline_analysis.sh` - Full pipeline wrapper
- `tests/test_null_baseline.py` - Comprehensive tests

**Key Features**:
- Reproducible permutation with local RNG (no global state)
- Multiple null permutations for robust baseline
- Statistical tests (KS, Mann-Whitney)
- Enrichment factors at multiple p-value thresholds
- Automatic significance annotation

## Coding Guidelines

### Testing Requirements

**ALWAYS test code before pushing!** Workflow:
1. Write code
2. **Run tests** ← MANDATORY
3. Fix any failures
4. Commit
5. Push

### File Structure Convention

```python
# src/module/file.py

"""
Module docstring explaining purpose.

Author: Lescai Lab
"""

import standard_library
import third_party
from typing import Type, Optional

import local_modules

# Constants
CONSTANT_VALUE = 42

# Main classes/functions
class MainClass:
    """Class docstring in NumPy format."""

    def method(self, arg: Type) -> ReturnType:
        """
        Method docstring.

        Parameters
        ----------
        arg : Type
            Description

        Returns
        -------
        ReturnType
            Description
        """
        pass
```

### Import Order

1. Standard library (e.g., `argparse`, `pathlib`)
2. Third-party (e.g., `numpy`, `torch`, `pandas`)
3. Local modules (e.g., `from src.data import ...`)

### Type Hints

- Use type hints on all public function signatures
- Use `Optional[T]` for nullable types
- Use `List[T]`, `Dict[K, V]` for collections
- Remove unused imports (ruff/pyflakes will flag)

### Random Number Generation

**Always use local RNG, never mutate global state:**

```python
# ✓ GOOD
rng = np.random.default_rng(seed)
permuted = rng.permutation(array)

# ✗ BAD
np.random.seed(seed)  # Mutates global state!
permuted = np.random.permutation(array)
```

### Error Handling

```python
# Be explicit about expected errors
try:
    result = risky_operation()
except SpecificError as e:
    logger.warning(f"Expected error handled: {e}")
    result = fallback_value
# Don't catch broad Exception unless re-raising
```

### matplotlib Backend

**Always set backend before importing pyplot:**

```python
# ✓ GOOD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ✗ BAD
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Too late!
```

## Common Issues and Solutions

### Memory Issues with Large VCF

**Problem**: VCF with millions of variants exhausts memory.

**Solution**: Process chromosomes independently, use chunking:

```python
dataset = ChunkedVariantDataset(
    samples=all_samples,
    annotation_level=level,
    chunk_size=3000,  # Adjust based on available memory
    overlap=0
)
```

### Sparse Tensor Construction

**Problem**: Variable number of variants per sample breaks batching.

**Solution**: Use ChunkedVariantDataset which handles padding automatically:

```python
def collate_chunks(batch):
    """Handles variable-length chunks with automatic padding."""
    # Implementation in src/data/dataset.py
```

### Attention Numerical Stability

**Problem**: Softmax over all-masked positions produces NaN.

**Solution**: Use large negative (not -inf) and handle NaN:

```python
attn = attn.masked_fill(mask == 0, -1e9)  # Not -inf
attn_weights = F.softmax(attn, dim=-1)
attn_weights = torch.nan_to_num(attn_weights, 0.0)
```

### Model Not Learning (AUC ≈ 0.5)

**Checklist**:
1. Verify labels are correctly loaded (check metadata in preprocessed file)
2. Check case/control balance (should not be extremely skewed)
3. Try lower learning rate (`--lr 0.000001`)
4. Reduce attribution regularization (`--lambda-attr 0.01`)
5. Verify data is not all-zeros (check feature distributions)
6. Check for data leakage in preprocessing

### Position Collision Bug

**Fixed in Phase 2**: Variants on different chromosomes can have the same numerical position. Solution: Always include chromosome in variant keys.

```python
# ✓ GOOD
variant_key = (chrom, pos, gene_id)

# ✗ BAD
variant_key = (pos, gene_id)  # Will collide across chroms!
```

## Testing Patterns

### Unit Tests

```python
# tests/test_module.py

import pytest
from src.module import function_to_test

class TestFunctionName:
    """Tests for function_to_test."""

    def test_normal_case(self):
        """Test with typical input."""
        result = function_to_test(normal_input)
        assert result == expected_output

    def test_edge_case(self):
        """Test with edge case input."""
        result = function_to_test(edge_input)
        assert result == expected_edge_output

    def test_error_case(self):
        """Test that errors are raised correctly."""
        with pytest.raises(ExpectedError):
            function_to_test(bad_input)
```

### Integration Tests

Create end-to-end tests with mock data:

```python
def test_full_pipeline():
    """Test complete workflow with synthetic data."""
    # 1. Create mock data
    # 2. Run preprocessing
    # 3. Train model
    # 4. Run explainability
    # 5. Validate outputs
```

### Testing with Synthetic Data

```python
def create_synthetic_dataset(n_samples=100, n_variants=1000, n_genes=50):
    """Create synthetic data for testing."""
    # Genotypes: mostly 0, some 1, few 2
    genotypes = np.random.choice([0, 1, 2], size=(n_samples, n_variants),
                                  p=[0.95, 0.04, 0.01])

    # Positions: sorted within chromosome
    positions = np.sort(np.random.randint(0, 10_000_000, size=n_variants))

    # Gene assignments: random but contiguous
    gene_assignments = np.repeat(np.arange(n_genes), n_variants // n_genes + 1)[:n_variants]

    # Labels: binary with some variants associated
    causal_variants = np.random.choice(n_variants, size=10, replace=False)
    risk_scores = genotypes[:, causal_variants].sum(axis=1)
    labels = (risk_scores > np.median(risk_scores)).astype(int)

    return {
        'genotypes': genotypes,
        'positions': positions,
        'gene_assignments': gene_assignments,
        'labels': labels,
        'causal_variants': causal_variants  # For validation
    }
```

## Commit Guidelines

Use semantic commit messages:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `test:` adding tests
- `refactor:` code change that doesn't add feature or fix bug

Always include session URL in commit message:
```
feat: implement null baseline attribution analysis

Adds complete pipeline for statistical validation of variant
discoveries by comparing against null distribution from permuted labels.

https://claude.ai/code/session_<SESSION_ID>
```

## Questions to Ask Before Coding

Before implementing a component, verify:

1. **Input/output shapes**: What tensors go in and come out?
2. **Edge cases**: What if a gene has zero variants? A sample has zero variants?
3. **Gradients**: Does this operation need to be differentiable?
4. **Memory**: Will this fit in GPU memory for expected data sizes?
5. **Interpretability**: Can we extract meaningful information afterward?
6. **Testing**: How will I test this before pushing?

## When Stuck

1. **Check ARCHITECTURE.md** for mathematical specifications
2. **Check EXPERIMENTS.md** for expected behaviors
3. **Check CLAUDE.md** for project context
4. **Run existing tests** to verify what works
5. **Create minimal reproduction** of the issue
6. **Search literature** for how others solved similar problems

## Performance Optimization Tips

### Memory Optimization

- Use `chunk_size` to control memory usage
- Process chromosomes independently if needed
- Use gradient checkpointing for very deep models
- Clear CUDA cache between large operations: `torch.cuda.empty_cache()`

### Speed Optimization

- Increase `chunk_size` if memory allows
- Use larger `batch_size` with gradient accumulation
- Profile with `torch.profiler` to find bottlenecks
- Consider mixed precision training (fp16) for faster training

### Debugging Performance

```python
import time

start = time.time()
result = expensive_operation()
print(f"Operation took {time.time() - start:.2f}s")
```

## Code Review Checklist

Before pushing code, verify:

- [ ] All tests pass
- [ ] No unused imports (check with `ruff` or `flake8`)
- [ ] Type hints on public functions
- [ ] Docstrings in NumPy format
- [ ] No global RNG state mutation
- [ ] matplotlib backend set correctly (if using plots)
- [ ] No hard-coded paths (use arguments or env vars)
- [ ] Session URL in commit message

## Resources

- **CLAUDE.md**: Full project context
- **ARCHITECTURE.md**: Model specifications
- **EXPERIMENTS.md**: Experimental protocol
- **README.md**: User documentation
- **tests/**: Unit and integration tests
- **GitHub Issues**: https://github.com/lescailab/sieve-project/issues
