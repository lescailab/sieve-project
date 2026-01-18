# INSTRUCTIONS.md - Claude Code Session Guide

## Quick Start for Claude Code

When starting a Claude Code session on this project, read CLAUDE.md first for full context. This file provides specific task instructions.

## Current Project State

**Completed**:
- Project documentation (CLAUDE.md, ARCHITECTURE.md, EXPERIMENTS.md)
- Theoretical framework for SIEVE architecture
- Experimental protocol design

**In Progress**:
- None yet - implementation begins now

**Next Priority**:
- Phase 1A: Data pipeline implementation

## Development Phases

### Phase 1A: Data Pipeline (START HERE)

**Goal**: Create robust VCF parsing and feature extraction.

**Tasks**:

1. Create `src/data/vcf_parser.py`:
   - Parse multi-sample VCF using cyvcf2
   - Implement CSQ field parsing with sanitization fix
   - Handle contig harmonization (remove 'chr' prefix)
   - Extract per-sample genotypes efficiently

2. Create `src/data/annotation.py`:
   - Extract VEP annotations: consequence, SIFT, PolyPhen, gene symbol
   - Map consequences to ordinal severity scores
   - Handle missing annotation values gracefully

3. Create `src/encoding/levels.py`:
   - Define AnnotationLevel enum (L0 through L4)
   - Implement feature vector construction for each level
   - Document which features are included at each level

4. Create `src/data/dataset.py`:
   - PyTorch Dataset class for sample iteration
   - Handle variable number of variants per sample
   - Implement collate function for batching with padding

**Acceptance Criteria**:
- Can load a VCF and iterate over samples
- Each sample produces correct feature tensors
- Unit tests pass for known edge cases

**Key Code Pattern for VCF Parsing**:
```python
import cyvcf2

def parse_vcf(vcf_path, phenotype_map):
    """
    Parse VCF and yield per-sample variant data.
    
    Args:
        vcf_path: Path to VEP-annotated multi-sample VCF
        phenotype_map: Dict mapping sample_id -> label (0 or 1)
    
    Yields:
        sample_id, label, variants_df
    """
    vcf = cyvcf2.VCF(vcf_path)
    samples = vcf.samples
    
    # Build variant-sample genotype matrix in chunks to manage memory
    # Implementation details in vcf_parser.py
```

### Phase 1B: Model Implementation

**Goal**: Implement SIEVE architecture components.

**Tasks**:

1. Create `src/encoding/positional.py`:
   - Sinusoidal positional encoding for genomic positions
   - Relative position bucketing for attention bias

2. Create `src/models/attention.py`:
   - PositionAwareSparseAttention class
   - Follow specification in ARCHITECTURE.md exactly
   - Include attention weight output for interpretability

3. Create `src/models/aggregation.py`:
   - GeneAggregator with max pooling
   - Handle variable variants per gene via scatter operations

4. Create `src/models/sieve.py`:
   - Full SIEVE model combining components
   - Forward pass returns logits and attention weights

**Acceptance Criteria**:
- Model instantiates without errors
- Forward pass produces correct output shapes
- Gradients flow through all components

### Phase 1C: Training Pipeline

**Goal**: Implement training loop with attribution regularization.

**Tasks**:

1. Create `src/training/loss.py`:
   - Binary cross-entropy loss
   - Attribution sparsity loss (differentiable)
   - Combined loss with configurable λ_attr

2. Create `src/training/trainer.py`:
   - Training loop with early stopping
   - Logging of losses and metrics
   - Model checkpointing

3. Create `src/training/validation.py`:
   - Nested cross-validation implementation
   - Metric computation (AUC, etc.)

**Acceptance Criteria**:
- Training completes without errors
- Loss decreases over epochs
- Validation metrics are computed correctly

### Phase 1D: Explainability

**Goal**: Implement variant attribution and epistasis detection.

**Tasks**:

1. Create `src/explain/gradients.py`:
   - Integrated gradients using Captum
   - Per-variant attribution scores

2. Create `src/explain/attention.py`:
   - Extract and analyze attention patterns
   - Identify high-attention variant pairs

3. Create `src/explain/counterfactual.py`:
   - Epistasis validation via perturbation
   - Statistical testing for non-additivity

**Acceptance Criteria**:
- Attribution scores computed for all variants
- Epistasis scores match expected additivity formula
- Statistical tests produce sensible p-values

## Coding Guidelines

### File Structure Convention
```python
# src/module/file.py

"""
Module docstring explaining purpose.
"""

import standard_library
import third_party
import local_modules

# Constants
CONSTANT_VALUE = 42

# Main classes/functions
class MainClass:
    """Class docstring."""
    
    def method(self, arg: Type) -> ReturnType:
        """Method docstring in NumPy format."""
        pass

# Helper functions (private)
def _helper_function():
    pass

# Module-level execution (rare)
if __name__ == "__main__":
    pass
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

### Testing Pattern
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

## Common Issues and Solutions

### Memory Issues with Large VCF

**Problem**: VCF with millions of variants exhausts memory.

**Solution**: Process chromosomes independently, use generators:
```python
def iter_variants_by_chrom(vcf_path):
    for chrom in CHROMOSOMES:
        for variant in vcf.query(chrom):
            yield variant
```

### Sparse Tensor Construction

**Problem**: Variable number of variants per sample breaks batching.

**Solution**: Pad to max length within batch, use mask tensor:
```python
def collate_fn(batch):
    max_variants = max(len(sample['variants']) for sample in batch)
    padded = []
    masks = []
    for sample in batch:
        n = len(sample['variants'])
        padded.append(F.pad(sample['variants'], (0, 0, 0, max_variants - n)))
        mask = torch.zeros(max_variants)
        mask[:n] = 1
        masks.append(mask)
    return torch.stack(padded), torch.stack(masks)
```

### Attention Numerical Stability

**Problem**: Softmax over all-masked positions produces NaN.

**Solution**: Replace -inf with large negative, handle NaN:
```python
attn = attn.masked_fill(mask == 0, -1e9)  # Not -inf
attn_weights = F.softmax(attn, dim=-1)
attn_weights = torch.nan_to_num(attn_weights, 0.0)
```

### Gradient Computation for Attribution

**Problem**: Captum requires specific input format.

**Solution**: Wrap model to match Captum interface:
```python
class ModelWrapper(nn.Module):
    def __init__(self, model, positions, gene_assignments, mask):
        super().__init__()
        self.model = model
        self.positions = positions
        self.gene_assignments = gene_assignments
        self.mask = mask
    
    def forward(self, variant_features):
        logits, _ = self.model(variant_features, self.positions, 
                               self.gene_assignments, self.mask)
        return logits
```

## Testing with Synthetic Data

For development, create synthetic data that mimics real VCF structure:

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

## Questions to Ask Before Coding

Before implementing a component, verify:

1. **Input/output shapes**: What tensors go in and come out?
2. **Edge cases**: What if a gene has zero variants? A sample has zero variants?
3. **Gradients**: Does this operation need to be differentiable?
4. **Memory**: Will this fit in GPU memory for expected data sizes?
5. **Interpretability**: Can we extract meaningful information afterward?

## Commit Guidelines

Use semantic commit messages:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `test:` adding tests
- `refactor:` code change that doesn't add feature or fix bug

Example: `feat: implement VCF parser with CSQ field handling`

## When Stuck

1. **Check ARCHITECTURE.md** for mathematical specifications
2. **Check EXPERIMENTS.md** for expected behaviors
3. **Run existing tests** to verify what works
4. **Create minimal reproduction** of the issue
5. **Search literature** for how others solved similar problems
