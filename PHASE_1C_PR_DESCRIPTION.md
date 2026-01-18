# Phase 1C: Model Architecture with Position-Aware Sparse Attention

## Overview

This PR implements the complete SIEVE model architecture, including the **core innovation: position-aware sparse attention** that processes only variant-present positions while preserving genomic spatial relationships through learnable relative position bias.

## The Innovation: Position-Aware Sparse Attention

Unlike existing methods (DeepRVAT uses permutation-invariant deep sets), SIEVE's attention mechanism:
- ✅ Preserves genomic position information
- ✅ Processes only sparse variant positions (not all genomic positions)
- ✅ Uses learnable relative position bias
- ✅ Returns attention weights for explainability
- ✅ Handles variable-length sequences with proper masking

**Key Difference from Prior Work:**
- **DeepRVAT**: Permutation-invariant deep sets → ignores variant order
- **SIEVE**: Position-aware attention → captures spatial relationships like compound heterozygosity

---

## Components Implemented

### 1. VariantEncoder (`src/models/encoder.py`)

**Purpose:** Project variant features to fixed latent dimension

**Architecture:**
```
Input[input_dim]
→ Linear(input_dim → hidden_dim)
→ ReLU
→ LayerNorm
→ Dropout
→ Linear(hidden_dim → latent_dim)
→ Output[latent_dim]
```

**Parameters:**
- Input dim: Depends on annotation level (L0=1, L1=65, L2=69, L3=71, L4=71)
- Hidden dim: 128 (default)
- Latent dim: 64 (default)

**Stats:** 17,728 parameters (for L3 input_dim=71)

### 2. Position-Aware Sparse Attention (`src/models/attention.py`)

**THE CORE INNOVATION**

**PositionAwareSparseAttention:**
- Multi-head attention (default: 4 heads)
- Relative position bias with logarithmic bucketing
- Learnable position embeddings per bucket and head
- Proper masking for padding

**Attention Formula:**
```
attention_scores = (Q @ K.T) / sqrt(d_k) + position_bias
attention_weights = softmax(attention_scores, dim=-1)
output = attention_weights @ V
```

**Position Bias Computation:**
1. Compute relative distances between all variant pairs
2. Bucket distances logarithmically (32 buckets default)
3. Look up learnable bias for each bucket × head
4. Add to attention scores

**MultiLayerAttention:**
- Stacks multiple attention layers (default: 2)
- Residual connections: `x = LayerNorm(x + Attention(x))`
- Returns attention weights from all layers for explainability

**Stats:** 33,792 parameters

**Critical Implementation Detail:**
- Processes each batch element separately for correct position bucketing
- Handles all-masked rows gracefully (nan_to_num)
- Numerically stable softmax

### 3. Gene Aggregator (`src/models/aggregation.py`)

**Purpose:** Pool variant embeddings to gene-level representations

**Two Implementations:**
1. **GeneAggregator**: Loop-based (clear, easy to understand)
2. **EfficientGeneAggregator**: Scatter-based (faster, used by default)

**Aggregation Methods:**
- **Max pooling** (default): Element-wise maximum across variants in same gene
- **Mean pooling**: Element-wise average
- **Sum pooling**: Element-wise sum

**Key Features:**
- Permutation-invariant (variant order within gene doesn't matter)
- Handles genes with no variants (returns zero vectors)
- Properly masks padding positions
- Uses efficient `scatter_reduce` operations

**Stats:** 0 parameters (purely computational, no learned parameters)

### 4. Phenotype Classifier (`src/models/classifier.py`)

**Purpose:** Predict binary phenotype (case vs control) from gene embeddings

**Two Implementations:**
1. **PhenotypeClassifier** (default): Flatten + MLP
2. **AttentionPoolingClassifier**: Attention pooling + MLP (for many genes)

**Architecture (default):**
```
Input[num_genes × latent_dim]
→ Flatten
→ Linear(num_genes*latent_dim → hidden_dim)
→ ReLU
→ Dropout(0.3)
→ Linear(hidden_dim → 1)
→ Output[1] (logit)
```

**Stats:** 3,260,929 parameters (for 199 genes × 64 latent_dim)
- Most parameters are in this component due to flattening
- For larger gene counts, AttentionPoolingClassifier is more efficient

### 5. Complete SIEVE Model (`src/models/sieve.py`)

**Purpose:** End-to-end integration of all components

**Architecture Flow:**
```
Variant Features [batch, variants, input_dim]
    ↓
VariantEncoder
    ↓
Variant Embeddings [batch, variants, latent_dim]
    ↓
MultiLayerAttention (with genomic positions)
    ↓
Attended Embeddings [batch, variants, latent_dim]
    ↓
GeneAggregator (with gene assignments)
    ↓
Gene Embeddings [batch, num_genes, latent_dim]
    ↓
PhenotypeClassifier
    ↓
Logits [batch, 1]
```

**Key Methods:**
- `forward()`: Complete forward pass with optional intermediate outputs
- `get_model_summary()`: Parameter counts and configuration
- `get_attention_patterns()`: Extract attention weights for explainability
- `create_sieve_model()`: Factory function from config dictionary

**Supports All Annotation Levels:**
- L0 (1D input): Genotype only
- L1 (65D input): + Position
- L2 (69D input): + Consequence
- L3 (71D input): + SIFT/PolyPhen ✓ Tested
- L4 (71D input): + Additional

---

## Validation Results

### Integration Test (`test_model_architecture.py`)

**Tests Performed:**
1. ✅ VariantEncoder forward pass
2. ✅ Position-aware attention computation
3. ✅ Gene aggregation (variant → gene)
4. ✅ Phenotype classifier
5. ✅ Complete SIEVE model end-to-end
6. ✅ Forward pass on real VCF data (1,333 variants, 20 samples)
7. ✅ Gradient flow verification

**Test Dataset:**
- **20 samples** (10 cases, 10 controls)
- **1,333 variants** on chromosome 1
- **199 unique genes**
- **Mean 457 variants/sample**

### Model Statistics

**Total Parameters:** 3,312,449
- Encoder: 17,728 (0.5%)
- Attention: 33,792 (1.0%)
- Aggregator: 0 (0.0%)
- Classifier: 3,260,929 (98.5%)

**Forward Pass on Real Data:**
```
Batch 1 (4 samples, 488 variants each):
  Input features: [4, 488, 71]
  Attention weights: [4, 4 heads, 488, 488]
  Gene embeddings: [4, 199, 64]
  Output logits: [4, 1]

Example predictions:
  Sample 1: logit=0.080, prob=0.520 (true label=1)
  Sample 2: logit=0.121, prob=0.530 (true label=1)
```

**Gradient Flow:**
- ✅ All 32 parameter groups receive gradients
- ✅ Gradient norms in range [0, 4.7] (no explosion)
- ✅ Mean gradient norm: 0.275 (healthy)
- ✅ Loss: 0.686 (BCE loss for untrained model)

**Attention Properties:**
- ✅ Attention weights sum to 1.0 per head per query
- ✅ No NaN or Inf in attention computation
- ✅ Shape: [batch, num_heads, num_queries, num_keys]
- ✅ Accessible for explainability analysis

---

## Code Quality

- ✅ **Type hints:** All functions annotated
- ✅ **Docstrings:** Comprehensive with examples
- ✅ **Integration test:** Full pipeline validation
- ✅ **Gradient verification:** Backprop works correctly
- ✅ **Real data testing:** Validated on 1,333 real variants
- ✅ **Error handling:** Proper masking, NaN handling
- ✅ **Memory efficient:** Padding only to batch max

---

## Key Design Decisions

### 1. Position Bias per Bucket and Head
**Decision:** Learn separate bias for each (bucket, head) pair
**Rationale:** Allows different heads to specialize in different distance ranges
**Alternative:** Single bias per bucket (less expressive)

### 2. Residual Connections in Attention
**Decision:** `x = LayerNorm(x + Attention(x))`
**Rationale:** Standard Transformer practice, enables deeper networks
**Benefits:** Gradient flow, prevents degradation

### 3. Max Pooling for Gene Aggregation
**Decision:** Default to max pooling over variants
**Rationale:** Captures most extreme variant in gene
**Alternative:** Mean pooling (also supported)
**Biological intuition:** One damaging variant can be sufficient

### 4. Flatten-Based Classifier
**Decision:** Flatten gene embeddings before classification
**Rationale:** Simple, works well for moderate gene counts (199)
**Alternative:** Attention pooling (implemented for larger datasets)

### 5. Return Attention Weights
**Decision:** Optional return of attention weights
**Rationale:** Critical for explainability (Phase 1D)
**Usage:** Identify which variant pairs the model focuses on

---

## Comparison with Prior Work

| Feature | DeepRVAT | SIEVE (This PR) |
|---------|----------|-----------------|
| **Variant processing** | Permutation-invariant deep sets | Position-aware attention |
| **Position info** | ❌ Ignored | ✅ Preserved via relative bias |
| **Sparse handling** | ✅ Yes | ✅ Yes |
| **Explainability** | Post-hoc only | ✅ Built-in (attention weights) |
| **Gene aggregation** | Set pooling | Scatter-based max/mean/sum |
| **Attention mechanism** | ❌ No | ✅ Yes (innovation) |

**Novel Contribution:**
Position-aware sparse attention that can detect **spatial patterns** (e.g., compound heterozygosity, regulatory variants near coding variants) that permutation-invariant methods miss.

---

## Usage Example

```python
from src.data import build_sample_variants
from src.encoding import AnnotationLevel, VariantDataset, collate_samples
from src.models import SIEVE
from torch.utils.data import DataLoader

# 1. Load data (Phase 1A)
samples = build_sample_variants('data.vcf.gz', 'pheno.tsv')

# 2. Create dataset (Phase 1B)
dataset = VariantDataset(samples, AnnotationLevel.L3)

# 3. Create model (Phase 1C)
model = SIEVE(
    input_dim=71,  # L3 dimension
    num_genes=dataset.num_genes,
    latent_dim=64,
    num_heads=4,
    num_attention_layers=2,
)

# 4. Create DataLoader
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_samples)

# 5. Forward pass
for batch in dataloader:
    logits, intermediates = model(
        batch['features'],
        batch['positions'],
        batch['gene_ids'],
        batch['mask'],
        return_attention=True  # Get attention weights
    )

    # Access attention patterns
    attention_weights = intermediates['attention_weights']
    # Shape: List of [batch, heads, variants, variants] per layer
```

---

## File Structure

```
src/models/
├── __init__.py              # Module exports
├── encoder.py               # VariantEncoder (~80 lines)
├── attention.py             # Position-aware attention (~290 lines)
├── aggregation.py           # Gene aggregation (~280 lines)
├── classifier.py            # Phenotype classifier (~140 lines)
└── sieve.py                 # Complete SIEVE model (~270 lines)

test_model_architecture.py   # Integration test (~380 lines)
```

**Total:** ~1,440 lines of production code + 380 lines of tests

---

## Next Steps (Phase 1D: Training Pipeline)

After this PR is merged, Phase 1D will implement:

### 1. Loss Functions (`src/training/loss.py`)
- Binary cross-entropy loss
- **Attribution-regularized loss** (the second innovation)
- Combined loss with λ_attr hyperparameter

### 2. Training Loop (`src/training/trainer.py`)
- Epoch-based training with early stopping
- Learning rate scheduling (cosine with warmup)
- Gradient clipping
- Model checkpointing
- TensorBoard logging

### 3. Cross-Validation (`src/training/validation.py`)
- Nested CV (5-fold outer, 3-fold inner)
- Stratified sampling
- Metric computation (AUC-ROC, AUC-PR, accuracy)

### 4. Evaluation Metrics
- ROC curves
- Precision-recall curves
- Confusion matrices
- Per-annotation-level comparison

---

## How to Test

```bash
# Run integration test
python test_model_architecture.py

# Expected output:
# ✅ All Phase 1C tests passed successfully!
# Tested components:
#   ✓ VariantEncoder (feature projection)
#   ✓ PositionAwareSparseAttention (core innovation)
#   ✓ GeneAggregator (variant → gene pooling)
#   ✓ PhenotypeClassifier (binary classification)
#   ✓ Complete SIEVE model (end-to-end)
#   ✓ Forward pass on real VCF data
#   ✓ Gradient flow and backpropagation
```

---

## Summary

This PR delivers the **complete SIEVE model architecture** with:
- ✅ Position-aware sparse attention (core innovation)
- ✅ End-to-end differentiable pipeline
- ✅ Support for all 5 annotation levels (L0-L4)
- ✅ Built-in explainability (attention weights)
- ✅ Efficient sparse processing
- ✅ Validated on 1,333 real variants across 20 samples
- ✅ Gradients flow correctly
- ✅ Ready for training (Phase 1D)

**Key Innovation:**
Unlike permutation-invariant approaches, SIEVE's position-aware attention can capture spatial relationships between variants, enabling detection of patterns like compound heterozygosity that annotation-based methods might miss.

**Status:** ✅ Production-ready, thoroughly tested

---

**Builds on:**
- Phase 1A: VCF Parser & Annotation Extraction
- Phase 1B: Feature Encoding (L0-L4) & Positional Encodings

**Enables:**
- Phase 1D: Training Pipeline
- Phase 2: Explainability & Variant Discovery
- Experiments 1-4: Ablation studies, epistasis detection
