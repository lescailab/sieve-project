# ARCHITECTURE.md - SIEVE Model Architecture

## Overview

SIEVE (Sparse Interpretable Exome Variant Explainer) is designed around three core innovations:
1. Position-aware sparse attention that preserves spatial relationships between variants
2. Attribution-regularized training that builds interpretability into the objective
3. Annotation-ablation protocol that distinguishes genuine discovery from annotation recovery

This document provides the mathematical foundations and implementation details.

## Data Representation

### Input: Annotated Multi-sample VCF

The input is a VCF file with N samples and V variant sites. Each variant site v has:
- Chromosome and position (chrom_v, pos_v)
- Reference and alternate alleles
- Per-sample genotypes g_{v,n} ∈ {0, 1, 2} (reference homozygote, heterozygote, alternate homozygote)
- VEP annotations including gene assignment, consequence, and functional scores

### Variant Feature Vector

Each variant v is represented by a feature vector x_v whose composition depends on the annotation level:

**Level L0** (genotype only):
```
x_v = [g_v]  # Just genotype dosage, dimension 1
```

**Level L1** (genotype + position):
```
x_v = [g_v, PE(pos_v)]  # Genotype + positional encoding, dimension 1 + d_pos
```

**Level L2** (L1 + consequence):
```
x_v = [g_v, PE(pos_v), one_hot(consequence_v)]  # Add consequence class
```

**Level L3** (L2 + SIFT/PolyPhen):
```
x_v = [g_v, PE(pos_v), one_hot(consequence_v), sift_v, polyphen_v]
```

**Level L4** (full annotations):
```
x_v = [g_v, PE(pos_v), one_hot(consequence_v), sift_v, polyphen_v, lof_v, ...]
```

### Positional Encoding

We use sinusoidal positional encodings adapted for genomic positions:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_pos))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_pos))
```

where d_pos is the positional embedding dimension (default 64).

This encoding allows the model to learn functions of relative position (important for detecting compound heterozygosity or clustered variants).

### Per-Sample Sparse Representation

For sample n, we construct a sparse representation S_n containing only their non-reference variants:

```
S_n = {(v, x_v, g_{v,n}) : g_{v,n} > 0}
```

This is the key to handling sparsity: we never materialize the full V-dimensional tensor, only the positions where the individual has variants.

## Model Architecture

### Component 1: Variant Encoder

A small MLP that projects variant features into a latent space:

```python
class VariantEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        # x: (batch, num_variants, input_dim)
        return self.mlp(x)  # (batch, num_variants, latent_dim)
```

### Component 2: Position-Aware Sparse Attention

This is the core architectural innovation. Standard self-attention is O(n²) in sequence length, infeasible for millions of genomic positions. Standard sparse attention (like in BigBird) uses fixed sparsity patterns. Our approach is naturally sparse because we only attend among variant-present positions.

The key insight: we want to preserve positional information without requiring dense encoding. We achieve this by:
1. Operating only on positions with variants (natural sparsity)
2. Encoding relative distances in attention computation

```python
class PositionAwareSparseAttention(nn.Module):
    def __init__(self, latent_dim, num_heads, max_relative_distance=100000):
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        
        # Relative position bias (bucketed)
        self.num_buckets = 32
        self.rel_pos_bias = nn.Embedding(self.num_buckets, num_heads)
        self.max_distance = max_relative_distance
    
    def _relative_position_bucket(self, relative_position):
        """
        Bucket relative positions into discrete bins.
        Uses logarithmic bucketing for large distances.
        """
        # Separate positive and negative
        sign = torch.sign(relative_position)
        relative_position = torch.abs(relative_position)
        
        # Linear for small distances, log for large
        max_exact = self.num_buckets // 4
        is_small = relative_position < max_exact
        
        # Small distances: linear buckets
        small_bucket = relative_position
        
        # Large distances: logarithmic buckets
        large_bucket = max_exact + (
            torch.log(relative_position.float() / max_exact) /
            torch.log(torch.tensor(self.max_distance / max_exact)) *
            (self.num_buckets // 2 - max_exact)
        ).long()
        large_bucket = torch.min(large_bucket, 
                                  torch.full_like(large_bucket, self.num_buckets // 2 - 1))
        
        bucket = torch.where(is_small, small_bucket, large_bucket)
        
        # Offset negative buckets
        bucket = torch.where(sign < 0, bucket + self.num_buckets // 2, bucket)
        
        return bucket.clamp(0, self.num_buckets - 1)
    
    def forward(self, x, positions, mask=None):
        """
        Args:
            x: (batch, num_variants, latent_dim) - variant embeddings
            positions: (batch, num_variants) - genomic positions of variants
            mask: (batch, num_variants) - valid variant mask (1 = valid, 0 = padding)
        
        Returns:
            attended: (batch, num_variants, latent_dim)
            attention_weights: (batch, num_heads, num_variants, num_variants)
        """
        batch, num_var, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch, num_var, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, num_var, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, num_var, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, num_var, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        # Compute pairwise distances
        pos_i = positions.unsqueeze(2)  # (batch, num_var, 1)
        pos_j = positions.unsqueeze(1)  # (batch, 1, num_var)
        relative_pos = pos_i - pos_j    # (batch, num_var, num_var)
        
        # Bucket and lookup bias
        buckets = self._relative_position_bucket(relative_pos)
        rel_bias = self.rel_pos_bias(buckets)  # (batch, num_var, num_var, num_heads)
        rel_bias = rel_bias.permute(0, 3, 1, 2)  # (batch, num_heads, num_var, num_var)
        
        attn = attn + rel_bias
        
        # Apply mask (for padded positions)
        if mask is not None:
            # Expand mask for attention: invalid positions shouldn't attend or be attended to
            mask_2d = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)
            attn = attn.masked_fill(mask_2d == 0, float('-inf'))
        
        # Softmax and attend
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)  # Handle all-masked rows
        
        attended = torch.matmul(attn_weights, v)
        
        # Reshape and project out
        attended = attended.transpose(1, 2).contiguous().view(batch, num_var, -1)
        attended = self.out_proj(attended)
        
        return attended, attn_weights
```

**Why this matters**: The relative position bias allows the model to learn that variants close together (potential compound heterozygosity) or at specific distances (potential haplotype patterns) are informative. This is impossible with permutation-invariant deep sets.

### Component 3: Gene Aggregation

After attention layers process variant relationships, we aggregate to gene level:

```python
class GeneAggregator(nn.Module):
    def __init__(self, latent_dim, num_genes, aggregation='max'):
        self.aggregation = aggregation
        self.num_genes = num_genes
        # Optional: learnable gene embeddings for context
        self.gene_embeddings = nn.Embedding(num_genes, latent_dim)
    
    def forward(self, variant_embeddings, gene_assignments, mask=None):
        """
        Args:
            variant_embeddings: (batch, num_variants, latent_dim)
            gene_assignments: (batch, num_variants) - gene index for each variant
            mask: (batch, num_variants) - valid variant mask
        
        Returns:
            gene_embeddings: (batch, num_genes, latent_dim)
        """
        batch, num_var, latent_dim = variant_embeddings.shape
        
        # Initialize gene representations
        gene_repr = torch.zeros(batch, self.num_genes, latent_dim, 
                                device=variant_embeddings.device)
        
        if self.aggregation == 'max':
            # Scatter max: for each gene, take element-wise max of its variants
            gene_repr = gene_repr.fill_(float('-inf'))
            gene_repr = torch.scatter_reduce(
                gene_repr, 1, 
                gene_assignments.unsqueeze(-1).expand(-1, -1, latent_dim),
                variant_embeddings,
                reduce='amax'
            )
            gene_repr = torch.where(gene_repr == float('-inf'), 
                                     torch.zeros_like(gene_repr), gene_repr)
        
        elif self.aggregation == 'mean':
            # Scatter mean
            gene_repr = torch.scatter_reduce(
                gene_repr, 1,
                gene_assignments.unsqueeze(-1).expand(-1, -1, latent_dim),
                variant_embeddings,
                reduce='mean'
            )
        
        elif self.aggregation == 'attention':
            # Attention-weighted aggregation (more complex, described separately)
            pass
        
        return gene_repr
```

### Component 4: Phenotype Classifier

Simple classification head on gene representations:

```python
class PhenotypeClassifier(nn.Module):
    def __init__(self, num_genes, latent_dim, hidden_dim=256):
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(num_genes * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, gene_embeddings):
        """
        Args:
            gene_embeddings: (batch, num_genes, latent_dim)
        
        Returns:
            logits: (batch, 1)
        """
        x = self.flatten(gene_embeddings)
        return self.classifier(x)
```

### Full SIEVE Model

```python
class SIEVE(nn.Module):
    def __init__(self, config):
        self.variant_encoder = VariantEncoder(
            config.input_dim, 
            config.hidden_dim, 
            config.latent_dim
        )
        
        self.attention_layers = nn.ModuleList([
            PositionAwareSparseAttention(
                config.latent_dim, 
                config.num_heads
            )
            for _ in range(config.num_attention_layers)
        ])
        
        self.gene_aggregator = GeneAggregator(
            config.latent_dim,
            config.num_genes,
            config.aggregation
        )
        
        self.classifier = PhenotypeClassifier(
            config.num_genes,
            config.latent_dim,
            config.classifier_hidden_dim
        )
    
    def forward(self, variant_features, positions, gene_assignments, mask=None):
        """
        Args:
            variant_features: (batch, num_variants, input_dim)
            positions: (batch, num_variants) - genomic positions
            gene_assignments: (batch, num_variants) - gene index per variant
            mask: (batch, num_variants) - valid variant mask
        
        Returns:
            logits: (batch, 1)
            attention_weights: list of attention weight tensors (for interpretability)
        """
        # Encode variants
        x = self.variant_encoder(variant_features)
        
        # Apply attention layers
        all_attention_weights = []
        for attn_layer in self.attention_layers:
            x_attn, attn_weights = attn_layer(x, positions, mask)
            x = x + x_attn  # Residual connection
            all_attention_weights.append(attn_weights)
        
        # Aggregate to genes
        gene_repr = self.gene_aggregator(x, gene_assignments, mask)
        
        # Classify
        logits = self.classifier(gene_repr)
        
        return logits, all_attention_weights
```

## Attribution-Regularized Training

### Standard Loss

Binary cross-entropy for case-control classification:

```python
bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
```

### Attribution Sparsity Loss

We want the model to produce sparse attributions—most of its prediction should depend on a small number of variants. This is implemented through a differentiable approximation.

During training, we compute a simplified attribution score (gradient × input) and penalize its entropy:

```python
def attribution_sparsity_loss(model, variant_features, positions, gene_assignments, 
                               mask, labels, temperature=1.0):
    """
    Compute attribution sparsity loss to encourage interpretability.
    """
    # Enable gradient computation for inputs
    variant_features.requires_grad_(True)
    
    # Forward pass
    logits, _ = model(variant_features, positions, gene_assignments, mask)
    
    # Compute gradients w.r.t. inputs
    grads = torch.autograd.grad(
        logits.sum(), variant_features, 
        create_graph=True  # Need second-order gradients
    )[0]
    
    # Attribution = gradient × input (simplified integrated gradients)
    attribution = (grads * variant_features).sum(dim=-1)  # (batch, num_variants)
    
    # Apply mask
    if mask is not None:
        attribution = attribution * mask
    
    # Compute "soft" sparsity: we want attribution to concentrate on few variants
    # Use entropy of normalized attribution as sparsity measure
    attr_abs = torch.abs(attribution) + 1e-8
    attr_normalized = attr_abs / attr_abs.sum(dim=-1, keepdim=True)
    
    # Entropy: high entropy = spread out (bad), low entropy = concentrated (good)
    entropy = -(attr_normalized * torch.log(attr_normalized)).sum(dim=-1)
    
    # We want LOW entropy, so this is the loss
    sparsity_loss = entropy.mean()
    
    return sparsity_loss


def combined_loss(model, variant_features, positions, gene_assignments, 
                  mask, labels, lambda_attr=0.1):
    """
    Combined classification + attribution sparsity loss.
    """
    # Classification loss
    logits, _ = model(variant_features, positions, gene_assignments, mask)
    cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
    
    # Attribution sparsity loss
    attr_loss = attribution_sparsity_loss(
        model, variant_features, positions, gene_assignments, mask, labels
    )
    
    # Combined
    total_loss = cls_loss + lambda_attr * attr_loss
    
    return total_loss, cls_loss, attr_loss
```

**Why this matters**: Without this regularization, a model might achieve good classification by using many weak signals spread across the genome. Such a model is hard to interpret—which variants really matter? The sparsity loss encourages the model to "commit" to a small set of variants, making explainability more meaningful.

### Hyperparameter: λ_attr

The attribution regularization weight λ_attr controls the trade-off:
- λ_attr = 0: Standard training, no sparsity constraint
- λ_attr small (0.01-0.1): Mild encouragement toward sparsity
- λ_attr large (>0.5): Strong sparsity, may hurt classification performance

We recommend starting with λ_attr = 0.1 and tuning based on validation performance.

## Explainability Methods

### 1. Integrated Gradients (Variant-Level Attribution)

For a trained model, compute integrated gradients from a zero baseline to the actual input:

```python
from captum.attr import IntegratedGradients

def compute_variant_attributions(model, variant_features, positions, 
                                  gene_assignments, mask, n_steps=50):
    """
    Compute integrated gradients attribution for each variant.
    """
    ig = IntegratedGradients(model)
    
    # Baseline: zero features (no variants)
    baseline = torch.zeros_like(variant_features)
    
    # Compute attributions
    attributions = ig.attribute(
        variant_features,
        baselines=baseline,
        additional_forward_args=(positions, gene_assignments, mask),
        n_steps=n_steps
    )
    
    # Sum over feature dimension to get per-variant score
    variant_scores = attributions.sum(dim=-1)  # (batch, num_variants)
    
    return variant_scores
```

### 2. Attention Weight Analysis (Positional Patterns)

The attention weights reveal which variant pairs the model considers together:

```python
def analyze_attention_patterns(attention_weights, positions, threshold=0.1):
    """
    Identify variant pairs with strong mutual attention.
    
    Args:
        attention_weights: (batch, num_heads, num_variants, num_variants)
        positions: (batch, num_variants)
        threshold: minimum attention weight to consider
    
    Returns:
        pairs: list of (variant_i, variant_j, attention_score, distance)
    """
    # Average over heads and batch
    avg_attention = attention_weights.mean(dim=(0, 1))  # (num_var, num_var)
    
    # Symmetrize (attend to each other)
    symmetric = (avg_attention + avg_attention.T) / 2
    
    # Find strong pairs
    pairs = []
    for i in range(symmetric.shape[0]):
        for j in range(i+1, symmetric.shape[1]):
            if symmetric[i, j] > threshold:
                dist = abs(positions[0, i] - positions[0, j]).item()
                pairs.append((i, j, symmetric[i, j].item(), dist))
    
    return sorted(pairs, key=lambda x: -x[2])  # Sort by attention score
```

### 3. Epistasis Detection via Counterfactual Perturbation

To validate that detected interactions are truly epistatic (non-additive):

```python
def test_epistasis(model, sample, variant_i, variant_j):
    """
    Test for epistasis between two variants using counterfactual perturbation.
    
    True epistasis: effect(i+j) ≠ effect(i) + effect(j)
    """
    # Original prediction
    pred_original = model(sample).item()
    
    # Remove variant i only
    sample_no_i = remove_variant(sample, variant_i)
    pred_no_i = model(sample_no_i).item()
    effect_i = pred_original - pred_no_i
    
    # Remove variant j only
    sample_no_j = remove_variant(sample, variant_j)
    pred_no_j = model(sample_no_j).item()
    effect_j = pred_original - pred_no_j
    
    # Remove both variants
    sample_no_ij = remove_variant(sample_no_i, variant_j)
    pred_no_ij = model(sample_no_ij).item()
    effect_ij = pred_original - pred_no_ij
    
    # Epistasis score: deviation from additivity
    epistasis_score = effect_ij - (effect_i + effect_j)
    
    return {
        'effect_i': effect_i,
        'effect_j': effect_j,
        'effect_ij': effect_ij,
        'epistasis_score': epistasis_score,
        'additive_expected': effect_i + effect_j
    }
```

## Model Complexity and Scalability

### Parameter Count

For a typical configuration:
- num_genes = 20,000
- latent_dim = 128
- num_attention_layers = 2
- num_heads = 4

Approximate parameters:
- Variant encoder: ~50K
- Attention layers: ~200K
- Gene aggregator: ~2.5M (gene embeddings)
- Classifier: ~2.6M

Total: ~5M parameters (much smaller than foundation models)

### Computational Complexity

The key computational step is attention, which is O(V²) where V is the number of variants per sample. But V is typically 10,000-50,000 (not millions), making this tractable.

Per-sample forward pass: ~0.1-1 second on GPU
Training 1,000 samples for 100 epochs: ~3-10 hours on single GPU

### Memory Requirements

Main memory consumers:
- Attention weights: O(batch × heads × V²) 
- Variant embeddings: O(batch × V × latent_dim)

For batch=32, V=30,000, heads=4, latent_dim=128:
- Attention: ~14 GB (may need gradient checkpointing)
- Embeddings: ~0.5 GB

Recommendation: Use gradient checkpointing for attention layers, batch size 16-32.

## Configuration

Default configuration (YAML):

```yaml
# Model architecture
model:
  latent_dim: 128
  hidden_dim: 256
  num_attention_layers: 2
  num_heads: 4
  aggregation: 'max'
  classifier_hidden_dim: 256

# Training
training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.01
  num_epochs: 100
  lambda_attr: 0.1  # Attribution sparsity weight
  early_stopping_patience: 10

# Data
data:
  annotation_level: 'L2'  # Start with minimal annotations
  max_variants_per_sample: 50000
  min_variants_per_gene: 1

# Explainability
explain:
  integrated_gradients_steps: 50
  attention_threshold: 0.1
  epistasis_top_k: 100
```
