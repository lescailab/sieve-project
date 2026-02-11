# Appendix A: Model Architecture Details

### Overview

SIEVE (Sparse Interpretable Exome Variant Explainer) is designed around three core innovations:
1. Position-aware sparse attention that preserves spatial relationships between variants
2. Attribution-regularised training that builds interpretability into the objective
3. Annotation-ablation protocol that distinguishes genuine discovery from annotation recovery

This appendix provides the mathematical foundations and implementation details.

### Data Representation

#### Input: Annotated Multi-sample VCF

The input is a VCF file with N samples and V variant sites. Each variant site v has:
- Chromosome and position (chrom_v, pos_v)
- Reference and alternate alleles
- Per-sample genotypes g_{v,n} ∈ {0, 1, 2} (reference homozygote, heterozygote, alternate homozygote)
- VEP annotations including gene assignment, consequence, and functional scores

#### Variant Feature Vector

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

#### Positional Encoding

We use sinusoidal positional encodings adapted for genomic positions:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_pos))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_pos))
```

where d_pos is the positional embedding dimension (default 64).

This encoding allows the model to learn functions of relative position (important for detecting compound heterozygosity or clustered variants).

#### Per-Sample Sparse Representation

For sample n, we construct a sparse representation S_n containing only their non-reference variants:

```
S_n = {(v, x_v, g_{v,n}) : g_{v,n} > 0}
```

This is the key to handling sparsity: we never materialise the full V-dimensional tensor, only the positions where the individual has variants.

### Model Architecture Components

#### Component 1: Variant Encoder

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

#### Component 2: Position-Aware Sparse Attention

This is the core architectural innovation. Standard self-attention is O(n²) in sequence length, infeasible for millions of genomic positions. Standard sparse attention (like in BigBird) uses fixed sparsity patterns. Our approach is naturally sparse because we only attend among variant-present positions.

The key insight: we want to preserve positional information without requiring dense encoding. We achieve this by:
1. Operating only on positions with variants (natural sparsity)
2. Encoding relative distances in attention computation

**Why this matters**: The relative position bias allows the model to learn that variants close together (potential compound heterozygosity) or at specific distances (potential haplotype patterns) are informative. This is impossible with permutation-invariant deep sets.

#### Component 3: Gene Aggregation

After attention layers process variant relationships, we aggregate to gene level using permutation-invariant pooling (max or mean).

#### Component 4: Phenotype Classifier

Simple classification head on gene representations with dropout for regularisation.

### Attribution-Regularised Training

#### Standard Loss

Binary cross-entropy for case-control classification:

```python
bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
```

#### Attribution Sparsity Loss

We want the model to produce sparse attributions—most of its prediction should depend on a small number of variants. This is implemented through a differentiable approximation.

During training, we compute a simplified attribution score (gradient × input) and penalise its entropy:

**Why this matters**: Without this regularisation, a model might achieve good classification by using many weak signals spread across the genome. Such a model is hard to interpret—which variants really matter? The sparsity loss encourages the model to "commit" to a small set of variants, making explainability more meaningful.

#### Hyperparameter: λ_attr

The attribution regularisation weight λ_attr controls the trade-off:
- λ_attr = 0: Standard training, no sparsity constraint
- λ_attr small (0.01-0.1): Mild encouragement toward sparsity
- λ_attr large (>0.5): Strong sparsity, may hurt classification performance

We recommend starting with λ_attr = 0.1 and tuning based on validation performance.

### Explainability Methods

#### 1. Integrated Gradients (Variant-Level Attribution)

For a trained model, compute integrated gradients from a zero baseline to the actual input using the Captum library. The baseline represents "no variants" and the integration path interpolates from this baseline to the actual genotype.

#### 2. Attention Weight Analysis (Positional Patterns)

The attention weights reveal which variant pairs the model considers together. High mutual attention between variant pairs can indicate:
- Compound heterozygosity (nearby variants in same gene)
- Haplotype structure (variants inherited together)
- Potential epistatic interactions

#### 3. Epistasis Detection via Counterfactual Perturbation

To validate that detected interactions are truly epistatic (non-additive):

```
effect_i = f(with variant i) - f(without variant i)
effect_j = f(with variant j) - f(without variant j)
effect_ij = f(with both) - f(without both)
epistasis_score = effect_ij - (effect_i + effect_j)
```

Non-zero epistasis score indicates non-additive interaction.

### Model Complexity and Scalability

#### Parameter Count

For a typical configuration (20,000 genes, latent_dim=128, 2 attention layers, 4 heads):
- Total parameters: ~5M (much smaller than foundation models)

#### Computational Complexity

The key computational step is attention, which is O(V²) where V is the number of variants per sample. But V is typically 10,000-50,000 (not millions), making this tractable.

- Per-sample forward pass: ~0.1-1 second on GPU
- Training 1,000 samples for 100 epochs: ~3-10 hours on single GPU

#### Memory Requirements

For batch=32, V=30,000, heads=4, latent_dim=128:
- Attention: ~14 GB (may need gradient checkpointing)
- Embeddings: ~0.5 GB

**Recommendation**: Use gradient checkpointing for attention layers, batch size 16-32.

---

