# Appendix A: Model Architecture Details

This appendix explains the mathematical structure implemented by SIEVE. The
order follows the workflow of the model: sparse variant encoding, positional
attention, gene aggregation, phenotype prediction, training, explanation, and
interaction validation.

## 1. Sparse Variant Representation

For sample $n$, SIEVE keeps only non-reference variants:

$$
S_n = \{(x_{nv}, \mathrm{pos}_v, \mathrm{chrom}_v, g(v)) :
d_{nv} > 0\}
$$

where $x_{nv}$ is the encoded feature vector, $\mathrm{pos}_v$ is the
chromosome-local coordinate, $\mathrm{chrom}_v$ is the chromosome identifier,
$g(v)$ is the gene index, and $d_{nv}$ is genotype dosage. This sparse
representation avoids materialising a genome-wide dense tensor.

## 2. Annotation Levels

Each variant is encoded at one of five ablation levels:

$$
\begin{aligned}
x_v^{L0} &= [d_v] \\
x_v^{L1} &= [d_v, \mathrm{PE}(\mathrm{pos}_v)] \\
x_v^{L2} &= [d_v, \mathrm{PE}(\mathrm{pos}_v), c_v] \\
x_v^{L3} &= [d_v, \mathrm{PE}(\mathrm{pos}_v), c_v,
              \mathrm{sift}^{\mathrm{norm}}_v,
              \mathrm{polyphen}^{\mathrm{norm}}_v] \\
x_v^{L4} &= x_v^{L3}
\end{aligned}
$$

Current feature dimensions are:

| Level | Features | Dimension |
|-------|----------|-----------|
| L0 | genotype dosage | 1 |
| L1 | L0 + 64-dimensional position encoding | 65 |
| L2 | L1 + 4-dimensional consequence one-hot vector | 69 |
| L3 | L2 + SIFT + PolyPhen | 71 |
| L4 | currently identical to L3; reserved for future features | 71 |

SIFT is inverted so that larger values mean more deleterious. PolyPhen is kept
on the same direction:

$$
\mathrm{sift}^{\mathrm{norm}}_v = 1-\mathrm{sift}^{\mathrm{raw}}_v,
\qquad
\mathrm{polyphen}^{\mathrm{norm}}_v =
\mathrm{polyphen}^{\mathrm{raw}}_v
$$

Missing functional scores are imputed to the neutral value $0.5$.

## 3. Position Encoding

For L1-L4, chromosome-local coordinates are converted to sinusoidal features:

$$
\begin{aligned}
\mathrm{PE}(\mathrm{pos}, 2i)
&= \sin\!\left(\mathrm{pos}\cdot
   \exp\!\left(-\log(10000)\frac{2i}{d}\right)\right) \\
\mathrm{PE}(\mathrm{pos}, 2i+1)
&= \cos\!\left(\mathrm{pos}\cdot
   \exp\!\left(-\log(10000)\frac{2i}{d}\right)\right)
\end{aligned}
$$

with $d=64$. The model also builds chromosome indices. During attention,
chromosome embeddings can be added to variant embeddings, and
cross-chromosome pairs are routed to a dedicated learned relative-position
bias bucket. Cross-chromosome attention is not masked; the separate bucket only
prevents coordinate differences between chromosomes from being treated as
within-chromosome distances.

## 4. Variant Encoder

The input feature vector is projected to the latent dimension by a two-layer
MLP:

$$
h_v =
\mathrm{Linear}_2\!\left(
  \mathrm{Dropout}\!\left(
    \mathrm{LayerNorm}\!\left(
      \mathrm{ReLU}(\mathrm{Linear}_1(x_v))
    \right)
  \right)
\right)
$$

If chromosome embeddings are enabled, the attention input is:

$$
\tilde{h}_v = h_v + e_{\mathrm{chrom}(v)}
$$

## 5. Position-Aware Sparse Attention

For each attention layer:

$$
Q = \tilde{H}W_Q,\qquad K=\tilde{H}W_K,\qquad V=\tilde{H}W_V
$$

Within a chromosome, relative position is bucketed from
$r_{ij}=\mathrm{pos}_i-\mathrm{pos}_j$. Small distances use near-exact buckets;
larger distances use logarithmic buckets up to the configured maximum distance.
For variants on different chromosomes, a dedicated cross-chromosome bucket is
used.

The attention score is:

$$
s_{ijh} =
\frac{Q_{ih}\cdot K_{jh}}{\sqrt{d_{\mathrm{head}}}}
+ \beta_{\mathrm{bucket}(i,j),h}
$$

Padding is masked on the key side, then attention is:

$$
\alpha_{ijh}=\mathrm{softmax}_j(s_{ijh}),\qquad
a_{ih}=\sum_j \alpha_{ijh}V_{jh}
$$

Each layer applies output projection, residual connection, and layer
normalisation:

$$
H_{\mathrm{next}} =
\mathrm{LayerNorm}\!\left(H + \mathrm{Attention}(H)\right)
$$

## 6. Gene Aggregation

After attention, every variant embedding is pooled into its assigned gene.
Supported aggregations are:

$$
\begin{aligned}
E^{\max}_{gk} &= \max_{v:g(v)=g} H_{vk} \\
E^{\mathrm{mean}}_{gk}
&= \frac{1}{|\{v:g(v)=g\}|}\sum_{v:g(v)=g}H_{vk} \\
E^{\mathrm{sum}}_{gk} &= \sum_{v:g(v)=g}H_{vk}
\end{aligned}
$$

Genes without observed variants receive zero embeddings. In chunked training,
each chunk produces a gene embedding matrix. The default wrapper averages
non-zero gene embeddings across chunks:

$$
E_{ng} =
\frac{\sum_c E_{ncg}}
     {\#\{c:\|E_{ncg}\|_2 > 0\}}
$$

## 7. Phenotype Classifier

The classifier flattens the gene embedding matrix and optionally concatenates
sample-level covariates such as sex or ancestry principal components:

$$
u_n = [\mathrm{vec}(E_n), c_n]
$$

The logit and predicted case probability are:

$$
\begin{aligned}
z_n &= W_2\,\mathrm{Dropout}\!\left(
  \mathrm{ReLU}(W_1u_n+b_1)
\right)+b_2 \\
p_n &= \sigma(z_n)
\end{aligned}
$$

## 8. Training Objective

The classification term is binary cross-entropy with logits:

$$
\mathcal{L}_{\mathrm{BCE}}(z,y)
= -y\log\sigma(z) - (1-y)\log(1-\sigma(z))
$$

Optional class weighting uses the positive-class weight:

$$
w_+ = \frac{n_{\mathrm{total}}}{2n_+}
$$

When `--lambda-attr` is greater than zero, SIEVE adds embedding sparsity
regularisation:

$$
\mathcal{L}_{\mathrm{total}}
= \mathcal{L}_{\mathrm{BCE}}
+ \lambda_{\mathrm{attr}}\mathcal{L}_{\mathrm{sparse}}
$$

The public result key is still named `attribution_sparsity` for backward
compatibility, but the implemented regulariser is not gradient entropy and does
not compute Integrated Gradients during training.

For non-chunked batches, the sparsity term is the mean normalised sum of
variant embedding L2 norms:

$$
\begin{aligned}
m_{nv} &= \|H_{nv}\|_2 \\
\mathcal{L}^{(n)}_{\mathrm{sparse}}
&= \frac{\sum_v m_{nv}M_{nv}}
        {\max(1,\sum_v M_{nv})} \\
\mathcal{L}_{\mathrm{sparse}}
&= \frac{1}{N}\sum_n \mathcal{L}^{(n)}_{\mathrm{sparse}}
\end{aligned}
$$

For chunked training, the same idea is applied to aggregated gene embeddings:

$$
\begin{aligned}
m_{ng} &= \|E_{ng}\|_2 \\
\mathcal{L}^{(n)}_{\mathrm{sparse}}
&= \frac{\sum_g m_{ng}}
        {\max(1,\#\{g:m_{ng}>0\})} \\
\mathcal{L}_{\mathrm{sparse}}
&= \frac{1}{N}\sum_n \mathcal{L}^{(n)}_{\mathrm{sparse}}
\end{aligned}
$$

## 9. Explanation

After training, variant attributions are computed with Integrated Gradients
using a zero feature baseline:

$$
\mathrm{IG}_k(x)
= (x_k-x'_k)
\int_0^1
\frac{\partial F(x' + \alpha(x-x'))}{\partial x_k}
\,d\alpha,\qquad x'=0
$$

Feature-level attributions are collapsed to a variant score, usually by the L2
norm:

$$
\mathrm{score}_v=\|\mathrm{IG}_v\|_2
$$

Population-level rankings aggregate these scores across carrier samples, for
example by mean attribution:

$$
\bar{a}_v =
\frac{1}{N_v}\sum_{n:v\in S_n}\mathrm{score}_{nv}
$$

## 10. Epistasis Validation

Attention-derived pairs are hypotheses. Counterfactual validation evaluates
the model with both variants present, each variant removed alone, and both
variants removed:

$$
\Delta_{ij} = p_{11}-p_{10}-p_{01}+p_{00}
$$

where $p_{11}$ is the original prediction, $p_{10}$ removes variant $j$,
$p_{01}$ removes variant $i$, and $p_{00}$ removes both. A non-zero
$\Delta_{ij}$ means the joint model effect is not additive under this
counterfactual perturbation.
