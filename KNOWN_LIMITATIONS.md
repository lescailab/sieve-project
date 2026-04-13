# Known Limitations

This document records methodological limitations discovered during development
and analysis. Each section states the issue, its observed impact, and the
recommended workaround.

---

## 1. Empirical p-value resolution floor

### What it is

`empirical_p_variant` and `empirical_p_gene` are computed using the
Phipson & Smyth (2010) convention:

```
p_i = (k + 1) / (N_null + 1)
```

where `k` is the number of null values ≥ `mean_attribution_i` and
`N_null` is the size of the null pool. The minimum achievable value is
therefore `1 / (N_null + 1)`.

### When it causes problems

When a model is genuinely informative (real attribution distribution lies
substantially above the null), most real variants will exceed every null
value and receive `p = 1 / (N_null + 1)`. This is the **resolution
floor**: a large fraction of variants are tied at the same minimum p-value,
making rank-based operations (top-K selection, Jaccard comparison across
annotation levels) effectively random within the tied set.

This was observed in practice for annotation levels L1 and L2 in CAD
analysis: the median empirical p-value was pinned at `1/(N+1)`, and the
entire top-100 list was drawn from the tied-floor set.

### Recommended workaround

Use `z_attribution` (variant-level) or `gene_z_score` (gene-level) for
any top-K selection or cross-level comparison. These columns contain
per-chromosome z-normalised attributions produced by `correct_chrx_bias.py`
and have continuous resolution irrespective of null pool size.

`compare_ablation_rankings.py` defaults to `z_attribution` for this
reason. To explicitly use empirical p-values, pass
`--score-column empirical_p_variant` and acknowledge the resolution-floor
warning that is printed to stderr.

---

## 2. Cross-level attribution scale incomparability

### What it is

Integrated gradients magnitudes depend on model depth, embedding
dimensionality, normalisation layers, and the scale of the classification
head output. Models trained at different annotation levels (L0–L3) differ
on all of these axes because the input feature dimension changes and the
network may converge to a different loss landscape.

### Why this matters

Raw `mean_attribution` values from two different annotation levels (e.g.,
L0 vs. L3) are not directly comparable. A variant with
`mean_attribution = 0.05` at L0 and `mean_attribution = 0.03` at L3 does
not necessarily carry more signal at L0 — the scales are incommensurable.

This means:

- **Never compute raw score differences across annotation levels.** A
  statement like "this variant has 40% higher attribution at L1 than L0"
  is not interpretable.
- **Never threshold on a fixed attribution value across levels.** A
  threshold of 0.01 may be stringent at L0 and permissive at L3.

### Recommended workaround

Use **rank-based metrics** for cross-level comparison:

- **Jaccard similarity** on top-K variant sets (as computed by
  `compare_ablation_rankings.py`) is invariant to absolute scale.
- **Spearman correlation** of variant ranks across levels.
- **Level-specific variant discovery** (variants in top-K at one level but
  outside top-K at all others) as implemented in `compare_ablation_rankings.py`.

Within a single annotation level, `z_attribution` from `correct_chrx_bias.py`
provides a scale that is at least comparable across chromosomes.
