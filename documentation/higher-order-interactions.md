# Higher-Order Interactions

`discover_interactions.py` recovers epistatic variant sets of order greater than two from a
fitted model checkpoint. Attention restricts the search space, and only the candidate sets
that survive that restriction are tested with a counterfactual experiment. The procedure runs
on artefacts a completed run has already written, and requires neither retraining nor a second
explainability pass.

## Position in the workflow

`explain.py` writes variant pairs whose attention exceeds a percentile threshold, and
`validate_epistasis.py` computes a counterfactual synergy for those pairs. Neither addresses
sets of three or more loci.

The thresholded pair list cannot be used to reconstruct such sets. Retaining only pairs above a
high percentile leaves the expected number of retained pairs internal to any given small set
below one, so the absence of a set from `sieve_interactions.csv` carries no information about
whether its loci are mutually attended.

The alternative that the restriction exists to avoid is an exhaustive scan, in which the number
of subsets of order `k` grows as the `k`-th power of the variant count.

## Inputs

| Artefact | Path within a run directory | Use |
|---|---|---|
| Model checkpoint | `experiments/<name>/best_model.pt` | attention extraction |
| Model configuration | `experiments/<name>/config.yaml` | model reconstruction |
| Preprocessed cohort | `preprocessed.pt` | forward passes |
| Thresholded pairs | `results/sieve_interactions.csv` | seeding, optional |
| Per-sample attributions | `results/attributions_per_sample/` | variant restriction, optional |

Where an experiment holds `cv_results.yaml` rather than a single `best_model.pt`, the fold with
the highest AUC is selected, matching how `explain.py` picks a checkpoint. Any path can be
overridden with `--checkpoint`, `--config` or `--preprocessed-data`.

## Stages

### Attention graph

A weighted undirected graph is built over retained variants, with edge weight the attention
between two variants averaged across individuals, layers and heads. Attention is directed but
an interaction is not, so each matrix is symmetrised before any edge is read from it.

Attention over all retained variants is quadratic in variant count, so the variant set is
restricted before the graph is built. The first pass over the cohort scores each variant by its
mean attention mass, the sum of its attention to every other valid variant in the same chunk,
averaged over the individuals carrying it. Variants above `--variant-percentile` form the pool,
capped at `--max-graph-variants`. The threshold and the resulting variant count are recorded in
`discovery_summary.yaml`. The second pass fills the pairwise weights over that pool.

Restriction by a percentile of mean attention mass keeps the criterion internal to the model and
reproducible from the checkpoint alone. Passing `--attributions-dir` narrows the pool further to
the variants carrying the largest mean absolute attribution, before the percentile is applied.

An edge observed in fewer than `--min-edge-support` individuals is set to zero, since an edge
seen once cannot be distinguished from a single individual's private configuration.

### Candidate sets

Seed edges are taken from the highest-weight edges of the graph, and from
`sieve_interactions.csv` when that file is present. Each seed is expanded by repeatedly adding
the variant with the greatest mean attention to **every** current member rather than to any
single member, which is what separates a clique from a chain. By default a variant joins only
when it has a non-zero edge to every current member; `--allow-missing-edges` relaxes this.

The set at each size `k` is recorded with its internal density, the mean edge weight over all
`C(k,2)` internal edges. Absent edges count as zero rather than being skipped, so a set held
together by a single strong link is scored as the chain it is.

Cost is the number of seeds multiplied by expansion depth, not the number of combinations.

### Order selection

The density-against-size curve determines the order of each candidate. A set whose loci
interact holds its internal density as it grows to its order and loses density on the next
addition; a chain of unrelated pairwise attention loses density immediately.

Expansion is read up to the first size whose internal density falls below the distribution
obtained from random sets of the same size drawn from the same variant pool, at
`--null-quantile`. The emitted size is the one maximising the gap between observed and random
density.

A density curve that stays flat has not shown the set to have ended, so growth that holds the
gap to within `--gap-tolerance` is emitted at the larger order. Setting that tolerance to zero
emits the smallest size instead whenever the null mean drifts upward with set size.

### Counterfactual test

The pairwise synergy of `validate_epistasis.py` is generalised to order `k` as the joint
knockout of all `k` loci against the knockouts of every proper subset, following the
inclusion-exclusion form of an order-`k` interaction:

$$I_K = \sum_{T \subseteq K} (-1)^{|T|} f(\text{ablate } T)$$

At `k = 2` this reduces to `f(both) - f(only v1) - f(only v2) + f(neither)`, the expression
`validate_epistasis.py` already computes, which fixes the sign convention.

Each candidate is tested in an individual carrying every member of the set, chosen as the
carrier with the fewest variants. Large samples are cut to a window centred on the members,
following the same approximation the pairwise validation makes: excluded context can influence
the members' embeddings, but the interaction is a `k`-th order difference in which shared
context largely cancels. A candidate whose members cannot share a window of `--test-chunk-size`
variants in any carrier is reported untested rather than approximated further.

Where the model was fitted with covariates, each condition is scored at the carrier's own
covariate profile, sex in column 0 followed by any further covariates stored on the sample,
which is the convention training and integrated gradients both apply. Covariates are
concatenated ahead of a non-linear classifier, so scoring at a profile the carrier does not
have would move the operating point and change the interaction rather than only its baseline.
The profile enters the classifier alone, leaving the attention graph unaffected.

Cost grows as `2^k` per candidate, which bounds the practical order and is exposed as
`--max-test-order`.

## Running the discovery

```bash
python scripts/discover_interactions.py \
    --run-dir /path/to/run \
    --experiment-name L3_run \
    --max-order 5 \
    --n-seeds 1000 \
    --output-dir /path/to/run/results/interaction_discovery \
    --device cuda
```

Full option tables are in `command-reference.md`.

## Outputs

| File | Contents |
|---|---|
| `attention_graph_variants.csv` | Retained pool with each variant's mean attention mass |
| `attention_graph_edges.csv` | Strongest edges with attention and contributing individuals |
| `candidate_sets.csv` | One row per emitted set: order, members, density and null statistics |
| `expansion_trace.csv` | Density and weakest internal edge at every size of every seed |
| `density_null.csv` | Mean, standard deviation and quantile of random-set density per size |
| `higher_order_synergy.csv` | Counterfactual result per candidate |
| `discovery_summary.yaml` | Parameters, thresholds and counts for the run |

Members are written as `pos:gene_id` entries joined by semicolons, using the same variant key as
`sieve_interactions.csv`, so candidate members join against existing outputs without
translation.

`density_gap` is the quantity the candidate list is ranked by. `density_z` expresses the same
gap in standard deviations of the null, and is infinite where the null has no spread.

Every candidate appears in `higher_order_synergy.csv`, including those that were not tested, so
that the two tables stay aligned. The `status` column carries the reason:

| Status | Meaning |
|---|---|
| `tested` | Synergy computed in at least one carrier |
| `no_carrier` | No individual carries every member of the set |
| `no_testable_carrier` | Carriers exist, but none admits a window holding every member |
| `above_max_order` | Order exceeds `--max-test-order`, so the `2^k` test was not run |

A multi-allelic locus, at which one `(pos, gene)` key resolves to more than one variant in a
sample, is not a usable carrier, and such a sample is excluded from the carrier set rather than
resolved to one of its alleles.

## Evaluation against a known architecture

Where the generating architecture is recorded, `--truth` takes a CSV in long form with one row
per locus, columns `set_id` and `pos`, and either `gene_id` for the run's internal gene index or
`gene` for a symbol resolved through it:

```
set_id,pos,gene
1,11575398,BRCA1
1,16344466,TP53
1,10342629,PTEN
```

The evaluation writes `evaluation/truth_recovery.csv`, giving the rank at which each true set
first appears among the emitted candidates, both as an exact match and as a subset of a larger
candidate; `evaluation/recovery_by_order.csv`, giving the proportion recovered at each order;
and `evaluation/evaluation_summary.yaml`, giving the false discovery rate among the candidates
surviving the counterfactual. A cohort simulated without interactions fixes the false-positive
rate, since every set it emits is spurious.

Truth is read by this stage alone. It lives in a module the discovery code does not import, so
the discovery path cannot reach it.

## Conditions of applicability

Attention weights are trained through the classification loss alone, and the gene aggregation
following attention is order-invariant. A set of loci whose effect is entirely non-additive
need not therefore be mutually attended, and the recall of the expansion stage is bounded by
what attention already encodes. The procedure restricts a search; it does not recover
information the model did not represent.

Detectability of an order-`k` set by any pairwise method depends on the two-locus marginal
effects the set induces, which fall as order rises. Recovery of sets whose pairwise marginals
are negligible is the case that distinguishes this procedure from a pairwise scan.

Synergy magnitudes reflect the fitted model. A checkpoint trained to a low AUC returns
interactions near zero across every candidate, and the density statistics then describe the
attention geometry rather than an epistatic architecture.
