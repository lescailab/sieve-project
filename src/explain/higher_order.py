"""
Higher-order epistatic interaction discovery from attention.

Recovers epistatic variant sets of order greater than two from a trained SIEVE
model.  Attention restricts the search space; only the candidates that survive
that restriction are tested with a counterfactual experiment.

The four discovery stages are:

1. :func:`accumulate_attention_mass` and :func:`build_attention_graph` build a
   weighted undirected graph over a restricted variant pool, with edge weight
   the attention between two variants averaged across individuals and layers.
2. :func:`expand_seed` grows each seed edge by repeatedly adding the variant
   with the greatest mean attention to *every* current member, which is what
   separates a clique from a chain.
3. :func:`random_density_null` and :func:`select_candidate_order` choose the
   order of each candidate from its density-against-size curve, against the
   density of random sets of the same size drawn from the same pool.
4. :class:`HigherOrderCounterfactual` generalises the pairwise synergy of
   :class:`~src.explain.counterfactual_epistasis.CounterfactualEpistasisDetector`
   to order ``k`` through the inclusion-exclusion form of an order-``k``
   interaction.

Nothing here reads a known architecture: candidate generation is a function of
the model and the cohort alone, so the procedure applies to cohorts where no
truth exists.  Evaluation against a truth file lives in
:mod:`src.explain.higher_order_evaluation`.

Author: Francesco Lescai
"""

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.data import SampleVariants
from src.data.covariates import encode_sex_for_covariate
from src.encoding.chunked_dataset import collate_chunks
from src.encoding.sparse_tensor import build_variant_tensor

# A variant is addressed by genomic position and gene index, the same key the
# pairwise path uses in sieve_interactions.csv, so candidate members can be
# joined against existing outputs without translation.
VariantKey = Tuple[int, int]


# ---------------------------------------------------------------------------
# Stage 1: attention graph
# ---------------------------------------------------------------------------


def aggregate_attention(
    attention_weights: Sequence[Tensor],
    aggregate_layers: str = 'mean',
    aggregate_heads: str = 'mean',
) -> Tensor:
    """
    Reduce per-layer, per-head attention to one symmetric matrix per individual.

    Parameters
    ----------
    attention_weights : Sequence[Tensor]
        One tensor per layer, each ``(batch, heads, n_variants, n_variants)``.
    aggregate_layers : str
        'mean', 'max' or 'last'.
    aggregate_heads : str
        'mean' or 'max'.

    Returns
    -------
    Tensor
        ``(batch, n_variants, n_variants)``, symmetrised with a zero diagonal.

    Notes
    -----
    Attention is directed but an interaction is not, so the matrix is
    symmetrised as ``(A + A^T) / 2`` before any edge is read from it.
    """
    if len(attention_weights) == 0:
        raise ValueError("No attention weights were returned by the model.")

    if aggregate_layers == 'mean':
        attn = torch.mean(torch.stack(list(attention_weights)), dim=0)
    elif aggregate_layers == 'max':
        attn = torch.max(torch.stack(list(attention_weights)), dim=0)[0]
    elif aggregate_layers == 'last':
        attn = attention_weights[-1]
    else:
        raise ValueError(f"Unknown layer aggregation: {aggregate_layers}")

    if aggregate_heads == 'mean':
        attn = attn.mean(dim=1)
    elif aggregate_heads == 'max':
        attn = attn.max(dim=1)[0]
    else:
        raise ValueError(f"Unknown head aggregation: {aggregate_heads}")

    attn = 0.5 * (attn + attn.transpose(1, 2))
    diagonal = torch.arange(attn.shape[-1], device=attn.device)
    attn[:, diagonal, diagonal] = 0.0
    return attn


def _chunk_batches(
    model: nn.Module,
    dataset,
    device: str,
    batch_size: int,
    aggregate_layers: str,
    aggregate_heads: str,
    progress_every: int = 10,
    label: str = 'chunks',
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run the cohort through the model and yield symmetric attention per chunk.

    Yields
    ------
    tuple
        ``(attention, positions, gene_ids, n_valid)`` for one chunk, where
        ``attention`` is ``(n_valid, n_valid)`` and positions and gene ids are
        the first ``n_valid`` entries of the chunk.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_chunks,
        num_workers=0,
    )
    total_chunks = len(dataset)

    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        features = batch['features'].to(device)
        positions = batch['positions'].to(device)
        gene_ids = batch['gene_ids'].to(device)
        mask = batch['mask'].to(device)
        chrom_ids = (
            batch['chrom_ids'].to(device) if 'chrom_ids' in batch else None
        )

        if features.shape[1] == 0:
            continue

        with torch.no_grad():
            attention_weights = model.get_attention_patterns(
                features, positions, gene_ids, mask, chrom_ids=chrom_ids
            )
            attn = aggregate_attention(
                attention_weights, aggregate_layers, aggregate_heads
            ).cpu().numpy()

        positions_np = positions.cpu().numpy()
        gene_ids_np = gene_ids.cpu().numpy()
        mask_np = mask.cpu().numpy()

        for b in range(attn.shape[0]):
            n_valid = int(mask_np[b].sum())
            if n_valid < 2:
                continue
            yield (
                attn[b][:n_valid, :n_valid],
                positions_np[b][:n_valid],
                gene_ids_np[b][:n_valid],
                n_valid,
            )

        del features, positions, gene_ids, mask, attention_weights
        if chrom_ids is not None:
            del chrom_ids
        if device == 'cuda':
            torch.cuda.empty_cache()

        if progress_every and (batch_idx + 1) % progress_every == 0:
            done = min((batch_idx + 1) * batch_size, total_chunks)
            print(f"  Processed {done}/{total_chunks} {label}")


def accumulate_attention_mass(
    model: nn.Module,
    dataset,
    device: str = 'cuda',
    batch_size: int = 4,
    aggregate_layers: str = 'mean',
    aggregate_heads: str = 'mean',
) -> Dict[VariantKey, float]:
    """
    Mean attention mass of every variant, averaged over the individuals carrying it.

    A variant's mass in one individual is the sum of its symmetrised attention
    to all other valid variants in the same chunk.  This is the criterion that
    restricts the variant pool before the graph is built, so it is internal to
    the model and reproducible from the checkpoint alone.

    Parameters
    ----------
    model : nn.Module
        Trained SIEVE model (base or chunked wrapper).
    dataset : ChunkedVariantDataset
        Cohort to pass over.
    device : str
        'cuda' or 'cpu'.
    batch_size : int
        Chunks per forward pass.
    aggregate_layers, aggregate_heads : str
        Reduction across attention layers and heads.

    Returns
    -------
    Dict[VariantKey, float]
        Mean attention mass keyed by ``(pos, gene_id)``.
    """
    mass_sum: Dict[VariantKey, float] = defaultdict(float)
    mass_count: Dict[VariantKey, int] = defaultdict(int)

    for attn, positions, gene_ids, n_valid in _chunk_batches(
        model, dataset, device, batch_size,
        aggregate_layers, aggregate_heads, label='chunks (mass pass)',
    ):
        row_mass = attn.sum(axis=1)
        for i in range(n_valid):
            key = (int(positions[i]), int(gene_ids[i]))
            mass_sum[key] += float(row_mass[i])
            mass_count[key] += 1

    return {key: mass_sum[key] / mass_count[key] for key in mass_sum}


def select_graph_variants(
    mass: Dict[VariantKey, float],
    percentile: float = 99.0,
    max_variants: Optional[int] = 2000,
    restrict_to: Optional[Iterable[VariantKey]] = None,
) -> Tuple[List[VariantKey], float]:
    """
    Restrict the variant pool before the pairwise graph is built.

    Attention over all retained variants is quadratic in variant count, so the
    pool is cut to the variants above a percentile of mean attention mass.

    Parameters
    ----------
    mass : Dict[VariantKey, float]
        Mean attention mass per variant, from :func:`accumulate_attention_mass`.
    percentile : float
        Percentile of the mass distribution below which variants are dropped.
    max_variants : Optional[int]
        Hard cap on the pool size, applied after the percentile. ``None``
        leaves the percentile as the only criterion.
    restrict_to : Optional[Iterable[VariantKey]]
        Further restriction applied before the percentile, for example the
        variants carrying the largest attributions.

    Returns
    -------
    keys : List[VariantKey]
        Retained variants, ordered by decreasing mass.
    threshold : float
        The mass value at the percentile cut.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100.")

    candidates = dict(mass)
    if restrict_to is not None:
        allowed = set(restrict_to)
        candidates = {k: v for k, v in candidates.items() if k in allowed}

    if not candidates:
        return [], float('nan')

    values = np.fromiter(candidates.values(), dtype=float)
    threshold = float(np.percentile(values, percentile))

    keys = [k for k, v in candidates.items() if v >= threshold]
    keys.sort(key=lambda k: candidates[k], reverse=True)

    if max_variants is not None and len(keys) > max_variants:
        keys = keys[:max_variants]

    return keys, threshold


@dataclass
class AttentionGraph:
    """
    Weighted undirected attention graph over a restricted variant pool.

    Attributes
    ----------
    keys : List[VariantKey]
        Graph nodes, ordered as the matrix rows.
    weights : np.ndarray
        ``(V, V)`` mean attention between two variants over the individuals
        carrying both. Pairs never observed together hold zero.
    support : np.ndarray
        ``(V, V)`` number of individuals contributing to each edge.
    mass : np.ndarray
        ``(V,)`` mean attention mass, the quantity the pool was cut on.
    """

    keys: List[VariantKey]
    weights: np.ndarray
    support: np.ndarray
    mass: np.ndarray
    key_to_index: Dict[VariantKey, int] = field(init=False)

    def __post_init__(self):
        self.key_to_index = {key: i for i, key in enumerate(self.keys)}

    @property
    def n_variants(self) -> int:
        return len(self.keys)

    def density(self, members: Sequence[int]) -> float:
        """
        Internal density of a set: mean edge weight over all ``C(k, 2)`` edges.

        Absent edges count as zero rather than being skipped, so a set held
        together by a single strong link is scored as the chain it is.
        """
        if len(members) < 2:
            return 0.0
        idx = np.asarray(members, dtype=int)
        block = self.weights[np.ix_(idx, idx)]
        k = len(idx)
        return float(block.sum() / (k * (k - 1)))

    def min_edge(self, members: Sequence[int]) -> float:
        """Weakest internal edge of a set, zero for sets smaller than two."""
        if len(members) < 2:
            return 0.0
        return min(
            float(self.weights[i, j]) for i, j in combinations(sorted(members), 2)
        )

    def top_edges(self, n_edges: int, min_support: int = 1) -> List[Tuple[int, int, float]]:
        """
        Highest-weight edges, as ``(i, j, weight)`` with ``i < j``.

        Parameters
        ----------
        n_edges : int
            Number of edges to return.
        min_support : int
            Minimum number of individuals contributing to an edge.
        """
        upper = np.triu(self.weights, k=1)
        if min_support > 1:
            upper = np.where(self.support >= min_support, upper, 0.0)
        flat = upper.ravel()
        n_edges = int(min(n_edges, np.count_nonzero(flat)))
        if n_edges == 0:
            return []
        order = np.argpartition(flat, -n_edges)[-n_edges:]
        order = order[np.argsort(flat[order])[::-1]]
        edges = []
        for pos in order:
            i, j = divmod(int(pos), self.weights.shape[1])
            edges.append((i, j, float(flat[pos])))
        return edges


def build_attention_graph(
    model: nn.Module,
    dataset,
    keys: Sequence[VariantKey],
    mass: Optional[Dict[VariantKey, float]] = None,
    device: str = 'cuda',
    batch_size: int = 4,
    aggregate_layers: str = 'mean',
    aggregate_heads: str = 'mean',
    min_support: int = 2,
) -> AttentionGraph:
    """
    Accumulate pairwise attention between retained variants across the cohort.

    Parameters
    ----------
    model : nn.Module
        Trained SIEVE model.
    dataset : ChunkedVariantDataset
        Cohort to pass over.
    keys : Sequence[VariantKey]
        Retained variant pool from :func:`select_graph_variants`.
    mass : Optional[Dict[VariantKey, float]]
        Mean attention mass, carried onto the graph for reporting.
    device, batch_size, aggregate_layers, aggregate_heads
        As for :func:`accumulate_attention_mass`.
    min_support : int
        Edges observed in fewer than this many individuals are zeroed. An edge
        seen once cannot be distinguished from a single individual's private
        configuration, which is the same reason the pairwise path requires a
        variant pair to recur across samples.

    Returns
    -------
    AttentionGraph
    """
    key_to_index = {key: i for i, key in enumerate(keys)}
    n = len(keys)
    weight_sum = np.zeros((n, n), dtype=np.float64)
    pair_count = np.zeros((n, n), dtype=np.int32)

    for attn, positions, gene_ids, n_valid in _chunk_batches(
        model, dataset, device, batch_size,
        aggregate_layers, aggregate_heads, label='chunks (graph pass)',
    ):
        local: List[int] = []
        global_idx: List[int] = []
        for i in range(n_valid):
            key = (int(positions[i]), int(gene_ids[i]))
            gi = key_to_index.get(key)
            if gi is not None:
                local.append(i)
                global_idx.append(gi)

        if len(local) < 2:
            continue

        block = attn[np.ix_(local, local)]
        rows = np.asarray(global_idx, dtype=int)
        weight_sum[np.ix_(rows, rows)] += block
        pair_count[np.ix_(rows, rows)] += 1

    # A pair never carried by the same individual has no attention to average.
    with np.errstate(invalid='ignore', divide='ignore'):
        weights = np.where(pair_count > 0, weight_sum / np.maximum(pair_count, 1), 0.0)
    if min_support > 1:
        weights = np.where(pair_count >= min_support, weights, 0.0)
    np.fill_diagonal(weights, 0.0)

    mass_array = np.array(
        [float(mass.get(key, 0.0)) if mass else 0.0 for key in keys], dtype=float
    )

    return AttentionGraph(
        keys=list(keys),
        weights=weights,
        support=pair_count,
        mass=mass_array,
    )


def load_attribution_mass(
    per_sample_dir: str | Path,
    dataset,
    max_samples: Optional[int] = None,
) -> Dict[VariantKey, float]:
    """
    Mean absolute attribution per variant, from the per-sample files of explain.py.

    Used only to narrow the variant pool before Stage 1; it plays no part in
    scoring candidates.

    Parameters
    ----------
    per_sample_dir : str or Path
        The ``attributions_per_sample/`` directory written by explain.py.
    dataset : object
        Dataset exposing ``samples`` and ``gene_index``; variant scores are
        aligned positionally with each sample's variant list.
    max_samples : Optional[int]
        Read only the first N samples.

    Returns
    -------
    Dict[VariantKey, float]
        Mean absolute variant score keyed by ``(pos, gene_id)``.
    """
    per_sample_dir = Path(per_sample_dir)
    score_sum: Dict[VariantKey, float] = defaultdict(float)
    score_count: Dict[VariantKey, int] = defaultdict(int)

    samples = dataset.samples
    if max_samples is not None:
        samples = samples[:max_samples]

    gene_index = dataset.gene_index
    n_missing = 0
    n_mismatched = 0

    for sample_idx, sample in enumerate(samples):
        path = per_sample_dir / f'sample_{sample_idx}.npz'
        if not path.exists():
            n_missing += 1
            continue
        with np.load(path, allow_pickle=False) as data:
            scores = np.abs(np.asarray(data['variant_scores'], dtype=float))

        # The per-sample files carry no variant coordinates, so alignment rests
        # on the dataset's variant order being the one explain.py scored.
        if scores.shape[0] != len(sample.variants):
            n_mismatched += 1
            continue

        for variant, score in zip(sample.variants, scores):
            gene_id = gene_index.get(variant.gene, -1)
            if gene_id < 0:
                continue
            key = (int(variant.pos), int(gene_id))
            score_sum[key] += float(score)
            score_count[key] += 1

    if n_missing:
        print(f"  {n_missing} sample(s) had no attribution file and were skipped")
    if n_mismatched:
        print(f"  {n_mismatched} sample(s) had attribution lengths that did not "
              "match the dataset variant order and were skipped")

    return {key: score_sum[key] / score_count[key] for key in score_sum}


# ---------------------------------------------------------------------------
# Stage 2: candidate sets by greedy expansion
# ---------------------------------------------------------------------------


def expand_seed(
    graph: AttentionGraph,
    seed: Tuple[int, int],
    max_order: int,
    require_all_edges: bool = True,
) -> List[Dict]:
    """
    Grow one seed edge into a nested family of sets, recording density by size.

    At every step the variant added is the one with the greatest *mean*
    attention to all current members, not to any single member. Scoring
    against the whole set is what distinguishes a clique from a chain of
    unrelated pairwise attention.

    Parameters
    ----------
    graph : AttentionGraph
        Graph to expand within.
    seed : Tuple[int, int]
        Node indices of the seed edge.
    max_order : int
        Largest set size to grow to.
    require_all_edges : bool
        Admit a candidate only when it has a non-zero edge to every current
        member. With this off, a variant can join on a strong mean while
        being unconnected to part of the set.

    Returns
    -------
    List[Dict]
        One entry per size from 2 to ``max_order``, each with ``members``,
        ``size``, ``density`` and ``min_edge``. The list stops early when no
        admissible variant remains.
    """
    members = [int(seed[0]), int(seed[1])]
    trace = [{
        'members': tuple(members),
        'size': 2,
        'density': graph.density(members),
        'min_edge': graph.min_edge(members),
    }]

    while len(members) < max_order:
        member_idx = np.asarray(members, dtype=int)
        # Mean attention of every node to all current members at once.
        mean_to_members = graph.weights[:, member_idx].mean(axis=1)
        if require_all_edges:
            connected = (graph.weights[:, member_idx] > 0).all(axis=1)
            mean_to_members = np.where(connected, mean_to_members, -1.0)
        mean_to_members[member_idx] = -1.0

        best = int(np.argmax(mean_to_members))
        if mean_to_members[best] <= 0.0:
            break

        members.append(best)
        trace.append({
            'members': tuple(sorted(members)),
            'size': len(members),
            'density': graph.density(members),
            'min_edge': graph.min_edge(members),
        })

    return trace


def collect_seeds(
    graph: AttentionGraph,
    n_seeds: int,
    min_support: int = 2,
    extra_seeds: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """
    Seed edges for expansion: the highest-weight edges, plus any supplied pairs.

    Parameters
    ----------
    graph : AttentionGraph
        Graph to seed from.
    n_seeds : int
        Number of graph edges to take.
    min_support : int
        Minimum number of individuals contributing to a seed edge.
    extra_seeds : Optional[Sequence[Tuple[int, int]]]
        Node-index pairs to prepend, for example the thresholded pairs of a
        previous explain.py run mapped onto the graph.

    Returns
    -------
    List[Tuple[int, int]]
        Deduplicated seed edges, ordered with the supplied pairs first.
    """
    seeds: List[Tuple[int, int]] = []
    seen = set()

    for pair in (extra_seeds or []):
        key = tuple(sorted(pair))
        if key not in seen:
            seen.add(key)
            seeds.append((int(key[0]), int(key[1])))

    for i, j, _weight in graph.top_edges(n_seeds, min_support=min_support):
        key = tuple(sorted((i, j)))
        if key not in seen:
            seen.add(key)
            seeds.append((int(key[0]), int(key[1])))

    return seeds


# ---------------------------------------------------------------------------
# Stage 3: order selection
# ---------------------------------------------------------------------------


def random_density_null(
    graph: AttentionGraph,
    sizes: Iterable[int],
    n_draws: int = 1000,
    seed: int = 0,
) -> Dict[int, np.ndarray]:
    """
    Density of random sets of each size drawn from the same variant pool.

    Parameters
    ----------
    graph : AttentionGraph
        Pool to draw from.
    sizes : Iterable[int]
        Set sizes to build a null for.
    n_draws : int
        Random sets per size.
    seed : int
        Seed for the draw, so a run is reproducible.

    Returns
    -------
    Dict[int, np.ndarray]
        Density samples per size.
    """
    rng = np.random.default_rng(seed)
    null: Dict[int, np.ndarray] = {}
    n = graph.n_variants

    for size in sizes:
        if size < 2 or size > n:
            null[size] = np.zeros(0, dtype=float)
            continue
        densities = np.empty(n_draws, dtype=float)
        for d in range(n_draws):
            members = rng.choice(n, size=size, replace=False)
            densities[d] = graph.density(members)
        null[size] = densities

    return null


@dataclass
class CandidateSet:
    """A candidate interacting set with the statistics that selected its order."""

    members: Tuple[int, ...]
    order: int
    density: float
    min_edge: float
    null_mean_density: float
    null_quantile_density: float
    density_gap: float
    density_z: float
    seed: Tuple[int, int]


def select_candidate_order(
    graph: AttentionGraph,
    trace: Sequence[Dict],
    null: Dict[int, np.ndarray],
    seed: Tuple[int, int],
    quantile: float = 0.95,
    gap_tolerance: float = 0.01,
) -> Optional[CandidateSet]:
    """
    Choose the order of one expanded seed from its density-against-size curve.

    A set whose loci interact holds its internal density as it grows to its
    order and loses density on the next addition; a chain of unrelated
    pairwise attention loses density immediately. Expansion is therefore read
    only up to the first size whose density falls below the random-set
    distribution, and the emitted size is the one maximising the gap between
    observed and random density.

    Parameters
    ----------
    graph : AttentionGraph
        Graph the trace was grown in.
    trace : Sequence[Dict]
        Output of :func:`expand_seed`.
    null : Dict[int, np.ndarray]
        Random-set densities per size, from :func:`random_density_null`.
    seed : Tuple[int, int]
        The seed edge, carried onto the result.
    quantile : float
        Quantile of the null that terminates expansion.
    gap_tolerance : float
        Relative tolerance on the gap when comparing sizes. A set whose
        density has not fallen has not been shown to have ended, so growth
        that holds the gap to within this tolerance is preferred at the
        larger order; without it, a flat density curve emits the seed pair.

    Returns
    -------
    Optional[CandidateSet]
        ``None`` when no size in the trace clears the null.
    """
    best: Optional[CandidateSet] = None

    for entry in trace:
        size = entry['size']
        null_densities = null.get(size)
        if null_densities is None or null_densities.size == 0:
            continue

        null_mean = float(null_densities.mean())
        null_sd = float(null_densities.std())
        null_q = float(np.quantile(null_densities, quantile))

        if entry['density'] < null_q:
            # Density has fallen into the random-set distribution: everything
            # beyond this size is a chain, so the curve is not read further.
            break

        gap = entry['density'] - null_mean
        if null_sd > 0:
            z = gap / null_sd
        else:
            # A null with no spread makes the z-score undefined; report the
            # direction of the gap rather than a division by zero.
            z = float('inf') if gap > 0 else 0.0

        keeps_the_gap = (
            best is not None
            and gap >= best.density_gap - gap_tolerance * abs(best.density_gap)
        )
        if best is None or gap > best.density_gap or keeps_the_gap:
            best = CandidateSet(
                members=tuple(sorted(entry['members'])),
                order=size,
                density=entry['density'],
                min_edge=entry['min_edge'],
                null_mean_density=null_mean,
                null_quantile_density=null_q,
                density_gap=gap,
                density_z=z,
                seed=seed,
            )

    return best


def discover_candidate_sets(
    graph: AttentionGraph,
    seeds: Sequence[Tuple[int, int]],
    max_order: int = 5,
    null: Optional[Dict[int, np.ndarray]] = None,
    n_null_draws: int = 1000,
    quantile: float = 0.95,
    null_seed: int = 0,
    require_all_edges: bool = True,
    gap_tolerance: float = 0.01,
) -> Tuple[List[CandidateSet], List[Dict], Dict[int, np.ndarray]]:
    """
    Run Stages 2 and 3 over every seed and return the deduplicated candidates.

    Cost is the number of seeds multiplied by expansion depth, not the number
    of combinations of the variant pool.

    Parameters
    ----------
    graph : AttentionGraph
        Graph to expand within.
    seeds : Sequence[Tuple[int, int]]
        Seed edges, from :func:`collect_seeds`.
    max_order : int
        Largest set size to grow to.
    null : Optional[Dict[int, np.ndarray]]
        Precomputed random-set densities; drawn here when absent.
    n_null_draws : int
        Random sets per size when the null is drawn here.
    quantile : float
        Quantile of the null that terminates expansion.
    null_seed : int
        Seed for the random draw.
    require_all_edges : bool
        Passed to :func:`expand_seed`.
    gap_tolerance : float
        Passed to :func:`select_candidate_order`.

    Returns
    -------
    candidates : List[CandidateSet]
        Ordered by decreasing density gap, one entry per distinct member set.
    traces : List[Dict]
        Every ``(seed, size)`` step, for inspection of the density curves.
    null : Dict[int, np.ndarray]
        The null actually used.
    """
    if null is None:
        null = random_density_null(
            graph, range(2, max_order + 1), n_draws=n_null_draws, seed=null_seed
        )

    candidates: Dict[Tuple[int, ...], CandidateSet] = {}
    traces: List[Dict] = []

    for seed in seeds:
        trace = expand_seed(graph, seed, max_order, require_all_edges=require_all_edges)
        for entry in trace:
            traces.append({
                'seed_variant1': graph.keys[seed[0]],
                'seed_variant2': graph.keys[seed[1]],
                'size': entry['size'],
                'density': entry['density'],
                'min_edge': entry['min_edge'],
                'members': tuple(sorted(entry['members'])),
            })

        candidate = select_candidate_order(
            graph, trace, null, seed, quantile=quantile, gap_tolerance=gap_tolerance
        )
        if candidate is None:
            continue

        # Distinct seeds routinely converge on the same set; the one with the
        # larger gap is kept so the ranking is not diluted by duplicates.
        existing = candidates.get(candidate.members)
        if existing is None or candidate.density_gap > existing.density_gap:
            candidates[candidate.members] = candidate

    ordered = sorted(candidates.values(), key=lambda c: c.density_gap, reverse=True)
    return ordered, traces, null


# ---------------------------------------------------------------------------
# Stage 4: counterfactual test of order k
# ---------------------------------------------------------------------------


class HigherOrderCounterfactual:
    """
    Counterfactual interaction of order ``k`` by inclusion-exclusion.

    The joint knockout of all ``k`` loci is tested against the knockouts of
    every proper subset:

    .. math::

        I_K = \\sum_{T \\subseteq K} (-1)^{|T|} f(\\mathrm{ablate}\\ T)

    At ``k = 2`` this is
    ``f(both) - f(only v1) - f(only v2) + f(neither)``, the expression
    :meth:`~src.explain.counterfactual_epistasis.CounterfactualEpistasisDetector.validate_interaction_with_perturbation`
    already computes, which fixes the sign convention.

    Cost grows as ``2^k`` forward passes per candidate, which is what bounds
    the practical order.

    Parameters
    ----------
    model : nn.Module
        Trained SIEVE model.
    device : str
        'cuda' or 'cpu'.
    synergy_threshold : float
        ``|I_K|`` above which a candidate is called significant.
    independence_epsilon : float
        ``|I_K|`` below which a candidate is called independent.

    Examples
    --------
    >>> tester = HigherOrderCounterfactual(model, device='cpu')
    >>> result = tester.compute_interaction(
    ...     features, positions, gene_ids, mask, [3, 17, 42]
    ... )
    >>> result['order']
    3
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        synergy_threshold: float = 0.05,
        independence_epsilon: float = 0.01,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.synergy_threshold = synergy_threshold
        self.independence_epsilon = independence_epsilon

    def compute_interaction(
        self,
        features: Tensor,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor,
        variant_indices: Sequence[int],
        chrom_ids: Optional[Tensor] = None,
        covariates: Optional[Tensor] = None,
    ) -> Dict:
        """
        Interaction of one variant set in one individual.

        Parameters
        ----------
        features, positions, gene_ids, mask : Tensor
            A single individual's variants, with or without a batch dimension.
        variant_indices : Sequence[int]
            Indices of the set members within these tensors.
        chrom_ids : Optional[Tensor]
            Chromosome indices, when the model is chromosome-aware.
        covariates : Optional[Tensor]
            The individual's covariate vector, shape ``(num_covariates,)`` or
            ``(1, num_covariates)``. Required when the model was fitted with
            ``num_covariates > 0``: the classifier substitutes zeros for a
            missing vector, which would evaluate every condition at a
            covariate profile the carrier does not have. Since covariates are
            concatenated ahead of a non-linear classifier, that shifts the
            operating point and changes the interaction, not merely its
            baseline.

        Returns
        -------
        Dict
            ``synergy`` (the order-``k`` interaction), ``pred_all_present``,
            ``pred_none_present``, ``effect_joint``, ``interaction_type``,
            ``is_significant``, ``order``, ``n_conditions``, and
            ``predictions`` mapping each ablated subset to its prediction.
        """
        indices = [int(i) for i in variant_indices]
        if len(set(indices)) != len(indices):
            raise ValueError("variant_indices must be distinct.")
        k = len(indices)
        if k < 2:
            raise ValueError("An interaction needs at least two loci.")

        self.model.eval()

        if features.dim() == 2:
            features = features.unsqueeze(0)
            positions = positions.unsqueeze(0)
            gene_ids = gene_ids.unsqueeze(0)
            mask = mask.unsqueeze(0)
            if chrom_ids is not None and chrom_ids.dim() == 1:
                chrom_ids = chrom_ids.unsqueeze(0)

        if covariates is not None:
            if covariates.dim() == 1:
                covariates = covariates.unsqueeze(0)
            covariates = covariates.to(self.device)

        features = features.to(self.device)
        positions = positions.to(self.device)
        gene_ids = gene_ids.to(self.device)
        mask = mask.to(self.device)
        if chrom_ids is not None:
            chrom_ids = chrom_ids.to(self.device)

        n_positions = features.shape[1]
        for idx in indices:
            if not 0 <= idx < n_positions:
                raise IndexError(
                    f"Variant index {idx} is outside the {n_positions} positions supplied."
                )

        predictions: Dict[Tuple[int, ...], float] = {}
        synergy = 0.0

        with torch.no_grad():
            for size in range(k + 1):
                for ablated in combinations(range(k), size):
                    perturbed_features = features.clone()
                    perturbed_mask = mask.clone()
                    for slot in ablated:
                        # Ablation matches the pairwise path: the feature row is
                        # zeroed and the position is removed from the mask, so
                        # the variant leaves attention rather than becoming a
                        # zero-valued variant the model still attends to.
                        perturbed_features[0, indices[slot], :] = 0
                        perturbed_mask[0, indices[slot]] = False

                    logits, _ = self.model(
                        perturbed_features, positions, gene_ids, perturbed_mask,
                        covariates=covariates,
                        chrom_ids=chrom_ids,
                    )
                    pred = torch.sigmoid(logits).item()

                    ablated_variants = tuple(indices[slot] for slot in ablated)
                    predictions[ablated_variants] = pred
                    synergy += ((-1) ** size) * pred

        pred_all = predictions[()]
        pred_none = predictions[tuple(indices)]

        if abs(synergy) <= self.independence_epsilon:
            interaction_type = 'independent'
        elif synergy > 0:
            interaction_type = 'synergistic'
        else:
            interaction_type = 'antagonistic'

        return {
            'order': k,
            'n_conditions': len(predictions),
            'pred_all_present': pred_all,
            'pred_none_present': pred_none,
            'effect_joint': pred_all - pred_none,
            'synergy': synergy,
            'interaction_type': interaction_type,
            'is_significant': abs(synergy) > self.synergy_threshold,
            'predictions': predictions,
        }


def build_variant_sample_index(dataset) -> Tuple[Dict[VariantKey, set], List[Dict[VariantKey, int]]]:
    """
    Inverted index from variant key to carriers, and per-sample key to position.

    Parameters
    ----------
    dataset : object
        Dataset exposing ``samples`` and ``gene_index``.

    Returns
    -------
    key_to_samples : Dict[VariantKey, set]
        Sample indices carrying each variant.
    sample_key_to_idx : List[Dict[VariantKey, int]]
        Per sample, the index of each key in that sample's variant list. A
        multi-allelic site that resolves to the same key twice is recorded as
        ``-1``, so an ambiguous locus is skipped rather than guessed at.
    """
    key_to_samples: Dict[VariantKey, set] = defaultdict(set)
    sample_key_to_idx: List[Dict[VariantKey, int]] = []
    gene_index = dataset.gene_index

    for sample_idx, sample in enumerate(dataset.samples):
        key_to_idx: Dict[VariantKey, int] = {}
        for variant_idx, variant in enumerate(sample.variants):
            gene_id = gene_index.get(variant.gene, -1)
            if gene_id < 0:
                continue
            key = (int(variant.pos), int(gene_id))
            if key in key_to_idx:
                key_to_idx[key] = -1
                # The locus cannot be resolved to one variant in this sample,
                # so the sample is not a carrier the counterfactual can use;
                # leaving it in would report a candidate as untestable rather
                # than as having no carrier.
                key_to_samples[key].discard(sample_idx)
            else:
                key_to_idx[key] = variant_idx
                key_to_samples[key].add(sample_idx)
        sample_key_to_idx.append(key_to_idx)

    return key_to_samples, sample_key_to_idx


def extract_carrier_window(
    dataset,
    sample_idx: int,
    local_indices: Sequence[int],
    chunk_size: int,
) -> Optional[Tuple[Dict[str, Tensor], List[int], int, int]]:
    """
    A window of one sample's variants that holds every member of a set.

    Large samples are cut to a window centred on the members to keep the
    quadratic attention affordable. This is the same approximation the
    pairwise validation makes: excluded context can influence the members'
    embeddings, but the interaction is a ``k``-th order difference in which
    shared context largely cancels.

    Parameters
    ----------
    dataset : object
        Dataset exposing ``samples``, ``annotation_level``, ``gene_index``,
        ``impute_value`` and ``chrom_index``.
    sample_idx : int
        Sample to cut the window from.
    local_indices : Sequence[int]
        Indices of the set members in that sample's variant list.
    chunk_size : int
        Maximum variants per forward pass.

    Returns
    -------
    Optional[tuple]
        ``(tensors, remapped_indices, start, end)``, or ``None`` when the
        members span more than ``chunk_size`` variants and cannot share a
        window.
    """
    sample = dataset.samples[sample_idx]
    n_variants = len(sample.variants)
    indices = sorted(int(i) for i in local_indices)

    if n_variants <= chunk_size:
        tensors = build_variant_tensor(
            sample, dataset.annotation_level, dataset.gene_index,
            impute_value=dataset.impute_value, chrom_index=dataset.chrom_index,
        )
        return tensors, indices, 0, n_variants

    span = indices[-1] - indices[0] + 1
    if span > chunk_size:
        return None

    pad = (chunk_size - span) // 2
    start = max(0, indices[0] - pad)
    end = min(n_variants, start + chunk_size)
    start = max(0, end - chunk_size)

    window = SampleVariants(
        sample_id=sample.sample_id,
        label=sample.label,
        variants=sample.variants[start:end],
    )
    tensors = build_variant_tensor(
        window, dataset.annotation_level, dataset.gene_index,
        impute_value=dataset.impute_value, chrom_index=dataset.chrom_index,
    )
    return tensors, [i - start for i in indices], start, end


def resolve_num_covariates(model: nn.Module) -> int:
    """
    Number of sample-level covariates the fitted model expects.

    Parameters
    ----------
    model : nn.Module
        A SIEVE model or a chunked wrapper around one.

    Returns
    -------
    int
        ``num_covariates`` of the underlying model, zero when it has none.
    """
    base = getattr(model, 'base_model', model)
    return int(getattr(base, 'num_covariates', 0))


def build_carrier_covariates(
    dataset,
    sample_idx: int,
    num_covariates: int,
    device: str = 'cpu',
) -> Optional[Tensor]:
    """
    One individual's covariate vector, assembled as training assembled it.

    Sex occupies column 0 and any further columns carry the covariates stored
    on the sample, which is the convention
    :func:`~src.models.chunked_sieve.build_sample_covariates` applies for
    training and integrated gradients alike.

    Parameters
    ----------
    dataset : object
        Dataset exposing ``samples``.
    sample_idx : int
        Individual to build the vector for.
    num_covariates : int
        Number of covariates the model expects.
    device : str
        Device the returned tensor is placed on.

    Returns
    -------
    Optional[Tensor]
        ``(1, num_covariates)``, or ``None`` for a model with no covariates.
    """
    if num_covariates == 0:
        return None

    # Imported here rather than at module level, as the gradients path does,
    # to keep src.explain from importing src.models at import time.
    from src.models.chunked_sieve import build_sample_covariates

    target_device = torch.device(device)
    sample = dataset.samples[sample_idx]

    stored = getattr(sample, 'covariates', None)
    batch_covariates = None
    if stored is not None:
        batch_covariates = torch.as_tensor(
            np.asarray(stored, dtype=np.float32)
        ).reshape(1, -1).to(target_device)

    sex = torch.tensor(
        [encode_sex_for_covariate(getattr(sample, 'sex', None))],
        dtype=torch.float32,
    ).to(target_device)

    return build_sample_covariates(
        sex, num_covariates, 1, target_device,
        batch_covariates=batch_covariates,
    )


def test_candidate_sets(
    tester: HigherOrderCounterfactual,
    dataset,
    candidates: Sequence[CandidateSet],
    graph: AttentionGraph,
    chunk_size: int = 3000,
    max_carriers: int = 1,
    max_order: int = 5,
) -> List[Dict]:
    """
    Run the order-``k`` counterfactual on every candidate that has carriers.

    Parameters
    ----------
    tester : HigherOrderCounterfactual
        Configured counterfactual tester.
    dataset : object
        Cohort the candidates were discovered in.
    candidates : Sequence[CandidateSet]
        Candidates from :func:`discover_candidate_sets`.
    graph : AttentionGraph
        Graph whose node indices the candidates refer to.
    chunk_size : int
        Maximum variants per forward pass.
    max_carriers : int
        Individuals to test each candidate in. The reported synergy is the
        mean over the carriers that ran without error.
    max_order : int
        Candidates above this order are recorded as untested rather than
        costing ``2^k`` forward passes.

    Returns
    -------
    List[Dict]
        One row per candidate, including candidates that could not be tested,
        so the emitted candidate list and this table stay aligned.
    """
    key_to_samples, sample_key_to_idx = build_variant_sample_index(dataset)
    # A model fitted with covariates scores a carrier at that carrier's own
    # profile; leaving them out would silently evaluate every condition at an
    # all-zero profile.
    num_covariates = resolve_num_covariates(tester.model)
    results: List[Dict] = []

    for candidate_idx, candidate in enumerate(candidates):
        keys = [graph.keys[m] for m in candidate.members]
        row = {
            'set_id': candidate_idx,
            'order': candidate.order,
            'members': ';'.join(f"{pos}:{gene}" for pos, gene in keys),
            'density': candidate.density,
            'density_gap': candidate.density_gap,
            'density_z': candidate.density_z,
            'min_edge': candidate.min_edge,
            'n_carriers_tested': 0,
            'synergy': float('nan'),
            'interaction_type': 'untested',
            'is_significant': False,
            'status': 'untested',
        }

        if candidate.order > max_order:
            row['status'] = 'above_max_order'
            results.append(row)
            continue

        carriers = set.intersection(
            *(key_to_samples.get(key, set()) for key in keys)
        ) if keys else set()

        if not carriers:
            row['status'] = 'no_carrier'
            results.append(row)
            continue

        # Smaller carriers first: the counterfactual is quadratic in the
        # window, so the cheapest individual that carries the set is preferred.
        ordered_carriers = sorted(
            carriers, key=lambda si: len(dataset.samples[si].variants)
        )

        synergies: List[float] = []
        effects: List[float] = []
        tested_samples: List[int] = []

        for sample_idx in ordered_carriers:
            if len(synergies) >= max_carriers:
                break

            key_to_idx = sample_key_to_idx[sample_idx]
            local = [key_to_idx.get(key, -1) for key in keys]
            if any(i < 0 for i in local):
                continue

            window = extract_carrier_window(dataset, sample_idx, local, chunk_size)
            if window is None:
                continue
            tensors, remapped, _start, _end = window
            carrier_covariates = build_carrier_covariates(
                dataset, sample_idx, num_covariates, tester.device
            )

            try:
                result = tester.compute_interaction(
                    features=tensors['features'],
                    positions=tensors['positions'],
                    gene_ids=tensors['gene_ids'],
                    mask=tensors['mask'],
                    variant_indices=remapped,
                    chrom_ids=tensors.get('chrom_ids'),
                    covariates=carrier_covariates,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  Warning: CUDA OOM on set {candidate_idx}, sample {sample_idx}")
                continue
            except Exception as error:
                print(f"  Warning: set {candidate_idx} failed on sample {sample_idx}: {error}")
                continue

            synergies.append(result['synergy'])
            effects.append(result['effect_joint'])
            tested_samples.append(sample_idx)

        if not synergies:
            row['status'] = 'no_testable_carrier'
            results.append(row)
            continue

        mean_synergy = float(np.mean(synergies))
        if abs(mean_synergy) <= tester.independence_epsilon:
            interaction_type = 'independent'
        elif mean_synergy > 0:
            interaction_type = 'synergistic'
        else:
            interaction_type = 'antagonistic'

        row.update({
            'n_carriers_tested': len(synergies),
            'carrier_samples': ';'.join(str(s) for s in tested_samples),
            'synergy': mean_synergy,
            'synergy_sd': float(np.std(synergies)) if len(synergies) > 1 else 0.0,
            'effect_joint': float(np.mean(effects)),
            'interaction_type': interaction_type,
            'is_significant': bool(abs(mean_synergy) > tester.synergy_threshold),
            'status': 'tested',
        })
        results.append(row)

    return results
