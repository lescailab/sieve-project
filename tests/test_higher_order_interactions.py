"""
Tests for higher-order epistatic interaction discovery.

Cover the four discovery stages and the evaluation stage:

- the attention graph and the restriction that precedes it,
- greedy expansion, which must separate a clique from a chain,
- order selection against the random-set density null,
- the order-k counterfactual, whose k = 2 case must reproduce the pairwise
  synergy the existing detector computes,
- recovery ranks and false discovery rate against a known architecture.

The pairwise path is also checked to be unchanged by the addition.

Author: Francesco Lescai
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import SampleVariants, VariantRecord
from src.encoding.levels import AnnotationLevel
from src.encoding.sparse_tensor import VariantDataset
from src.explain.counterfactual_epistasis import CounterfactualEpistasisDetector
from src.explain.higher_order import (
    AttentionGraph,
    HigherOrderCounterfactual,
    aggregate_attention,
    build_variant_sample_index,
    collect_seeds,
    discover_candidate_sets,
    expand_seed,
    extract_carrier_window,
    random_density_null,
    select_candidate_order,
    select_graph_variants,
)
from src.explain import higher_order_evaluation as evaluation
from src.models.sieve import SIEVE


# ----- Fixtures ---------------------------------------------------------------


def make_graph(weights: np.ndarray, keys=None) -> AttentionGraph:
    """Build a graph directly from a weight matrix, bypassing the model pass."""
    n = weights.shape[0]
    if keys is None:
        keys = [(1000 + 10 * i, i % 3) for i in range(n)]
    support = np.where(weights > 0, 5, 0).astype(np.int32)
    return AttentionGraph(
        keys=keys,
        weights=weights.astype(float),
        support=support,
        mass=weights.sum(axis=1),
    )


def clique_and_chain_graph() -> AttentionGraph:
    """
    Six variants: 0-1-2-3 a clique, 4 and 5 a chain hanging off it.

    Node 4 attends strongly to node 0 alone and node 5 to node 4 alone, so a
    rule that adds the variant with the greatest attention to any single
    member would pull them in, while a rule scoring against every member
    would not.
    """
    weights = np.zeros((6, 6))
    for i, j in combinations(range(4), 2):
        weights[i, j] = weights[j, i] = 0.8
    weights[0, 4] = weights[4, 0] = 0.95
    weights[4, 5] = weights[5, 4] = 0.95
    return make_graph(weights)


def planted_clique_graph(
    n_variants: int = 30,
    clique=(0, 1, 2, 3),
    clique_weight: float = 0.8,
    background: float = 0.01,
    seed: int = 0,
) -> AttentionGraph:
    """
    A clique planted in a pool of weak background attention.

    A pool this size keeps the random-set null distinct from the planted
    signal, which a handful of nodes cannot do: with six variants the strongest
    pairs dominate their own null.
    """
    rng = np.random.default_rng(seed)
    weights = rng.uniform(0.0, background, size=(n_variants, n_variants))
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0.0)
    for i, j in combinations(clique, 2):
        weights[i, j] = weights[j, i] = clique_weight
    return make_graph(weights)


def tiny_model(seed: int = 0) -> SIEVE:
    """A small SIEVE on CPU, deterministic and in eval mode."""
    torch.manual_seed(seed)
    model = SIEVE(
        input_dim=4,
        num_genes=3,
        latent_dim=8,
        hidden_dim=16,
        num_heads=2,
        num_attention_layers=2,
        classifier_hidden_dim=16,
        dropout=0.0,
    )
    model.eval()
    return model


def tiny_inputs(n_variants: int = 6, seed: int = 1):
    """Feature, position, gene and mask tensors for one individual."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.rand(1, n_variants, 4, generator=generator)
    positions = torch.arange(1, n_variants + 1).unsqueeze(0) * 1000
    gene_ids = torch.tensor([[i % 3 for i in range(n_variants)]])
    mask = torch.ones(1, n_variants, dtype=torch.bool)
    return features, positions, gene_ids, mask


class AdditiveModel(nn.Module):
    """
    A model whose logit is a chosen polynomial in the presence of three loci.

    Used to check the inclusion-exclusion sum recovers the coefficient of the
    highest-order term exactly, independently of any trained network.
    """

    def __init__(self, indices, coefficients):
        super().__init__()
        self.indices = list(indices)
        self.coefficients = dict(coefficients)

    def forward(self, features, positions, gene_ids, mask, **kwargs):
        present = tuple(bool(mask[0, i]) for i in self.indices)
        value = 0.0
        for subset, coefficient in self.coefficients.items():
            if all(present[s] for s in subset):
                value += coefficient
        # Inverted sigmoid, so the reported prediction is exactly `value`.
        logit = torch.log(torch.tensor(value) / (1 - torch.tensor(value)))
        return logit.reshape(1), None


# ----- Stage 1: attention graph ------------------------------------------------


def test_aggregate_attention_is_symmetric_with_zero_diagonal():
    """An interaction is undirected, so the reduced matrix must be symmetric."""
    attn = torch.rand(2, 3, 5, 5)
    reduced = aggregate_attention([attn, torch.rand(2, 3, 5, 5)])

    assert reduced.shape == (2, 5, 5)
    assert torch.allclose(reduced, reduced.transpose(1, 2))
    assert torch.allclose(torch.diagonal(reduced, dim1=1, dim2=2), torch.zeros(2, 5))


def test_aggregate_attention_rejects_empty_and_unknown_reductions():
    with pytest.raises(ValueError):
        aggregate_attention([])
    with pytest.raises(ValueError):
        aggregate_attention([torch.rand(1, 2, 3, 3)], aggregate_layers='median')
    with pytest.raises(ValueError):
        aggregate_attention([torch.rand(1, 2, 3, 3)], aggregate_heads='median')


def test_select_graph_variants_applies_percentile_then_cap():
    """The percentile is the criterion; the cap only bounds the graph size."""
    mass = {(i * 10, 0): float(i) for i in range(101)}

    keys, threshold = select_graph_variants(mass, percentile=90.0, max_variants=None)
    assert threshold == pytest.approx(90.0)
    assert len(keys) == 11
    assert keys[0] == (1000, 0)

    capped, _ = select_graph_variants(mass, percentile=90.0, max_variants=3)
    assert capped == keys[:3]


def test_select_graph_variants_honours_an_external_restriction():
    """An attribution-based pool is applied before the percentile cut."""
    mass = {(i * 10, 0): float(i) for i in range(101)}
    allowed = [(i * 10, 0) for i in range(10)]

    keys, _ = select_graph_variants(
        mass, percentile=50.0, max_variants=None, restrict_to=allowed
    )

    assert set(keys) <= set(allowed)
    assert (1000, 0) not in keys


def test_select_graph_variants_on_empty_input():
    keys, threshold = select_graph_variants({}, percentile=99.0)
    assert keys == []
    assert np.isnan(threshold)


def test_graph_density_counts_absent_edges_as_zero():
    """A set held together by one link must score as the chain it is."""
    weights = np.zeros((3, 3))
    weights[0, 1] = weights[1, 0] = 0.9
    graph = make_graph(weights)

    # Only one of the three possible edges carries weight.
    assert graph.density([0, 1, 2]) == pytest.approx(2 * 0.9 / 6)
    assert graph.min_edge([0, 1, 2]) == pytest.approx(0.0)
    assert graph.density([0]) == 0.0


def test_graph_top_edges_respect_support():
    weights = np.zeros((3, 3))
    weights[0, 1] = weights[1, 0] = 0.9
    weights[0, 2] = weights[2, 0] = 0.5
    graph = make_graph(weights)
    graph.support[0, 1] = graph.support[1, 0] = 1

    all_edges = graph.top_edges(5, min_support=1)
    assert [(i, j) for i, j, _ in all_edges] == [(0, 1), (0, 2)]

    supported = graph.top_edges(5, min_support=2)
    assert [(i, j) for i, j, _ in supported] == [(0, 2)]


# ----- Stage 2: greedy expansion ----------------------------------------------


def test_expansion_grows_the_clique_and_not_the_chain():
    """
    Scoring a candidate against every member is what separates a clique from
    a chain: node 4 outweighs any clique member on its single edge to node 0,
    yet must not join before nodes 2 and 3.
    """
    graph = clique_and_chain_graph()

    trace = expand_seed(graph, (0, 1), max_order=4)

    assert [entry['size'] for entry in trace] == [2, 3, 4]
    assert set(trace[-1]['members']) == {0, 1, 2, 3}


def test_expansion_stops_when_no_variant_connects_to_every_member():
    """With the all-edges requirement, an unconnected pool ends expansion."""
    graph = clique_and_chain_graph()

    trace = expand_seed(graph, (0, 1), max_order=6)

    # Nodes 4 and 5 have no edge to nodes 1, 2 or 3, so the clique is maximal.
    assert set(trace[-1]['members']) == {0, 1, 2, 3}


def test_expansion_without_the_all_edges_requirement_admits_a_chain():
    graph = clique_and_chain_graph()

    trace = expand_seed(graph, (0, 1), max_order=6, require_all_edges=False)

    assert len(trace[-1]['members']) == 6
    # Admitting unconnected variants costs density, which is the signal
    # order selection later reads.
    assert trace[-1]['density'] < trace[2]['density']


def test_clique_holds_density_while_a_chain_loses_it():
    graph = clique_and_chain_graph()

    trace = expand_seed(graph, (0, 1), max_order=4)
    densities = [entry['density'] for entry in trace]

    assert densities == pytest.approx([0.8, 0.8, 0.8])


def test_collect_seeds_puts_supplied_pairs_first_and_deduplicates():
    graph = clique_and_chain_graph()

    seeds = collect_seeds(graph, n_seeds=3, extra_seeds=[(2, 3), (3, 2)])

    assert seeds[0] == (2, 3)
    assert len(seeds) == len(set(seeds))


# ----- Stage 3: order selection ------------------------------------------------


def test_random_density_null_is_reproducible_and_sized_per_draw():
    graph = clique_and_chain_graph()

    first = random_density_null(graph, [3, 4], n_draws=50, seed=7)
    second = random_density_null(graph, [3, 4], n_draws=50, seed=7)

    assert set(first) == {3, 4}
    assert first[3].shape == (50,)
    assert np.allclose(first[3], second[3])


def test_random_density_null_returns_empty_for_impossible_sizes():
    graph = make_graph(np.zeros((3, 3)))
    null = random_density_null(graph, [1, 4], n_draws=10)
    assert null[1].size == 0
    assert null[4].size == 0


def test_order_selection_emits_the_size_maximising_the_density_gap():
    """
    A set that holds density to size 4 and loses it at size 5 must be emitted
    at order 4.
    """
    graph = clique_and_chain_graph()
    trace = [
        {'members': (0, 1), 'size': 2, 'density': 0.8, 'min_edge': 0.8},
        {'members': (0, 1, 2), 'size': 3, 'density': 0.8, 'min_edge': 0.8},
        {'members': (0, 1, 2, 3), 'size': 4, 'density': 0.8, 'min_edge': 0.8},
        {'members': (0, 1, 2, 3, 4), 'size': 5, 'density': 0.05, 'min_edge': 0.0},
    ]
    null = {
        2: np.full(100, 0.30),
        3: np.full(100, 0.25),
        4: np.full(100, 0.20),
        5: np.full(100, 0.15),
    }

    candidate = select_candidate_order(graph, trace, null, seed=(0, 1))

    assert candidate.order == 4
    assert candidate.members == (0, 1, 2, 3)
    assert candidate.density_gap == pytest.approx(0.6)


def test_order_selection_returns_nothing_when_the_curve_never_clears_the_null():
    graph = clique_and_chain_graph()
    trace = [{'members': (0, 1), 'size': 2, 'density': 0.01, 'min_edge': 0.01}]
    null = {2: np.full(100, 0.5)}

    assert select_candidate_order(graph, trace, null, seed=(0, 1)) is None


def test_discovery_deduplicates_sets_reached_from_different_seeds():
    graph = planted_clique_graph()
    seeds = [(0, 1), (2, 3), (0, 2)]

    candidates, traces, null = discover_candidate_sets(
        graph, seeds, max_order=4, n_null_draws=200, null_seed=3
    )

    member_sets = [candidate.members for candidate in candidates]
    assert len(member_sets) == len(set(member_sets))
    assert (0, 1, 2, 3) in member_sets
    assert set(null) == {2, 3, 4}
    assert len(traces) >= len(seeds)


def test_discovery_emits_a_planted_clique_at_its_own_order():
    """
    Growing past the planted order dilutes the density, so the emitted size is
    the clique's own, not the largest the expansion reached.
    """
    graph = planted_clique_graph()

    candidates, _, _ = discover_candidate_sets(
        graph, [(0, 1)], max_order=6, n_null_draws=500, null_seed=1
    )

    assert candidates[0].members == (0, 1, 2, 3)
    assert candidates[0].order == 4
    assert candidates[0].density == pytest.approx(0.8)


def test_discovered_candidates_are_ranked_by_density_gap():
    graph = planted_clique_graph()
    candidates, _, _ = discover_candidate_sets(
        graph, collect_seeds(graph, n_seeds=10), max_order=4, n_null_draws=200
    )

    gaps = [candidate.density_gap for candidate in candidates]
    assert gaps == sorted(gaps, reverse=True)


# ----- Stage 4: counterfactual of order k --------------------------------------


def test_order_two_reproduces_the_existing_pairwise_synergy():
    """
    The k = 2 case of the inclusion-exclusion form must equal the synergy the
    pairwise detector already computes, which fixes the sign convention.
    """
    model = tiny_model()
    features, positions, gene_ids, mask = tiny_inputs()

    pairwise = CounterfactualEpistasisDetector(model=model, device='cpu')
    pairwise_result = pairwise.validate_interaction_with_perturbation(
        features=features, positions=positions, gene_ids=gene_ids, mask=mask,
        variant1_idx=1, variant2_idx=4,
    )

    higher_order = HigherOrderCounterfactual(model=model, device='cpu')
    higher_order_result = higher_order.compute_interaction(
        features=features, positions=positions, gene_ids=gene_ids, mask=mask,
        variant_indices=[1, 4],
    )

    assert higher_order_result['synergy'] == pytest.approx(
        pairwise_result['synergy'], abs=1e-6
    )
    assert higher_order_result['pred_all_present'] == pytest.approx(
        pairwise_result['pred_both'], abs=1e-6
    )
    assert higher_order_result['pred_none_present'] == pytest.approx(
        pairwise_result['pred_neither'], abs=1e-6
    )
    assert higher_order_result['interaction_type'] == pairwise_result['interaction_type']


def test_counterfactual_runs_two_to_the_k_conditions():
    model = tiny_model()
    features, positions, gene_ids, mask = tiny_inputs()
    tester = HigherOrderCounterfactual(model=model, device='cpu')

    for order in (2, 3, 4):
        result = tester.compute_interaction(
            features=features, positions=positions, gene_ids=gene_ids, mask=mask,
            variant_indices=list(range(order)),
        )
        assert result['order'] == order
        assert result['n_conditions'] == 2 ** order


def test_inclusion_exclusion_recovers_the_third_order_coefficient():
    """
    On a model whose prediction is a known polynomial in locus presence, the
    order-3 sum must return the coefficient of the three-way term and cancel
    every lower-order term.
    """
    indices = [0, 1, 2]
    coefficients = {
        (): 0.10,
        (0,): 0.05,
        (1,): 0.07,
        (0, 1): 0.03,
        (0, 1, 2): 0.20,
    }
    model = AdditiveModel(indices, coefficients)
    features, positions, gene_ids, mask = tiny_inputs(n_variants=4)

    tester = HigherOrderCounterfactual(model=model, device='cpu')
    result = tester.compute_interaction(
        features=features, positions=positions, gene_ids=gene_ids, mask=mask,
        variant_indices=indices,
    )

    assert result['synergy'] == pytest.approx(0.20, abs=1e-5)
    assert result['interaction_type'] == 'synergistic'
    assert result['is_significant']


def test_a_purely_additive_model_has_no_higher_order_interaction():
    model = AdditiveModel([0, 1, 2], {(): 0.2, (0,): 0.1, (1,): 0.05, (2,): 0.07})
    features, positions, gene_ids, mask = tiny_inputs(n_variants=4)

    result = HigherOrderCounterfactual(model=model, device='cpu').compute_interaction(
        features=features, positions=positions, gene_ids=gene_ids, mask=mask,
        variant_indices=[0, 1, 2],
    )

    assert result['synergy'] == pytest.approx(0.0, abs=1e-6)
    assert result['interaction_type'] == 'independent'
    assert not result['is_significant']


def test_counterfactual_rejects_degenerate_sets_and_out_of_range_indices():
    model = tiny_model()
    features, positions, gene_ids, mask = tiny_inputs()
    tester = HigherOrderCounterfactual(model=model, device='cpu')

    with pytest.raises(ValueError):
        tester.compute_interaction(features, positions, gene_ids, mask, [1, 1, 2])
    with pytest.raises(ValueError):
        tester.compute_interaction(features, positions, gene_ids, mask, [1])
    with pytest.raises(IndexError):
        tester.compute_interaction(features, positions, gene_ids, mask, [1, 99])


# ----- Carrier lookup ----------------------------------------------------------


def make_dataset(n_samples: int = 4, n_variants: int = 8) -> VariantDataset:
    """A small cohort in which every sample carries the same variant keys."""
    samples = []
    for s in range(n_samples):
        variants = [
            VariantRecord('1', 1000 + 10 * v, 'A', 'T', f'GENE{v % 3}',
                          'missense_variant', 1, {})
            for v in range(n_variants)
        ]
        samples.append(SampleVariants(f'sample{s}', s % 2, variants))
    return VariantDataset(samples, AnnotationLevel.L0)


def test_variant_sample_index_finds_carriers_and_flags_ambiguity():
    dataset = make_dataset()
    key_to_samples, sample_key_to_idx = build_variant_sample_index(dataset)

    gene_id = dataset.gene_index['GENE0']
    assert key_to_samples[(1000, gene_id)] == {0, 1, 2, 3}
    assert sample_key_to_idx[0][(1000, gene_id)] == 0

    # A second record at the same position in the same gene is multi-allelic
    # and must be marked ambiguous rather than resolved to one of the two.
    duplicate = VariantRecord('1', 1000, 'A', 'G', 'GENE0', 'missense_variant', 1, {})
    dataset.samples[0].variants.append(duplicate)
    recarried, reindexed = build_variant_sample_index(dataset)
    assert reindexed[0][(1000, gene_id)] == -1
    # An ambiguous locus is not a carrier the counterfactual can use, so the
    # sample must leave the carrier set rather than be filtered out later.
    assert recarried[(1000, gene_id)] == {1, 2, 3}


def test_carrier_window_keeps_every_member_inside_the_chunk():
    dataset = make_dataset(n_variants=20)

    window = extract_carrier_window(dataset, 0, [2, 9, 11], chunk_size=12)
    tensors, remapped, start, end = window

    assert end - start == 12
    assert tensors['features'].shape[0] == 12
    assert all(0 <= i < 12 for i in remapped)
    assert [i + start for i in remapped] == [2, 9, 11]


def test_carrier_window_declines_when_members_cannot_share_a_chunk():
    dataset = make_dataset(n_variants=40)
    assert extract_carrier_window(dataset, 0, [0, 39], chunk_size=10) is None


def test_carrier_window_returns_the_whole_sample_when_it_fits():
    dataset = make_dataset(n_variants=6)
    tensors, remapped, start, end = extract_carrier_window(
        dataset, 0, [1, 4], chunk_size=100
    )
    assert (start, end) == (0, 6)
    assert remapped == [1, 4]
    assert tensors['features'].shape[0] == 6


# ----- Stage 5: evaluation -----------------------------------------------------


def candidate_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {'members': '100:0;200:1;300:2', 'density_gap': 0.6,
         'synergy': 0.20, 'is_significant': True, 'status': 'tested'},
        {'members': '400:0;500:1', 'density_gap': 0.4,
         'synergy': 0.01, 'is_significant': False, 'status': 'tested'},
        {'members': '600:0;700:1;800:2;900:0', 'density_gap': 0.2,
         'synergy': -0.30, 'is_significant': True, 'status': 'tested'},
    ])


def test_truth_file_is_read_with_gene_ids_or_symbols(tmp_path):
    by_id = tmp_path / 'truth_ids.csv'
    by_id.write_text(
        'set_id,pos,gene_id\n1,100,0\n1,200,1\n1,300,2\n2,400,0\n2,500,1\n'
    )
    sets = evaluation.load_truth_sets(by_id)
    assert sets == [((100, 0), (200, 1), (300, 2)), ((400, 0), (500, 1))]

    by_symbol = tmp_path / 'truth_symbols.csv'
    by_symbol.write_text('set_id,pos,gene\n1,100,GENE0\n1,200,GENE1\n')
    resolved = evaluation.load_truth_sets(
        by_symbol, gene_index={'GENE0': 0, 'GENE1': 1}
    )
    assert resolved == [((100, 0), (200, 1))]


def test_truth_file_errors_are_explicit(tmp_path):
    no_gene = tmp_path / 'no_gene.csv'
    no_gene.write_text('set_id,pos\n1,100\n1,200\n')
    with pytest.raises(ValueError, match="gene_id"):
        evaluation.load_truth_sets(no_gene)

    unknown = tmp_path / 'unknown.csv'
    unknown.write_text('set_id,pos,gene\n1,100,MISSING\n1,200,GENE1\n')
    with pytest.raises(ValueError, match="gene index"):
        evaluation.load_truth_sets(unknown, gene_index={'GENE1': 1})

    no_set = tmp_path / 'no_set.csv'
    no_set.write_text('pos,gene_id\n100,0\n')
    with pytest.raises(ValueError, match="set_id"):
        evaluation.load_truth_sets(no_set)


def test_recovery_reports_the_rank_of_each_true_set():
    truth = [((100, 0), (200, 1), (300, 2)), ((1000, 0), (1100, 1))]

    recovery = evaluation.evaluate_recovery(candidate_frame(), truth)

    recovered = recovery.set_index('order')
    assert recovered.loc[3, 'exact_rank'] == 1
    assert bool(recovered.loc[3, 'recovered_exact'])
    assert not bool(recovered.loc[2, 'recovered_exact'])
    assert pd.isna(recovered.loc[2, 'containing_rank'])


def test_recovery_credits_a_true_set_contained_in_a_larger_candidate():
    truth = [((600, 0), (700, 1))]

    recovery = evaluation.evaluate_recovery(candidate_frame(), truth)

    assert not bool(recovery.loc[0, 'recovered_exact'])
    assert bool(recovery.loc[0, 'recovered_containing'])
    assert recovery.loc[0, 'containing_rank'] == 3


def test_recovery_by_order_reports_proportions():
    truth = [
        ((100, 0), (200, 1), (300, 2)),
        ((1000, 0), (1100, 1), (1200, 2)),
        ((400, 0), (500, 1)),
    ]

    recovery = evaluation.evaluate_recovery(candidate_frame(), truth)
    by_order = evaluation.recovery_by_order(recovery)

    third = by_order[by_order['order'] == 3].iloc[0]
    assert third['n_true'] == 2
    assert third['proportion_exact'] == pytest.approx(0.5)

    second = by_order[by_order['order'] == 2].iloc[0]
    assert second['proportion_exact'] == pytest.approx(1.0)


def test_false_discovery_rate_counts_only_surviving_candidates():
    truth = [((100, 0), (200, 1), (300, 2))]

    rates = evaluation.false_discovery_rate(candidate_frame(), truth)

    assert rates['n_tested'] == 3
    assert rates['n_surviving'] == 2
    assert rates['n_true_positive'] == 1
    assert rates['false_discovery_rate'] == pytest.approx(0.5)


def test_a_cohort_without_interactions_makes_every_survivor_a_false_positive():
    rates = evaluation.false_discovery_rate(candidate_frame(), [])

    assert rates['n_true_positive'] == 0
    assert rates['false_discovery_rate'] == pytest.approx(1.0)


def test_false_discovery_rate_is_undefined_with_no_survivors():
    frame = candidate_frame().assign(is_significant=False)
    rates = evaluation.false_discovery_rate(frame, [])
    assert np.isnan(rates['false_discovery_rate'])


def test_recovery_by_order_on_an_empty_truth_file():
    by_order = evaluation.recovery_by_order(pd.DataFrame())
    assert by_order.empty
    assert 'proportion_exact' in by_order.columns


# ----- The pairwise path is unchanged ------------------------------------------


def test_pairwise_detector_keeps_its_fields_and_synergy_expression():
    """
    The addition must not alter the existing pairwise output: the four
    conditions, their derived effects and the synergy identity are checked
    against the values the detector itself reports.
    """
    model = tiny_model()
    features, positions, gene_ids, mask = tiny_inputs()

    result = CounterfactualEpistasisDetector(
        model=model, device='cpu'
    ).validate_interaction_with_perturbation(
        features=features, positions=positions, gene_ids=gene_ids, mask=mask,
        variant1_idx=0, variant2_idx=3,
    )

    expected_fields = {
        'pred_both', 'pred_variant1_only', 'pred_variant2_only', 'pred_neither',
        'effect_variant1', 'effect_variant2', 'effect_combined', 'synergy',
        'interaction_type', 'is_significant',
    }
    assert expected_fields <= set(result)
    assert result['synergy'] == pytest.approx(
        result['pred_both'] - result['pred_variant1_only']
        - result['pred_variant2_only'] + result['pred_neither'],
        abs=1e-9,
    )


def test_order_selection_prefers_the_larger_order_when_density_holds():
    """
    A density curve that stays flat has not shown the set to have ended, so
    the emitted order is the largest size the gap survives to, not the seed.
    """
    graph = planted_clique_graph()
    trace = [
        {'members': (0, 1), 'size': 2, 'density': 0.8, 'min_edge': 0.8},
        {'members': (0, 1, 2), 'size': 3, 'density': 0.8, 'min_edge': 0.8},
        {'members': (0, 1, 2, 3), 'size': 4, 'density': 0.8, 'min_edge': 0.8},
    ]
    # A null whose mean rises slightly with size makes the gap shrink as the
    # set grows, which without a tolerance would emit the seed pair.
    null = {2: np.full(100, 0.010), 3: np.full(100, 0.012), 4: np.full(100, 0.014)}

    candidate = select_candidate_order(graph, trace, null, seed=(0, 1))
    assert candidate.order == 4

    strict = select_candidate_order(graph, trace, null, seed=(0, 1), gap_tolerance=0.0)
    assert strict.order == 2


# ----- End to end over a fitted model ------------------------------------------


def test_attention_graph_is_built_over_the_cohort_and_drives_a_test():
    """
    Stages 1 to 4 run together on a small model and cohort: mass restricts the
    pool, the graph is accumulated over individuals, expansion emits sets, and
    the counterfactual reports a synergy for the sets that have carriers.
    """
    from src.encoding.chunked_dataset import ChunkedVariantDataset
    from src.explain.higher_order import (
        accumulate_attention_mass, build_attention_graph, test_candidate_sets
    )

    samples = []
    for s in range(6):
        variants = [
            VariantRecord('1', 1000 + 10 * v, 'A', 'T', f'GENE{v % 3}',
                          'missense_variant', 1, {})
            for v in range(10)
        ]
        samples.append(SampleVariants(f'sample{s}', s % 2, variants))

    dataset = ChunkedVariantDataset(
        samples=samples, annotation_level=AnnotationLevel.L0, chunk_size=10, overlap=0
    )
    model = SIEVE(
        input_dim=dataset[0]['features'].shape[1],
        num_genes=dataset.num_genes,
        latent_dim=8, hidden_dim=16, num_heads=2, num_attention_layers=1,
        classifier_hidden_dim=16, dropout=0.0,
        num_chromosomes=dataset.num_chromosomes,
    )
    model.eval()

    mass = accumulate_attention_mass(model, dataset, device='cpu', batch_size=2)
    assert len(mass) == 10
    assert all(value > 0 for value in mass.values())

    keys, _threshold = select_graph_variants(mass, percentile=0.0, max_variants=None)
    graph = build_attention_graph(
        model, dataset, keys, mass=mass, device='cpu', batch_size=2, min_support=2
    )

    assert graph.n_variants == 10
    assert np.allclose(graph.weights, graph.weights.T)
    assert np.all(np.diagonal(graph.weights) == 0)
    # Every variant is carried by all six individuals in this cohort.
    assert int(graph.support[0, 1]) == 6

    candidates, _traces, _null = discover_candidate_sets(
        graph, collect_seeds(graph, n_seeds=5), max_order=3, n_null_draws=100
    )

    if candidates:
        tester = HigherOrderCounterfactual(model=model, device='cpu')
        results = test_candidate_sets(
            tester, dataset, candidates[:2], graph, chunk_size=10, max_carriers=1
        )
        assert len(results) == len(candidates[:2])
        assert all(row['status'] == 'tested' for row in results)
        assert all(np.isfinite(row['synergy']) for row in results)


def test_a_set_with_no_carrier_is_recorded_rather_than_dropped():
    """
    The emitted candidate list and the synergy table stay aligned, so a set
    that cannot be tested is reported with its reason.
    """
    from src.explain.higher_order import CandidateSet, test_candidate_sets

    dataset = make_dataset(n_samples=2, n_variants=4)
    model = tiny_model()
    graph = make_graph(np.full((2, 2), 0.5) - np.eye(2) * 0.5,
                       keys=[(999999, 0), (888888, 1)])
    candidate = CandidateSet(
        members=(0, 1), order=2, density=0.5, min_edge=0.5,
        null_mean_density=0.1, null_quantile_density=0.2,
        density_gap=0.4, density_z=2.0, seed=(0, 1),
    )

    results = test_candidate_sets(
        HigherOrderCounterfactual(model=model, device='cpu'),
        dataset, [candidate], graph, chunk_size=10,
    )

    assert len(results) == 1
    assert results[0]['status'] == 'no_carrier'
    assert not results[0]['is_significant']


def test_the_discovery_module_does_not_import_the_evaluation_module():
    """
    Stages 1 to 4 must not be able to reach a truth file, which is enforced by
    the evaluation living in a module the discovery code never imports.
    """
    import src.explain.higher_order as discovery

    source = Path(discovery.__file__).read_text()
    assert 'higher_order_evaluation' not in source.replace(
        ':mod:`src.explain.higher_order_evaluation`', ''
    )


# ----- Output tables stay readable when a run emits nothing --------------------


def test_empty_output_tables_still_carry_their_headers(tmp_path):
    """
    A run that emits no edge and no candidate must still write files the same
    downstream reader can parse, not headerless empty ones.
    """
    from scripts.discover_interactions import (
        CANDIDATE_COLUMNS, EDGE_COLUMNS, candidates_to_frame, write_graph_tables
    )

    # Every edge falls below the support threshold, so no edge is reported.
    graph = make_graph(np.zeros((3, 3)))
    write_graph_tables(graph, tmp_path, max_edges=10, min_support=2)

    edges = pd.read_csv(tmp_path / 'attention_graph_edges.csv')
    assert edges.empty
    assert list(edges.columns) == EDGE_COLUMNS

    variants = pd.read_csv(tmp_path / 'attention_graph_variants.csv')
    assert len(variants) == 3

    empty_candidates = candidates_to_frame([], graph)
    assert empty_candidates.empty
    assert list(empty_candidates.columns) == CANDIDATE_COLUMNS


def test_evaluation_runs_when_no_candidate_cleared_the_null():
    """
    Stage 5 must report every true set as a miss rather than fail when the
    discovery emitted nothing.
    """
    from scripts.discover_interactions import CANDIDATE_COLUMNS, candidates_to_frame

    empty = candidates_to_frame([], make_graph(np.zeros((3, 3))))
    scored = empty.assign(is_significant=False, status='untested')
    truth = [((100, 0), (200, 1), (300, 2))]

    recovery, by_order, rates = evaluation.evaluate(scored, truth)

    assert len(recovery) == 1
    assert not bool(recovery.loc[0, 'recovered_exact'])
    assert not bool(recovery.loc[0, 'recovered_containing'])
    assert by_order.loc[0, 'proportion_exact'] == pytest.approx(0.0)
    assert rates['n_surviving'] == 0
    assert np.isnan(rates['false_discovery_rate'])
    assert 'density_gap' in CANDIDATE_COLUMNS


def test_a_candidate_whose_only_carrier_is_ambiguous_reports_no_carrier():
    """
    With the ambiguous sample removed from the carrier set, the candidate is
    reported as having no carrier rather than as untestable.
    """
    from src.explain.higher_order import CandidateSet, test_candidate_sets

    dataset = make_dataset(n_samples=1, n_variants=4)
    gene_id = dataset.gene_index['GENE0']
    # A second allele at the same position in the same gene makes the locus
    # ambiguous in the only sample that carries it.
    dataset.samples[0].variants.append(
        VariantRecord('1', 1000, 'A', 'G', 'GENE0', 'missense_variant', 1, {})
    )

    graph = make_graph(
        np.array([[0.0, 0.5], [0.5, 0.0]]),
        keys=[(1000, gene_id), (1010, dataset.gene_index['GENE1'])],
    )
    candidate = CandidateSet(
        members=(0, 1), order=2, density=0.5, min_edge=0.5,
        null_mean_density=0.1, null_quantile_density=0.2,
        density_gap=0.4, density_z=2.0, seed=(0, 1),
    )

    results = test_candidate_sets(
        HigherOrderCounterfactual(model=tiny_model(), device='cpu'),
        dataset, [candidate], graph, chunk_size=10,
    )

    assert results[0]['status'] == 'no_carrier'


# ----- Covariates reach the counterfactual -------------------------------------


class CovariateSensitiveModel(nn.Module):
    """
    A model whose three-way interaction depends on the covariate value.

    Mirrors the fitted architecture in the way that matters here: covariates
    are concatenated ahead of a non-linear head, so an omitted covariate
    vector moves the operating point and changes the interaction rather than
    only its baseline.
    """

    num_covariates = 2

    def __init__(self, indices):
        super().__init__()
        self.indices = list(indices)

    def forward(self, features, positions, gene_ids, mask, covariates=None, **kwargs):
        present = [bool(mask[0, i]) for i in self.indices]
        if covariates is None:
            # What the fitted classifier does with a missing vector.
            covariate_value = 0.0
        else:
            covariate_value = float(covariates.reshape(-1)[0])
        value = 0.1 + 0.4 * covariate_value * float(all(present))
        return torch.log(torch.tensor(value) / (1 - torch.tensor(value))).reshape(1), None


def test_covariates_are_threaded_into_the_counterfactual():
    """
    The synergy reported for a carrier must be the one at that carrier's
    covariate profile, not at the all-zero profile the classifier falls back on.
    """
    indices = [0, 1, 2]
    model = CovariateSensitiveModel(indices)
    features, positions, gene_ids, mask = tiny_inputs(n_variants=4)
    tester = HigherOrderCounterfactual(model=model, device='cpu')

    without = tester.compute_interaction(
        features, positions, gene_ids, mask, indices,
    )
    with_covariates = tester.compute_interaction(
        features, positions, gene_ids, mask, indices,
        covariates=torch.tensor([1.0, 0.0]),
    )

    assert without['synergy'] == pytest.approx(0.0, abs=1e-6)
    assert with_covariates['synergy'] == pytest.approx(0.4, abs=1e-5)


def test_a_one_dimensional_covariate_vector_is_accepted():
    model = CovariateSensitiveModel([0, 1, 2])
    features, positions, gene_ids, mask = tiny_inputs(n_variants=4)
    tester = HigherOrderCounterfactual(model=model, device='cpu')

    flat = tester.compute_interaction(
        features, positions, gene_ids, mask, [0, 1, 2],
        covariates=torch.tensor([1.0, 0.0]),
    )
    batched = tester.compute_interaction(
        features, positions, gene_ids, mask, [0, 1, 2],
        covariates=torch.tensor([[1.0, 0.0]]),
    )

    assert flat['synergy'] == pytest.approx(batched['synergy'])


def test_resolve_num_covariates_reads_through_a_chunked_wrapper():
    from src.models.chunked_sieve import ChunkedSIEVEModel
    from src.explain.higher_order import resolve_num_covariates

    plain = SIEVE(input_dim=4, num_genes=3, latent_dim=8, hidden_dim=16,
                  num_heads=2, num_attention_layers=1, classifier_hidden_dim=16,
                  dropout=0.0, num_covariates=3)
    assert resolve_num_covariates(plain) == 3
    assert resolve_num_covariates(ChunkedSIEVEModel(base_model=plain)) == 3
    assert resolve_num_covariates(tiny_model()) == 0


def test_carrier_covariates_follow_the_training_convention():
    """Sex occupies column 0, with stored covariates filling the vector."""
    from src.explain.higher_order import build_carrier_covariates

    dataset = make_dataset(n_samples=2, n_variants=4)
    dataset.samples[0].sex = 'M'
    dataset.samples[1].sex = 'F'

    assert build_carrier_covariates(dataset, 0, 0) is None

    male = build_carrier_covariates(dataset, 0, 3)
    female = build_carrier_covariates(dataset, 1, 3)
    assert male.shape == (1, 3)
    assert float(male[0, 0]) == pytest.approx(1.0)
    assert float(female[0, 0]) == pytest.approx(0.0)

    # A stored vector is used as it stands, sex included, as training built it.
    dataset.samples[0].covariates = np.array([1.0, 0.5, -0.25], dtype=np.float32)
    stored = build_carrier_covariates(dataset, 0, 3)
    assert np.allclose(stored.numpy().reshape(-1), [1.0, 0.5, -0.25])


def test_covariate_model_end_to_end_through_test_candidate_sets():
    """A covariate-bearing model is scored at each carrier's own profile."""
    from src.explain.higher_order import CandidateSet, test_candidate_sets

    dataset = make_dataset(n_samples=2, n_variants=4)
    for sample in dataset.samples:
        sample.sex = 'M'
        sample.covariates = np.array([1.0, 0.0], dtype=np.float32)

    keys = [(1000, dataset.gene_index['GENE0']),
            (1010, dataset.gene_index['GENE1']),
            (1020, dataset.gene_index['GENE2'])]
    weights = np.full((3, 3), 0.5) - np.eye(3) * 0.5
    graph = make_graph(weights, keys=keys)
    candidate = CandidateSet(
        members=(0, 1, 2), order=3, density=0.5, min_edge=0.5,
        null_mean_density=0.1, null_quantile_density=0.2,
        density_gap=0.4, density_z=2.0, seed=(0, 1),
    )

    tester = HigherOrderCounterfactual(
        model=CovariateSensitiveModel([0, 1, 2]), device='cpu'
    )
    results = test_candidate_sets(tester, dataset, [candidate], graph, chunk_size=10)

    assert results[0]['status'] == 'tested'
    # 0.4 at the carrier's profile; an omitted vector would report 0.0.
    assert results[0]['synergy'] == pytest.approx(0.4, abs=1e-5)
    assert results[0]['is_significant']
