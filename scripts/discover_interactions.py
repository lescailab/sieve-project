#!/usr/bin/env python3
"""
Discover higher-order epistatic interactions from a fitted SIEVE model.

Recovers epistatic variant sets of order greater than two from artefacts a
completed run already wrote, without retraining or re-running explain.py.
Attention restricts the search space; only the candidates that survive that
restriction are tested with a counterfactual experiment.

The pairwise path is untouched: explain.py still writes thresholded pairs and
validate_epistasis.py still validates them.  This entry point reads those
outputs and never writes to them.

Usage:
    python3 scripts/discover_interactions.py \
        --run-dir <run> \
        --experiment-name <name> \
        --max-order 5 \
        --n-seeds 1000 \
        --output-dir <run>/results/interaction_discovery \
        --device cuda

The run directory is expected to hold:
    experiments/<name>/best_model.pt   model checkpoint
    experiments/<name>/config.yaml     model configuration
    preprocessed.pt                    preprocessed cohort
    results/sieve_interactions.csv     thresholded pairs (optional, seeding)
    results/attributions_per_sample/   per-sample attributions (optional)

Any of these can be overridden with an explicit path.  A truth file is a
separate optional argument used only by the evaluation stage, so the discovery
path cannot reach it.

Author: Francesco Lescai
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from src.encoding import get_feature_dimension, AnnotationLevel
from src.encoding.chunked_dataset import ChunkedVariantDataset
from src.models import ChunkedSIEVEModel
from src.models.sieve import create_sieve_model, load_state_dict_with_legacy_upgrade
from src.explain.higher_order import (
    HigherOrderCounterfactual,
    resolve_num_covariates,
    accumulate_attention_mass,
    build_attention_graph,
    collect_seeds,
    discover_candidate_sets,
    load_attribution_mass,
    select_graph_variants,
    test_candidate_sets,
)
from src.explain import higher_order_evaluation as evaluation


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Discover higher-order epistatic interactions from attention',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--run-dir', type=str, required=True,
                        help='Run directory holding experiments/, preprocessed.pt and results/')
    parser.add_argument('--experiment-name', type=str,
                        help='Experiment under <run-dir>/experiments/ to read the model from')
    parser.add_argument('--checkpoint', type=str,
                        help='Model checkpoint (overrides the run-directory layout)')
    parser.add_argument('--config', type=str,
                        help='Model config.yaml (overrides the run-directory layout)')
    parser.add_argument('--preprocessed-data', type=str,
                        help='Preprocessed cohort (overrides the run-directory layout)')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory (default: <run-dir>/results/interaction_discovery)')

    graph = parser.add_argument_group('attention graph (stage 1)')
    graph.add_argument('--variant-percentile', type=float, default=99.0,
                       help='Percentile of mean attention mass below which variants are dropped')
    graph.add_argument('--max-graph-variants', type=int, default=2000,
                       help='Hard cap on the restricted variant pool')
    graph.add_argument('--min-edge-support', type=int, default=2,
                       help='Minimum individuals contributing to an edge')
    graph.add_argument('--attributions-dir', type=str,
                       help='attributions_per_sample/ directory, to narrow the pool further')
    graph.add_argument('--attribution-top-k', type=int, default=20000,
                       help='Variants kept by mean absolute attribution before the percentile cut')
    graph.add_argument('--chunk-size', type=int, default=2000,
                       help='Variants per chunk for the attention passes')
    graph.add_argument('--batch-size', type=int, default=4,
                       help='Chunks per forward pass')
    graph.add_argument('--aggregate-layers', type=str, default='mean',
                       choices=['mean', 'max', 'last'],
                       help='Reduction across attention layers')
    graph.add_argument('--aggregate-heads', type=str, default='mean',
                       choices=['mean', 'max'],
                       help='Reduction across attention heads')
    graph.add_argument('--max-edges-reported', type=int, default=10000,
                       help='Edges written to the graph edge table')

    search = parser.add_argument_group('candidate search (stages 2 and 3)')
    search.add_argument('--max-order', type=int, default=5,
                        help='Largest set size to grow to')
    search.add_argument('--n-seeds', type=int, default=1000,
                        help='Seed edges expanded')
    search.add_argument('--seed-pairs', type=str,
                        help='sieve_interactions.csv whose pairs are added as seeds '
                             '(default: <run-dir>/results/sieve_interactions.csv when present)')
    search.add_argument('--no-seed-pairs', action='store_true',
                        help='Seed from graph edges alone, ignoring any pair file')
    search.add_argument('--n-null-draws', type=int, default=1000,
                        help='Random sets drawn per size for the density null')
    search.add_argument('--null-quantile', type=float, default=0.95,
                        help='Quantile of the null at which expansion terminates')
    search.add_argument('--null-seed', type=int, default=0,
                        help='Seed for the random draw, so a run is reproducible')
    search.add_argument('--gap-tolerance', type=float, default=0.01,
                        help='Relative tolerance on the density gap; growth that holds '
                             'the gap to within it is emitted at the larger order')
    search.add_argument('--allow-missing-edges', action='store_true',
                        help='Admit a variant with no edge to some current members')

    test = parser.add_argument_group('counterfactual test (stage 4)')
    test.add_argument('--skip-counterfactual', action='store_true',
                      help='Emit candidates without testing them')
    test.add_argument('--top-k-sets', type=int, default=100,
                      help='Candidates tested, by decreasing density gap')
    test.add_argument('--max-test-order', type=int, default=5,
                      help='Candidates above this order are recorded as untested; '
                           'the test costs 2^k forward passes')
    test.add_argument('--n-carriers', type=int, default=1,
                      help='Individuals each candidate is tested in')
    test.add_argument('--test-chunk-size', type=int, default=3000,
                      help='Maximum variants per counterfactual forward pass')
    test.add_argument('--synergy-threshold', type=float, default=0.05,
                      help='|synergy| above which a candidate is called significant')
    test.add_argument('--independence-epsilon', type=float, default=0.01,
                      help='|synergy| below which a candidate is called independent')

    evaluate = parser.add_argument_group('evaluation (stage 5)')
    evaluate.add_argument('--truth', type=str,
                          help='CSV of the generating architecture: set_id, pos and '
                               'gene_id or gene. Read only by the evaluation stage')

    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    """
    Locate the run artefacts, honouring any explicit override.

    Falls back to the best fold when an experiment holds cross-validation
    results rather than a single ``best_model.pt``, matching how explain.py
    picks a checkpoint.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    Dict[str, Optional[Path]]
        Paths for 'checkpoint', 'config', 'preprocessed', 'output_dir',
        'seed_pairs' and 'attributions'.

    Raises
    ------
    FileNotFoundError
        When a required artefact is absent.
    ValueError
        When neither --experiment-name nor the explicit paths are given.
    """
    run_dir = Path(args.run_dir)

    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        if not args.experiment_name:
            raise ValueError(
                "--experiment-name is required unless --checkpoint is given."
            )
        exp_dir = run_dir / 'experiments' / args.experiment_name
        checkpoint = exp_dir / 'best_model.pt'
        if not checkpoint.exists():
            cv_path = exp_dir / 'cv_results.yaml'
            if cv_path.exists():
                with open(cv_path) as handle:
                    cv_results = yaml.safe_load(handle)
                best_fold = max(
                    range(len(cv_results['fold_results'])),
                    key=lambda i: cv_results['fold_results'][i]['auc'],
                )
                checkpoint = exp_dir / f'fold_{best_fold}' / 'best_model.pt'
                print(f"Using fold {best_fold} "
                      f"(AUC: {cv_results['fold_results'][best_fold]['auc']:.4f})")

    if args.config:
        config = Path(args.config)
    elif args.experiment_name:
        config = run_dir / 'experiments' / args.experiment_name / 'config.yaml'
    else:
        raise ValueError(
            "--config is required when --checkpoint is given without --experiment-name."
        )

    preprocessed = (
        Path(args.preprocessed_data) if args.preprocessed_data
        else run_dir / 'preprocessed.pt'
    )

    for label, path in (('checkpoint', checkpoint), ('config', config),
                        ('preprocessed cohort', preprocessed)):
        if not path.exists():
            raise FileNotFoundError(f"Required {label} not found: {path}")

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else run_dir / 'results' / 'interaction_discovery'
    )

    seed_pairs: Optional[Path] = None
    if not args.no_seed_pairs:
        seed_pairs = (
            Path(args.seed_pairs) if args.seed_pairs
            else run_dir / 'results' / 'sieve_interactions.csv'
        )
        if not seed_pairs.exists():
            if args.seed_pairs:
                raise FileNotFoundError(f"Seed pair file not found: {seed_pairs}")
            seed_pairs = None

    attributions: Optional[Path] = None
    if args.attributions_dir:
        attributions = Path(args.attributions_dir)
        if not attributions.exists():
            raise FileNotFoundError(f"Attributions directory not found: {attributions}")

    return {
        'checkpoint': checkpoint,
        'config': config,
        'preprocessed': preprocessed,
        'output_dir': output_dir,
        'seed_pairs': seed_pairs,
        'attributions': attributions,
    }


def load_model(config_path: Path, checkpoint_path: Path, dataset, device: str):
    """
    Rebuild the fitted model from a checkpoint and its configuration.

    Parameters
    ----------
    config_path : Path
        The run's config.yaml.
    checkpoint_path : Path
        The run's checkpoint.
    dataset : ChunkedVariantDataset
        Cohort, whose gene and chromosome counts size the embeddings.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    tuple
        ``(model, config)``.
    """
    with open(config_path) as handle:
        config = yaml.safe_load(handle)

    annotation_level = AnnotationLevel[config['level']]
    if 'input_dim' not in config:
        config['input_dim'] = get_feature_dimension(annotation_level)
    # Chromosome embedding and cross-chromosome bias bucket are sized from the
    # dataset rather than stored in the config, so surface them here for the
    # constructed model to match the checkpoint's tensor shapes.
    config['num_chromosomes'] = dataset.num_chromosomes

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    base_model = create_sieve_model(config, num_genes=dataset.num_genes)
    if any(key.startswith('base_model.') for key in state_dict):
        model = ChunkedSIEVEModel(
            base_model=base_model,
            aggregation_method=config.get('aggregation_method', 'mean'),
        )
    else:
        model = base_model
    load_state_dict_with_legacy_upgrade(model, state_dict)

    model = model.to(device)
    model.eval()
    return model, config


def load_seed_pairs(path: Path, key_to_index: Dict) -> List[Tuple[int, int]]:
    """
    Map a thresholded pair file onto graph node indices.

    Pairs whose loci fell outside the restricted pool are dropped, since a set
    cannot be expanded from a seed the graph does not hold.

    Parameters
    ----------
    path : Path
        sieve_interactions.csv from explain.py.
    key_to_index : Dict
        Graph node lookup by ``(pos, gene_id)``.

    Returns
    -------
    List[Tuple[int, int]]
        Seed edges as node-index pairs.
    """
    frame = pd.read_csv(path)
    required = {'variant1_pos', 'variant2_pos', 'variant1_gene', 'variant2_gene'}
    missing = required - set(frame.columns)
    if missing:
        print(f"  Pair file lacks column(s) {', '.join(sorted(missing))}; not seeding from it")
        return []

    seeds = []
    for _, row in frame.iterrows():
        i = key_to_index.get((int(row['variant1_pos']), int(row['variant1_gene'])))
        j = key_to_index.get((int(row['variant2_pos']), int(row['variant2_gene'])))
        if i is not None and j is not None and i != j:
            seeds.append((i, j))
    return seeds


# Column orders are named so that a table with no rows is still written with
# its headers: a run that emits no edge or no candidate must produce a file
# downstream readers can parse, not a headerless empty one.
EDGE_COLUMNS = [
    'variant1_pos', 'variant1_gene', 'variant2_pos', 'variant2_gene',
    'attention', 'n_individuals',
]

CANDIDATE_COLUMNS = [
    'set_id', 'order', 'members', 'density', 'min_edge',
    'null_mean_density', 'null_quantile_density', 'density_gap', 'density_z',
    'seed_variant1_pos', 'seed_variant1_gene',
    'seed_variant2_pos', 'seed_variant2_gene',
]

TRACE_COLUMNS = [
    'seed_variant1_pos', 'seed_variant1_gene',
    'seed_variant2_pos', 'seed_variant2_gene',
    'size', 'density', 'min_edge', 'members',
]


def write_graph_tables(graph, output_dir: Path, max_edges: int, min_support: int) -> None:
    """Write the retained variant pool and its strongest edges."""
    variants = pd.DataFrame({
        'node_index': range(graph.n_variants),
        'pos': [key[0] for key in graph.keys],
        'gene_id': [key[1] for key in graph.keys],
        'attention_mass': graph.mass,
    })
    variants.to_csv(output_dir / 'attention_graph_variants.csv', index=False)

    edges = graph.top_edges(max_edges, min_support=min_support)
    edge_frame = pd.DataFrame([
        {
            'variant1_pos': graph.keys[i][0],
            'variant1_gene': graph.keys[i][1],
            'variant2_pos': graph.keys[j][0],
            'variant2_gene': graph.keys[j][1],
            'attention': weight,
            'n_individuals': int(graph.support[i, j]),
        }
        for i, j, weight in edges
    ], columns=EDGE_COLUMNS)
    edge_frame.to_csv(output_dir / 'attention_graph_edges.csv', index=False)
    print(f"  Wrote {len(variants)} variants and {len(edge_frame)} edges")


def candidates_to_frame(candidates, graph) -> pd.DataFrame:
    """Flatten candidate sets into one row per set, members as ``pos:gene``."""
    rows = []
    for set_id, candidate in enumerate(candidates):
        keys = [graph.keys[m] for m in candidate.members]
        rows.append({
            'set_id': set_id,
            'order': candidate.order,
            'members': ';'.join(f"{pos}:{gene}" for pos, gene in keys),
            'density': candidate.density,
            'min_edge': candidate.min_edge,
            'null_mean_density': candidate.null_mean_density,
            'null_quantile_density': candidate.null_quantile_density,
            'density_gap': candidate.density_gap,
            'density_z': candidate.density_z,
            'seed_variant1_pos': graph.keys[candidate.seed[0]][0],
            'seed_variant1_gene': graph.keys[candidate.seed[0]][1],
            'seed_variant2_pos': graph.keys[candidate.seed[1]][0],
            'seed_variant2_gene': graph.keys[candidate.seed[1]][1],
        })
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        paths = resolve_paths(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1

    output_dir = paths['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Higher-Order Interaction Discovery")
    print("=" * 60)
    print(f"Checkpoint:   {paths['checkpoint']}")
    print(f"Config:       {paths['config']}")
    print(f"Cohort:       {paths['preprocessed']}")
    print(f"Output:       {output_dir}")

    print("\nLoading cohort...")
    preprocessed = torch.load(paths['preprocessed'], weights_only=False)
    all_samples = preprocessed['samples']
    print(f"Loaded {len(all_samples)} samples")

    with open(paths['config']) as handle:
        raw_config = yaml.safe_load(handle)
    annotation_level = AnnotationLevel[raw_config['level']]

    dataset = ChunkedVariantDataset(
        samples=all_samples,
        annotation_level=annotation_level,
        chunk_size=args.chunk_size,
        overlap=0,
    )

    print("\nLoading model...")
    model, config = load_model(paths['config'], paths['checkpoint'], dataset, args.device)

    # === STAGE 1: ATTENTION GRAPH ===
    print("\n" + "=" * 60)
    print("Stage 1: attention graph")
    print("=" * 60)

    restrict_to = None
    if paths['attributions'] is not None:
        print(f"Reading attributions from {paths['attributions']}...")
        attribution_mass = load_attribution_mass(paths['attributions'], dataset)
        ranked = sorted(attribution_mass, key=attribution_mass.get, reverse=True)
        restrict_to = ranked[:args.attribution_top_k]
        print(f"  Restricted to {len(restrict_to)} variants by attribution")

    print("Accumulating attention mass over the cohort...")
    mass = accumulate_attention_mass(
        model, dataset, device=args.device, batch_size=args.batch_size,
        aggregate_layers=args.aggregate_layers, aggregate_heads=args.aggregate_heads,
    )
    print(f"  Attention mass for {len(mass)} variants")

    keys, mass_threshold = select_graph_variants(
        mass,
        percentile=args.variant_percentile,
        max_variants=args.max_graph_variants,
        restrict_to=restrict_to,
    )
    print(f"  Retained {len(keys)} variants at percentile {args.variant_percentile} "
          f"(mass threshold {mass_threshold:.6g})")

    if len(keys) < 3:
        print("\nERROR: Fewer than three variants survived the restriction; "
              "there is no set to search for. Lower --variant-percentile.")
        return 1

    print("Accumulating pairwise attention over the retained pool...")
    graph = build_attention_graph(
        model, dataset, keys, mass=mass, device=args.device,
        batch_size=args.batch_size,
        aggregate_layers=args.aggregate_layers,
        aggregate_heads=args.aggregate_heads,
        min_support=args.min_edge_support,
    )
    write_graph_tables(graph, output_dir, args.max_edges_reported, args.min_edge_support)

    # === STAGES 2 AND 3: CANDIDATE SETS AND ORDER SELECTION ===
    print("\n" + "=" * 60)
    print("Stages 2 and 3: candidate sets and order selection")
    print("=" * 60)

    extra_seeds: List[Tuple[int, int]] = []
    if paths['seed_pairs'] is not None:
        extra_seeds = load_seed_pairs(paths['seed_pairs'], graph.key_to_index)
        print(f"Seeded {len(extra_seeds)} pair(s) from {paths['seed_pairs']}")

    seeds = collect_seeds(
        graph, args.n_seeds,
        min_support=args.min_edge_support,
        extra_seeds=extra_seeds,
    )
    print(f"Expanding {len(seeds)} seed edge(s) to order {args.max_order}...")

    candidates, traces, null = discover_candidate_sets(
        graph, seeds,
        max_order=args.max_order,
        n_null_draws=args.n_null_draws,
        quantile=args.null_quantile,
        null_seed=args.null_seed,
        require_all_edges=not args.allow_missing_edges,
        gap_tolerance=args.gap_tolerance,
    )
    print(f"Emitted {len(candidates)} distinct candidate set(s)")

    candidate_frame = candidates_to_frame(candidates, graph)
    candidate_frame.to_csv(output_dir / 'candidate_sets.csv', index=False)

    trace_frame = pd.DataFrame([
        {
            'seed_variant1_pos': trace['seed_variant1'][0],
            'seed_variant1_gene': trace['seed_variant1'][1],
            'seed_variant2_pos': trace['seed_variant2'][0],
            'seed_variant2_gene': trace['seed_variant2'][1],
            'size': trace['size'],
            'density': trace['density'],
            'min_edge': trace['min_edge'],
            'members': ';'.join(
                f"{graph.keys[m][0]}:{graph.keys[m][1]}" for m in trace['members']
            ),
        }
        for trace in traces
    ], columns=TRACE_COLUMNS)
    trace_frame.to_csv(output_dir / 'expansion_trace.csv', index=False)

    null_frame = pd.DataFrame([
        {
            'size': size,
            'n_draws': int(densities.size),
            'mean_density': float(densities.mean()) if densities.size else float('nan'),
            'sd_density': float(densities.std()) if densities.size else float('nan'),
            'quantile_density': (
                float(np.quantile(densities, args.null_quantile))
                if densities.size else float('nan')
            ),
        }
        for size, densities in sorted(null.items())
    ])
    null_frame.to_csv(output_dir / 'density_null.csv', index=False)

    if candidate_frame.empty:
        print("\nNo candidate set cleared the random-set density distribution.")

    # === STAGE 4: COUNTERFACTUAL TEST ===
    tested_frame = pd.DataFrame()
    if not args.skip_counterfactual and candidates:
        print("\n" + "=" * 60)
        print("Stage 4: counterfactual test")
        print("=" * 60)

        to_test = candidates[:args.top_k_sets]
        print(f"Testing {len(to_test)} candidate(s), up to 2^k forward passes each...")

        tester = HigherOrderCounterfactual(
            model=model,
            device=args.device,
            synergy_threshold=args.synergy_threshold,
            independence_epsilon=args.independence_epsilon,
        )
        results = test_candidate_sets(
            tester, dataset, to_test, graph,
            chunk_size=args.test_chunk_size,
            max_carriers=args.n_carriers,
            max_order=args.max_test_order,
        )
        tested_frame = pd.DataFrame(results)
        tested_frame.to_csv(output_dir / 'higher_order_synergy.csv', index=False)

        n_tested = int((tested_frame['status'] == 'tested').sum())
        n_significant = int(tested_frame['is_significant'].sum())
        print(f"  Tested {n_tested}, significant {n_significant}")
        if n_significant:
            significant = tested_frame[tested_frame['is_significant']].copy()
            top = significant.reindex(
                significant['synergy'].abs().sort_values(ascending=False).index
            ).head(5)
            print(top[['order', 'members', 'synergy', 'interaction_type']].to_string(index=False))

    # === SUMMARY ===
    summary = {
        'checkpoint': str(paths['checkpoint']),
        'config': str(paths['config']),
        'preprocessed_data': str(paths['preprocessed']),
        'n_samples': len(all_samples),
        'annotation_level': config['level'],
        'variant_percentile': args.variant_percentile,
        'variant_mass_threshold': float(mass_threshold),
        'n_variants_scored': len(mass),
        'n_variants_in_graph': graph.n_variants,
        'min_edge_support': args.min_edge_support,
        'aggregate_layers': args.aggregate_layers,
        'aggregate_heads': args.aggregate_heads,
        'restricted_by_attributions': paths['attributions'] is not None,
        'num_covariates': resolve_num_covariates(model),
        'n_seeds': len(seeds),
        'n_seed_pairs_from_file': len(extra_seeds),
        'max_order': args.max_order,
        'null_quantile': args.null_quantile,
        'n_null_draws': args.n_null_draws,
        'null_seed': args.null_seed,
        'require_all_edges': not args.allow_missing_edges,
        'gap_tolerance': args.gap_tolerance,
        'n_candidate_sets': len(candidates),
        'counterfactual_run': not args.skip_counterfactual and bool(candidates),
    }
    if not tested_frame.empty:
        summary['n_sets_tested'] = int((tested_frame['status'] == 'tested').sum())
        summary['n_sets_significant'] = int(tested_frame['is_significant'].sum())

    if candidates:
        orders = candidate_frame['order'].value_counts().sort_index()
        summary['candidates_by_order'] = {int(k): int(v) for k, v in orders.items()}

    with open(output_dir / 'discovery_summary.yaml', 'w') as handle:
        yaml.dump(summary, handle, default_flow_style=False, sort_keys=False)

    # === STAGE 5: EVALUATION ===
    if args.truth:
        print("\n" + "=" * 60)
        print("Stage 5: evaluation against a known architecture")
        print("=" * 60)

        truth_sets = evaluation.load_truth_sets(args.truth, gene_index=dataset.gene_index)
        print(f"Loaded {len(truth_sets)} true set(s)")

        scored = tested_frame if not tested_frame.empty else candidate_frame
        if 'is_significant' not in scored.columns:
            scored = scored.assign(is_significant=False, status='untested')

        recovery, by_order, rates = evaluation.evaluate(scored, truth_sets)

        eval_dir = output_dir / 'evaluation'
        eval_dir.mkdir(parents=True, exist_ok=True)
        recovery.to_csv(eval_dir / 'truth_recovery.csv', index=False)
        by_order.to_csv(eval_dir / 'recovery_by_order.csv', index=False)
        with open(eval_dir / 'evaluation_summary.yaml', 'w') as handle:
            yaml.dump(rates, handle, default_flow_style=False, sort_keys=False)

        if not by_order.empty:
            print(by_order.to_string(index=False))
        print(f"False discovery rate among survivors: {rates['false_discovery_rate']}")
        print(f"Evaluation written to {eval_dir}")

    print("\n" + "=" * 60)
    print(f"Discovery written to {output_dir}")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
