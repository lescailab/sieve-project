#!/usr/bin/env python3
"""
Validate epistatic interactions using counterfactual perturbation.

Takes interactions discovered by attention analysis and validates them
using counterfactual perturbation experiments.

Usage:
    python scripts/validate_epistasis.py \
        --interactions results/sieve_interactions.csv \
        --checkpoint outputs/L3_attr_medium/fold_0/best_model.pt \
        --config outputs/L3_attr_medium/config.yaml \
        --preprocessed-data data/preprocessed.pt \
        --output-dir results/epistasis_validation \
        --top-k 50

Author: Francesco Lescai
"""

import argparse
from pathlib import Path
import yaml
import torch
import pandas as pd

from src.encoding import VariantDataset, get_feature_dimension, AnnotationLevel
from src.encoding.sparse_tensor import build_variant_tensor
from src.models.sieve import create_sieve_model
from src.models import ChunkedSIEVEModel
from src.data import SampleVariants
from src.explain.counterfactual_epistasis import CounterfactualEpistasisDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate epistatic interactions with counterfactual perturbation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required inputs
    parser.add_argument('--interactions', type=str, required=True,
                        help='Path to interactions CSV from explain.py')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml')
    parser.add_argument('--preprocessed-data', type=str, required=True,
                        help='Path to preprocessed data')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for validation results')

    # Options
    parser.add_argument('--top-k', type=int, default=50,
                        help='Number of top interactions to validate')
    parser.add_argument('--synergy-threshold', type=float, default=0.05,
                        help='Minimum synergy to consider significant')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--chunk-size', type=int, default=3000,
                        help='Max variants per forward pass (default: 3000, '
                             'matching training chunk size)')

    # Genome build
    parser.add_argument('--genome-build', type=str, default='GRCh37',
                        help='Reference genome build (GRCh37 or GRCh38)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Epistasis Validation via Counterfactual Perturbation")
    print("="*60)

    # Load interactions
    print("\nLoading interactions...")

    # Check if file exists and is not empty
    interactions_path = Path(args.interactions)
    if not interactions_path.exists():
        print(f"ERROR: Interactions file not found: {args.interactions}")
        print("Please run explain.py first to generate interactions.")
        return

    # Check if file has content (more than just header)
    file_size = interactions_path.stat().st_size
    if file_size < 10:  # Less than 10 bytes means empty or just newline
        print(f"ERROR: Interactions file is empty: {args.interactions}")
        print("\nNo interactions were found during attention analysis.")
        print("This can happen when:")
        print("  - The model doesn't use attention mechanisms")
        print("  - Attention weights are too uniform (no strong patterns)")
        print("  - The attention threshold is too high")
        print("\nTo proceed, you can:")
        print("  1. Run explain.py with --attention-threshold 0.01 (lower threshold)")
        print("  2. Skip epistasis validation if your model doesn't rely on interactions")
        return

    try:
        interactions_df = pd.read_csv(args.interactions)
    except pd.errors.EmptyDataError:
        print(f"ERROR: Interactions file has no data: {args.interactions}")
        print("\nThe CSV file exists but contains no interaction records.")
        print("This means no variant-variant interactions were detected.")
        return

    if len(interactions_df) == 0:
        print("ERROR: Interactions file has no rows.")
        print("\nNo interactions available for validation.")
        return

    print(f"Loaded {len(interactions_df)} interactions")

    # Validate required columns
    required_cols = {'variant1_pos', 'variant2_pos', 'variant1_gene', 'variant2_gene'}
    missing_cols = required_cols - set(interactions_df.columns)
    if missing_cols:
        print(f"ERROR: Interactions CSV missing required columns: {', '.join(sorted(missing_cols))}")
        print("Expected columns from explain.py attention analysis: "
              "variant1_pos, variant2_pos, variant1_gene, variant2_gene")
        return

    # Sort by attention and take top K
    if 'mean_attention' in interactions_df.columns:
        interactions_df = interactions_df.sort_values('mean_attention', ascending=False)
    elif 'attention_score' in interactions_df.columns:
        interactions_df = interactions_df.sort_values('attention_score', ascending=False)

    top_interactions = interactions_df.head(args.top_k)
    print(f"Validating top {len(top_interactions)} interactions")

    # Load model and config
    print("\nLoading model...")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    chunk_size = config.get('chunk_size', args.chunk_size)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Load data
    print("Loading data...")
    preprocessed = torch.load(args.preprocessed_data, weights_only=False)
    all_samples = preprocessed['samples']
    metadata = preprocessed.get('metadata', {})

    print(f"Loaded {len(all_samples)} samples")
    if metadata:
        print(f"  Cases: {metadata.get('num_cases', 'unknown')}")
        print(f"  Controls: {metadata.get('num_controls', 'unknown')}")

    # Get annotation level
    annotation_level = AnnotationLevel[config['level']]

    # Create dataset (same way as training)
    dataset = VariantDataset(all_samples, annotation_level=annotation_level)

    # Create model (add input_dim if missing)
    if 'input_dim' not in config:
        config['input_dim'] = get_feature_dimension(annotation_level)

    base_model = create_sieve_model(config, num_genes=dataset.num_genes)

    # Check if checkpoint has chunked model (base_model. prefix) or plain model
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('base_model.') for k in state_dict.keys()):
        model = ChunkedSIEVEModel(
            base_model=base_model,
            aggregation_method=config.get('aggregation_method', 'mean')
        )
        model.load_state_dict(state_dict)
    else:
        model = base_model
        model.load_state_dict(state_dict)

    model = model.to(args.device)
    model.eval()

    # Initialize detector
    detector = CounterfactualEpistasisDetector(model=model, device=args.device)

    # Validate interactions
    print("\n" + "="*60)
    print("Validating Interactions")
    print("="*60)

    validation_results = []

    # Build a (pos, gene_id) -> list of sample indices inverted index from
    # raw sample data, avoiding full feature encoding for every sample upfront.
    print("Building position index across samples...")
    from collections import defaultdict
    variant_key_to_samples = defaultdict(set)
    # Also store per-sample (pos, gene_id) -> tensor index for later lookup
    sample_pg_to_idx = []
    gene_index = dataset.gene_index
    for si, sv in enumerate(dataset.samples):
        pg_to_idx = {}
        for ti, variant in enumerate(sv.variants):
            gene_id = gene_index.get(variant.gene, -1)
            if gene_id < 0:
                continue
            key = (variant.pos, gene_id)
            if key in pg_to_idx:
                # Multi-allelic site: same (pos, gene_id) already seen in sample.
                # Mark as ambiguous (-1) so we skip rather than pick the wrong index.
                pg_to_idx[key] = -1
            else:
                pg_to_idx[key] = ti
                variant_key_to_samples[key].add(si)
        sample_pg_to_idx.append(pg_to_idx)

    last_reported = 0

    for idx, interaction in top_interactions.iterrows():
        v1_pos = int(interaction['variant1_pos'])
        v2_pos = int(interaction['variant2_pos'])
        v1_gene = int(interaction['variant1_gene'])
        v2_gene = int(interaction['variant2_gene'])

        v1_key = (v1_pos, v1_gene)
        v2_key = (v2_pos, v2_gene)

        # Intersect candidate samples that carry both variants
        candidates_v1 = variant_key_to_samples.get(v1_key, set())
        candidates_v2 = variant_key_to_samples.get(v2_key, set())
        candidate_samples = candidates_v1 & candidates_v2

        # Sort candidates by variant count (fewest first) to prefer smaller
        # samples that are cheaper for the O(n²) attention computation.
        sorted_candidates = sorted(
            candidate_samples,
            key=lambda si: len(dataset.samples[si].variants)
        )

        validated = False
        for sample_idx in sorted_candidates:
            pg_to_idx = sample_pg_to_idx[sample_idx]
            v1_idx = pg_to_idx[v1_key]
            v2_idx = pg_to_idx[v2_key]

            # Skip if either variant is ambiguous (multi-allelic)
            if v1_idx < 0 or v2_idx < 0:
                print(f"Warning: Multi-allelic ambiguity for interaction {idx} "
                      f"in sample {sample_idx}, trying next sample")
                continue

            sv = dataset.samples[sample_idx]
            n_variants = len(sv.variants)

            # For large samples, extract a chunk around the two target variants
            # to keep attention O(n²) manageable.  This is an approximation:
            # with global attention, excluded variants can influence the target
            # pair's embeddings.  In practice the effect is small because the
            # synergy formula is a second-order difference and most distant
            # context is shared across all 4 perturbation conditions.
            if n_variants > chunk_size:
                # Ensure both target indices stay inside the window
                lo = min(v1_idx, v2_idx)
                hi = max(v1_idx, v2_idx)
                span = hi - lo + 1

                if span > chunk_size:
                    # Target variants are further apart than chunk_size in the
                    # variant list — cannot fit both in one chunk, skip sample.
                    print(f"Warning: Interaction {idx} variants are {span} apart "
                          f"in sample {sample_idx} (chunk_size={chunk_size}), "
                          f"trying next sample")
                    continue

                # Centre the window around the pair
                pad = (chunk_size - span) // 2
                start = max(0, lo - pad)
                end = min(n_variants, start + chunk_size)
                start = max(0, end - chunk_size)  # adjust if we hit the right edge

                chunk_variants = sv.variants[start:end]
                chunk_sv = SampleVariants(
                    sample_id=sv.sample_id,
                    label=sv.label,
                    variants=chunk_variants,
                )
                chunk_tensor = build_variant_tensor(
                    chunk_sv, dataset.annotation_level,
                    dataset.gene_index, impute_value=dataset.impute_value,
                )
                # Remap target indices into the chunk
                chunk_v1_idx = v1_idx - start
                chunk_v2_idx = v2_idx - start
                was_chunked = True
                chunk_start_idx = start
                chunk_end_idx = end
            else:
                chunk_tensor = build_variant_tensor(
                    sv, dataset.annotation_level,
                    dataset.gene_index, impute_value=dataset.impute_value,
                )
                chunk_v1_idx = v1_idx
                chunk_v2_idx = v2_idx
                was_chunked = False
                chunk_start_idx = 0
                chunk_end_idx = n_variants

            try:
                validation = detector.validate_interaction_with_perturbation(
                    features=chunk_tensor['features'],
                    positions=chunk_tensor['positions'],
                    gene_ids=chunk_tensor['gene_ids'],
                    mask=chunk_tensor['mask'],
                    variant1_idx=chunk_v1_idx,
                    variant2_idx=chunk_v2_idx,
                )

                result = {
                    'sample_idx': sample_idx,
                    'variant1_pos': v1_pos,
                    'variant2_pos': v2_pos,
                    'variant1_gene': v1_gene,
                    'variant2_gene': v2_gene,
                    'same_gene': interaction.get('same_gene', False),
                    'was_chunked': was_chunked,
                    'chunk_start_idx': chunk_start_idx,
                    'chunk_end_idx': chunk_end_idx,
                    'n_variants_in_chunk': chunk_tensor['features'].shape[0],
                    **validation
                }

                validation_results.append(result)
                validated = True

                if len(validation_results) % 10 == 0 and len(validation_results) > last_reported:
                    last_reported = len(validation_results)
                    print(f"  Validated {len(validation_results)} interactions...")
                break

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"Warning: CUDA OOM for interaction {idx} on sample {sample_idx} "
                      f"({chunk_tensor['features'].shape[0]} variants in chunk), "
                      f"trying next sample")
                continue

            except Exception as e:
                print(f"Warning: Could not validate interaction {idx} on sample {sample_idx}: {e}")
                continue

        if not validated:
            print(f"Warning: No sample found carrying both variants for interaction {idx}, skipping")

    print(f"\nCompleted validation of {len(validation_results)} interactions")

    # Create results DataFrame
    if len(validation_results) == 0:
        print("\nERROR: No interactions were successfully validated.")
        print("\nPossible reasons:")
        print("  - No samples carry both variant positions for any interaction")
        print("  - All perturbation attempts failed")
        return

    results_df = pd.DataFrame(validation_results)

    # Filter for significant interactions
    significant = results_df[results_df['is_significant']]
    synergistic = significant[significant['interaction_type'] == 'synergistic']
    antagonistic = significant[significant['interaction_type'] == 'antagonistic']

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    print(f"\nTotal validated: {len(results_df)}")
    print(f"Significant (|synergy| > {args.synergy_threshold}): {len(significant)}")
    print(f"  Synergistic: {len(synergistic)}")
    print(f"  Antagonistic: {len(antagonistic)}")

    if len(synergistic) > 0:
        print(f"\nTop synergistic interactions:")
        print(synergistic.nlargest(5, 'synergy')[[
            'variant1_pos', 'variant2_pos', 'synergy', 'effect_combined'
        ]])

    if len(antagonistic) > 0:
        print(f"\nTop antagonistic interactions:")
        print(antagonistic.nsmallest(5, 'synergy')[[
            'variant1_pos', 'variant2_pos', 'synergy', 'effect_combined'
        ]])

    # Save results
    results_path = output_dir / 'epistasis_validation.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nValidation results saved to {results_path}")

    # Save significant only
    significant_path = output_dir / 'epistasis_significant.csv'
    significant.to_csv(significant_path, index=False)
    print(f"Significant interactions saved to {significant_path}")

    # Summary statistics
    summary = {
        'genome_build': args.genome_build,
        'total_interactions_tested': len(results_df),
        'n_significant': len(significant),
        'n_synergistic': len(synergistic),
        'n_antagonistic': len(antagonistic),
        'mean_synergy': float(results_df['synergy'].mean()),
        'max_synergy': float(results_df['synergy'].max()),
        'min_synergy': float(results_df['synergy'].min()),
    }

    summary_path = output_dir / 'epistasis_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

    print(f"Summary saved to {summary_path}")
    print("="*60)


if __name__ == '__main__':
    main()
