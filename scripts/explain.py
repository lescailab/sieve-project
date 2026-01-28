#!/usr/bin/env python3
"""
Explainability analysis for trained SIEVE models.

Computes variant attributions using integrated gradients and analyzes
attention patterns to discover disease-associated variants and epistatic
interactions.

Usage:
    # Basic usage - analyze best model from experiment
    python scripts/explain.py \
        --experiment-dir outputs/L3_attr_medium \
        --preprocessed-data data/preprocessed.pt \
        --output-dir results/explainability

    # Analyze specific fold
    python scripts/explain.py \
        --checkpoint outputs/L3_attr_medium/fold_0/best_model.pt \
        --config outputs/L3_attr_medium/config.yaml \
        --preprocessed-data data/preprocessed.pt \
        --output-dir results/explainability_fold0

    # Only compute attributions (skip attention analysis)
    python scripts/explain.py \
        --experiment-dir outputs/L3_attr_medium \
        --preprocessed-data data/preprocessed.pt \
        --output-dir results/explainability \
        --skip-attention

Author: Lescai Lab
"""

import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.encoding import VariantDataset, collate_samples, get_feature_dimension, AnnotationLevel
from src.models.sieve import create_sieve_model
from src.explain.gradients import IntegratedGradientsExplainer
from src.explain.attention_analysis import AttentionAnalyzer
from src.explain.variant_ranking import VariantRanker


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run explainability analysis on trained SIEVE model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model input
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--experiment-dir', type=str,
                             help='Path to experiment directory (will use best fold)')
    model_group.add_argument('--checkpoint', type=str,
                             help='Path to specific model checkpoint')

    parser.add_argument('--config', type=str,
                        help='Path to config.yaml (required if using --checkpoint)')

    # Data input
    parser.add_argument('--preprocessed-data', type=str, required=True,
                        help='Path to preprocessed data (.pt file)')

    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')

    # Analysis options
    parser.add_argument('--n-steps', type=int, default=50,
                        help='Number of integration steps for IG')
    parser.add_argument('--max-variants', type=int, default=2000,
                        help='Maximum variants per sample for IG (to avoid OOM). Samples with more variants are randomly sampled.')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for dataloader (samples processed individually during IG to avoid OOM)')
    parser.add_argument('--skip-attention', action='store_true',
                        help='Skip attention analysis (faster)')
    parser.add_argument('--top-k-variants', type=int, default=100,
                        help='Number of top variants to extract')
    parser.add_argument('--top-k-interactions', type=int, default=100,
                        help='Number of top interactions to extract')
    parser.add_argument('--attention-threshold', type=float, default=0.1,
                        help='Minimum attention weight for interactions')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    return parser.parse_args()


def load_model_and_config(args):
    """Load model and configuration."""
    if args.experiment_dir:
        # Load from experiment directory
        exp_dir = Path(args.experiment_dir)

        # Load config
        config_path = exp_dir / 'config.yaml'
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Find best fold (highest AUC in CV results)
        cv_results_path = exp_dir / 'cv_results.yaml'
        if cv_results_path.exists():
            with open(cv_results_path) as f:
                cv_results = yaml.safe_load(f)

            # Find best fold
            best_fold = 0
            best_auc = 0
            for i, result in enumerate(cv_results['fold_results']):
                if result['auc'] > best_auc:
                    best_auc = result['auc']
                    best_fold = i

            checkpoint_path = exp_dir / f'fold_{best_fold}' / 'best_model.pt'
            print(f"Using fold {best_fold} (AUC: {best_auc:.4f})")
        else:
            # Single run - use best_model.pt directly
            checkpoint_path = exp_dir / 'best_model.pt'
            print("Using single run model")

    else:
        # Load specific checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not args.config:
            raise ValueError("--config required when using --checkpoint")

        config_path = Path(args.config)
        with open(config_path) as f:
            config = yaml.safe_load(f)

    print(f"Loading model from {checkpoint_path}")

    # Load checkpoint
    # Note: weights_only=False is safe here since these are our own trusted checkpoints
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    return config, checkpoint


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("SIEVE Explainability Analysis")
    print("="*60)

    # Load model and config
    config, checkpoint = load_model_and_config(args)

    # Load data
    print("\nLoading data...")
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

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples
    )


    # Create model (add input_dim if missing from config)
    print("\nCreating model...")
    if 'input_dim' not in config:
        config['input_dim'] = get_feature_dimension(annotation_level)

    model = create_sieve_model(config, num_genes=dataset.num_genes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # === INTEGRATED GRADIENTS ===
    print("\n" + "="*60)
    print("Computing Integrated Gradients Attributions")
    print("="*60)

    explainer = IntegratedGradientsExplainer(
        model=model,
        device=args.device,
        n_steps=args.n_steps,
        max_variants=args.max_variants
    )

    print(f"IG Configuration:")
    print(f"  Integration steps: {args.n_steps}")
    print(f"  Max variants per sample: {args.max_variants}")
    print(f"  Note: Samples with >{args.max_variants} variants will be randomly sampled")

    print(f"Computing attributions with {args.n_steps} integration steps...")
    attributions, variant_scores, metadata = explainer.attribute_batch(
        dataloader=dataloader,
        aggregate='l2'
    )

    print(f"Computed attributions for {len(attributions)} samples")

    # Save raw attributions
    attributions_path = output_dir / 'attributions.npz'
    np.savez(
        attributions_path,
        attributions=np.array(attributions, dtype=object),
        variant_scores=np.array(variant_scores, dtype=object),
        metadata=np.array(metadata, dtype=object)
    )
    print(f"Saved attributions to {attributions_path}")

    # === BUILD VARIANT INFO MAP ===
    # Map (position, gene_id) -> {chromosome, gene_name} for annotation
    print("\nBuilding variant info map...")

    # Build gene index (same as in encoding)
    gene_symbols = sorted(set(v.gene for s in all_samples for v in s.variants))
    gene_index = {gene: idx for idx, gene in enumerate(gene_symbols)}

    variant_info_map = {}
    for sample in all_samples:
        for variant in sample.variants:
            pos = variant.pos
            gene_symbol = variant.gene
            chrom = variant.chrom
            gene_id = gene_index[gene_symbol]

            key = (pos, gene_id)
            if key not in variant_info_map:
                variant_info_map[key] = {
                    'chromosome': chrom,
                    'gene_name': gene_symbol
                }

    print(f"Mapped {len(variant_info_map)} unique (position, gene_id) combinations")

    # === VARIANT RANKING ===
    print("\n" + "="*60)
    print("Ranking Variants")
    print("="*60)

    ranker = VariantRanker(aggregation='rank_average', variant_info_map=variant_info_map)

    # Separate cases and controls
    case_indices = [i for i, m in enumerate(metadata) if m.get('label') == 1]
    control_indices = [i for i, m in enumerate(metadata) if m.get('label') == 0]

    print(f"Cases: {len(case_indices)}, Controls: {len(control_indices)}")

    # Rank variants
    variant_rankings = ranker.rank_variants(
        attributions=variant_scores,
        metadata=metadata,
        case_indices=case_indices,
        control_indices=control_indices
    )

    print(f"Ranked {len(variant_rankings)} unique variants")

    # Rank genes
    gene_rankings = ranker.rank_genes(
        variant_rankings=variant_rankings,
        aggregation='max'
    )

    print(f"Ranked {len(gene_rankings)} genes")

    # Get case-enriched variants
    if case_indices and control_indices:
        try:
            case_enriched = ranker.get_case_enriched_variants(
                variant_rankings=variant_rankings,
                min_case_samples=min(5, len(case_indices) // 4),
                min_diff=0.05,
                top_k=args.top_k_variants
            )
            print(f"Identified {len(case_enriched)} case-enriched variants")
        except (ValueError, KeyError):
            print("Not enough data for case-enriched analysis")
            case_enriched = None
    else:
        case_enriched = None

    # Export rankings
    ranker.export_rankings(
        variant_rankings=variant_rankings,
        gene_rankings=gene_rankings,
        output_dir=str(output_dir),
        prefix='sieve'
    )

    # === ATTENTION ANALYSIS ===
    if not args.skip_attention:
        print("\n" + "="*60)
        print("Analyzing Attention Patterns")
        print("="*60)

        analyzer = AttentionAnalyzer(
            model=model,
            device=args.device,
            attention_threshold=args.attention_threshold
        )

        all_interactions = []

        print("Extracting attention weights...")
        for batch_idx, batch in enumerate(dataloader):
            # Extract attention
            attention_weights = analyzer.extract_attention_weights(
                variant_features=batch['features'],
                positions=batch['positions'],
                gene_ids=batch['gene_ids'],
                mask=batch['mask']
            )

            # Find interactions
            interactions = analyzer.find_top_interactions(
                attention_weights=attention_weights,
                positions=batch['positions'],
                gene_ids=batch['gene_ids'],
                mask=batch['mask'],
                top_k=args.top_k_interactions,
                aggregate_layers='mean',
                aggregate_heads='mean'
            )

            all_interactions.extend(interactions)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * args.batch_size} samples")

        print(f"Extracted {len(all_interactions)} interactions")

        # Aggregate across samples
        aggregated_interactions = analyzer.aggregate_interactions_across_samples(
            all_sample_interactions=[all_interactions],
            min_samples=2
        )

        print(f"Found {len(aggregated_interactions)} recurring interactions")

        # Save interactions
        import pandas as pd
        interactions_df = pd.DataFrame(aggregated_interactions)
        interactions_path = output_dir / 'sieve_interactions.csv'
        interactions_df.to_csv(interactions_path, index=False)
        print(f"Saved interactions to {interactions_path}")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    print(f"\nTop 10 Variants by Attribution:")
    print(variant_rankings.head(10)[['position', 'gene_id', 'mean_attribution', 'num_samples']])

    print(f"\nTop 10 Genes:")
    print(gene_rankings.head(10)[['gene_id', 'num_variants', 'gene_score', 'top_variant_pos']])

    if case_enriched is not None and len(case_enriched) > 0:
        print(f"\nTop 10 Case-Enriched Variants:")
        print(case_enriched.head(10)[[
            'position', 'gene_id', 'case_attribution', 'control_attribution', 'case_control_diff'
        ]])

    print(f"\nResults saved to {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
