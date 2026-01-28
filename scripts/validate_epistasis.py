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

Author: Lescai Lab
"""

import argparse
from pathlib import Path
import yaml
import torch
import pandas as pd

from src.encoding import VariantDataset, get_feature_dimension, AnnotationLevel
from src.models.sieve import create_sieve_model
from src.explain.shap_epistasis import SHAPEpistasisDetector


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

    model = create_sieve_model(config, num_genes=dataset.num_genes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    # Initialize detector
    detector = SHAPEpistasisDetector(model=model, device=args.device)

    # Validate interactions
    print("\n" + "="*60)
    print("Validating Interactions")
    print("="*60)

    validation_results = []

    for idx, interaction in top_interactions.iterrows():
        # Get sample and variant indices
        sample_idx = int(interaction.get('sample_idx', 0))

        # This is tricky - we need to map back to the actual variant indices
        # For now, we'll use the positions to find variants
        # In practice, you'd store variant indices in the interactions

        # Skip if we don't have the required fields
        if 'variant1_idx' not in interaction or 'variant2_idx' not in interaction:
            print(f"Warning: Interaction {idx} missing variant indices, skipping")
            continue

        v1_idx = int(interaction['variant1_idx'])
        v2_idx = int(interaction['variant2_idx'])

        # Get sample data
        sample = dataset[sample_idx]

        # Validate interaction
        try:
            validation = detector.validate_interaction_with_perturbation(
                features=sample['features'],
                positions=sample['positions'],
                gene_ids=sample['gene_ids'],
                mask=sample['mask'],
                variant1_idx=v1_idx,
                variant2_idx=v2_idx
            )

            # Add to results
            result = {
                'sample_idx': sample_idx,
                'variant1_pos': int(interaction['variant1_pos']),
                'variant2_pos': int(interaction['variant2_pos']),
                'variant1_gene': int(interaction['variant1_gene']),
                'variant2_gene': int(interaction['variant2_gene']),
                'same_gene': interaction.get('same_gene', False),
                **validation
            }

            validation_results.append(result)

            # Print progress
            if (len(validation_results)) % 10 == 0:
                print(f"  Validated {len(validation_results)} interactions...")

        except Exception as e:
            print(f"Warning: Could not validate interaction {idx}: {e}")
            continue

    print(f"\nCompleted validation of {len(validation_results)} interactions")

    # Create results DataFrame
    if len(validation_results) == 0:
        print("\nERROR: No interactions were successfully validated.")
        print("\nPossible reasons:")
        print("  - Interaction file missing required columns: variant1_idx, variant2_idx")
        print("  - No variant indices matched the dataset")
        print("  - All samples had too few variants to validate interactions")
        print("\nNote: Interactions file must include variant indices, not just positions.")
        print("These are typically generated by explain.py attention analysis.")
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
