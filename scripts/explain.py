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

Notes:
    Attribution magnitudes (mean_attribution, score) are model-specific and
    not directly comparable across annotation levels or model architectures.
    For cross-level ablation comparison use rank-based metrics (Jaccard on
    top-K sets) rather than raw score differences. See KNOWN_LIMITATIONS.md
    for details on the empirical p-value resolution floor and cross-level
    scale incomparability.

Author: Francesco Lescai
"""

import argparse
import gc
import shutil
from collections import Counter
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.encoding import (
    ChunkedVariantDataset,
    collate_chunks,
    get_feature_dimension,
    AnnotationLevel
)
from src.models.sieve import create_sieve_model
from src.models import ChunkedSIEVEModel
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
    parser.add_argument('--skip-ig', action='store_true',
                        help='Skip Integrated Gradients computation (use if you only need attention analysis)')
    parser.add_argument('--top-k-variants', type=int, default=100,
                        help='Number of top variants to extract')
    parser.add_argument('--top-k-interactions', type=int, default=100,
                        help='Number of top interactions to extract')
    parser.add_argument('--attention-threshold', type=float, default=0.1,
                        help='Minimum attention weight for interactions')
    parser.add_argument('--attention-threshold-mode', type=str, default='absolute',
                        choices=['absolute', 'percentile'],
                        help='How to threshold pairwise attention scores')
    parser.add_argument('--attention-percentile', type=float, default=99.9,
                        help='Percentile cutoff for attention interactions when using percentile mode')
    parser.add_argument('--aggregation-method', type=str, default='mean',
                        choices=['mean', 'max', 'rank_average'],
                        help=(
                            'How to aggregate per-sample variant scores into a '
                            'population-level ranking score. '
                            "'mean': score == mean_attribution (default, most transparent). "
                            "'max': score == max_attribution. "
                            "'rank_average': composite rank across mean, max, and sample count."
                        ))
    parser.add_argument('--is-null-baseline', action='store_true',
                        help='Flag indicating this is a null baseline analysis (for metadata)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    # Genome build
    parser.add_argument('--genome-build', type=str, default='GRCh37',
                        help='Reference genome build (GRCh37 or GRCh38)')

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

    # Create CHUNKED dataset for whole-genome coverage
    print("\nCreating CHUNKED dataset for FULL GENOME explainability...")
    chunk_size = min(args.max_variants, 2000)  # Smaller chunks for IG (memory-intensive)
    print(f"  Chunk size: {chunk_size}")
    print(f"  This ensures ALL chromosomes are analyzed, not just chr1/chr2!")

    dataset = ChunkedVariantDataset(
        samples=all_samples,
        annotation_level=annotation_level,
        chunk_size=chunk_size,
        overlap=0
    )

    # Create model (add input_dim if missing from config)
    print("\nCreating model...")
    if 'input_dim' not in config:
        config['input_dim'] = get_feature_dimension(annotation_level)

    # Load base model
    base_model = create_sieve_model(config, num_genes=dataset.num_genes)

    # Check if checkpoint has chunked model or base model
    state_dict = checkpoint['model_state_dict']

    # Try to detect if this is a chunked model checkpoint
    if any(k.startswith('base_model.') for k in state_dict.keys()):
        # Checkpoint is from chunked model - need to wrap base model
        model = ChunkedSIEVEModel(
            base_model=base_model,
            aggregation_method=config.get('aggregation_method', 'mean')
        )
        model.load_state_dict(state_dict)
    else:
        # Checkpoint is from base model only - just use base model for IG
        # (IG works on individual chunks, doesn't need aggregation)
        model = base_model
        model.load_state_dict(state_dict)

    model = model.to(args.device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # For IG, we need the base model (not wrapped)
    if isinstance(model, ChunkedSIEVEModel):
        ig_model = model.base_model
        print("  Using base model for Integrated Gradients (chunk-level attributions)")
    else:
        ig_model = model
        print("  Using model directly for Integrated Gradients")

    # === INTEGRATED GRADIENTS (CHUNKED) ===
    if args.skip_ig:
        print("\n" + "="*60)
        print("Skipping Integrated Gradients (--skip-ig specified)")
        print("="*60)
        variant_rankings = None
        gene_rankings = None
        case_enriched = None
    else:
        print("\n" + "="*60)
        print("Computing Integrated Gradients Attributions (CHUNKED)")
        print("="*60)

        explainer = IntegratedGradientsExplainer(
            model=ig_model,
            device=args.device,
            n_steps=args.n_steps,
            max_variants=chunk_size  # Process full chunks (no truncation within chunks)
        )

        print(f"IG Configuration:")
        print(f"  Integration steps: {args.n_steps}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Processing ALL chunks per sample for FULL GENOME coverage")

        # === BUILD VARIANT INFO MAP (before IG loop, needed for ranker) ===
        # Map (chrom, position, gene_id) -> {gene_name} for annotation
        # CRITICAL: Include chromosome in key to prevent position collisions!
        # Same position number can exist on different chromosomes.
        print("\nBuilding variant info map...")

        # Use the dataset's gene_index (not a new one!)
        gene_index = dataset.gene_index

        variant_info_map = {}
        for sample in all_samples:
            for variant in sample.variants:
                pos = variant.pos
                gene_symbol = variant.gene
                chrom = variant.chrom

                # Skip genes not in dataset's gene_index (shouldn't happen but be safe)
                if gene_symbol not in gene_index:
                    print(f"WARNING: Gene {gene_symbol} not in dataset gene_index!")
                    continue

                gene_id = gene_index[gene_symbol]

                # FIXED: Include chromosome in key to prevent collisions
                key = (chrom, pos, gene_id)
                if key not in variant_info_map:
                    variant_info_map[key] = {
                        'gene_name': gene_symbol
                    }

        print(f"Mapped {len(variant_info_map)} unique (chrom, position, gene_id) combinations")

        # Diagnostic: Check chromosome distribution in variant_info_map
        # Chromosome is now part of the KEY (chrom, pos, gene_id), not the value
        chrom_counts = {}
        for (chrom, pos, gene_id) in variant_info_map.keys():
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1

        print(f"Variant info map chromosome distribution:")
        for chrom in sorted(chrom_counts.keys(), key=lambda x: (x.isdigit() and int(x) or 999, x))[:10]:
            print(f"  Chr {chrom}: {chrom_counts[chrom]} unique variants")
        if len(chrom_counts) > 10:
            print(f"  ... and {len(chrom_counts) - 10} more chromosomes")

        # === PREPARE INCREMENTAL PROCESSING ===
        # Create ranker upfront for incremental sample accumulation
        ranker = VariantRanker(aggregation=args.aggregation_method, variant_info_map=variant_info_map)

        # Determine case/control sets upfront
        case_indices = set(i for i in range(len(all_samples)) if all_samples[i].label == 1)
        control_indices = set(i for i in range(len(all_samples)) if all_samples[i].label == 0)
        print(f"Cases: {len(case_indices)}, Controls: {len(control_indices)}")

        # Temp directory for incremental attribution saving (avoids holding all in RAM)
        tmp_dir = output_dir / '_tmp_attributions'
        tmp_dir.mkdir(exist_ok=True)

        # Lightweight metadata list (small 1D arrays + scalars per sample)
        all_metadata = []

        num_samples = len(all_samples)
        metadata_variant_count = 0
        print(f"\nProcessing {num_samples} samples (chunk-by-chunk)...")

        for sample_idx in range(num_samples):
            if (sample_idx + 1) % 10 == 0 or (sample_idx + 1) == num_samples:
                print(f"  Sample {sample_idx + 1}/{num_samples}...")

            # Get all chunks for this sample
            chunk_indices = dataset.get_chunks_for_sample(sample_idx)

            # Process each chunk
            chunk_attributions = []
            chunk_positions = []
            chunk_gene_ids = []
            chunk_chromosomes = []

            for chunk_idx in chunk_indices:
                chunk = dataset[chunk_idx]

                # Get chunk info to map back to original variants
                chunk_info = dataset.chunk_info[chunk_idx]
                start_idx = chunk_info['start_idx']
                end_idx = chunk_info['end_idx']
                original_variants = all_samples[sample_idx].variants[start_idx:end_idx]

                # Move to device
                features = chunk['features'].unsqueeze(0).to(args.device)
                positions = chunk['positions'].unsqueeze(0).to(args.device)
                gene_ids = chunk['gene_ids'].unsqueeze(0).to(args.device)
                mask = chunk['mask'].unsqueeze(0).to(args.device)

                # Compute attributions for this chunk
                attr = explainer.attribute(features, positions, gene_ids, mask)

                # Extract valid variants (non-padded) to CPU numpy immediately
                valid_mask = mask[0].cpu().numpy()
                attr_valid = attr[0][valid_mask].cpu().numpy()

                # Get chromosomes from original variants (matching valid positions)
                valid_chroms = np.array([v.chrom for v in original_variants])[valid_mask]

                chunk_attributions.append(attr_valid)
                chunk_positions.append(positions[0][valid_mask].cpu().numpy())
                chunk_gene_ids.append(gene_ids[0][valid_mask].cpu().numpy())
                chunk_chromosomes.append(valid_chroms)

                # Free GPU tensors immediately after extracting to CPU
                del features, positions, gene_ids, mask, attr

            # Combine all chunks for this sample
            sample_attributions = np.concatenate(chunk_attributions, axis=0)
            sample_positions = np.concatenate(chunk_positions, axis=0)
            sample_gene_ids = np.concatenate(chunk_gene_ids, axis=0)
            sample_chromosomes = np.concatenate(chunk_chromosomes, axis=0)

            # Aggregate to variant scores (L2 norm across features)
            if sample_attributions.ndim > 1:
                sample_variant_scores = np.linalg.norm(sample_attributions, ord=2, axis=1)
            else:
                sample_variant_scores = np.abs(sample_attributions)

            # Save this sample's full data to disk immediately (freed from RAM after)
            np.savez(
                tmp_dir / f'sample_{sample_idx}.npz',
                attributions=sample_attributions,
                variant_scores=sample_variant_scores,
            )

            # Feed scores into ranker incrementally (then discard per-sample arrays)
            is_case = True if sample_idx in case_indices else (
                False if sample_idx in control_indices else None
            )
            ranker.accumulate_sample(
                variant_scores=sample_variant_scores,
                positions=sample_positions,
                gene_ids=sample_gene_ids,
                chromosomes=sample_chromosomes,
                sample_idx=sample_idx,
                is_case=is_case,
            )

            # Keep only lightweight metadata (small 1D arrays + scalars)
            sample_meta = {
                'positions': sample_positions,
                'gene_ids': sample_gene_ids,
                'chromosomes': sample_chromosomes,
                'sample_idx': sample_idx,
                'sample_id': all_samples[sample_idx].sample_id,
                'label': all_samples[sample_idx].label
            }
            all_metadata.append(sample_meta)
            metadata_variant_count += len(sample_positions)

            # Free per-sample arrays (full attributions + scores now on disk)
            del sample_attributions, sample_variant_scores
            del chunk_attributions, chunk_positions, chunk_gene_ids, chunk_chromosomes

            # Periodic garbage collection to reclaim Python overhead
            if (sample_idx + 1) % 50 == 0:
                gc.collect()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()

        print(f"\nComputed attributions for {num_samples} samples")
        print(f"CRITICAL: All chunks processed - FULL GENOME coverage achieved!")

        # Diagnostic: Check what variants are in the metadata
        print(f"\nDiagnostic: Checking metadata variant distribution...")
        print(f"  Total variants in metadata across all samples: {metadata_variant_count:,}")

        # Check chromosome distribution in original data
        chrom_dist = Counter()
        for sample in all_samples:
            for variant in sample.variants:
                chrom_dist[variant.chrom] += 1

        print(f"  Chromosomes in original data: {len(chrom_dist)}")
        for chrom in sorted(chrom_dist.keys(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 999, x))[:10]:
            print(f"    Chr {chrom}: {chrom_dist[chrom]:,} variants")
        if len(chrom_dist) > 10:
            print(f"    ... and {len(chrom_dist) - 10} more chromosomes")

        # Check a sample of positions and genes from metadata
        if len(all_metadata) > 0 and len(all_metadata[0]['positions']) > 0:
            sample_meta = all_metadata[0]
            print(f"\n  First sample has {len(sample_meta['positions']):,} variants in attributions")
            print(f"  First 5 positions: {sample_meta['positions'][:5].tolist()}")
            print(f"  First 5 gene_ids: {sample_meta['gene_ids'][:5].tolist()}")

        # Recombine only lightweight variant_scores + metadata into attributions.npz
        # Raw per-feature attributions stay in per-sample files (too large to fit in RAM together)
        print("\nSaving variant scores and metadata...")
        all_variant_scores = []
        for sidx in range(num_samples):
            with np.load(tmp_dir / f'sample_{sidx}.npz', allow_pickle=False) as data:
                all_variant_scores.append(data['variant_scores'])

        attributions_path = output_dir / 'attributions.npz'
        np.savez(
            attributions_path,
            variant_scores=np.array(all_variant_scores, dtype=object),
            metadata=np.array(all_metadata, dtype=object),
        )
        print(f"Saved variant scores + metadata to {attributions_path}")

        del all_variant_scores, all_metadata
        gc.collect()

        # Promote temp dir to permanent per-sample output (rename, no copy)
        per_sample_dir = output_dir / 'attributions_per_sample'
        if per_sample_dir.exists():
            shutil.rmtree(per_sample_dir)
        tmp_dir.rename(per_sample_dir)
        print(f"Per-sample raw attributions preserved in {per_sample_dir}/")
        print(f"  {num_samples} files, each containing 'attributions' and 'variant_scores' arrays")

        # === VARIANT RANKING (from incremental accumulation) ===
        print("\n" + "="*60)
        print("Ranking Variants")
        print("="*60)

        variant_rankings = ranker.finalize_rankings()

        print(f"Ranked {len(variant_rankings)} unique variants")

        # Diagnostic: Check chromosome distribution in variant rankings
        if 'chromosome' in variant_rankings.columns:
            ranking_chrom_counts = variant_rankings['chromosome'].value_counts()
            print(f"\nDiagnostic: Variant rankings chromosome distribution:")
            print(f"  Unique chromosomes: {len(ranking_chrom_counts)}")
            for chrom in sorted(ranking_chrom_counts.index, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 999, x))[:10]:
                print(f"  Chr {chrom}: {ranking_chrom_counts[chrom]} variants")
            if len(ranking_chrom_counts) > 10:
                print(f"  ... and {len(ranking_chrom_counts) - 10} more chromosomes")

            if len(ranking_chrom_counts) == 1:
                print(f"  ❌ CRITICAL: Only 1 chromosome in rankings! This is the bug we're looking for.")
        else:
            print("  ⚠️ WARNING: No chromosome column in variant rankings!")

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

        # Create dataloader for attention analysis
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_chunks,
            num_workers=0
        )
        total_chunks = len(dataset)
        num_samples = len(all_samples)
        print(f"Created dataloader with {len(dataloader)} batches ({total_chunks} chunks from {num_samples} samples)")

        analyzer = AttentionAnalyzer(
            model=model,
            device=args.device,
            attention_threshold=args.attention_threshold,
            threshold_mode=args.attention_threshold_mode,
            attention_percentile=args.attention_percentile,
        )

        all_interactions = []
        interactions_by_sample = {}

        print("Extracting attention weights...")
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            features = batch['features'].to(args.device)
            positions = batch['positions'].to(args.device)
            gene_ids = batch['gene_ids'].to(args.device)
            mask = batch['mask'].to(args.device)

            # Extract attention
            attention_weights = analyzer.extract_attention_weights(
                variant_features=features,
                positions=positions,
                gene_ids=gene_ids,
                mask=mask
            )

            # Find interactions
            interactions = analyzer.find_top_interactions(
                attention_weights=attention_weights,
                positions=positions,
                gene_ids=gene_ids,
                mask=mask,
                top_k=args.top_k_interactions,
                aggregate_layers='mean',
                aggregate_heads='mean',
                sample_indices=batch['original_sample_indices'],
                chunk_indices=batch['chunk_indices'],
            )

            all_interactions.extend(interactions)
            for interaction in interactions:
                interactions_by_sample.setdefault(interaction['sample_idx'], []).append(interaction)

            # Free GPU tensors after each batch
            del features, positions, gene_ids, mask, attention_weights
            if args.device == 'cuda':
                torch.cuda.empty_cache()

            if (batch_idx + 1) % 10 == 0:
                chunks_done = min((batch_idx + 1) * args.batch_size, total_chunks)
                print(f"  Processed {chunks_done}/{total_chunks} chunks ({num_samples} samples)")

        print(f"Extracted {len(all_interactions)} interactions")

        # Aggregate across samples
        aggregated_interactions = analyzer.aggregate_interactions_across_samples(
            all_sample_interactions=list(interactions_by_sample.values()),
            min_samples=2
        )

        print(f"Found {len(aggregated_interactions)} recurring interactions")

        # Save interactions
        import pandas as pd
        interactions_df = pd.DataFrame(aggregated_interactions)
        interactions_path = output_dir / 'sieve_interactions.csv'
        interactions_df.to_csv(interactions_path, index=False)
        print(f"Saved interactions to {interactions_path}")

        # Free attention analysis data
        del all_interactions, interactions_by_sample, aggregated_interactions
        gc.collect()

    # === SAVE ANALYSIS METADATA ===
    analysis_metadata = {
        'is_null_baseline': args.is_null_baseline,
        'experiment_dir': str(args.experiment_dir) if args.experiment_dir else str(args.checkpoint),
        'genome_build': args.genome_build,
        'n_samples': len(all_samples),
        'annotation_level': config['level'],
        'n_integration_steps': args.n_steps,
        'max_variants_per_sample': args.max_variants,
        'aggregation_method': args.aggregation_method,
        'skip_attention': args.skip_attention,
        'skip_ig': args.skip_ig,
        'attention_threshold_mode': args.attention_threshold_mode,
        'attention_threshold': args.attention_threshold,
        'attention_percentile': args.attention_percentile,
    }

    if variant_rankings is not None:
        analysis_metadata['n_ranked_variants'] = len(variant_rankings)
        analysis_metadata['n_ranked_genes'] = len(gene_rankings)

    metadata_path = output_dir / 'analysis_metadata.yaml'
    with open(metadata_path, 'w') as f:
        yaml.dump(analysis_metadata, f, default_flow_style=False, sort_keys=False)
    print(f"Analysis metadata saved to {metadata_path}")

    # === SUMMARY ===
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    if variant_rankings is not None:
        print(f"\nTop 10 Variants by Attribution:")
        print(variant_rankings.head(10)[['position', 'gene_id', 'mean_attribution', 'num_samples']])

        print(f"\nTop 10 Genes:")
        print(gene_rankings.head(10)[['gene_id', 'num_variants', 'gene_score', 'top_variant_pos']])

        if case_enriched is not None and len(case_enriched) > 0:
            print(f"\nTop 10 Case-Enriched Variants:")
            print(case_enriched.head(10)[[
                'position', 'gene_id', 'case_attribution', 'control_attribution', 'case_control_diff'
            ]])
    else:
        print("\n(Integrated Gradients skipped - no variant rankings to display)")

    print(f"\nResults saved to {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
