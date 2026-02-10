#!/usr/bin/env python3
"""
Create null baseline datasets with permuted labels for attribution calibration.

This script creates copies of preprocessed data with randomly shuffled labels,
breaking any real genotype-phenotype relationship. Models trained on this data
establish the null distribution of attributions.

Usage:
    # Single permutation
    python scripts/create_null_baseline.py \
        --input data/preprocessed.pt \
        --output data/preprocessed_NULL.pt \
        --seed 42

    # Multiple permutations for robust null distribution
    python scripts/create_null_baseline.py \
        --input data/preprocessed.pt \
        --output-dir data/null_permutations \
        --n-permutations 5

Author: Francesco Lescai
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def create_single_permutation(
    input_path: str,
    output_path: str,
    seed: int = 42
) -> dict:
    """
    Create a single permuted dataset.

    Parameters
    ----------
    input_path : str
        Path to original preprocessed data
    output_path : str
        Path to save permuted data
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Statistics about the permutation
    """
    print(f"Loading original data from {input_path}...")
    data = torch.load(input_path, weights_only=False)

    # Handle different data structures
    if 'labels' in data:
        labels = data['labels']
    elif 'samples' in data:
        # If samples are stored as list of objects
        labels = torch.tensor([s.label if hasattr(s, 'label') else s['label']
                               for s in data['samples']])
    else:
        raise ValueError("Cannot find labels in preprocessed data. "
                        f"Available keys: {list(data.keys())}")

    n_samples = len(labels)
    n_cases = (labels == 1).sum().item()
    n_controls = (labels == 0).sum().item()

    print(f"Original data: {n_samples} samples ({n_cases} cases, {n_controls} controls)")

    # Report all keys found in the data — everything except 'labels'
    # (or the label field inside 'samples') will be copied verbatim.
    all_keys = sorted(data.keys())
    print(f"Data keys ({len(all_keys)}): {all_keys}")

    # Permute labels using a local RNG to avoid mutating global NumPy state
    rng = np.random.default_rng(seed)
    permuted_indices = rng.permutation(n_samples)

    if isinstance(labels, torch.Tensor):
        permuted_labels = labels[permuted_indices].clone()
    else:
        permuted_labels = [labels[i] for i in permuted_indices]

    # Verify permutation changed positions
    if isinstance(labels, torch.Tensor):
        same_position = (labels == permuted_labels).sum().item()
    else:
        same_position = sum(1 for i, l in enumerate(labels) if l == permuted_labels[i])

    print(f"Labels in same position after permutation: {same_position}/{n_samples} "
          f"({100*same_position/n_samples:.1f}%)")

    # Create permuted dataset
    permuted_data = {}
    for key, value in data.items():
        if key == 'labels':
            permuted_data[key] = permuted_labels
        elif key == 'samples' and isinstance(value, list):
            # Need to update labels within sample objects
            permuted_samples = []
            for i, sample in enumerate(value):
                if hasattr(sample, '_replace'):  # namedtuple
                    permuted_samples.append(sample._replace(label=permuted_labels[i].item()))
                elif isinstance(sample, dict):
                    new_sample = sample.copy()
                    new_sample['label'] = permuted_labels[i].item() if isinstance(permuted_labels[i], torch.Tensor) else permuted_labels[i]
                    permuted_samples.append(new_sample)
                else:
                    # Try to set attribute directly
                    sample_copy = sample  # May need deep copy depending on structure
                    sample_copy.label = permuted_labels[i].item() if isinstance(permuted_labels[i], torch.Tensor) else permuted_labels[i]
                    permuted_samples.append(sample_copy)
            permuted_data[key] = permuted_samples
        else:
            permuted_data[key] = value

    # Add metadata
    permuted_data['_null_baseline_metadata'] = {
        'is_null_baseline': True,
        'permutation_seed': seed,
        'original_path': str(input_path),
        'n_samples': n_samples,
        'n_cases': n_cases,
        'n_controls': n_controls,
        'same_position_count': same_position,
    }

    # Verify all original keys are preserved (only labels should differ)
    preserved_keys = sorted(k for k in permuted_data.keys()
                            if k != '_null_baseline_metadata')
    original_keys = sorted(data.keys())
    if preserved_keys != original_keys:
        missing = set(original_keys) - set(preserved_keys)
        extra = set(preserved_keys) - set(original_keys)
        raise RuntimeError(
            f"Key mismatch after permutation! "
            f"Missing: {missing}, Extra: {extra}"
        )
    print(f"Verified: all {len(original_keys)} original keys preserved")

    # Report non-label fields that were copied verbatim
    for key in original_keys:
        if key in ('labels', 'samples'):
            continue
        value = permuted_data[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor {value.shape} {value.dtype} (unchanged)")
        elif isinstance(value, list):
            print(f"  {key}: list[{len(value)}] (unchanged)")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} entries (unchanged)")
        else:
            print(f"  {key}: {type(value).__name__} (unchanged)")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving permuted data to {output_path}...")
    torch.save(permuted_data, output_path)

    stats = {
        'n_samples': n_samples,
        'n_cases': n_cases,
        'n_controls': n_controls,
        'same_position': same_position,
        'seed': seed,
    }

    print("Done!")
    return stats


def create_multiple_permutations(
    input_path: str,
    output_dir: str,
    n_permutations: int = 5,
    base_seed: int = 42
) -> list:
    """
    Create multiple permuted datasets for robust null distribution.

    Parameters
    ----------
    input_path : str
        Path to original preprocessed data
    output_dir : str
        Directory to save permuted datasets
    n_permutations : int
        Number of permutations to create
    base_seed : int
        Base random seed (each permutation uses base_seed + i)

    Returns
    -------
    list
        List of statistics dicts for each permutation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {n_permutations} permuted datasets...")
    all_stats = []

    for i in range(n_permutations):
        seed = base_seed + i
        output_path = output_dir / f"preprocessed_NULL_perm{i}.pt"

        print(f"\n--- Permutation {i+1}/{n_permutations} (seed={seed}) ---")
        stats = create_single_permutation(input_path, str(output_path), seed)
        stats['permutation_index'] = i
        stats['output_path'] = str(output_path)
        all_stats.append(stats)

    # Save summary
    summary_path = output_dir / "permutation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Null Baseline Permutation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Original data: {input_path}\n")
        f.write(f"Number of permutations: {n_permutations}\n")
        f.write(f"Base seed: {base_seed}\n\n")
        for stats in all_stats:
            f.write(f"Permutation {stats['permutation_index']}: seed={stats['seed']}, "
                   f"same_position={stats['same_position']}/{stats['n_samples']}\n")

    print(f"\nSummary saved to {summary_path}")
    return all_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create null baseline datasets with permuted labels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', type=str, required=True,
                       help='Path to original preprocessed data (.pt file)')

    # Single permutation mode
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for single permuted dataset')

    # Multiple permutation mode
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for multiple permuted datasets')
    parser.add_argument('--n-permutations', type=int, default=5,
                       help='Number of permutations (only with --output-dir)')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (or base seed for multiple permutations)')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output and args.output_dir:
        raise ValueError("Specify either --output (single) or --output-dir (multiple), not both")

    if not args.output and not args.output_dir:
        # Default to single permutation with auto-named output
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_NULL{input_path.suffix}")

    if args.output:
        create_single_permutation(args.input, args.output, args.seed)
    else:
        create_multiple_permutations(args.input, args.output_dir,
                                    args.n_permutations, args.seed)


if __name__ == '__main__':
    main()
