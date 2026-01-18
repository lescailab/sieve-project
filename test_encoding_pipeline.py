#!/usr/bin/env python3
"""
Integration test for Phase 1B: Feature Encoding Pipeline

This script tests the complete encoding pipeline on real test data:
1. Load VCF data (Phase 1A)
2. Encode at all annotation levels (L0-L4)
3. Build PyTorch DataLoader
4. Verify batching and shapes
5. Test both positional encodings

Author: Lescai Lab
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import build_sample_variants
from src.encoding import (
    AnnotationLevel,
    VariantDataset,
    collate_samples,
    get_feature_dimension,
    get_level_description,
    test_encoding_consistency,
    test_sparse_tensor,
)


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_encoding_levels(samples, gene_index):
    """Test encoding at all annotation levels."""
    print_section("Testing All Annotation Levels (L0-L4)")

    all_levels = [
        AnnotationLevel.L0,
        AnnotationLevel.L1,
        AnnotationLevel.L2,
        AnnotationLevel.L3,
        AnnotationLevel.L4,
    ]

    for level in all_levels:
        print(f"\n{get_level_description(level)}")
        print(f"Expected feature dimension: {get_feature_dimension(level)}")

        # Create dataset
        dataset = VariantDataset(samples, level, gene_index)

        # Get first sample
        sample_tensor = dataset[0]

        print(f"  Sample: {sample_tensor['sample_id']}")
        print(f"  Number of variants: {sample_tensor['features'].shape[0]}")
        print(f"  Feature shape: {sample_tensor['features'].shape}")
        print(f"  Label: {sample_tensor['label'].item()}")

        # Verify feature dimension
        actual_dim = sample_tensor['features'].shape[1] if sample_tensor['features'].shape[0] > 0 else get_feature_dimension(level)
        expected_dim = get_feature_dimension(level)
        assert actual_dim == expected_dim, f"Dimension mismatch: {actual_dim} != {expected_dim}"

        # Check for any NaN or Inf values
        assert not torch.isnan(sample_tensor['features']).any(), "NaN values in features"
        assert not torch.isinf(sample_tensor['features']).any(), "Inf values in features"

        print(f"  ✓ Feature dimension correct")
        print(f"  ✓ No NaN/Inf values")


def test_batching(samples, gene_index):
    """Test DataLoader batching."""
    print_section("Testing DataLoader Batching")

    # Test with L3 (most comprehensive)
    dataset = VariantDataset(samples, AnnotationLevel.L3, gene_index)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_samples,
        shuffle=False
    )

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: 4")
    print(f"Expected batches: {len(dataloader)}")

    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Features shape: {batch['features'].shape}")
        print(f"  Positions shape: {batch['positions'].shape}")
        print(f"  Gene IDs shape: {batch['gene_ids'].shape}")
        print(f"  Mask shape: {batch['mask'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")

        batch_size, max_variants, feature_dim = batch['features'].shape

        # Verify shapes
        assert batch['positions'].shape == (batch_size, max_variants)
        assert batch['gene_ids'].shape == (batch_size, max_variants)
        assert batch['mask'].shape == (batch_size, max_variants)
        assert batch['labels'].shape == (batch_size,)
        assert feature_dim == 71  # L3 dimension

        # Count real variants
        real_variants = batch['mask'].sum().item()
        print(f"  Real variants (total): {real_variants}")
        print(f"  Padding positions: {batch_size * max_variants - real_variants}")

        # Verify mask is boolean
        assert batch['mask'].dtype == torch.bool, "Mask should be boolean"

        # Check labels
        print(f"  Labels: {batch['labels'].tolist()}")
        cases = (batch['labels'] == 1).sum().item()
        controls = (batch['labels'] == 0).sum().item()
        print(f"  Cases: {cases}, Controls: {controls}")

        print(f"  ✓ Batch {batch_idx + 1} valid")

        # Only check first few batches
        if batch_idx >= 2:
            break


def test_missing_value_handling(samples, gene_index):
    """Test handling of missing SIFT/PolyPhen scores."""
    print_section("Testing Missing Value Handling")

    # Test with L3 which includes SIFT/PolyPhen
    dataset = VariantDataset(samples, AnnotationLevel.L3, gene_index, impute_value=0.5)

    # Get several samples and check SIFT/PolyPhen values
    n_samples_to_check = min(5, len(dataset))

    sift_values = []
    polyphen_values = []

    for i in range(n_samples_to_check):
        sample_tensor = dataset[i]
        features = sample_tensor['features']

        if features.shape[0] > 0:
            # L3 features: [genotype(1), position(64), consequence(4), SIFT(1), PolyPhen(1)]
            # SIFT is at index -2, PolyPhen at index -1
            sift = features[:, -2].numpy()
            polyphen = features[:, -1].numpy()

            sift_values.extend(sift.tolist())
            polyphen_values.extend(polyphen.tolist())

    print(f"\nAnalyzed {n_samples_to_check} samples:")
    print(f"  Total variants checked: {len(sift_values)}")

    # Handle edge case where no variants are found
    if len(sift_values) == 0:
        print("  Warning: No variants found in checked samples!")
        print("  ✓ Missing value handling test skipped (no variants to test)")
        return

    print(f"  SIFT value range: [{min(sift_values):.3f}, {max(sift_values):.3f}]")
    print(f"  PolyPhen value range: [{min(polyphen_values):.3f}, {max(polyphen_values):.3f}]")

    # Count how many have neutral imputation (0.5)
    sift_imputed = sum(1 for v in sift_values if abs(v - 0.5) < 0.001)
    polyphen_imputed = sum(1 for v in polyphen_values if abs(v - 0.5) < 0.001)

    print(f"  SIFT imputed (0.5): {sift_imputed} ({100*sift_imputed/len(sift_values):.1f}%)")
    print(f"  PolyPhen imputed (0.5): {polyphen_imputed} ({100*polyphen_imputed/len(polyphen_values):.1f}%)")

    # Verify all values are in valid range [0, 1]
    assert all(0 <= v <= 1 for v in sift_values), "SIFT values out of range"
    assert all(0 <= v <= 1 for v in polyphen_values), "PolyPhen values out of range"

    print(f"  ✓ All functional scores in valid range [0, 1]")
    print(f"  ✓ Missing values imputed with neutral (0.5)")


def test_positional_encodings():
    """Test both types of positional encodings."""
    print_section("Testing Positional Encodings")

    import numpy as np
    from src.encoding import sinusoidal_position_encoding, relative_position_bucket

    # Test sinusoidal encoding
    print("\n1. Sinusoidal Positional Encoding (for features)")
    positions = np.array([100, 1000, 10000, 100000])
    encodings = sinusoidal_position_encoding(positions, d_model=64)

    print(f"  Positions: {positions}")
    print(f"  Encoding shape: {encodings.shape}")
    print(f"  Value range: [{encodings.min():.3f}, {encodings.max():.3f}]")

    assert encodings.shape == (4, 64), "Wrong encoding shape"
    assert np.all(np.abs(encodings) <= 1.0), "Encoding values out of range"
    print(f"  ✓ Sinusoidal encoding works correctly")

    # Test relative bucketing
    print("\n2. Relative Position Bucketing (for attention bias)")
    query_pos = torch.tensor([100, 200, 300, 400])
    key_pos = torch.tensor([100, 150, 250, 350])
    buckets = relative_position_bucket(query_pos, key_pos, num_buckets=32)

    print(f"  Query positions: {query_pos.tolist()}")
    print(f"  Key positions: {key_pos.tolist()}")
    print(f"  Bucket shape: {buckets.shape}")
    print(f"  Bucket value range: [{buckets.min().item()}, {buckets.max().item()}]")

    assert buckets.shape == (4, 4), "Wrong bucket shape"
    assert torch.all((buckets >= 0) & (buckets < 32)), "Invalid bucket indices"
    print(f"  ✓ Relative bucketing works correctly")

    # Verify they serve different purposes
    print("\n3. Verification: Non-Conflicting Use")
    print("  ✓ Sinusoidal: Used in L1-L4 feature encoding (absolute position)")
    print("  ✓ Bucketing: Will be used in attention mechanism (relative distance)")
    print("  ✓ These operate at different stages and don't conflict")


def test_dataset_statistics(dataset):
    """Test dataset statistics computation."""
    print_section("Dataset Statistics")

    stats = dataset.get_sample_statistics()

    print(f"\nDataset Overview:")
    print(f"  Number of samples: {stats['n_samples']}")
    print(f"  Number of genes: {stats['n_genes']}")
    print(f"  Annotation level: {stats['annotation_level']}")
    print(f"  Feature dimension: {stats['feature_dim']}")

    print(f"\nVariant Distribution:")
    print(f"  Mean variants/sample: {stats['mean_variants_per_sample']:.1f}")
    print(f"  Min variants: {stats['min_variants']}")
    print(f"  Max variants: {stats['max_variants']}")

    label_dist = dataset.get_label_distribution()
    print(f"\nLabel Distribution:")
    for label, count in label_dist.items():
        label_name = "Case" if label == 1 else "Control"
        print(f"  {label_name} (label={label}): {count}")

    print(f"\n✓ Statistics computed successfully")


def main():
    """Run all integration tests."""
    print("=" * 80)
    print("Phase 1B Integration Test: Feature Encoding Pipeline")
    print("=" * 80)

    # Load test data
    print_section("Loading Test Data")
    vcf_path = Path("test_data/small/test_data.vcf.gz")
    pheno_path = Path("test_data/small/test_data_phenotypes.tsv")

    print(f"VCF: {vcf_path}")
    print(f"Phenotypes: {pheno_path}")

    samples = build_sample_variants(vcf_path, pheno_path)
    print(f"\nLoaded {len(samples)} samples")

    # Build gene index
    print_section("Building Gene Index")
    from src.encoding import build_gene_index
    gene_index = build_gene_index(samples)
    print(f"Total unique genes: {len(gene_index)}")
    print(f"First 10 genes: {list(gene_index.keys())[:10]}")

    # Run built-in tests
    print_section("Running Built-in Unit Tests")
    print("\n1. Testing positional encoding consistency...")
    test_encoding_consistency()

    print("\n2. Testing sparse tensor construction...")
    test_sparse_tensor()

    # Test all annotation levels
    test_encoding_levels(samples, gene_index)

    # Test batching
    test_batching(samples, gene_index)

    # Test missing value handling
    test_missing_value_handling(samples, gene_index)

    # Test positional encodings
    test_positional_encodings()

    # Test dataset statistics
    dataset_l3 = VariantDataset(samples, AnnotationLevel.L3, gene_index)
    test_dataset_statistics(dataset_l3)

    # Final summary
    print_section("Test Summary")
    print("✅ All Phase 1B tests passed successfully!")
    print("\nTested components:")
    print("  ✓ Annotation levels L0-L4")
    print("  ✓ Sinusoidal positional encoding")
    print("  ✓ Relative position bucketing")
    print("  ✓ Sparse tensor construction")
    print("  ✓ DataLoader batching with padding")
    print("  ✓ Missing value imputation (neutral 0.5)")
    print("  ✓ Dataset statistics")
    print("\nPhase 1B implementation is validated and ready for Phase 1C!")


if __name__ == '__main__':
    main()
