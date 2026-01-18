#!/usr/bin/env python3
"""
Integration test for Phase 1D: Training Pipeline

This script tests the complete training pipeline on real data:
1. Load and encode data (Phase 1A + 1B)
2. Create train/val loaders
3. Test loss functions
4. Test trainer initialization
5. Test single epoch training
6. Test validation
7. Test checkpointing

Author: Lescai Lab
"""

from pathlib import Path
import tempfile

import torch
import numpy as np

from src.data import build_sample_variants
from src.encoding import AnnotationLevel, VariantDataset, get_feature_dimension
from src.models import SIEVE
from src.training import (
    SIEVELoss,
    Trainer,
    attribution_sparsity_loss,
    compute_class_weights,
    create_stratified_folds,
    get_train_val_loaders,
)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")


def test_loss_functions():
    """Test loss function implementations."""
    print_section("Testing Loss Functions")

    batch_size = 4
    num_variants = 10
    latent_dim = 64

    # Create dummy data
    logits = torch.randn(batch_size, 1)
    labels = torch.randint(0, 2, (batch_size,))
    variant_embeddings = torch.randn(batch_size, num_variants, latent_dim, requires_grad=True)
    mask = torch.ones(batch_size, num_variants, dtype=torch.bool)

    print("\n1. Testing SIEVELoss without attribution regularization...")
    loss_fn = SIEVELoss(lambda_attr=0.0)
    loss_dict = loss_fn(logits=logits, labels=labels)
    print(f"  Classification loss: {loss_dict['classification'].item():.4f}")
    print(f"  Attribution loss: {loss_dict['attribution_sparsity'].item():.4f}")
    print(f"  Total loss: {loss_dict['total'].item():.4f}")
    assert loss_dict['attribution_sparsity'].item() == 0.0
    print("  ✓ Loss without attribution works")

    print("\n2. Testing SIEVELoss with attribution regularization...")
    loss_fn = SIEVELoss(lambda_attr=0.1)
    loss_dict = loss_fn(
        logits=logits,
        labels=labels,
        variant_embeddings=variant_embeddings,
        mask=mask,
    )
    print(f"  Classification loss: {loss_dict['classification'].item():.4f}")
    print(f"  Attribution loss: {loss_dict['attribution_sparsity'].item():.4f}")
    print(f"  Total loss: {loss_dict['total'].item():.4f}")
    assert loss_dict['attribution_sparsity'].item() > 0.0
    print("  ✓ Loss with attribution works")

    print("\n3. Testing attribution_sparsity_loss directly...")
    attr_loss = attribution_sparsity_loss(
        variant_embeddings=variant_embeddings,
        logits=logits.squeeze(),
        mask=mask,
    )
    print(f"  Attribution sparsity loss: {attr_loss.item():.4f}")
    print("  ✓ Attribution sparsity loss works")

    print("\n4. Testing compute_class_weights...")
    labels_imbalanced = torch.tensor([0, 0, 0, 1])  # 3:1 imbalance
    pos_weight = compute_class_weights(labels_imbalanced)
    print(f"  Labels: {labels_imbalanced.tolist()}")
    print(f"  Positive class weight: {pos_weight.item():.2f}")
    assert pos_weight.item() == 2.0  # 4 / (2 * 1) = 2.0
    print("  ✓ Class weights computed correctly")


def test_cross_validation():
    """Test cross-validation utilities."""
    print_section("Testing Cross-Validation Utilities")

    # Create dummy labels
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 50-50 split

    print("\n1. Testing stratified fold creation...")
    folds = create_stratified_folds(labels, n_folds=5, random_state=42)
    print(f"  Created {len(folds)} folds")
    for i, (train_idx, val_idx) in enumerate(folds):
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_ratio = train_labels.sum() / len(train_labels)
        val_ratio = val_labels.sum() / len(val_labels)
        print(f"  Fold {i+1}: Train {len(train_idx)} ({train_ratio:.1%} cases), "
              f"Val {len(val_idx)} ({val_ratio:.1%} cases)")
    print("  ✓ Stratified folds created correctly")


def test_training_loop():
    """Test training loop on real data."""
    print_section("Testing Training Loop on Real Data")

    # Paths
    vcf_path = "test_data/small/test_data.vcf.gz"
    phenotype_path = "test_data/small/test_data_phenotypes.tsv"

    # Check if files exist
    if not Path(vcf_path).exists():
        print(f"  Warning: Test data not found at {vcf_path}")
        print("  Skipping training loop test")
        return

    # Load data
    print("\n1. Loading data...")
    all_samples = build_sample_variants(vcf_path, phenotype_path)
    print(f"  Loaded {len(all_samples)} samples")

    # Create dataset
    print("\n2. Creating dataset with L3 encoding...")
    annotation_level = AnnotationLevel.L3
    dataset = VariantDataset(all_samples, annotation_level=annotation_level)
    input_dim = get_feature_dimension(annotation_level)
    num_genes = dataset.num_genes
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of genes: {num_genes}")

    # Create train/val split
    print("\n3. Creating train/val split...")
    labels = np.array([sample.label for sample in all_samples])
    n_samples = len(labels)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(n_samples * 0.8)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")

    # Create data loaders
    print("\n4. Creating data loaders...")
    train_loader, val_loader = get_train_val_loaders(
        dataset=dataset,
        train_indices=train_idx,
        val_indices=val_idx,
        batch_size=8,
        num_workers=0,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    print("\n5. Creating SIEVE model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    model = SIEVE(
        input_dim=input_dim,
        latent_dim=64,
        num_genes=num_genes,
        num_attention_layers=2,
        num_heads=4,
        hidden_dim=128,
    )
    print(f"  Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer and loss
    print("\n6. Creating optimizer and loss function...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = SIEVELoss(lambda_attr=0.0)  # No attribution for quick test
    print("  ✓ Optimizer and loss created")

    # Create trainer
    print("\n7. Creating trainer...")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            checkpoint_dir=checkpoint_dir,
            early_stopping_patience=5,
        )
        print("  ✓ Trainer created")

        # Test single epoch
        print("\n8. Testing single training epoch...")
        train_metrics = trainer.train_epoch(train_loader)
        print(f"  Train loss: {train_metrics['loss']:.4f}")
        print(f"  Train AUC: {train_metrics['auc']:.4f}")
        print(f"  Train accuracy: {train_metrics['accuracy']:.4f}")
        print("  ✓ Training epoch completed")

        # Test validation
        print("\n9. Testing validation...")
        val_metrics = trainer.validate(val_loader)
        print(f"  Val loss: {val_metrics['loss']:.4f}")
        print(f"  Val AUC: {val_metrics['auc']:.4f}")
        print(f"  Val accuracy: {val_metrics['accuracy']:.4f}")
        print("  ✓ Validation completed")

        # Test checkpointing
        print("\n10. Testing checkpointing...")
        trainer.save_checkpoint('test_checkpoint.pt', val_metrics)
        checkpoint_path = checkpoint_dir / 'test_checkpoint.pt'
        assert checkpoint_path.exists()
        print(f"  Checkpoint saved to {checkpoint_path}")

        # Load checkpoint
        loaded_metrics = trainer.load_checkpoint('test_checkpoint.pt')
        print(f"  Checkpoint loaded successfully")
        print(f"  Loaded metrics AUC: {loaded_metrics['auc']:.4f}")
        print("  ✓ Checkpointing works")

        # Test short training run
        print("\n11. Testing short training run (3 epochs)...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            verbose=True,
        )
        print(f"\n  Training completed!")
        print(f"  Final train AUC: {history['train_auc'][-1]:.4f}")
        print(f"  Final val AUC: {history['val_auc'][-1]:.4f}")
        print(f"  Best val AUC: {trainer.best_val_auc:.4f}")
        print("  ✓ Multi-epoch training works")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Phase 1D: Training Pipeline Integration Test")
    print("="*60)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Run tests
    test_loss_functions()
    test_cross_validation()
    test_training_loop()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == '__main__':
    main()
