#!/usr/bin/env python3
"""
Training script for SIEVE models.

This script trains a SIEVE model on VCF data with specified annotation level.
Supports single train/val split or k-fold cross-validation.

Usage:
    python scripts/train.py --vcf data.vcf.gz --phenotypes phenotypes.tsv --level L3
    python scripts/train.py --vcf data.vcf.gz --phenotypes phenotypes.tsv --level L3 --cv 5

Author: Lescai Lab
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import build_sample_variants
from src.encoding import AnnotationLevel, VariantDataset, get_feature_dimension
from src.models import SIEVE
from src.training import (
    SIEVELoss,
    Trainer,
    create_stratified_folds,
    get_train_val_loaders,
    print_fold_stats,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SIEVE model on VCF data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--vcf', type=str, default=None,
                        help='Path to VCF file (required if not using --preprocessed-data)')
    parser.add_argument('--phenotypes', type=str, default=None,
                        help='Path to phenotypes file (required if not using --preprocessed-data)')
    parser.add_argument('--preprocessed-data', type=str, default=None,
                        help='Path to preprocessed data file (.pt from preprocess.py)')
    parser.add_argument('--level', type=str, required=True,
                        choices=['L0', 'L1', 'L2', 'L3', 'L4'],
                        help='Annotation level to use')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--lambda-attr', type=float, default=0.0,
                        help='Attribution regularization weight')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--gradient-clip', type=float, default=None,
                        help='Gradient clipping value')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of batches to accumulate gradients over (simulates larger batch)')
    parser.add_argument('--max-variants-per-batch', type=int, default=3000,
                        help='Maximum variants to process per batch (prevents OOM)')

    # Model arguments
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Latent dimension for embeddings')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num-attention-layers', type=int, default=2,
                        help='Number of attention layers')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension in encoder')

    # Cross-validation arguments
    parser.add_argument('--cv', type=int, default=None,
                        help='Number of CV folds (if None, use single train/val split)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio (if not using CV)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for checkpoints and results')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (default: {level}_run)')

    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(
    input_dim: int,
    num_genes: int,
    latent_dim: int,
    num_heads: int,
    num_attention_layers: int,
    hidden_dim: int,
) -> SIEVE:
    """Create SIEVE model."""
    model = SIEVE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_genes=num_genes,
        num_attention_layers=num_attention_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
    )
    return model


def train_single_fold(
    train_loader,
    val_loader,
    model: SIEVE,
    args,
    checkpoint_dir: Path,
) -> Dict[str, float]:
    """Train model on a single fold."""
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize AUC
        factor=0.5,
        patience=5,
    )

    # Create loss function
    loss_fn = SIEVELoss(lambda_attr=args.lambda_attr)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        checkpoint_dir=checkpoint_dir,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping,
        gradient_clip_value=args.gradient_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Train
    print("\nTraining...")
    train_start = time.time()
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        verbose=True,
    )
    train_time = time.time() - train_start

    # Capture actual epochs trained before loading checkpoint
    actual_epochs_trained = trainer.current_epoch + 1

    # Load best model
    best_metrics = trainer.load_checkpoint('best_model.pt')

    # Add timing information (use actual epochs, not best checkpoint epoch)
    best_metrics['training_time_seconds'] = train_time
    best_metrics['epochs_trained'] = actual_epochs_trained
    best_metrics['time_per_epoch_seconds'] = train_time / actual_epochs_trained
    best_metrics['best_epoch'] = trainer.current_epoch + 1  # When best model was found

    return best_metrics


def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)

    # Validate arguments
    if args.preprocessed_data is None and (args.vcf is None or args.phenotypes is None):
        raise ValueError("Must provide either --preprocessed-data OR both --vcf and --phenotypes")

    # Create experiment name
    if args.experiment_name is None:
        args.experiment_name = f"{args.level}_run"

    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(vars(args), f)
    print(f"Config saved to {config_path}")

    # Load data
    if args.preprocessed_data is not None:
        # Load from preprocessed file
        print(f"\nLoading preprocessed data from {args.preprocessed_data}...")
        start_time = time.time()

        preprocessed = torch.load(args.preprocessed_data, weights_only=False)
        all_samples = preprocessed['samples']
        metadata = preprocessed.get('metadata', {})

        load_time = time.time() - start_time
        print(f"Loaded {len(all_samples)} samples in {load_time:.1f} seconds")
        if metadata:
            print(f"  Original VCF: {metadata.get('vcf_path', 'unknown')}")
            print(f"  Cases: {metadata.get('num_cases', 'unknown')}")
            print(f"  Controls: {metadata.get('num_controls', 'unknown')}")
    else:
        # Load from VCF
        print(f"\nLoading data from {args.vcf}...")
        start_time = time.time()
        all_samples = build_sample_variants(
            vcf_path=args.vcf,
            phenotype_file=args.phenotypes,
        )
        load_time = time.time() - start_time
        print(f"Loaded {len(all_samples)} samples in {load_time:.1f} seconds")

    # Create dataset
    annotation_level = AnnotationLevel[args.level]
    print(f"\nCreating dataset with annotation level {args.level}...")
    dataset = VariantDataset(all_samples, annotation_level=annotation_level)

    # Get dimensions
    input_dim = get_feature_dimension(annotation_level)
    num_genes = dataset.num_genes
    print(f"Input dimension: {input_dim}")
    print(f"Number of genes: {num_genes}")

    # Get labels
    labels = np.array([sample.label for sample in all_samples])
    n_cases = labels.sum()
    n_controls = len(labels) - n_cases
    print(f"Cases: {n_cases}, Controls: {n_controls} ({n_cases/len(labels):.1%} case rate)")

    if args.cv is not None:
        # Cross-validation
        print(f"\n{'='*60}")
        print(f"Running {args.cv}-fold cross-validation")
        print(f"{'='*60}")

        folds = create_stratified_folds(labels, n_folds=args.cv, random_state=args.seed)
        cv_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/{args.cv}")
            print(f"{'='*60}")

            # Print fold statistics
            train_labels = labels[train_idx]
            val_labels = labels[val_idx]
            print_fold_stats(fold_idx, train_labels, val_labels)

            # Create data loaders
            train_loader, val_loader = get_train_val_loaders(
                dataset=dataset,
                train_indices=train_idx,
                val_indices=val_idx,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_variants_per_batch=args.max_variants_per_batch,
            )

            # Create model
            model = create_model(
                input_dim=input_dim,
                num_genes=num_genes,
                latent_dim=args.latent_dim,
                num_heads=args.num_heads,
                num_attention_layers=args.num_attention_layers,
                hidden_dim=args.hidden_dim,
            )

            # Create fold checkpoint directory
            fold_dir = output_dir / f'fold_{fold_idx}'

            # Train
            fold_metrics = train_single_fold(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                args=args,
                checkpoint_dir=fold_dir,
            )

            cv_results.append(fold_metrics)

            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Val AUC: {fold_metrics['auc']:.4f}")
            print(f"  Val Accuracy: {fold_metrics['accuracy']:.4f}")

        # Print CV summary
        print(f"\n{'='*60}")
        print(f"Cross-Validation Summary")
        print(f"{'='*60}")
        mean_auc = np.mean([r['auc'] for r in cv_results])
        std_auc = np.std([r['auc'] for r in cv_results])
        mean_acc = np.mean([r['accuracy'] for r in cv_results])
        std_acc = np.std([r['accuracy'] for r in cv_results])

        print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

        # Save CV results
        results_path = output_dir / 'cv_results.yaml'
        with open(results_path, 'w') as f:
            yaml.dump({
                'mean_auc': float(mean_auc),
                'std_auc': float(std_auc),
                'mean_accuracy': float(mean_acc),
                'std_accuracy': float(std_acc),
                'data_loading_time_seconds': float(load_time),
                'fold_results': [
                    {k: float(v) for k, v in r.items()}
                    for r in cv_results
                ],
            }, f)
        print(f"\nResults saved to {results_path}")

    else:
        # Single train/val split
        print(f"\nUsing single train/val split ({1-args.val_split:.0%}/{args.val_split:.0%})")

        # Create stratified split
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=args.val_split,
            stratify=labels,
            random_state=args.seed,
        )

        # Print split statistics
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        print_fold_stats(0, train_labels, val_labels)

        # Create data loaders
        train_loader, val_loader = get_train_val_loaders(
            dataset=dataset,
            train_indices=train_idx,
            val_indices=val_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_variants_per_batch=args.max_variants_per_batch,
        )

        # Create model
        model = create_model(
            input_dim=input_dim,
            num_genes=num_genes,
            latent_dim=args.latent_dim,
            num_heads=args.num_heads,
            num_attention_layers=args.num_attention_layers,
            hidden_dim=args.hidden_dim,
        )

        # Train
        metrics = train_single_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            args=args,
            checkpoint_dir=output_dir,
        )

        print(f"\nFinal Results:")
        print(f"  Val AUC: {metrics['auc']:.4f}")
        print(f"  Val Accuracy: {metrics['accuracy']:.4f}")

        # Save results
        results_path = output_dir / 'results.yaml'
        results_with_timing = {k: float(v) for k, v in metrics.items()}
        results_with_timing['data_loading_time_seconds'] = float(load_time)
        with open(results_path, 'w') as f:
            yaml.dump(results_with_timing, f)
        print(f"\nResults saved to {results_path}")

    print(f"\nTraining complete! Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
