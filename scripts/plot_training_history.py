#!/usr/bin/env python3
"""
Visualize training history for SIEVE models.

Creates plots showing training vs validation metrics (loss, AUC, accuracy) over epochs.
Supports both single training runs and cross-validation with multiple folds.

Usage:
    # Single run
    python scripts/plot_training_history.py --history-file outputs/experiment/training_history.yaml --output training_curves.png

    # Cross-validation (all folds)
    python scripts/plot_training_history.py --experiment-dir outputs/experiment --output training_curves_cv.png

Author: Lescai Lab
"""

import argparse
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot training history',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--history-file', type=str, default=None,
                        help='Path to training_history.yaml file (single run)')
    parser.add_argument('--experiment-dir', type=str, default=None,
                        help='Directory containing fold_N subdirectories (CV runs)')
    parser.add_argument('--output', type=str, default='training_history.png',
                        help='Output plot filename')
    parser.add_argument('--max-folds', type=int, default=10,
                        help='Maximum number of folds to plot')
    return parser.parse_args()


def load_history(history_path):
    """Load training history from YAML file."""
    try:
        with open(history_path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"History file not found: {history_path}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied while accessing history file: {history_path}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML history file '{history_path}': {e}") from e
    return data


def validate_history_data(history_data, history_path):
    """
    Validate that history data contains all required keys.

    Args:
        history_data: Loaded history data dictionary
        history_path: Path to history file (for error messages)

    Raises:
        ValueError: If required keys are missing or data is invalid
    """
    required_top_keys = ['history']
    required_history_keys = [
        'train_loss', 'val_loss',
        'train_auc', 'val_auc',
        'train_accuracy', 'val_accuracy',
        'learning_rate'
    ]

    # Check if history_data is None or not a dict
    if history_data is None:
        raise ValueError(f"Empty or invalid YAML file: {history_path}")
    if not isinstance(history_data, dict):
        raise ValueError(
            f"Invalid history file format in {history_path}: expected a dictionary"
        )

    # Check top-level keys
    for key in required_top_keys:
        if key not in history_data:
            raise ValueError(f"Missing required key '{key}' in {history_path}")

    history = history_data['history']

    # Check that history is a dictionary
    if not isinstance(history, dict):
        raise ValueError(f"'history' must be a dictionary in {history_path}")

    # Check history keys
    missing_keys = [key for key in required_history_keys if key not in history]
    if missing_keys:
        raise ValueError(
            f"Missing required history keys in {history_path}: {', '.join(missing_keys)}"
        )

    # Check that history entries are non-empty lists
    for key in required_history_keys:
        value = history[key]
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"History key '{key}' must be a non-empty list in {history_path}"
            )

    # Check that all metrics have the same length
    lengths = {key: len(history[key]) for key in required_history_keys}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"Inconsistent history lengths in {history_path}: {lengths}"
        )


def plot_single_run(history_data, output_path, history_path=""):
    """Plot training curves for a single run."""
    # Validate data
    validate_history_data(history_data, history_path)

    history = history_data['history']
    best_epoch = history_data.get('best_epoch', 0)

    epochs = np.arange(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'o-', label='Train', color='#3498db', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_loss'], 's-', label='Validation', color='#e74c3c', linewidth=2, markersize=4)
    if best_epoch > 0:
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best (epoch {best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')

    # AUC
    ax = axes[0, 1]
    ax.plot(epochs, history['train_auc'], 'o-', label='Train', color='#3498db', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_auc'], 's-', label='Validation', color='#e74c3c', linewidth=2, markersize=4)
    if best_epoch > 0:
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best (epoch {best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Validation AUC', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0.4, 1.05)

    # Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, history['train_accuracy'], 'o-', label='Train', color='#3498db', linewidth=2, markersize=4)
    ax.plot(epochs, history['val_accuracy'], 's-', label='Validation', color='#e74c3c', linewidth=2, markersize=4)
    if best_epoch > 0:
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best (epoch {best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0.4, 1.05)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, history['learning_rate'], 'o-', color='#9b59b6', linewidth=2, markersize=4)
    if best_epoch > 0:
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best (epoch {best_epoch})')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')

    plt.suptitle('Training History', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close(fig)

    return fig


def plot_cv_runs(experiment_dir, output_path, max_folds=10):
    """Plot training curves for cross-validation runs."""
    experiment_dir = Path(experiment_dir)

    # Find all fold directories
    fold_dirs = sorted([d for d in experiment_dir.glob('fold_*') if d.is_dir()])

    if not fold_dirs:
        print(f"No fold directories found in {experiment_dir}")
        return

    # Limit number of folds
    fold_dirs = fold_dirs[:max_folds]

    # Load all histories
    histories = []
    for fold_dir in fold_dirs:
        history_path = fold_dir / 'training_history.yaml'
        if history_path.exists():
            try:
                history_data = load_history(history_path)
                validate_history_data(history_data, history_path)
                histories.append(history_data)
            except (FileNotFoundError, PermissionError, ValueError) as e:
                print(f"Warning: Skipping {fold_dir} due to invalid history: {e}")
        else:
            print(f"Warning: No training_history.yaml found in {fold_dir}")

    if not histories:
        print(f"No training histories found in {experiment_dir}")
        return

    # Determine max epochs (all folds might have different lengths due to early stopping)
    max_epochs = max(len(h['history']['train_loss']) for h in histories)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    # Loss
    ax = axes[0, 0]
    for fold_idx, (history_data, color) in enumerate(zip(histories, colors)):
        history = history_data['history']
        epochs = np.arange(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], '--', color=color, alpha=0.5, linewidth=1)
        ax.plot(epochs, history['val_loss'], '-', color=color, alpha=0.8, linewidth=1.5, label=f'Fold {fold_idx}')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Validation Loss (solid) vs Training Loss (dashed)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, linestyle='--')

    # AUC
    ax = axes[0, 1]
    for fold_idx, (history_data, color) in enumerate(zip(histories, colors)):
        history = history_data['history']
        epochs = np.arange(1, len(history['train_auc']) + 1)
        ax.plot(epochs, history['train_auc'], '--', color=color, alpha=0.5, linewidth=1)
        ax.plot(epochs, history['val_auc'], '-', color=color, alpha=0.8, linewidth=1.5, label=f'Fold {fold_idx}')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Validation AUC (solid) vs Training AUC (dashed)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0.4, 1.05)

    # Accuracy
    ax = axes[1, 0]
    for fold_idx, (history_data, color) in enumerate(zip(histories, colors)):
        history = history_data['history']
        epochs = np.arange(1, len(history['train_accuracy']) + 1)
        ax.plot(epochs, history['train_accuracy'], '--', color=color, alpha=0.5, linewidth=1)
        ax.plot(epochs, history['val_accuracy'], '-', color=color, alpha=0.8, linewidth=1.5, label=f'Fold {fold_idx}')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Validation Accuracy (solid) vs Training Accuracy (dashed)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0.4, 1.05)

    # Mean validation AUC with std
    ax = axes[1, 1]
    # Pad histories to same length with NaN
    val_aucs_padded = np.full((len(histories), max_epochs), np.nan)
    for fold_idx, history_data in enumerate(histories):
        history = history_data['history']
        n_epochs = len(history['val_auc'])
        val_aucs_padded[fold_idx, :n_epochs] = history['val_auc']

    # Compute mean and std (ignoring NaN)
    mean_val_auc = np.nanmean(val_aucs_padded, axis=0)
    std_val_auc = np.nanstd(val_aucs_padded, axis=0)
    epochs_mean = np.arange(1, max_epochs + 1)

    ax.plot(epochs_mean, mean_val_auc, 'o-', color='#e74c3c', linewidth=2, markersize=4, label='Mean Val AUC')
    ax.fill_between(epochs_mean, mean_val_auc - std_val_auc, mean_val_auc + std_val_auc,
                     color='#e74c3c', alpha=0.2, label='±1 std')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Mean Validation AUC Across Folds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0.4, 1.05)

    plt.suptitle(f'Training History - Cross-Validation ({len(histories)} folds)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close(fig)

    return fig


def print_training_analysis(history_data, history_path=""):
    """Print analysis of training dynamics."""
    # Validate data
    validate_history_data(history_data, history_path)

    history = history_data['history']
    best_epoch = history_data.get('best_epoch', 0)
    total_epochs = history_data.get('total_epochs', len(history['train_loss']))

    print("\n" + "="*60)
    print("Training Dynamics Analysis")
    print("="*60)

    print(f"\nTotal epochs: {total_epochs}")
    print(f"Best epoch: {best_epoch}")

    # Final metrics
    final_train_auc = history['train_auc'][-1]
    final_val_auc = history['val_auc'][-1]
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]

    print(f"\nFinal metrics (epoch {total_epochs}):")
    print(f"  Train AUC: {final_train_auc:.4f}")
    print(f"  Val AUC:   {final_val_auc:.4f}")
    print(f"  Train Loss: {final_train_loss:.4f}")
    print(f"  Val Loss:   {final_val_loss:.4f}")

    # Best metrics
    best_val_auc = max(history['val_auc'])
    best_train_auc = history['train_auc'][np.argmax(history['val_auc'])]

    print(f"\nBest validation metrics (epoch {best_epoch}):")
    print(f"  Train AUC: {best_train_auc:.4f}")
    print(f"  Val AUC:   {best_val_auc:.4f}")

    # Overfitting detection
    auc_gap = final_train_auc - final_val_auc
    loss_gap = final_val_loss - final_train_loss

    print(f"\nOverfitting indicators:")
    print(f"  AUC gap (train - val): {auc_gap:.4f}")
    print(f"  Loss gap (val - train): {loss_gap:.4f}")

    if auc_gap > 0.15:
        print(f"  ⚠ WARNING: Large AUC gap suggests overfitting")
    if final_train_auc > 0.95 and final_val_auc < 0.75:
        print(f"  ⚠ WARNING: Train AUC near perfect but validation much lower (overfitting)")
    if loss_gap > 0.5:
        print(f"  ⚠ WARNING: Large loss gap suggests overfitting")

    # Check if training improved over epochs
    early_val_auc = np.mean(history['val_auc'][:min(3, len(history['val_auc']))])
    improvement = final_val_auc - early_val_auc

    print(f"\nValidation AUC improvement from early epochs: {improvement:.4f}")
    if improvement < 0.01:
        print(f"  ⚠ WARNING: Little improvement - model may not be learning effectively")


def main():
    args = parse_args()

    if args.history_file is not None:
        # Single run
        history_data = load_history(args.history_file)
        print_training_analysis(history_data, args.history_file)
        plot_single_run(history_data, args.output, args.history_file)
    elif args.experiment_dir is not None:
        # Cross-validation
        plot_cv_runs(args.experiment_dir, args.output, max_folds=args.max_folds)
    else:
        print("Error: Must provide either --history-file or --experiment-dir")
        return


if __name__ == '__main__':
    main()
