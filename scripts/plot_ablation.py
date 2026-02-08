#!/usr/bin/env python3
"""
Visualize ablation experiment results.

Creates a bar plot showing AUC across annotation levels with error bars.

Usage:
    python scripts/plot_ablation.py --results-dir experiments/ablation --output ablation_results.png

Author: Francesco Lescai
"""

import argparse
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot ablation experiment results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing ablation results')
    parser.add_argument('--output', type=str, default='ablation_results.png',
                        help='Output plot filename')
    return parser.parse_args()


def load_results(results_dir):
    """Load results from all ablation experiments."""
    results_dir = Path(results_dir)

    levels = []
    mean_aucs = []
    std_aucs = []

    for level in ['L0', 'L1', 'L2', 'L3', 'L4']:
        cv_results_path = results_dir / f'ablation_{level}' / 'cv_results.yaml'
        if cv_results_path.exists():
            try:
                with open(cv_results_path) as f:
                    data = yaml.safe_load(f)
                # Ensure the expected keys are present
                mean_auc = data['mean_auc']
                std_auc = data['std_auc']
            except (FileNotFoundError, PermissionError) as e:
                print(f"Warning: Could not open '{cv_results_path}': {e}. Skipping {level}.")
                continue
            except yaml.YAMLError as e:
                print(f"Warning: Failed to parse YAML in '{cv_results_path}': {e}. Skipping {level}.")
                continue
            except (KeyError, TypeError) as e:
                print(f"Warning: Missing 'mean_auc'/'std_auc' in '{cv_results_path}': {e}. Skipping {level}.")
                continue

            levels.append(level)
            mean_aucs.append(mean_auc)
            std_aucs.append(std_auc)

    # Convert to numpy arrays for consistency with scientific computing
    mean_aucs = np.array(mean_aucs)
    std_aucs = np.array(std_aucs)

    return levels, mean_aucs, std_aucs


def create_plot(levels, mean_aucs, std_aucs, output_path):
    """Create bar plot with error bars."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(levels))
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db'][:len(levels)]

    bars = ax.bar(x, mean_aucs, yerr=std_aucs, capsize=5,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_xlabel('Annotation Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Experiment: Impact of Annotations on Model Performance',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=12)
    ax.set_ylim(0.5, max(mean_aucs) + max(std_aucs) + 0.05)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Random (AUC=0.5)')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, mean_aucs, std_aucs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add annotation descriptions
    descriptions = {
        'L0': 'Genotype only',
        'L1': 'Genotype + Position',
        'L2': 'L1 + Consequence',
        'L3': 'L2 + SIFT/PolyPhen',
        'L4': 'L3 + Additional'
    }

    desc_text = '\n'.join([f'{k}: {v}' for k, v in descriptions.items() if k in levels])
    if desc_text:
        ax.text(0.02, 0.98, desc_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Close figure to prevent memory leaks
    plt.close(fig)

    return fig


def print_summary(levels, mean_aucs, std_aucs):
    """Print text summary of results."""
    print("\n" + "="*60)
    print("Ablation Experiment Results Summary")
    print("="*60)

    for level, mean, std in zip(levels, mean_aucs, std_aucs):
        print(f"{level}: {mean:.4f} ± {std:.4f}")

    # Find best
    best_idx = np.argmax(mean_aucs)

    # Determine baseline: prefer L0 if available, otherwise use first level
    if 'L0' in levels:
        baseline_idx = levels.index('L0')
        baseline_label = levels[baseline_idx]
    else:
        baseline_idx = 0
        baseline_label = levels[baseline_idx]
        print(f"\n[Warning] Baseline level 'L0' not found; using first available level "
              f"('{baseline_label}') as baseline.")

    print(f"\nBest: {levels[best_idx]} with AUC = {mean_aucs[best_idx]:.4f}")
    improvement = mean_aucs[best_idx] - mean_aucs[baseline_idx]
    print(f"Improvement over baseline ({baseline_label}): {improvement:.4f} AUC")

    # Avoid division by zero
    if mean_aucs[baseline_idx] != 0:
        rel_improvement = 100 * improvement / mean_aucs[baseline_idx]
        print(f"Relative improvement: {rel_improvement:.1f}%")
    else:
        print(f"Relative improvement: N/A (baseline AUC is {mean_aucs[baseline_idx]:.4f})")


def main():
    args = parse_args()

    # Load results
    levels, mean_aucs, std_aucs = load_results(args.results_dir)

    if not levels:
        print(f"No results found in {args.results_dir}")
        return

    # Print summary
    print_summary(levels, mean_aucs, std_aucs)

    # Create plot
    create_plot(levels, mean_aucs, std_aucs, args.output)


if __name__ == '__main__':
    main()
