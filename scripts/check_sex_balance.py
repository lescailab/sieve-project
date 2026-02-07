#!/usr/bin/env python3
"""
Check sex balance across cases and controls.

Verifies that male/female proportions are balanced between case and
control groups. An imbalance could mean the model learns sex as a proxy
for disease.

Usage:
    python scripts/check_sex_balance.py \
        --sex-map /path/to/sample_sex.tsv \
        --phenotypes /path/to/phenotypes.tsv \
        --genome-build GRCh37 \
        --output-dir /path/to/diagnostics/

Author: Lescai Lab
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.genome import get_genome_build
from src.data.vcf_parser import load_phenotypes


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Check sex balance across cases and controls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--sex-map', type=str, required=True,
                        help='Path to sample_sex.tsv from infer_sex.py')
    parser.add_argument('--phenotypes', type=str, required=True,
                        help='Path to phenotypes file (TSV: sample_id, phenotype)')
    parser.add_argument('--genome-build', type=str, default='GRCh37',
                        help='Reference genome build')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for diagnostics')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    build = get_genome_build(args.genome_build)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SIEVE Sex Balance Diagnostic")
    print("=" * 60)
    print(f"Genome build: {build.name}")

    # Load sex map
    print(f"\nLoading sex map from {args.sex_map}")
    sex_df = pd.read_csv(args.sex_map, sep='\t')
    print(f"  {len(sex_df)} samples loaded")

    # Filter to M/F only (exclude ambiguous, insufficient)
    sex_map = dict(zip(sex_df['sample_id'], sex_df['inferred_sex']))
    mf_only = {k: v for k, v in sex_map.items() if v in ('M', 'F')}
    print(f"  {len(mf_only)} samples with definitive sex (M or F)")

    # Load phenotypes
    print(f"\nLoading phenotypes from {args.phenotypes}")
    phenotypes = load_phenotypes(Path(args.phenotypes))
    print(f"  {len(phenotypes)} samples loaded")
    n_cases = sum(1 for v in phenotypes.values() if v == 1)
    n_controls = sum(1 for v in phenotypes.values() if v == 0)
    print(f"  Cases: {n_cases}, Controls: {n_controls}")

    # Cross-tabulate
    overlap_ids = set(mf_only.keys()) & set(phenotypes.keys())
    print(f"\n  Overlapping samples: {len(overlap_ids)}")

    if len(overlap_ids) == 0:
        print("ERROR: No overlapping samples between sex map and phenotypes.")
        sys.exit(1)

    # Build contingency table
    male_case = sum(1 for sid in overlap_ids if mf_only[sid] == 'M' and phenotypes[sid] == 1)
    male_control = sum(1 for sid in overlap_ids if mf_only[sid] == 'M' and phenotypes[sid] == 0)
    female_case = sum(1 for sid in overlap_ids if mf_only[sid] == 'F' and phenotypes[sid] == 1)
    female_control = sum(1 for sid in overlap_ids if mf_only[sid] == 'F' and phenotypes[sid] == 0)

    table = np.array([[male_case, male_control],
                       [female_case, female_control]])

    total_cases = male_case + female_case
    total_controls = male_control + female_control
    total_male = male_case + male_control
    total_female = female_case + female_control

    print("\n  Cross-tabulation (Sex x Phenotype):")
    print(f"  {'':15s} {'Case':>10s} {'Control':>10s} {'Total':>10s}")
    print(f"  {'Male':15s} {male_case:10d} {male_control:10d} {total_male:10d}")
    print(f"  {'Female':15s} {female_case:10d} {female_control:10d} {total_female:10d}")
    print(f"  {'Total':15s} {total_cases:10d} {total_controls:10d} {len(overlap_ids):10d}")

    # Proportions
    pct_male_cases = male_case / total_cases * 100 if total_cases > 0 else 0
    pct_male_controls = male_control / total_controls * 100 if total_controls > 0 else 0

    print(f"\n  % Male in cases: {pct_male_cases:.1f}%")
    print(f"  % Male in controls: {pct_male_controls:.1f}%")

    # Fisher's exact test
    odds_ratio, fisher_pval = stats.fisher_exact(table)

    print(f"\n  Fisher's exact test:")
    print(f"    Odds ratio: {odds_ratio:.4f}")
    print(f"    P-value: {fisher_pval:.4e}")

    is_imbalanced = fisher_pval < 0.05
    if is_imbalanced:
        print("\n  WARNING: Significant sex imbalance detected (p < 0.05)!")
        print("  Recommendations:")
        print("    1. Sex-stratified analysis (train separate models for M and F)")
        print("    2. Add sex as a covariate in the classifier head")
    else:
        print("\n  Sex balance OK: no significant imbalance detected.")

    # Save contingency table
    table_df = pd.DataFrame(
        {'case': [male_case, female_case], 'control': [male_control, female_control]},
        index=['male', 'female'],
    )
    table_path = output_dir / 'sex_balance_table.csv'
    table_df.to_csv(table_path)
    print(f"\nSaved contingency table to {table_path}")

    # Save report
    report = {
        'genome_build': build.name,
        'n_samples_with_sex': len(mf_only),
        'n_samples_with_phenotype': len(phenotypes),
        'n_overlapping': int(len(overlap_ids)),
        'contingency_table': {
            'male_case': int(male_case),
            'male_control': int(male_control),
            'female_case': int(female_case),
            'female_control': int(female_control),
        },
        'pct_male_in_cases': float(pct_male_cases),
        'pct_male_in_controls': float(pct_male_controls),
        'fisher_exact_test': {
            'odds_ratio': float(odds_ratio),
            'p_value': float(fisher_pval),
        },
        'is_significantly_imbalanced': bool(is_imbalanced),
        'recommendation': (
            'Sex-stratified analysis or sex covariate recommended'
            if is_imbalanced
            else 'No action needed - sex is balanced across phenotype groups'
        ),
    }

    report_path = output_dir / 'sex_balance_report.yaml'
    with open(report_path, 'w') as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    print(f"Saved report to {report_path}")

    print("\nSex balance check complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
