#!/usr/bin/env python3
"""
Sex inference from X chromosome heterozygosity in multi-sample VCF.

Implements the PLINK-style F-statistic (X chromosome inbreeding coefficient)
with PAR exclusion. Produces a sex map for downstream ploidy-aware encoding.

Usage:
    python scripts/infer_sex.py \
        --vcf /path/to/cohort.vcf.gz \
        --output-dir /path/to/sex_inference/ \
        --genome-build GRCh37 \
        --min-gq 20 \
        --min-maf 0.05 \
        --f-male 0.8 \
        --f-female 0.2

Author: Francesco Lescai
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.genome import (
    GenomeBuild,
    get_genome_build,
    is_in_par,
    normalise_chrom,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Infer genetic sex from X chromosome heterozygosity',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--vcf', type=str, required=True,
                        help='Path to multi-sample VCF file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--genome-build', type=str, default='GRCh37',
                        help='Reference genome build')
    parser.add_argument('--min-gq', type=int, default=20,
                        help='Minimum genotype quality')
    parser.add_argument('--min-maf', type=float, default=0.05,
                        help='Minimum minor allele frequency')
    parser.add_argument('--max-missing', type=float, default=0.10,
                        help='Maximum missingness rate per variant')
    parser.add_argument('--f-male', type=float, default=0.8,
                        help='F-statistic threshold above which sample is male')
    parser.add_argument('--f-female', type=float, default=0.2,
                        help='F-statistic threshold below which sample is female')
    parser.add_argument('--known-sex', type=str, default=None,
                        help='Path to known sex file (sample_id\\tsex) for concordance check')

    return parser.parse_args()


def find_x_contig(vcf_contigs: list, build: GenomeBuild) -> str:
    """
    Find the chrX contig name used in the VCF.

    Parameters
    ----------
    vcf_contigs : list
        Contig names from the VCF header.
    build : GenomeBuild
        Genome build.

    Returns
    -------
    str
        The VCF contig name for chrX.

    Raises
    ------
    ValueError
        If no X chromosome contig is found.
    """
    for contig in vcf_contigs:
        normalised = normalise_chrom(contig, build)
        if normalised == 'X':
            return contig

    raise ValueError(
        f"No X chromosome contig found in VCF. "
        f"VCF contigs: {vcf_contigs[:20]}... "
        f"Expected one of: {build.x_contig_aliases}"
    )


def infer_sex_from_vcf(
    vcf_path: str,
    build: GenomeBuild,
    min_gq: int = 20,
    min_maf: float = 0.05,
    max_missing: float = 0.10,
    f_male: float = 0.8,
    f_female: float = 0.2,
) -> pd.DataFrame:
    """
    Infer genetic sex using X chromosome F-statistic.

    Parameters
    ----------
    vcf_path : str
        Path to VCF file.
    build : GenomeBuild
        Genome build for PAR coordinates.
    min_gq : int
        Minimum genotype quality.
    min_maf : float
        Minimum minor allele frequency.
    max_missing : float
        Maximum missingness rate per variant.
    f_male : float
        F > this -> male.
    f_female : float
        F < this -> female.

    Returns
    -------
    pd.DataFrame
        Per-sample sex inference results.
    """
    import cyvcf2

    vcf = cyvcf2.VCF(str(vcf_path))
    samples = vcf.samples
    n_samples = len(samples)

    # Find chrX contig
    vcf_contigs = vcf.seqnames
    x_contig = find_x_contig(vcf_contigs, build)
    print(f"  X chromosome contig in VCF: '{x_contig}'")

    # Per-sample accumulators
    n_called = np.zeros(n_samples, dtype=np.int64)
    n_het = np.zeros(n_samples, dtype=np.int64)
    sum_expected_het = np.zeros(n_samples, dtype=np.float64)

    # Iterate chrX variants only
    variant_count = 0
    used_count = 0

    for variant in vcf(x_contig):
        variant_count += 1

        # Only biallelic SNPs
        if len(variant.ALT) != 1 or len(variant.REF) != 1 or len(variant.ALT[0]) != 1:
            continue

        # Check FILTER
        filt = variant.FILTER
        if filt is not None and filt != '.' and filt != 'PASS':
            continue

        pos = variant.POS
        chrom = normalise_chrom(variant.CHROM, build)

        # Exclude PAR regions
        if is_in_par(pos, chrom, build):
            continue

        # Get genotypes: list of [allele1, allele2, phased]
        genotypes = variant.genotypes
        gq_values = variant.gt_quals

        # Compute allele frequency and missingness from the data.
        # On chrX non-PAR, males have haploid calls where cyvcf2 returns
        # a2 = -1.  We treat these as called with one allele (not missing).
        n_ref = 0
        n_alt = 0
        n_missing_var = 0

        for sample_idx in range(n_samples):
            gt = genotypes[sample_idx]
            a1, a2 = gt[0], gt[1]
            # Truly missing: both alleles unknown
            if a1 < 0:
                n_missing_var += 1
                continue
            if gq_values is not None and gq_values[sample_idx] < min_gq:
                n_missing_var += 1
                continue
            # Count allele 1
            n_ref += (a1 == 0)
            n_alt += (a1 > 0)
            # Count allele 2 only if present (not haploid)
            if a2 >= 0:
                n_ref += (a2 == 0)
                n_alt += (a2 > 0)

        total_alleles = n_ref + n_alt
        if total_alleles == 0:
            continue

        missingness = n_missing_var / n_samples
        if missingness > max_missing:
            continue

        alt_freq = n_alt / total_alleles
        maf = min(alt_freq, 1 - alt_freq)
        if maf < min_maf or maf > 0.5:
            continue

        # Expected heterozygosity for this variant
        exp_het_variant = 2 * alt_freq * (1 - alt_freq)

        used_count += 1

        # Accumulate per-sample stats.
        # Haploid calls (a2 = -1) are counted as called but never
        # heterozygous, which is the correct behaviour: males on chrX
        # are hemizygous so their F-statistic should trend toward 1.0.
        for sample_idx in range(n_samples):
            gt = genotypes[sample_idx]
            a1, a2 = gt[0], gt[1]

            # Skip truly missing (a1 unknown)
            if a1 < 0:
                continue
            if gq_values is not None and gq_values[sample_idx] < min_gq:
                continue

            n_called[sample_idx] += 1
            sum_expected_het[sample_idx] += exp_het_variant

            # Heterozygous only if diploid and alleles differ
            if a2 >= 0 and a1 != a2:
                n_het[sample_idx] += 1

    print(f"  Processed {variant_count} chrX variants, {used_count} passed filters")

    # Compute per-sample statistics
    results = []
    for sample_idx in range(n_samples):
        sample_id = samples[sample_idx]
        nc = int(n_called[sample_idx])
        nh = int(n_het[sample_idx])

        if nc < 50:
            results.append({
                'sample_id': sample_id,
                'inferred_sex': 'INSUFFICIENT_DATA',
                'F_statistic': np.nan,
                'n_chrx_variants': nc,
                'n_het': nh,
                'het_rate': np.nan,
                'confidence': 'LOW',
            })
            continue

        obs_het_rate = nh / nc
        exp_het_rate = sum_expected_het[sample_idx] / nc

        if exp_het_rate == 0:
            f_stat = np.nan
            sex = 'AMBIGUOUS'
            confidence = 'LOW'
        else:
            f_stat = 1.0 - (obs_het_rate / exp_het_rate)

            if f_stat > f_male:
                sex = 'M'
                confidence = 'HIGH'
            elif f_stat < f_female:
                sex = 'F'
                confidence = 'HIGH'
            else:
                sex = 'AMBIGUOUS'
                confidence = 'LOW'

        # Cross-validate: males should have near-zero het calls
        if sex == 'M' and obs_het_rate > 0.05:
            sex = 'DISCORDANT'
            confidence = 'LOW'

        results.append({
            'sample_id': sample_id,
            'inferred_sex': sex,
            'F_statistic': float(f_stat) if not np.isnan(f_stat) else np.nan,
            'n_chrx_variants': nc,
            'n_het': nh,
            'het_rate': float(obs_het_rate),
            'confidence': confidence,
        })

    return pd.DataFrame(results)


def create_diagnostic_plot(
    results: pd.DataFrame, output_path: Path,
    f_male: float = 0.8, f_female: float = 0.2,
) -> None:
    """
    Generate F-statistic histogram coloured by inferred sex.

    Parameters
    ----------
    results : pd.DataFrame
        Sex inference results with F_statistic and inferred_sex.
    output_path : Path
        Where to save the plot.
    f_male : float
        Male threshold line.
    f_female : float
        Female threshold line.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    valid = results.dropna(subset=['F_statistic'])
    if len(valid) == 0:
        ax.text(0.5, 0.5, 'No valid F-statistics', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    colors = {'M': '#4472C4', 'F': '#ED7D31', 'AMBIGUOUS': '#A5A5A5',
              'DISCORDANT': '#FF0000', 'INSUFFICIENT_DATA': '#CCCCCC'}

    for sex_label in ['F', 'M', 'AMBIGUOUS', 'DISCORDANT', 'INSUFFICIENT_DATA']:
        mask = valid['inferred_sex'] == sex_label
        if mask.sum() == 0:
            continue
        ax.hist(
            valid.loc[mask, 'F_statistic'], bins=50, alpha=0.7,
            label=f'{sex_label} (n={mask.sum()})', color=colors.get(sex_label, '#999999'),
        )

    ax.axvline(x=f_female, color='orange', linestyle='--', linewidth=1.5, label=f'F={f_female}')
    ax.axvline(x=f_male, color='blue', linestyle='--', linewidth=1.5, label=f'F={f_male}')

    ax.set_xlabel('X Chromosome F-statistic')
    ax.set_ylabel('Count')
    ax.set_title('Sex Inference Diagnostic: X Chromosome F-statistic Distribution')
    ax.legend(loc='upper center')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved diagnostic plot to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    build = get_genome_build(args.genome_build)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SIEVE Sex Inference from X Chromosome Heterozygosity")
    print("=" * 60)
    print(f"VCF: {args.vcf}")
    print(f"Genome build: {build.name}")
    print(f"PAR regions (X): {build.par_regions['X']}")
    print(f"F thresholds: male > {args.f_male}, female < {args.f_female}")

    # Run sex inference
    print("\nInferring sex...")
    results = infer_sex_from_vcf(
        vcf_path=args.vcf,
        build=build,
        min_gq=args.min_gq,
        min_maf=args.min_maf,
        max_missing=args.max_missing,
        f_male=args.f_male,
        f_female=args.f_female,
    )

    # Summary
    sex_counts = results['inferred_sex'].value_counts()
    n_male = int(sex_counts.get('M', 0))
    n_female = int(sex_counts.get('F', 0))
    n_ambiguous = int(sex_counts.get('AMBIGUOUS', 0))
    n_discordant = int(sex_counts.get('DISCORDANT', 0))
    n_insufficient = int(sex_counts.get('INSUFFICIENT_DATA', 0))

    print(f"\nSex inference results:")
    print(f"  Male: {n_male}")
    print(f"  Female: {n_female}")
    print(f"  Ambiguous: {n_ambiguous}")
    print(f"  Discordant: {n_discordant}")
    print(f"  Insufficient data: {n_insufficient}")

    valid = results.dropna(subset=['F_statistic'])
    males = valid[valid['inferred_sex'] == 'M']
    females = valid[valid['inferred_sex'] == 'F']

    median_f_male = float(males['F_statistic'].median()) if len(males) > 0 else None
    median_f_female = float(females['F_statistic'].median()) if len(females) > 0 else None

    # Save sample_sex.tsv
    sex_path = output_dir / 'sample_sex.tsv'
    results.to_csv(sex_path, sep='\t', index=False)
    print(f"\nSaved sex map to {sex_path}")

    # Diagnostic plot
    plot_path = output_dir / 'sex_inference_diagnostic.png'
    create_diagnostic_plot(results, plot_path, f_male=args.f_male, f_female=args.f_female)

    # Summary YAML
    summary = {
        'genome_build': build.name,
        'vcf_path': str(args.vcf),
        'n_samples': int(len(results)),
        'n_male': n_male,
        'n_female': n_female,
        'n_ambiguous': n_ambiguous,
        'n_discordant': n_discordant,
        'n_insufficient_data': n_insufficient,
        'median_F_male': median_f_male,
        'median_F_female': median_f_female,
        'f_male_threshold': float(args.f_male),
        'f_female_threshold': float(args.f_female),
        'min_gq': args.min_gq,
        'min_maf': args.min_maf,
        'max_missing': args.max_missing,
    }

    summary_path = output_dir / 'sex_inference_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"Saved summary to {summary_path}")

    # Known sex concordance check
    if args.known_sex:
        print("\nChecking concordance with known sex...")
        known = {}
        with open(args.known_sex, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    known[parts[0]] = parts[1].upper()

        concordant = 0
        discordant_list = []
        checked = 0
        for _, row in results.iterrows():
            sid = row['sample_id']
            if sid not in known:
                continue
            checked += 1
            known_sex = known[sid]
            inferred = row['inferred_sex']
            # Normalise known sex
            known_norm = 'M' if known_sex in ('M', 'MALE', '1') else 'F' if known_sex in ('F', 'FEMALE', '2') else known_sex
            if inferred == known_norm:
                concordant += 1
            else:
                discordant_list.append({
                    'sample_id': sid,
                    'known_sex': known_norm,
                    'inferred_sex': inferred,
                    'F_statistic': row['F_statistic'],
                })

        if checked > 0:
            rate = concordant / checked
            print(f"  Checked: {checked}, Concordant: {concordant} ({rate:.1%})")
            if discordant_list:
                print(f"  Discordant samples: {len(discordant_list)}")
                for d in discordant_list[:10]:
                    print(f"    {d['sample_id']}: known={d['known_sex']}, "
                          f"inferred={d['inferred_sex']}, F={d['F_statistic']:.3f}")
        else:
            print("  No matching sample IDs found between known sex and VCF")

    print("\nSex inference complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
