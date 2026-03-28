"""
Extract per-sample burden counts from a validation cohort VCF.

Parses a VEP-annotated VCF and counts non-reference alleles per sample within
SIEVE-derived target gene sets. Also computes consequence-stratified counts
(missense, LoF, synonymous) and optionally builds a full gene-level burden
matrix for fast permutation testing.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cyvcf2
import numpy as np
import pandas as pd
import yaml

from src.data.genome import get_genome_build, is_sex_chrom, normalise_chrom
from src.data.vcf_parser import (
    load_phenotypes,
    parse_csq_field,
    select_canonical_annotation,
)
from src.data.annotation import map_consequence_to_severity


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-sample burden counts from a validation VCF.",
    )
    parser.add_argument(
        "--vcf",
        type=Path,
        required=True,
        help="Path to validation VCF (bgzipped, tabix-indexed)",
    )
    parser.add_argument(
        "--phenotypes",
        type=Path,
        required=True,
        help="Phenotype TSV (sample_id \\t phenotype: 1=ctrl, 2=case)",
    )
    parser.add_argument(
        "--sieve-genes",
        type=Path,
        required=True,
        help="SIEVE gene list TSV (gene_name, gene_rank, gene_score)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--genome-build",
        default="GRCh37",
        help="Reference genome build (default: GRCh37)",
    )
    parser.add_argument(
        "--min-gq",
        type=int,
        default=20,
        help="Minimum genotype quality (default: 20)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="Gene set sizes to test (default: 50 100 200)",
    )
    parser.add_argument(
        "--consequence-stratify",
        action="store_true",
        default=False,
        help="Also compute burden stratified by consequence class",
    )
    parser.add_argument(
        "--include-sex-chroms",
        action="store_true",
        default=False,
        help="Include sex chromosome variants (excluded by default)",
    )
    parser.add_argument(
        "--from-variant-rankings",
        action="store_true",
        default=False,
        help="Input is a variant rankings CSV — aggregate to gene level internally",
    )
    parser.add_argument(
        "--compute-full-gene-matrix",
        action="store_true",
        default=False,
        help="Build full gene-level burden matrix for permutation testing",
    )
    return parser.parse_args(argv)


def load_sieve_genes(
    gene_file: Path,
    from_variant_rankings: bool = False,
) -> pd.DataFrame:
    """
    Load SIEVE gene list, optionally aggregating from variant rankings.

    Parameters
    ----------
    gene_file : Path
        Path to gene list TSV or variant rankings CSV.
    from_variant_rankings : bool
        If True, aggregate variant-level file to gene level.

    Returns
    -------
    pd.DataFrame
        Gene list with columns: gene_name, gene_rank, gene_score.
    """
    if from_variant_rankings:
        from scripts.generate_sieve_gene_list import generate_gene_list

        df = pd.read_csv(gene_file)
        return generate_gene_list(df)

    # Detect separator
    sep = "\t"
    with open(gene_file) as f:
        first_line = f.readline()
        if "," in first_line and "\t" not in first_line:
            sep = ","

    df = pd.read_csv(gene_file, sep=sep)

    required = {"gene_name", "gene_rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gene list missing required columns: {missing}")

    return df.sort_values("gene_rank")


def classify_consequence(consequence: str) -> str:
    """
    Classify a VEP consequence into broad categories.

    Returns one of: 'lof', 'missense', 'synonymous', 'other'.
    """
    severity = map_consequence_to_severity(consequence)
    if severity == 4:
        return "lof"
    if "missense_variant" in consequence:
        return "missense"
    if "synonymous_variant" in consequence:
        return "synonymous"
    return "other"


def extract_burden_from_vcf(
    vcf_path: Path,
    phenotypes: Dict[str, int],
    target_gene_sets: Dict[int, Set[str]],
    genome_build_name: str = "GRCh37",
    min_gq: int = 20,
    consequence_stratify: bool = False,
    include_sex_chroms: bool = False,
    compute_full_matrix: bool = False,
) -> Tuple[
    Dict[int, pd.DataFrame],
    Dict[int, dict],
    Optional[pd.DataFrame],
    Optional[dict],
    Optional[Dict[str, pd.DataFrame]],
]:
    """
    Parse validation VCF and compute per-sample burden counts.

    Parameters
    ----------
    vcf_path : Path
        Validation VCF path.
    phenotypes : Dict[str, int]
        Sample to label mapping (0=control, 1=case).
    target_gene_sets : Dict[int, Set[str]]
        Mapping from top-k threshold to set of target gene names (upper-cased).
    genome_build_name : str
        Genome build name.
    min_gq : int
        Minimum genotype quality.
    consequence_stratify : bool
        Whether to track consequence-stratified counts.
    include_sex_chroms : bool
        Whether to include sex chromosome variants.
    compute_full_matrix : bool
        Whether to build full gene × sample burden matrix.

    Returns
    -------
    burden_dfs : Dict[int, pd.DataFrame]
        Per top-k burden DataFrames.
    summaries : Dict[int, dict]
        Per top-k summary statistics.
    full_matrix : pd.DataFrame or None
        Full gene burden matrix (samples × genes) if compute_full_matrix.
    matrix_metadata : dict or None
        Serializable metadata for the full matrix.
    consequence_matrices : Dict[str, pd.DataFrame] or None
        Consequence-stratified burden matrices keyed by type
        ('missense', 'lof', 'synonymous', 'other'). Only populated when
        both compute_full_matrix and consequence_stratify are True.
    """
    build = get_genome_build(genome_build_name)
    vcf = cyvcf2.VCF(str(vcf_path))
    samples = vcf.samples

    # Validate CSQ presence
    has_csq = any(
        line.startswith("##INFO=<ID=CSQ")
        for line in vcf.raw_header.split("\n")
    )
    if not has_csq:
        raise ValueError(
            f"VCF '{vcf_path}' lacks VEP CSQ annotations. "
            "Run VEP before burden extraction."
        )

    # Map samples to phenotypes
    sample_indices = {}
    for idx, s in enumerate(samples):
        if s in phenotypes:
            sample_indices[idx] = s

    if not sample_indices:
        raise ValueError("No VCF samples found in phenotype file.")

    print(f"Matched {len(sample_indices)} samples with phenotypes")

    # Collect all target genes across all top-k sets (they may not be strictly nested)
    all_target_genes: Set[str] = set().union(*target_gene_sets.values())

    # Initialise per-sample burden counters for each top-k
    # Structure: {k: {sample_id: {burden_type: count}}}
    burden_counts: Dict[int, Dict[str, Dict[str, int]]] = {}
    for k in target_gene_sets:
        burden_counts[k] = {
            s: {"total": 0, "missense": 0, "lof": 0, "synonymous": 0, "other": 0, "n_genes": 0}
            for s in sample_indices.values()
        }

    # Track which genes each sample has variants in (for n_genes_with_variants)
    genes_with_variants: Dict[int, Dict[str, set]] = {
        k: {s: set() for s in sample_indices.values()} for k in target_gene_sets
    }

    # Full gene matrix tracking
    all_vcf_genes: Set[str] = set()
    full_gene_counts: Optional[Dict[str, Dict[str, int]]] = None
    full_gene_consequence: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None
    if compute_full_matrix:
        full_gene_counts = {s: {} for s in sample_indices.values()}
        if consequence_stratify:
            full_gene_consequence = {s: {} for s in sample_indices.values()}

    # Track genes found in VCF that match SIEVE targets
    found_target_genes: Set[str] = set()
    variant_count = 0
    assigned_count = 0

    for variant in vcf:
        variant_count += 1

        chrom = normalise_chrom(variant.CHROM, build)

        # Skip sex chromosomes unless requested
        if not include_sex_chroms and is_sex_chrom(chrom, build):
            continue

        # Get CSQ
        try:
            csq_raw = variant.INFO.get("CSQ")
            if not csq_raw:
                continue
        except KeyError:
            continue

        pos = variant.POS
        alts = variant.ALT

        for alt_idx, alt in enumerate(alts):
            csq_annotations = parse_csq_field(csq_raw, alt)
            if not csq_annotations:
                continue

            canonical = select_canonical_annotation(csq_annotations)
            gene = canonical.get("3", "").upper()
            consequence = canonical.get("1", "unknown")

            if not gene:
                continue

            all_vcf_genes.add(gene)
            is_target = gene in all_target_genes
            if is_target:
                found_target_genes.add(gene)

            csq_class = classify_consequence(consequence) if (
                consequence_stratify or is_target
            ) else None

            # Process genotypes
            genotypes = variant.genotypes
            gq_values = variant.gt_quals
            target_allele = alt_idx + 1

            for sample_idx, sample_id in sample_indices.items():
                gt = genotypes[sample_idx]
                allele1, allele2 = gt[0], gt[1]

                # GQ filter
                if gq_values is not None and gq_values[sample_idx] < min_gq:
                    continue

                # Compute dosage
                dosage = 0
                if allele1 == target_allele:
                    dosage += 1
                if allele2 >= 0 and allele2 == target_allele:
                    dosage += 1

                if dosage == 0:
                    continue

                assigned_count += 1

                # Update full gene matrix
                if compute_full_matrix and full_gene_counts is not None:
                    full_gene_counts[sample_id][gene] = (
                        full_gene_counts[sample_id].get(gene, 0) + dosage
                    )
                    if consequence_stratify and full_gene_consequence is not None:
                        if gene not in full_gene_consequence[sample_id]:
                            full_gene_consequence[sample_id][gene] = {
                                "missense": 0, "lof": 0, "synonymous": 0, "other": 0,
                            }
                        cat = csq_class or classify_consequence(consequence)
                        full_gene_consequence[sample_id][gene][cat] += dosage

                # Update target burden counts
                if is_target:
                    cat = csq_class or classify_consequence(consequence)
                    for k, gene_set in target_gene_sets.items():
                        if gene in gene_set:
                            burden_counts[k][sample_id]["total"] += dosage
                            burden_counts[k][sample_id][cat] += dosage
                            genes_with_variants[k][sample_id].add(gene)

    print(f"Processed {variant_count} VCF records")
    print(f"  Variant-sample assignments: {assigned_count}")
    print(f"  Unique genes in VCF: {len(all_vcf_genes)}")
    print(f"  SIEVE target genes found: {len(found_target_genes)}/{len(all_target_genes)}")

    # Build output DataFrames
    burden_dfs = {}
    summaries = {}

    for k, gene_set in target_gene_sets.items():
        rows = []
        for sample_id in sample_indices.values():
            row = {
                "sample_id": sample_id,
                "phenotype": phenotypes[sample_id],
                "total_burden": burden_counts[k][sample_id]["total"],
                "n_genes_with_variants": len(genes_with_variants[k][sample_id]),
            }
            if consequence_stratify:
                row["missense_burden"] = burden_counts[k][sample_id]["missense"]
                row["lof_burden"] = burden_counts[k][sample_id]["lof"]
                row["synonymous_burden"] = burden_counts[k][sample_id]["synonymous"]
                row["other_burden"] = burden_counts[k][sample_id]["other"]
            rows.append(row)

        df = pd.DataFrame(rows)
        burden_dfs[k] = df

        # Gene matching diagnostics
        genes_upper = gene_set
        missing = genes_upper - found_target_genes
        found = genes_upper & found_target_genes

        cases = df[df["phenotype"] == 1]
        controls = df[df["phenotype"] == 0]

        summary = {
            "top_k": k,
            "n_sieve_genes": len(gene_set),
            "n_sieve_genes_found_in_vcf": len(found),
            "n_sieve_genes_missing": len(missing),
            "missing_genes": sorted(missing),
            "n_samples": len(df),
            "n_cases": len(cases),
            "n_controls": len(controls),
            "mean_burden_cases": float(cases["total_burden"].mean()) if len(cases) > 0 else 0.0,
            "mean_burden_controls": float(controls["total_burden"].mean()) if len(controls) > 0 else 0.0,
            "total_variants_counted": int(df["total_burden"].sum()),
        }
        summaries[k] = summary

    # Build full gene matrix if requested
    full_matrix = None
    matrix_metadata = None
    consequence_matrices: Optional[Dict[str, pd.DataFrame]] = None
    if compute_full_matrix and full_gene_counts is not None:
        sorted_genes = sorted(all_vcf_genes)
        sorted_samples = sorted(sample_indices.values())
        matrix_data = np.zeros((len(sorted_samples), len(sorted_genes)), dtype=np.int32)

        gene_to_col = {g: i for i, g in enumerate(sorted_genes)}
        sample_to_row = {s: i for i, s in enumerate(sorted_samples)}

        for s, gene_dict in full_gene_counts.items():
            row_idx = sample_to_row[s]
            for g, count in gene_dict.items():
                col_idx = gene_to_col[g]
                matrix_data[row_idx, col_idx] = count

        full_matrix = pd.DataFrame(
            matrix_data, index=sorted_samples, columns=sorted_genes,
        )
        full_matrix.index.name = "sample_id"

        matrix_metadata = {
            "gene_names": sorted_genes,
            "sample_ids": sorted_samples,
            "n_genes": len(sorted_genes),
            "n_samples": len(sorted_samples),
            "genome_build": genome_build_name,
            "vcf_path": str(vcf_path),
            "include_sex_chroms": include_sex_chroms,
        }

        # Build consequence-stratified matrices if requested
        if consequence_stratify and full_gene_consequence is not None:
            consequence_matrices = {}
            for csq_type in ["missense", "lof", "synonymous", "other"]:
                csq_data = np.zeros(
                    (len(sorted_samples), len(sorted_genes)), dtype=np.int32,
                )
                for s, gene_dict in full_gene_consequence.items():
                    row_idx = sample_to_row[s]
                    for g, csq_counts in gene_dict.items():
                        col_idx = gene_to_col[g]
                        csq_data[row_idx, col_idx] = csq_counts.get(csq_type, 0)

                csq_df = pd.DataFrame(
                    csq_data, index=sorted_samples, columns=sorted_genes,
                )
                csq_df.index.name = "sample_id"
                consequence_matrices[csq_type] = csq_df

        print(f"Built full gene burden matrix: {full_matrix.shape}")

    return burden_dfs, summaries, full_matrix, matrix_metadata, consequence_matrices


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load phenotypes
    phenotypes = load_phenotypes(args.phenotypes)
    n_cases = sum(1 for v in phenotypes.values() if v == 1)
    n_controls = sum(1 for v in phenotypes.values() if v == 0)
    print(f"Loaded phenotypes: {len(phenotypes)} samples ({n_cases} cases, {n_controls} controls)")

    # Load SIEVE gene list
    gene_df = load_sieve_genes(args.sieve_genes, args.from_variant_rankings)
    print(f"Loaded {len(gene_df)} SIEVE genes")

    # Build target gene sets for each top-k
    target_gene_sets: Dict[int, Set[str]] = {}
    for k in args.top_k:
        actual_k = min(k, len(gene_df))
        top_genes = set(gene_df.head(actual_k)["gene_name"].str.upper())
        target_gene_sets[k] = top_genes
        print(f"  top-{k}: {len(top_genes)} genes")

    # Extract burden
    burden_dfs, summaries, full_matrix, matrix_metadata, csq_matrices = extract_burden_from_vcf(
        vcf_path=args.vcf,
        phenotypes=phenotypes,
        target_gene_sets=target_gene_sets,
        genome_build_name=args.genome_build,
        min_gq=args.min_gq,
        consequence_stratify=args.consequence_stratify,
        include_sex_chroms=args.include_sex_chroms,
        compute_full_matrix=args.compute_full_gene_matrix,
    )

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for k in args.top_k:
        burden_dfs[k].to_csv(args.output_dir / f"burden_topK{k}.tsv", sep="\t", index=False)

        with open(args.output_dir / f"burden_topK{k}_summary.yaml", "w") as f:
            yaml.dump(summaries[k], f, default_flow_style=False, sort_keys=False)

        print(f"Saved burden_topK{k}.tsv and summary")

    # Save full gene matrix
    if full_matrix is not None and matrix_metadata is not None:
        full_matrix.to_parquet(args.output_dir / "gene_burden_matrix.parquet")

        # Save consequence-stratified matrices
        if csq_matrices:
            for csq_type, csq_df in csq_matrices.items():
                csq_df.to_parquet(
                    args.output_dir / f"gene_burden_matrix_{csq_type}.parquet"
                )

        with open(args.output_dir / "gene_burden_matrix_metadata.yaml", "w") as f:
            yaml.dump(matrix_metadata, f, default_flow_style=False, sort_keys=False)

        print(f"Saved full gene burden matrix ({full_matrix.shape})")

    print("Done.")


if __name__ == "__main__":
    main()
