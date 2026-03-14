#!/usr/bin/env python3
"""
Co-occurrence audit for SIEVE epistasis detection.

Audits whether the variant co-occurrence structure in a cohort can support
epistasis detection by computing pairwise co-occurrence statistics across
MAF-stratified variant pairs.

Usage:
    python scripts/audit_cooccurrence.py \
        --preprocessed-data data/processed/cohort.pt \
        --output-dir results/cooccurrence_audit/
"""

import argparse
import csv
import logging
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import yaml

# Ensure project root is on sys.path so torch.load can reconstruct dataclasses
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

VariantKey = Tuple[str, int]  # (chrom, pos)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit variant co-occurrence structure for epistasis detection."
    )
    parser.add_argument(
        "--preprocessed-data",
        type=str,
        required=True,
        help="Path to preprocessed .pt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for audit results.",
    )
    parser.add_argument(
        "--maf-bins",
        type=str,
        default="0.001,0.01,0.05,0.1,0.5",
        help="Comma-separated MAF bin edges (default: 0.001,0.01,0.05,0.1,0.5).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=100000,
        help="Maximum number of pairs to evaluate (default: 100000).",
    )
    parser.add_argument(
        "--top-k-variants",
        type=int,
        default=500,
        help="Restrict pairwise analysis to top-K most frequent variants (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def make_maf_labels(bin_edges: List[float]) -> List[str]:
    """Create human-readable labels for sorted carrier-frequency bins."""
    labels = []
    for i in range(len(bin_edges) + 1):
        if i == 0:
            labels.append(f"<{bin_edges[0] * 100}%")
        elif i == len(bin_edges):
            labels.append(f">{bin_edges[-1] * 100}%")
        else:
            labels.append(f"{bin_edges[i - 1] * 100}-{bin_edges[i] * 100}%")
    return labels


def assign_maf_bin(freq: float, bin_edges: List[float]) -> str:
    """
    Assign a carrier frequency to a labelled MAF bin.

    Parameters
    ----------
    freq : float
        Carrier frequency (0 to 1).
    bin_edges : list of float
        Sorted bin edges, e.g. [0.001, 0.01, 0.05, 0.1, 0.5].

    Returns
    -------
    str
        Human-readable bin label, e.g. '<0.1%', '0.1-1%', '1-5%', etc.
    """
    labels = make_maf_labels(bin_edges)
    idx = int(np.digitize(freq, bin_edges))
    return labels[idx]


def maf_bin_sort_key(label: str) -> Tuple[int, float, float]:
    """Provide a stable ordering for human-readable MAF bin labels."""
    clean = label.rstrip("%")
    if clean.startswith("<"):
        return (0, 0.0, float(clean[1:]))
    if clean.startswith(">"):
        val = float(clean[1:])
        return (2, val, val)
    low, high = clean.split("-")
    return (1, float(low), float(high))


def build_variant_index(
    all_samples: List[Any],
) -> Tuple[
    Dict[VariantKey, Dict[str, Any]],
    Dict[int, Set[VariantKey]],
    int,
    int,
]:
    """
    First pass: build variant carrier index and per-sample variant sets.

    Parameters
    ----------
    all_samples : list of SampleVariants
        All samples from preprocessed data.

    Returns
    -------
    variant_info : dict
        Maps (chrom, pos) to gene, carrier_count, case_carriers, control_carriers.
    sample_variants : dict
        Maps sample index to set of (chrom, pos).
    n_cases : int
        Number of case samples.
    n_controls : int
        Number of control samples.
    """
    variant_info: Dict[VariantKey, Dict[str, Any]] = {}
    sample_variants: Dict[int, Set[VariantKey]] = {}
    n_cases = 0
    n_controls = 0

    for sample_idx, sample in enumerate(all_samples):
        label = sample.label
        if label == 1:
            n_cases += 1
        else:
            n_controls += 1

        variant_set: Set[VariantKey] = set()
        for variant in sample.variants:
            key: VariantKey = (variant.chrom, variant.pos)
            if key in variant_set:
                continue
            variant_set.add(key)

            if key not in variant_info:
                variant_info[key] = {
                    "gene": variant.gene,
                    "carrier_count": 0,
                    "case_carriers": 0,
                    "control_carriers": 0,
                }
            elif not variant_info[key]["gene"] and variant.gene:
                variant_info[key]["gene"] = variant.gene

            variant_info[key]["carrier_count"] += 1
            if label == 1:
                variant_info[key]["case_carriers"] += 1
            else:
                variant_info[key]["control_carriers"] += 1

        sample_variants[sample_idx] = variant_set

    return variant_info, sample_variants, n_cases, n_controls


def select_pairs(
    variant_info: Dict[VariantKey, Dict[str, Any]],
    top_k: int,
    max_pairs: int,
    bin_edges: List[float],
    total_samples: int,
    seed: int,
) -> List[Tuple[VariantKey, VariantKey]]:
    """
    Select variant pairs for co-occurrence evaluation.

    Selects the top-K most frequent variants, generates all pairs, and
    subsamples if needed. Additionally samples some pairs from lower-frequency
    bins to ensure coverage.

    Parameters
    ----------
    variant_info : dict
        Variant carrier index from first pass.
    top_k : int
        Number of top-frequency variants to use.
    max_pairs : int
        Maximum total pairs to return.
    bin_edges : list of float
        MAF bin edges.
    total_samples : int
        Total number of samples.
    seed : int
        Random seed.

    Returns
    -------
    list of tuple
        Selected pairs of variant keys.
    """
    rng = random.Random(seed)

    if max_pairs <= 0:
        return []

    # Sort variants by carrier_count descending
    sorted_variants = sorted(
        variant_info.keys(),
        key=lambda k: variant_info[k]["carrier_count"],
        reverse=True,
    )

    # Top-K most frequent variants
    top_variants = sorted_variants[:top_k]
    top_set = set(top_variants)

    # Generate all pairs from top-K
    all_top_pairs = list(combinations(top_variants, 2))
    logger.info(
        "Generated %d candidate pairs from top-%d variants.",
        len(all_top_pairs),
        len(top_variants),
    )

    # Reserve 10% of budget for lower-frequency pairs when the budget is large enough
    low_freq_budget = 0 if max_pairs < 10 else min(max_pairs, max(int(max_pairs * 0.1), 1))
    top_budget = max_pairs - low_freq_budget

    # Subsample top pairs if needed
    if len(all_top_pairs) > top_budget:
        rng.shuffle(all_top_pairs)
        selected_pairs = all_top_pairs[:top_budget]
    else:
        selected_pairs = all_top_pairs
        # Increase low-frequency budget with remaining slots
        low_freq_budget = max_pairs - len(selected_pairs)

    # Sample pairs from lower-frequency variants
    low_freq_variants = [
        v
        for v in sorted_variants
        if v not in top_set
        and variant_info[v]["carrier_count"] / total_samples < bin_edges[-1]
    ]

    if len(low_freq_variants) >= 2 and low_freq_budget > 0:
        # Sample individual low-frequency variants
        sample_size = min(len(low_freq_variants), top_k)
        sampled_low = rng.sample(low_freq_variants, sample_size)
        low_pairs = list(combinations(sampled_low, 2))
        if len(low_pairs) > low_freq_budget:
            rng.shuffle(low_pairs)
            low_pairs = low_pairs[:low_freq_budget]
        selected_pairs.extend(low_pairs)
        logger.info(
            "Added %d low-frequency pairs (from %d low-freq variants).",
            len(low_pairs),
            len(sampled_low),
        )

    # Deduplicate and enforce budget
    unique_pairs = []
    seen = set()
    for pair in selected_pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
        if len(unique_pairs) >= max_pairs:
            break

    logger.info("Total pairs selected: %d", len(unique_pairs))
    return unique_pairs


def compute_cooccurrence(
    pairs: List[Tuple[VariantKey, VariantKey]],
    sample_variants: Dict[int, Set[VariantKey]],
    variant_info: Dict[VariantKey, Dict[str, Any]],
    all_samples: List[Any],
    total_samples: int,
    bin_edges: List[float],
) -> List[Dict[str, Any]]:
    """
    Second pass: compute co-occurrence statistics for selected pairs.

    Parameters
    ----------
    pairs : list of tuple
        Selected variant pairs.
    sample_variants : dict
        Per-sample variant sets.
    variant_info : dict
        Variant carrier index.
    all_samples : list of SampleVariants
        All samples (for label lookup).
    total_samples : int
        Total number of samples.
    bin_edges : list of float
        MAF bin edges.

    Returns
    -------
    list of dict
        Co-occurrence records for each pair.
    """
    # Pre-build carrier sets per variant for fast intersection
    logger.info("Building per-variant carrier sets...")
    variant_carriers: Dict[VariantKey, Set[int]] = defaultdict(set)
    variant_carriers_case: Dict[VariantKey, Set[int]] = defaultdict(set)
    variant_carriers_control: Dict[VariantKey, Set[int]] = defaultdict(set)

    # Collect only the variants that appear in pairs
    needed_variants: Set[VariantKey] = set()
    for va, vb in pairs:
        needed_variants.add(va)
        needed_variants.add(vb)

    for sample_idx, vset in sample_variants.items():
        label = all_samples[sample_idx].label
        for vkey in vset:
            if vkey in needed_variants:
                variant_carriers[vkey].add(sample_idx)
                if label == 1:
                    variant_carriers_case[vkey].add(sample_idx)
                else:
                    variant_carriers_control[vkey].add(sample_idx)

    logger.info("Computing co-occurrence for %d pairs...", len(pairs))
    records: List[Dict[str, Any]] = []
    n_pairs = len(pairs)

    for i, (va, vb) in enumerate(pairs):
        if (i + 1) % 10000 == 0:
            logger.info("  Processed %d / %d pairs...", i + 1, n_pairs)

        carriers_a = variant_carriers[va]
        carriers_b = variant_carriers[vb]
        cooccur_all = carriers_a & carriers_b
        cooccur_cases = variant_carriers_case[va] & variant_carriers_case[vb]
        cooccur_controls = (
            variant_carriers_control[va] & variant_carriers_control[vb]
        )

        freq_a = variant_info[va]["carrier_count"] / total_samples
        freq_b = variant_info[vb]["carrier_count"] / total_samples
        expected = freq_a * freq_b * total_samples

        obs_exp = len(cooccur_all) / expected if expected > 0 else float("nan")

        info_a = variant_info[va]
        info_b = variant_info[vb]

        records.append(
            {
                "chrom_a": va[0],
                "pos_a": va[1],
                "gene_a": info_a["gene"],
                "chrom_b": vb[0],
                "pos_b": vb[1],
                "gene_b": info_b["gene"],
                "freq_a": round(freq_a, 6),
                "freq_b": round(freq_b, 6),
                "n_cooccur": len(cooccur_all),
                "n_cooccur_cases": len(cooccur_cases),
                "n_cooccur_controls": len(cooccur_controls),
                "expected_cooccur": round(expected, 4),
                "obs_exp_ratio": round(obs_exp, 4) if not np.isnan(obs_exp) else "NA",
                "maf_bin_a": assign_maf_bin(freq_a, bin_edges),
                "maf_bin_b": assign_maf_bin(freq_b, bin_edges),
                "same_gene": info_a["gene"] == info_b["gene"],
                "same_chrom": va[0] == vb[0],
            }
        )

    logger.info("Co-occurrence computation complete.")
    return records


def summarise_by_maf_bin(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Summarise co-occurrence statistics by MAF bin pair.

    Parameters
    ----------
    records : list of dict
        Per-pair co-occurrence records.

    Returns
    -------
    list of dict
        Summary rows grouped by (maf_bin_a, maf_bin_b).
    """
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for rec in records:
        bin_a = rec["maf_bin_a"]
        bin_b = rec["maf_bin_b"]
        if maf_bin_sort_key(bin_a) <= maf_bin_sort_key(bin_b):
            key = (bin_a, bin_b)
        else:
            key = (bin_b, bin_a)
        groups[key].append(rec["n_cooccur"])

    summaries: List[Dict[str, Any]] = []
    for (bin_a, bin_b), cooccur_counts in sorted(groups.items()):
        arr = np.array(cooccur_counts)
        n_pairs = len(arr)
        summaries.append(
            {
                "maf_bin_a": bin_a,
                "maf_bin_b": bin_b,
                "n_pairs": n_pairs,
                "mean_cooccur": round(float(np.mean(arr)), 2),
                "median_cooccur": round(float(np.median(arr)), 2),
                "n_zero_cooccur": int(np.sum(arr == 0)),
                "n_cooccur_gte5": int(np.sum(arr >= 5)),
                "n_cooccur_gte10": int(np.sum(arr >= 10)),
                "pct_testable": round(
                    float(np.sum(arr >= 5)) / n_pairs * 100, 1
                )
                if n_pairs > 0
                else 0.0,
            }
        )

    return summaries


def write_csv(
    rows: List[Dict[str, Any]], path: Path, columns: Optional[List[str]] = None
) -> None:
    """
    Write a list of dicts to CSV.

    Parameters
    ----------
    rows : list of dict
        Data rows.
    path : Path
        Output file path.
    columns : list of str, optional
        Column order. If None, uses keys from first row.
    """
    if columns is None:
        if not rows:
            logger.warning("No rows to write to %s", path)
            return
        columns = list(rows[0].keys())

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), path)


def write_summary_yaml(
    variant_info: Dict[VariantKey, Dict[str, Any]],
    total_samples: int,
    n_cases: int,
    n_controls: int,
    records: List[Dict[str, Any]],
    bin_summaries: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Write high-level summary statistics as YAML.

    Parameters
    ----------
    variant_info : dict
        Variant carrier index.
    total_samples : int
        Total number of samples.
    n_cases : int
        Number of case samples.
    n_controls : int
        Number of control samples.
    records : list of dict
        Per-pair co-occurrence records.
    output_path : Path
        Output YAML file path.
    """
    cooccur_values = [r["n_cooccur"] for r in records]
    arr = np.array(cooccur_values) if cooccur_values else np.array([0])

    n_zero = int(np.sum(arr == 0))
    n_testable = int(np.sum(arr >= 5))
    pct_testable = round(n_testable / len(arr) * 100, 1) if len(arr) > 0 else 0.0

    # Determine headline conclusion
    if pct_testable >= 30:
        conclusion = (
            "Sufficient co-occurrence for epistasis detection. "
            f"{pct_testable}% of evaluated pairs have >= 5 co-occurrences."
        )
    elif pct_testable >= 10:
        conclusion = (
            "Marginal co-occurrence for epistasis detection. "
            f"Only {pct_testable}% of evaluated pairs have >= 5 co-occurrences. "
            "Detection power is limited."
        )
    else:
        conclusion = (
            "Insufficient co-occurrence for epistasis detection. "
            f"Only {pct_testable}% of evaluated pairs have >= 5 co-occurrences. "
            "The cohort size and variant frequencies do not support reliable "
            "pairwise interaction testing."
        )

    summary = {
        "cohort": {
            "total_samples": total_samples,
            "n_cases": n_cases,
            "n_controls": n_controls,
        },
        "variants": {
            "total_unique_variants": len(variant_info),
            "mean_carrier_count": round(
                float(np.mean([v["carrier_count"] for v in variant_info.values()])), 2
            ),
            "median_carrier_count": round(
                float(np.median([v["carrier_count"] for v in variant_info.values()])),
                2,
            ),
        },
        "cooccurrence": {
            "total_pairs_evaluated": len(records),
            "n_zero_cooccurrence": n_zero,
            "pct_zero_cooccurrence": round(
                n_zero / len(arr) * 100, 1
            )
            if len(arr) > 0
            else 0.0,
            "n_pairs_gte5_cooccur": n_testable,
            "pct_testable": pct_testable,
            "n_pairs_gte10_cooccur": int(np.sum(arr >= 10)),
            "distribution": {
                "min": int(np.min(arr)),
                "p25": round(float(np.percentile(arr, 25)), 1),
                "median": round(float(np.median(arr)), 1),
                "p75": round(float(np.percentile(arr, 75)), 1),
                "max": int(np.max(arr)),
                "mean": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr)), 2),
            },
        },
        "conclusion": conclusion,
    }

    if bin_summaries:
        best_bin = max(bin_summaries, key=lambda row: row["pct_testable"])
        worst_bin = max(bin_summaries, key=lambda row: row["n_zero_cooccur"])
        summary["maf_bin_highlights"] = {
            "most_testable_bin_pair": {
                "maf_bin_a": best_bin["maf_bin_a"],
                "maf_bin_b": best_bin["maf_bin_b"],
                "pct_testable": best_bin["pct_testable"],
                "n_pairs": best_bin["n_pairs"],
            },
            "highest_zero_cooccurrence_bin_pair": {
                "maf_bin_a": worst_bin["maf_bin_a"],
                "maf_bin_b": worst_bin["maf_bin_b"],
                "n_zero_cooccur": worst_bin["n_zero_cooccur"],
                "n_pairs": worst_bin["n_pairs"],
            },
        }

    with open(output_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    logger.info("Wrote summary to %s", output_path)


def main() -> None:
    """Run the co-occurrence audit."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    bin_edges = [float(x) for x in args.maf_bins.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load preprocessed data
    logger.info("Loading preprocessed data from %s ...", args.preprocessed_data)
    preprocessed = torch.load(args.preprocessed_data, weights_only=False)
    all_samples = preprocessed["samples"]
    metadata = preprocessed.get("metadata", {})
    total_samples = len(all_samples)
    logger.info("Loaded %d samples.", total_samples)
    if metadata:
        logger.info("Metadata: %s", metadata)

    # First pass: build variant index
    logger.info("Building variant carrier index...")
    variant_info, sample_variants, n_cases, n_controls = build_variant_index(
        all_samples
    )
    logger.info(
        "Found %d unique variants across %d samples (%d cases, %d controls).",
        len(variant_info),
        total_samples,
        n_cases,
        n_controls,
    )

    # Compute carrier frequencies and assign MAF bins
    freq_rows: List[Dict[str, Any]] = []
    for (chrom, pos), info in sorted(variant_info.items()):
        carrier_freq = info["carrier_count"] / total_samples
        freq_rows.append(
            {
                "chrom": chrom,
                "pos": pos,
                "gene": info["gene"],
                "carrier_count": info["carrier_count"],
                "case_carriers": info["case_carriers"],
                "control_carriers": info["control_carriers"],
                "carrier_freq": round(carrier_freq, 6),
                "maf_bin": assign_maf_bin(carrier_freq, bin_edges),
            }
        )

    write_csv(
        freq_rows,
        output_dir / "variant_frequencies.csv",
        columns=[
            "chrom",
            "pos",
            "gene",
            "carrier_count",
            "case_carriers",
            "control_carriers",
            "carrier_freq",
            "maf_bin",
        ],
    )

    # Select pairs
    logger.info("Selecting variant pairs for evaluation...")
    pairs = select_pairs(
        variant_info,
        top_k=args.top_k_variants,
        max_pairs=args.max_pairs,
        bin_edges=bin_edges,
        total_samples=total_samples,
        seed=args.seed,
    )

    # Compute co-occurrence
    records = compute_cooccurrence(
        pairs, sample_variants, variant_info, all_samples, total_samples, bin_edges
    )

    # Write per-pair results
    write_csv(
        records,
        output_dir / "cooccurrence_per_pair.csv",
        columns=[
            "chrom_a",
            "pos_a",
            "gene_a",
            "chrom_b",
            "pos_b",
            "gene_b",
            "freq_a",
            "freq_b",
            "n_cooccur",
            "n_cooccur_cases",
            "n_cooccur_controls",
            "expected_cooccur",
            "obs_exp_ratio",
            "maf_bin_a",
            "maf_bin_b",
            "same_gene",
            "same_chrom",
        ],
    )

    # Summarise by MAF bin
    bin_summaries = summarise_by_maf_bin(records)
    write_csv(
        bin_summaries,
        output_dir / "cooccurrence_by_maf_bin.csv",
        columns=[
            "maf_bin_a",
            "maf_bin_b",
            "n_pairs",
            "mean_cooccur",
            "median_cooccur",
            "n_zero_cooccur",
            "n_cooccur_gte5",
            "n_cooccur_gte10",
            "pct_testable",
        ],
    )

    # Write summary YAML
    write_summary_yaml(
        variant_info,
        total_samples,
        n_cases,
        n_controls,
        records,
        bin_summaries,
        output_dir / "cooccurrence_summary.yaml",
    )

    logger.info("Co-occurrence audit complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
