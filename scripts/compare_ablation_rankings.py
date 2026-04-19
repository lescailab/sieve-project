#!/usr/bin/env python3
"""
Compare null-contrasted variant rankings across annotation levels L0-L3.

After running the per-level null baseline workflow, this script compares the
resulting significance-annotated variant rankings to quantify how much the
discovered variants depend on the annotation information provided. By default
it ranks variants by ``z_attribution`` from
``corrected_variant_rankings.csv`` (produced by ``correct_chrx_bias.py``).
Higher z-scores are treated as better ranks automatically. Key analyses:

1. Jaccard similarity matrices at multiple top-k thresholds
2. Level-specific variant discovery (high rank at one level, low at others)

Note on score column choice
---------------------------
``z_attribution`` is the recommended column for cross-level comparison
because it is comparable across chromosomes (per-chromosome z-normalised)
and is not subject to the resolution-floor problem of empirical p-values
(see KNOWN_LIMITATIONS.md).  ``empirical_p_variant`` is bounded below by
1/(N_null + 1), which pins most real variants at the floor when the model
is informative, making top-K selection a random draw from a tied set.

Usage:
    # From a directory with L{0..3}_sieve_variant_rankings.csv files
    python scripts/compare_ablation_rankings.py \\
        --ranking-dir results/ablation \\
        --score-column z_attribution \\
        --out-comparison ablation_ranking_comparison.yaml

    # With explicit per-level paths (using chrX-corrected files which contain z_attribution)
    python scripts/compare_ablation_rankings.py \\
        --rankings L0:results/null_baseline_L0/results/attribution_comparison/corrected/corrected_variant_rankings.csv \\
                   L1:results/null_baseline_L1/results/attribution_comparison/corrected/corrected_variant_rankings.csv \\
                   L2:results/null_baseline_L2/results/attribution_comparison/corrected/corrected_variant_rankings.csv \\
        --score-column z_attribution \\
        --out-comparison ablation_ranking_comparison.yaml

Author: Francesco Lescai
"""
from __future__ import annotations

import argparse
import csv
import itertools
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


LEVEL_ORDER = ["L0", "L1", "L2", "L3"]

# ---------------------------------------------------------------------------
# YAML output helper
# ---------------------------------------------------------------------------


def dump_yaml(value: Any, path: pathlib.Path) -> None:
    """
    Write a Python object as YAML to *path*.

    Parameters
    ----------
    value : Any
        Data structure to serialise.
    path : pathlib.Path
        Destination file.
    """
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


# ---------------------------------------------------------------------------
# CSV loading — flexible column matching
# ---------------------------------------------------------------------------

# Primary SIEVE columns produced by src/explain/variant_ranking.py and the
# null-contrast workflow.
VARIANT_ID_COLUMNS = [
    "variant_id",
    "feature",
    "feature_id",
]
SCORE_COLUMNS = [
    "empirical_p_variant",
    "fdr_variant",
    "z_attribution",
    "corrected_rank",
    "mean_attribution",
    "score",
    "attribution",
    "max_attribution",
    "mean_score",
    "importance",
]
GENE_COLUMNS = ["gene_name", "gene", "gene_symbol"]
GENE_ID_COLUMNS = ["gene_id"]
CHROM_COLUMNS = ["chromosome", "chrom", "chr"]
POS_COLUMNS = ["position", "pos", "start"]
_DESCENDING_RANK_COLUMNS = frozenset({"delta_rank"})


def _find_column(headers: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first matching column name (case-insensitive)."""
    lower_headers = {h.lower(): h for h in headers}
    for candidate in candidates:
        if candidate.lower() in lower_headers:
            return lower_headers[candidate.lower()]
    return None


def _build_variant_id(row: Dict[str, str], headers: List[str]) -> Optional[str]:
    """
    Build a unique variant identifier from a ranking CSV row.

    Preferred format is ``{chrom}:{pos}_{gene_id}`` which matches SIEVE's
    chromosome-aware variant keying and prevents position collisions across
    chromosomes.
    """
    # Try explicit variant_id column first
    vid_col = _find_column(headers, VARIANT_ID_COLUMNS)
    if vid_col and row.get(vid_col, "").strip():
        return row[vid_col].strip()

    # Build from components (chrom + pos + gene_id)
    chrom_col = _find_column(headers, CHROM_COLUMNS)
    pos_col = _find_column(headers, POS_COLUMNS)
    gene_id_col = _find_column(headers, GENE_ID_COLUMNS)

    if chrom_col and pos_col:
        chrom = row.get(chrom_col, "").strip()
        pos = row.get(pos_col, "").strip()
        gene_id = row.get(gene_id_col, "").strip() if gene_id_col else ""
        if chrom and pos:
            base = f"{chrom}:{pos}"
            if gene_id:
                return f"{base}_{gene_id}"
            return base

    return None


def _resolve_score_column(
    headers: List[str],
    score_column: Optional[str] = None,
) -> Tuple[str, bool]:
    """Resolve which score column to use for a given set of headers.

    Parameters
    ----------
    headers : List[str]
        Column headers from the CSV.
    score_column : str, optional
        Explicit column name override.

    Returns
    -------
    Tuple[str, bool]
        (resolved_column_name, was_explicit). *was_explicit* is True when
        the returned column came from *score_column* rather than auto-detection.
    """
    if score_column is not None:
        lower_headers = {h.lower(): h for h in headers}
        if score_column.lower() in lower_headers:
            return lower_headers[score_column.lower()], True
        raise ValueError(
            f"--score-column '{score_column}' not found in headers: {headers}"
        )

    # Auto-detect from SCORE_COLUMNS list
    col = _find_column(headers, SCORE_COLUMNS)
    return (col or ""), False


def _score_column_is_ascending(score_column: str) -> bool:
    """Return True when lower values indicate stronger ranking."""
    lowered = score_column.lower()
    if lowered in _DESCENDING_RANK_COLUMNS:
        return False
    return (
        lowered.startswith("empirical_p")
        or lowered.startswith("fdr")
        or lowered.endswith("_rank")
        or lowered == "rank"
    )


def _get_score(
    row: Dict[str, str],
    resolved_col: str,
    missing_value: float,
) -> float:
    """Extract attribution score from a row using a pre-resolved column.

    Parameters
    ----------
    row : Dict[str, str]
        A single CSV row.
    resolved_col : str
        Column name (already resolved by :func:`_resolve_score_column`).

    Returns
    -------
    float
        The numeric score value, or *missing_value* on failure.
    """
    if resolved_col:
        try:
            return float(row[resolved_col])
        except (ValueError, TypeError, KeyError):
            return missing_value
    return missing_value


def _get_field(
    row: Dict[str, str], headers: List[str], candidates: List[str]
) -> str:
    """Get a field value from a row, trying multiple column names."""
    col = _find_column(headers, candidates)
    if col:
        return row.get(col, "").strip()
    return ""


def load_rankings(
    csv_path: pathlib.Path,
    score_column: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, bool]:
    """
    Load a variant ranking CSV and return a list of dicts sorted by score.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to a SIEVE variant ranking CSV.
    score_column : str, optional
        Explicit column name to use for ranking. When *None*, auto-detects
        from :data:`SCORE_COLUMNS`.

    Returns
    -------
    Tuple[List[Dict[str, Any]], str, bool]
        (records, resolved_column_name, was_explicit). Records have keys
        ``variant_id``, ``score``, ``gene``, ``chrom``, ``pos``, ``rank``.
    """
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        resolved_col, was_explicit = _resolve_score_column(headers, score_column)
        ascending = _score_column_is_ascending(resolved_col)
        missing_value = float("inf") if ascending else float("-inf")
        rows: List[Dict[str, Any]] = []
        for row in reader:
            vid = _build_variant_id(row, headers)
            if vid is None:
                continue
            rows.append(
                {
                    "variant_id": vid,
                    "score": _get_score(row, resolved_col, missing_value),
                    "gene": _get_field(row, headers, GENE_COLUMNS),
                    "chrom": _get_field(row, headers, CHROM_COLUMNS),
                    "pos": _get_field(row, headers, POS_COLUMNS),
                }
            )

    rows.sort(key=lambda r: r["score"], reverse=not ascending)
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows, resolved_col, was_explicit


def find_ranking_files(ranking_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    """
    Discover variant ranking CSVs in *ranking_dir*.

    Looks for filenames matching ``L{0,1,2,3}_sieve_variant_rankings.csv``
    first, then falls back to more flexible globbing.

    Parameters
    ----------
    ranking_dir : pathlib.Path
        Directory to scan.

    Returns
    -------
    Dict[str, pathlib.Path]
        Mapping of level label to file path.
    """
    level_files: Dict[str, pathlib.Path] = {}
    for level in LEVEL_ORDER:
        # Exact match first
        pattern = f"{level}_sieve_variant_rankings.csv"
        candidates = list(ranking_dir.glob(pattern))
        if candidates:
            level_files[level] = candidates[0]
            continue
        # Flexible match
        for f in ranking_dir.glob(f"{level}_*variant*rank*.csv"):
            level_files[level] = f
            break
    return level_files


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def compute_jaccard(set_a: Set[str], set_b: Set[str]) -> Tuple[float, int, int]:
    """
    Compute Jaccard similarity between two sets.

    Parameters
    ----------
    set_a, set_b : Set[str]
        Sets of variant IDs.

    Returns
    -------
    Tuple[float, int, int]
        (jaccard_index, intersection_size, union_size)
    """
    if not set_a and not set_b:
        return 0.0, 0, 0
    intersection = set_a & set_b
    union = set_a | set_b
    jaccard = len(intersection) / len(union) if union else 0.0
    return jaccard, len(intersection), len(union)


def compute_jaccard_matrices(
    level_rankings: Dict[str, List[Dict[str, Any]]],
    top_k_values: List[int],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Compute pairwise Jaccard similarity matrices for each top-k threshold.

    Parameters
    ----------
    level_rankings : Dict[str, List[Dict[str, Any]]]
        Rankings per annotation level.
    top_k_values : List[int]
        Top-k thresholds to evaluate.

    Returns
    -------
    Dict[int, List[Dict[str, Any]]]
        Mapping of top_k to list of pairwise comparison records.
    """
    matrices: Dict[int, List[Dict[str, Any]]] = {}
    levels = sorted(
        level_rankings.keys(),
        key=lambda l: LEVEL_ORDER.index(l) if l in LEVEL_ORDER else 999,
    )

    for top_k in top_k_values:
        top_sets: Dict[str, Set[str]] = {}
        for level in levels:
            ranked = level_rankings[level]
            top_sets[level] = {r["variant_id"] for r in ranked[:top_k]}

        rows: List[Dict[str, Any]] = []
        for la, lb in itertools.combinations(levels, 2):
            jaccard, overlap, union_size = compute_jaccard(
                top_sets[la], top_sets[lb]
            )
            rows.append(
                {
                    "top_k": top_k,
                    "level_a": la,
                    "level_b": lb,
                    "jaccard": round(jaccard, 4),
                    "overlap": overlap,
                    "size_a": len(top_sets[la]),
                    "size_b": len(top_sets[lb]),
                    "union": union_size,
                }
            )
        matrices[top_k] = rows

    return matrices


def find_level_specific_variants(
    level_rankings: Dict[str, List[Dict[str, Any]]],
    high_rank_threshold: int,
    low_rank_threshold: int,
) -> List[Dict[str, Any]]:
    """
    Find variants ranked highly at one level but poorly at all others.

    A variant is *level-specific* if it appears in the top
    ``high_rank_threshold`` at exactly one level and is outside the top
    ``low_rank_threshold`` at every other level.

    Parameters
    ----------
    level_rankings : Dict[str, List[Dict[str, Any]]]
        Rankings per annotation level.
    high_rank_threshold : int
        Must be within this rank at the specific level.
    low_rank_threshold : int
        Must be outside this rank at all other levels.

    Returns
    -------
    List[Dict[str, Any]]
        Level-specific variant records.
    """
    levels = sorted(
        level_rankings.keys(),
        key=lambda l: LEVEL_ORDER.index(l) if l in LEVEL_ORDER else 999,
    )

    # Build rank lookup: level -> variant_id -> rank
    rank_lookup: Dict[str, Dict[str, int]] = {}
    info_lookup: Dict[str, Dict[str, Any]] = {}
    total_variants: Dict[str, int] = {}

    for level in levels:
        ranked = level_rankings[level]
        total_variants[level] = len(ranked)
        rank_lookup[level] = {}
        for r in ranked:
            rank_lookup[level][r["variant_id"]] = r["rank"]
            if r["variant_id"] not in info_lookup:
                info_lookup[r["variant_id"]] = {
                    "gene": r.get("gene", ""),
                    "chrom": r.get("chrom", ""),
                    "pos": r.get("pos", ""),
                }

    results: List[Dict[str, Any]] = []
    for level in levels:
        other_levels = [l for l in levels if l != level]
        ranked = level_rankings[level]
        top_at_level = [r for r in ranked if r["rank"] <= high_rank_threshold]

        for variant_row in top_at_level:
            vid = variant_row["variant_id"]
            is_specific = True
            for other in other_levels:
                other_rank = rank_lookup[other].get(
                    vid, total_variants[other] + 1
                )
                if other_rank <= low_rank_threshold:
                    is_specific = False
                    break

            if is_specific:
                info = info_lookup.get(vid, {})
                entry: Dict[str, Any] = {
                    "variant_id": vid,
                    "gene": info.get("gene", ""),
                    "chrom": info.get("chrom", ""),
                    "pos": info.get("pos", ""),
                    "specific_to_level": level,
                    "rank_at_specific_level": variant_row["rank"],
                    "score_at_specific_level": variant_row["score"],
                }
                # Cross-level ranks
                for l in LEVEL_ORDER:
                    if l in rank_lookup:
                        entry[f"rank_at_{l}"] = rank_lookup[l].get(
                            vid, total_variants.get(l, 0) + 1
                        )
                results.append(entry)

    return results


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ranking-dir",
        type=str,
        help=(
            "Directory containing variant ranking CSVs named as "
            "L{0,1,2,3}_sieve_variant_rankings.csv"
        ),
    )
    group.add_argument(
        "--rankings",
        nargs="+",
        metavar="LEVEL:PATH",
        help=(
            "Explicit per-level ranking files, e.g. "
            "L0:results/L0/sieve_variant_rankings.csv "
            "L1:results/L1/sieve_variant_rankings.csv"
        ),
    )
    parser.add_argument(
        "--top-k",
        default="50,100,200,500",
        help="Comma-separated top-k values for Jaccard computation (default: 50,100,200,500)",
    )
    parser.add_argument(
        "--high-rank-threshold",
        type=int,
        default=100,
        help="Threshold for high-ranking variants (default: 100)",
    )
    parser.add_argument(
        "--low-rank-threshold",
        type=int,
        default=500,
        help="Threshold for low-ranking variants at other levels (default: 500)",
    )
    parser.add_argument(
        "--out-comparison",
        default="ablation_ranking_comparison.yaml",
        help="Output YAML summary path",
    )
    parser.add_argument(
        "--out-jaccard",
        default="ablation_jaccard_matrix.tsv",
        help="Output Jaccard matrix TSV path",
    )
    parser.add_argument(
        "--out-level-specific",
        default="level_specific_variants.tsv",
        help="Output level-specific variants TSV path",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="z_attribution",
        help=(
            "Column name to use for ranking variants. "
            "Defaults to z_attribution (per-chromosome z-normalised attribution "
            "from correct_chrx_bias.py), which is recommended for cross-level "
            "comparison because it is not subject to the empirical p-value "
            "resolution floor (see KNOWN_LIMITATIONS.md). "
            "Columns such as empirical_p_variant, fdr_variant, and corrected_rank "
            "are ranked ascending automatically; attribution-like scores are "
            "ranked descending."
        ),
    )
    return parser.parse_args()


def _parse_rankings_arg(rankings: List[str]) -> Dict[str, pathlib.Path]:
    """Parse ``LEVEL:PATH`` arguments into a dict."""
    level_files: Dict[str, pathlib.Path] = {}
    for item in rankings:
        if ":" not in item:
            raise ValueError(
                f"Invalid --rankings format '{item}'. Expected LEVEL:PATH, "
                f"e.g. L0:results/L0/sieve_variant_rankings.csv"
            )
        level, path_str = item.split(":", 1)
        level = level.upper()
        if level not in LEVEL_ORDER:
            print(
                f"WARNING: Level '{level}' not in expected order {LEVEL_ORDER}",
                file=sys.stderr,
            )
        path = pathlib.Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Ranking file not found: {path}")
        level_files[level] = path
    return level_files


def main() -> int:
    """Entry point for the ablation ranking comparison."""
    args = parse_args()

    if args.score_column == "empirical_p_variant":
        print(
            "Warning: empirical_p_variant may be at the resolution floor for "
            "high-information annotation levels (median p pinned to 1/(N+1)), "
            "making top-K selection a draw from a tied set. "
            "Consider z_attribution for cross-level comparison "
            "(see KNOWN_LIMITATIONS.md).",
            file=sys.stderr,
        )

    top_k_values = [int(k.strip()) for k in args.top_k.split(",")]

    # Discover ranking files
    if args.ranking_dir:
        ranking_dir = pathlib.Path(args.ranking_dir)
        level_files = find_ranking_files(ranking_dir)
    else:
        level_files = _parse_rankings_arg(args.rankings)

    if not level_files:
        print(
            "ERROR: No ranking files found. Check --ranking-dir or --rankings.",
            file=sys.stderr,
        )
        return 1

    levels_found = sorted(
        level_files.keys(),
        key=lambda l: LEVEL_ORDER.index(l) if l in LEVEL_ORDER else 999,
    )
    print(
        f"Found ranking files for levels: {', '.join(levels_found)}",
        file=sys.stderr,
    )

    # Load all rankings
    level_rankings: Dict[str, List[Dict[str, Any]]] = {}
    resolved_score_col = ""
    score_was_explicit = False
    for level, fpath in level_files.items():
        try:
            rankings, col_name, was_explicit = load_rankings(
                fpath, score_column=args.score_column
            )
        except ValueError as exc:
            print(
                f"ERROR: Failed to load rankings for {level} from {fpath}: {exc}",
                file=sys.stderr,
            )
            return 1
        level_rankings[level] = rankings
        if not resolved_score_col and col_name:
            resolved_score_col = col_name
            score_was_explicit = was_explicit
        print(
            f"  {level}: {len(level_rankings[level])} variants from {fpath.name}",
            file=sys.stderr,
        )

    # Log which score column is in use
    if resolved_score_col:
        source = "from --score-column" if score_was_explicit else "auto-detected"
        print(
            f"Score column: {resolved_score_col} ({source}, "
            f"{'ascending' if _score_column_is_ascending(resolved_score_col) else 'descending'})",
            file=sys.stderr,
        )

    if len(level_rankings) < 2:
        print(
            f"WARNING: Need at least 2 levels for comparison, found {len(level_rankings)}",
            file=sys.stderr,
        )

    # Compute Jaccard matrices
    jaccard_matrices = compute_jaccard_matrices(level_rankings, top_k_values)

    # Find level-specific variants
    level_specific = find_level_specific_variants(
        level_rankings,
        args.high_rank_threshold,
        args.low_rank_threshold,
    )

    # Write Jaccard TSV
    jaccard_path = pathlib.Path(args.out_jaccard)
    jaccard_path.parent.mkdir(parents=True, exist_ok=True)
    with jaccard_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            ["top_k", "level_a", "level_b", "jaccard", "overlap", "size_a", "size_b", "union"]
        )
        for top_k in top_k_values:
            for row in jaccard_matrices.get(top_k, []):
                writer.writerow(
                    [
                        row["top_k"],
                        row["level_a"],
                        row["level_b"],
                        row["jaccard"],
                        row["overlap"],
                        row["size_a"],
                        row["size_b"],
                        row["union"],
                    ]
                )

    # Write level-specific variants TSV
    level_specific_path = pathlib.Path(args.out_level_specific)
    level_specific_path.parent.mkdir(parents=True, exist_ok=True)
    rank_cols = [f"rank_at_{l}" for l in LEVEL_ORDER if l in level_rankings]
    fieldnames = [
        "variant_id",
        "gene",
        "chrom",
        "pos",
        "specific_to_level",
        "rank_at_specific_level",
    ] + rank_cols + ["score_at_specific_level"]

    with level_specific_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in level_specific:
            writer.writerow(row)

    # Write YAML summary
    yaml_summary: Dict[str, Any] = {
        "levels_analysed": levels_found,
        "top_k_values": top_k_values,
        "high_rank_threshold": args.high_rank_threshold,
        "low_rank_threshold": args.low_rank_threshold,
        "score_column": resolved_score_col,
        "score_sort_order": (
            "ascending" if resolved_score_col and _score_column_is_ascending(resolved_score_col)
            else "descending"
        ),
        "variants_per_level": {
            level: len(rankings) for level, rankings in level_rankings.items()
        },
        "jaccard_matrices": {},
        "level_specific_variant_counts": {},
    }

    for top_k in top_k_values:
        key = f"top_{top_k}"
        yaml_summary["jaccard_matrices"][key] = {}
        for row in jaccard_matrices.get(top_k, []):
            pair_key = f"{row['level_a']}_vs_{row['level_b']}"
            yaml_summary["jaccard_matrices"][key][pair_key] = {
                "jaccard": row["jaccard"],
                "overlap": row["overlap"],
                "union": row["union"],
            }

    for level in levels_found:
        count = sum(1 for v in level_specific if v["specific_to_level"] == level)
        yaml_summary["level_specific_variant_counts"][level] = count

    yaml_summary["total_level_specific_variants"] = len(level_specific)

    comparison_path = pathlib.Path(args.out_comparison)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(yaml_summary, comparison_path)

    print(f"Jaccard matrix written to {args.out_jaccard}", file=sys.stderr)
    print(
        f"Level-specific variants written to {args.out_level_specific} "
        f"({len(level_specific)} variants)",
        file=sys.stderr,
    )
    print(f"Comparison summary written to {args.out_comparison}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
