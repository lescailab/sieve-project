#!/usr/bin/env python3
"""
Non-linear classifier validation of SIEVE gene sets.

Tests whether SIEVE-identified gene sets carry non-linear discriminative
information in independent validation cohorts by training a classifier on
the per-gene burden vector and comparing performance against a null
distribution from random gene sets of equal size.

Usage:
    python scripts/validate_nonlinear_classifier.py \
        --burden-matrix /path/to/gene_burden_matrix.parquet \
        --sieve-genes /path/to/sieve_genes_L1.tsv \
        --phenotypes /path/to/phenotypes.tsv \
        --output-dir /path/to/nonlinear_validation/ \
        --top-k 50 100 200 500 \
        --n-permutations 1000 \
        --cv-folds 5 \
        --seed 42 \
        --n-jobs 4
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.vcf_parser import load_phenotypes


LEVEL_ORDER = ["L0", "L1", "L2", "L3"]
CLASSIFIER_ORDER = ["rf", "lr"]
CLASSIFIER_LABELS = {
    "rf": "random_forest",
    "lr": "logistic_regression",
}
CV_REPEATS = 3
PERMUTATION_PROGRESS_EVERY = 100

FoldIndices = List[Tuple[np.ndarray, np.ndarray]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Non-linear classifier validation of SIEVE gene sets.",
    )
    parser.add_argument(
        "--burden-matrix",
        type=Path,
        required=True,
        help=(
            "Path to gene-burden matrix parquet file, or a directory containing "
            "gene_burden_matrix*.parquet outputs"
        ),
    )
    parser.add_argument(
        "--sieve-genes",
        type=Path,
        required=True,
        help=(
            "Path to SIEVE gene list TSV (single level), or directory "
            "containing sieve_genes_L{0,1,2,3}.tsv files (multi-level mode)"
        ),
    )
    parser.add_argument(
        "--phenotypes",
        type=Path,
        required=True,
        help="Phenotype TSV (sample_id \\t phenotype: 1=ctrl, 2=case)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[100],
        help="Gene set size(s) (default: 100)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of random gene set permutations (default: 1000)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel jobs (default: 4)",
    )
    parser.add_argument(
        "--consequence",
        default="total",
        choices=["total", "missense", "lof"],
        help="Which burden matrix to use (default: total)",
    )
    parser.add_argument(
        "--classifiers",
        default="rf",
        choices=["rf", "lr", "both"],
        help="Classifier(s): rf, lr, or both (default: rf)",
    )
    parser.add_argument(
        "--also-export-csv",
        action="store_true",
        default=False,
        help="Export SIEVE feature matrices as CSV for external analysis",
    )
    return parser.parse_args(argv)


def level_sort_key(level: str) -> tuple[int, str]:
    """Sort known ablation levels first, preserving L0-L3 order."""
    if level in LEVEL_ORDER:
        return LEVEL_ORDER.index(level), level
    return len(LEVEL_ORDER), level


def resolve_burden_matrix_path(path: Path, consequence: str) -> Path:
    """
    Resolve the requested burden matrix.

    Supports either:
    - a direct parquet path
    - a directory containing extract_validation_burden.py outputs
    """
    expected_name = (
        "gene_burden_matrix.parquet"
        if consequence == "total"
        else f"gene_burden_matrix_{consequence}.parquet"
    )

    if path.is_dir():
        resolved = path / expected_name
    else:
        resolved = path
        if consequence != "total" and path.name != expected_name:
            candidate = path.with_name(expected_name)
            if candidate.exists():
                resolved = candidate

    if not resolved.exists():
        raise FileNotFoundError(
            f"Burden matrix not found for consequence '{consequence}': {resolved}"
        )

    return resolved


def load_burden_matrix(path: Path) -> pd.DataFrame:
    """Load gene-burden matrix from parquet, ensuring sample IDs are the index."""
    df = pd.read_parquet(path)
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    df.index = df.index.astype(str)
    df.columns = [str(col) for col in df.columns]
    return df


def load_sieve_genes(gene_file: Path) -> pd.DataFrame:
    """Load a SIEVE gene list TSV or CSV."""
    sep = "\t"
    with open(gene_file) as handle:
        first_line = handle.readline()
        if "," in first_line and "\t" not in first_line:
            sep = ","

    df = pd.read_csv(gene_file, sep=sep)
    required = {"gene_name", "gene_rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gene list missing required columns: {sorted(missing)}")

    return df.sort_values("gene_rank").reset_index(drop=True)


def detect_sieve_gene_files(sieve_genes_path: Path) -> Dict[str, Path]:
    """Detect one or more SIEVE gene list files."""
    if sieve_genes_path.is_file():
        match = re.search(r"L(\d+)", sieve_genes_path.name)
        level = f"L{match.group(1)}" if match else "single"
        return {level: sieve_genes_path}

    if sieve_genes_path.is_dir():
        files: Dict[str, Path] = {}
        # Match both naming conventions:
        #   L0_sieve_genes.tsv  (from --ablation-level L0)
        #   sieve_genes_L0.tsv
        for gene_file in sorted(sieve_genes_path.glob("*sieve_genes*.tsv")):
            match = re.search(r"L(\d+)", gene_file.name)
            if match:
                level = f"L{match.group(1)}"
                if level not in files:
                    files[level] = gene_file
        if not files:
            raise FileNotFoundError(
                f"No SIEVE gene list files (L*_sieve_genes*.tsv or "
                f"sieve_genes_L*.tsv) found in {sieve_genes_path}"
            )
        return files

    raise FileNotFoundError(f"Path does not exist: {sieve_genes_path}")


def build_classifier(classifier_name: str, seed: int) -> Pipeline | RandomForestClassifier:
    """Construct a fresh classifier instance with a deterministic seed."""
    if classifier_name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )

    if classifier_name == "lr":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        l1_ratio=0,
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported classifier: {classifier_name}")


def generate_fixed_folds(
    y: np.ndarray,
    cv_folds: int,
    seed: int,
    n_repeats: int = CV_REPEATS,
) -> FoldIndices:
    """Generate fixed repeated stratified folds for reuse across all evaluations."""
    splitter = RepeatedStratifiedKFold(
        n_splits=cv_folds,
        n_repeats=n_repeats,
        random_state=seed,
    )
    return list(splitter.split(np.zeros(len(y), dtype=np.float64), y))


def evaluate_gene_set(
    X: np.ndarray,
    y: np.ndarray,
    fold_indices: FoldIndices,
    classifier_name: str,
    model_seed: int,
    warn_context: str = "",
) -> np.ndarray:
    """
    Evaluate a gene set using pre-defined CV folds.

    Returns an array of per-fold AUC values. Failed folds are recorded as NaN.
    """
    aucs: list[float] = []
    context = warn_context or classifier_name

    for fold_number, (train_idx, test_idx) in enumerate(fold_indices, start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        clf = build_classifier(classifier_name, model_seed)

        try:
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_prob))
        except ValueError as exc:
            print(
                f"  WARNING: AUC computation failed for {context} fold {fold_number}: {exc}",
                file=sys.stderr,
            )
            auc = float("nan")

        aucs.append(auc)

    return np.asarray(aucs, dtype=np.float64)


def run_single_permutation(
    perm_idx: int,
    burden_values: np.ndarray,
    n_genes_total: int,
    k_effective: int,
    y: np.ndarray,
    fold_indices: FoldIndices,
    classifier_name: str,
    seed: int,
    n_permutations: int,
) -> float:
    """Sample one random gene set and return its mean AUC."""
    rng = np.random.RandomState(seed + perm_idx)
    random_cols = rng.choice(n_genes_total, size=k_effective, replace=False)
    X_random = burden_values[:, random_cols]

    permutation_seed = seed + n_permutations + perm_idx
    aucs = evaluate_gene_set(
        X_random,
        y,
        fold_indices,
        classifier_name,
        permutation_seed,
        warn_context=f"{classifier_name} permutation {perm_idx + 1}",
    )
    return float(np.nanmean(aucs))


def format_duration(seconds: float) -> str:
    """Format a duration in a compact human-readable form."""
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, mins = divmod(minutes, 60)

    if hours:
        return f"{hours}h {mins}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def summarize_missing_genes(missing_genes: Sequence[str], preview: int = 20) -> str:
    """Format missing-gene diagnostics for stdout logging."""
    if not missing_genes:
        return "none"

    shown = ", ".join(missing_genes[:preview])
    if len(missing_genes) > preview:
        return f"{shown}, ... ({len(missing_genes)} total)"
    return shown


def select_sieve_features(
    burden_matrix: pd.DataFrame,
    sieve_genes_df: pd.DataFrame,
    top_k_requested: int,
) -> Dict[str, Any]:
    """Match top-ranked SIEVE genes to burden-matrix columns."""
    ranked = sieve_genes_df.sort_values("gene_rank").head(top_k_requested).copy()
    selected_gene_names = [str(gene) for gene in ranked["gene_name"].tolist()]

    burden_cols_upper = {str(column).upper(): str(column) for column in burden_matrix.columns}
    matched_genes: list[str] = []
    missing_genes: list[str] = []
    seen_matches: set[str] = set()

    for gene_name in selected_gene_names:
        gene_upper = gene_name.upper()
        if gene_upper in burden_cols_upper and gene_upper not in seen_matches:
            matched_genes.append(burden_cols_upper[gene_upper])
            seen_matches.add(gene_upper)
        elif gene_upper not in burden_cols_upper:
            missing_genes.append(gene_name)

    X_sieve = burden_matrix.loc[:, matched_genes].to_numpy(dtype=np.float64, copy=True)

    return {
        "top_k_requested": int(top_k_requested),
        "top_k_used": int(len(selected_gene_names)),
        "requested_gene_names": selected_gene_names,
        "matched_genes": matched_genes,
        "missing_genes": missing_genes,
        "k_effective": int(len(matched_genes)),
        "X_sieve": X_sieve,
    }


def summarize_null_distribution(null_aucs: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a null AUC distribution."""
    return {
        "mean": float(round(np.mean(null_aucs), 6)),
        "std": float(round(np.std(null_aucs), 6)),
        "median": float(round(np.median(null_aucs), 6)),
        "p5": float(round(np.percentile(null_aucs, 5), 6)),
        "p95": float(round(np.percentile(null_aucs, 95), 6)),
        "max": float(round(np.max(null_aucs), 6)),
    }


def run_permutations(
    burden_values: np.ndarray,
    k_effective: int,
    y: np.ndarray,
    fold_indices: FoldIndices,
    classifier_name: str,
    n_permutations: int,
    seed: int,
    n_jobs: int,
    progress_every: int = PERMUTATION_PROGRESS_EVERY,
) -> np.ndarray:
    """Evaluate the random-gene null distribution with batched progress reporting."""
    if n_permutations <= 0:
        return np.empty(0, dtype=np.float64)

    n_genes_total = burden_values.shape[1]
    all_results: list[float] = []
    start_time = time.time()

    for batch_start in range(0, n_permutations, progress_every):
        batch_end = min(batch_start + progress_every, n_permutations)
        batch_indices = list(range(batch_start, batch_end))

        batch_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
            delayed(run_single_permutation)(
                perm_idx,
                burden_values,
                n_genes_total,
                k_effective,
                y,
                fold_indices,
                classifier_name,
                seed,
                n_permutations,
            )
            for perm_idx in batch_indices
        )

        all_results.extend(batch_results)
        completed = len(all_results)
        elapsed = time.time() - start_time
        rate = elapsed / completed if completed else math.nan
        remaining = rate * (n_permutations - completed) if completed else math.nan

        print(
            f"  Permutation {completed}/{n_permutations} "
            f"({(completed / n_permutations) * 100:.1f}%) - "
            f"elapsed: {format_duration(elapsed)} - "
            f"estimated remaining: {format_duration(remaining)}"
        )

    return np.asarray(all_results, dtype=np.float64)


def plot_validation(
    observed_aucs: np.ndarray,
    null_aucs: np.ndarray,
    observed_mean: float,
    empirical_p: float,
    level: str,
    top_k: int,
    classifier_name: str,
    output_path: Path,
) -> None:
    """Create a two-panel diagnostic plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    classifier_label = CLASSIFIER_LABELS[classifier_name].replace("_", " ")

    axes[0].hist(null_aucs, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].axvline(
        observed_mean,
        color="red",
        linewidth=2,
        label=f"Observed = {observed_mean:.3f}",
    )
    axes[0].set_xlabel("Mean AUC")
    axes[0].set_ylabel("Count")
    axes[0].set_title(
        f"Null distribution - {level} top-{top_k} ({classifier_label})\n"
        f"empirical p = {empirical_p:.4f}"
    )
    axes[0].legend()

    valid_observed = observed_aucs[~np.isnan(observed_aucs)]
    box = axes[1].boxplot(
        [null_aucs, valid_observed],
        tick_labels=["Null (means)", "Observed (folds)"],
        patch_artist=True,
    )
    box["boxes"][0].set_facecolor("steelblue")
    box["boxes"][0].set_alpha(0.5)
    box["boxes"][1].set_facecolor("salmon")
    box["boxes"][1].set_alpha(0.7)
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Observed per-fold AUCs vs null distribution")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot observed AUC across levels and top-k values."""
    classifiers = [name for name in CLASSIFIER_ORDER if name in summary_df["classifier"].unique()]
    if not classifiers:
        return

    levels = sorted(summary_df["level"].unique(), key=level_sort_key)
    top_ks = sorted(summary_df["top_k"].unique())
    bonferroni_threshold = 0.05 / len(summary_df) if len(summary_df) else 0.05

    fig, axes = plt.subplots(
        1,
        len(classifiers),
        figsize=(max(6, len(top_ks) * 1.7 * len(classifiers)), max(4, len(levels) * 0.9 + 2)),
        squeeze=False,
    )

    matrices: list[np.ndarray] = []
    pval_matrices: list[np.ndarray] = []
    for classifier_name in classifiers:
        subset = summary_df[summary_df["classifier"] == classifier_name]
        matrix = np.full((len(levels), len(top_ks)), np.nan, dtype=np.float64)
        pval_matrix = np.full((len(levels), len(top_ks)), np.nan, dtype=np.float64)

        for i, level in enumerate(levels):
            for j, top_k in enumerate(top_ks):
                row = subset[(subset["level"] == level) & (subset["top_k"] == top_k)]
                if not row.empty:
                    matrix[i, j] = float(row["observed_auc"].iloc[0])
                    pval_matrix[i, j] = float(row["empirical_p"].iloc[0])

        matrices.append(matrix)
        pval_matrices.append(pval_matrix)

    finite_arrays = [matrix[np.isfinite(matrix)] for matrix in matrices if np.isfinite(matrix).any()]
    finite_values = np.concatenate(finite_arrays) if finite_arrays else np.empty(0, dtype=np.float64)
    vmin = max(0.40, float(np.min(finite_values)) - 0.02) if finite_values.size else 0.45
    vmax = min(0.90, float(np.max(finite_values)) + 0.02) if finite_values.size else 0.70
    if math.isclose(vmin, vmax):
        vmax = vmin + 0.05

    image = None
    for axis, classifier_name, matrix, pval_matrix in zip(
        axes.ravel(),
        classifiers,
        matrices,
        pval_matrices,
    ):
        image = axis.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_xticks(range(len(top_ks)))
        axis.set_xticklabels([str(top_k) for top_k in top_ks])
        axis.set_yticks(range(len(levels)))
        axis.set_yticklabels(levels)
        axis.set_xlabel("Top-k genes")
        axis.set_ylabel("Annotation level")
        axis.set_title(CLASSIFIER_LABELS[classifier_name].replace("_", " ").title())

        for i in range(len(levels)):
            for j in range(len(top_ks)):
                value = matrix[i, j]
                if np.isnan(value):
                    continue

                suffix = ""
                p_value = pval_matrix[i, j]
                if not np.isnan(p_value):
                    if p_value < bonferroni_threshold:
                        suffix = " **"
                    elif p_value < 0.05:
                        suffix = " *"

                axis.text(
                    j,
                    i,
                    f"{value:.3f}{suffix}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if value >= (vmin + vmax) / 2 else "white",
                )

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Observed AUC")

    fig.suptitle("Observed AUC across SIEVE ablation levels and top-k selections")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Write a concise markdown summary of the validation results."""
    n_tests = len(summary_df)
    bonferroni_threshold = 0.05 / n_tests if n_tests else 0.05

    lines = [
        "# Non-linear classifier validation report",
        "",
        f"- Tests performed: {n_tests}",
        f"- Bonferroni threshold: {bonferroni_threshold:.6f}",
        "",
    ]

    significant = summary_df[summary_df["bonferroni_significant"] == "yes"]
    if significant.empty:
        lines.extend(
            [
                "## Bonferroni-significant results",
                "",
                "None.",
                "",
            ]
        )
    else:
        lines.extend(["## Bonferroni-significant results", ""])
        for _, row in significant.sort_values(["empirical_p", "observed_auc"]).iterrows():
            lines.append(
                f"- {row['level']} top-{row['top_k']} ({row['classifier']}): "
                f"AUC {row['observed_auc']:.3f}, null mean {row['null_mean_auc']:.3f}, "
                f"empirical p {row['empirical_p']:.4f}"
            )
        lines.append("")

    nominal = summary_df[
        (summary_df["empirical_p"] < 0.05)
        & (summary_df["bonferroni_significant"] != "yes")
    ]
    if nominal.empty:
        lines.extend(["## Nominal signals", "", "None.", ""])
    else:
        lines.extend(["## Nominal signals", ""])
        for _, row in nominal.sort_values(["empirical_p", "observed_auc"]).iterrows():
            lines.append(
                f"- {row['level']} top-{row['top_k']} ({row['classifier']}): "
                f"AUC {row['observed_auc']:.3f}, empirical p {row['empirical_p']:.4f}"
            )
        lines.append("")

    if not summary_df.empty:
        best_auc = summary_df.loc[summary_df["observed_auc"].idxmax()]
        best_p = summary_df.loc[summary_df["empirical_p"].idxmin()]

        lines.extend(
            [
                "## Best results",
                "",
                (
                    f"- Best observed AUC: {best_auc['level']} top-{best_auc['top_k']} "
                    f"({best_auc['classifier']}) with AUC {best_auc['observed_auc']:.3f}"
                ),
                (
                    f"- Lowest empirical p-value: {best_p['level']} top-{best_p['top_k']} "
                    f"({best_p['classifier']}) with p {best_p['empirical_p']:.4f}"
                ),
                "",
            ]
        )

    if {"rf", "lr"}.issubset(set(summary_df["classifier"].unique())):
        lines.extend(["## Random forest vs logistic regression", ""])
        rf_rows = summary_df[summary_df["classifier"] == "rf"].set_index(["level", "top_k"])
        lr_rows = summary_df[summary_df["classifier"] == "lr"].set_index(["level", "top_k"])
        shared_index = rf_rows.index.intersection(lr_rows.index)

        if len(shared_index) == 0:
            lines.extend(["No overlapping level/top-k comparisons.", ""])
        else:
            for level, top_k in shared_index:
                rf_auc = float(rf_rows.loc[(level, top_k), "observed_auc"])
                lr_auc = float(lr_rows.loc[(level, top_k), "observed_auc"])
                auc_gap = rf_auc - lr_auc
                direction = "RF > LR" if auc_gap > 0 else "LR > RF" if auc_gap < 0 else "RF = LR"
                lines.append(
                    f"- {level} top-{top_k}: RF {rf_auc:.3f}, LR {lr_auc:.3f}, "
                    f"RF-LR {auc_gap:+.3f} ({direction})"
                )
            lines.append("")

    if significant.empty:
        conclusion = (
            "No level/top-k combination showed Bonferroni-corrected evidence that the "
            "SIEVE gene set transfers non-linear discriminative signal to this cohort."
        )
    else:
        conclusion = (
            "At least one level/top-k combination exceeded the matched random-gene null, "
            "supporting transfer of multi-gene cardiovascular signal."
        )

    lines.extend(["## Conclusion", "", conclusion, ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_yaml_result(
    result: Dict[str, Any],
    cv_folds: int,
    linear_baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the primary YAML payload for one level/top-k result."""
    payload: Dict[str, Any] = {
        "parameters": {
            "ablation_level": result["level"],
            "top_k_requested": int(result["top_k"]),
            "top_k_used": int(result["top_k_used"]),
            "k_effective": int(result["k_effective"]),
            "missing_genes": [str(gene) for gene in result["missing_genes"]],
            "n_samples": int(result["n_samples"]),
            "n_cases": int(result["n_cases"]),
            "n_controls": int(result["n_controls"]),
            "consequence_type": result["consequence"],
            "classifier": CLASSIFIER_LABELS[result["classifier"]],
            "n_estimators": 500 if result["classifier"] == "rf" else None,
            "min_samples_leaf": 5 if result["classifier"] == "rf" else None,
            "cv_folds": int(cv_folds),
            "cv_repeats": int(CV_REPEATS),
            "n_permutations": int(result["n_permutations"]),
            "seed": int(result["seed"]),
        },
        "observed": {
            "mean_auc": float(round(result["observed_auc"], 6)),
            "std_auc": float(round(result["observed_std"], 6)),
            "per_fold_aucs": [
                float(round(value, 6)) if not np.isnan(value) else None
                for value in result["observed_aucs"]
            ],
        },
        "null_distribution": summarize_null_distribution(result["null_aucs"]),
        "empirical_p": float(round(result["empirical_p"], 6)),
        "percentile_rank": float(round(result["percentile_rank"], 2)),
    }

    if linear_baseline is not None:
        payload["linear_baseline"] = {
            "mean_auc": float(round(linear_baseline["observed_auc"], 6)),
            "std_auc": float(round(linear_baseline["observed_std"], 6)),
            "empirical_p": float(round(linear_baseline["empirical_p"], 6)),
            "rf_minus_lr_auc": float(
                round(result["observed_auc"] - linear_baseline["observed_auc"], 6)
            ),
        }

    return payload


def write_validation_outputs(
    result: Dict[str, Any],
    output_dir: Path,
    cv_folds: int,
    linear_baseline: Optional[Dict[str, Any]] = None,
    also_export_csv: bool = False,
) -> None:
    """Persist YAML, NPZ, plots, and optional CSV export for one result."""
    yaml_payload = build_yaml_result(result, cv_folds, linear_baseline=linear_baseline)

    with open(result["yaml_path"], "w") as handle:
        yaml.safe_dump(yaml_payload, handle, default_flow_style=False, sort_keys=False)

    np.savez_compressed(result["npz_path"], null_aucs=result["null_aucs"])

    plot_validation(
        result["observed_aucs"],
        result["null_aucs"],
        result["observed_auc"],
        result["empirical_p"],
        result["level"],
        result["top_k"],
        result["classifier"],
        result["plot_path"],
    )

    if also_export_csv:
        csv_dir = output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        export_df = pd.DataFrame(
            result["X_sieve"],
            columns=result["matched_genes"],
            index=result["sample_ids"],
        )
        export_df.insert(0, "sample_id", result["sample_ids"])
        export_df["phenotype"] = result["y"]
        export_df.to_csv(result["csv_path"], index=False)


def run_validation(
    burden_matrix: pd.DataFrame,
    burden_values: np.ndarray,
    sieve_genes_df: pd.DataFrame,
    y: np.ndarray,
    sample_ids: np.ndarray,
    level: str,
    top_k_requested: int,
    top_k_used: int,
    classifier_name: str,
    fold_indices: FoldIndices,
    n_permutations: int,
    seed: int,
    n_jobs: int,
    output_dir: Path,
    consequence: str,
    output_suffix: str = "",
) -> Dict[str, Any]:
    """Run one level/top-k/classifier evaluation and return the full result payload."""
    feature_info = select_sieve_features(
        burden_matrix=burden_matrix,
        sieve_genes_df=sieve_genes_df,
        top_k_requested=top_k_used,
    )

    matched_genes = feature_info["matched_genes"]
    missing_genes = feature_info["missing_genes"]
    k_effective = feature_info["k_effective"]
    X_sieve = feature_info["X_sieve"]

    print(f"  Gene matching: {k_effective}/{top_k_used} found, {len(missing_genes)} missing")
    if missing_genes:
        print(f"  Missing genes: {summarize_missing_genes(missing_genes)}")

    if k_effective < 10:
        print(
            "  WARNING: Very few SIEVE genes found in validation VCF - "
            "results may be unreliable"
        )

    if k_effective == 0:
        print("  ERROR: No SIEVE genes found in burden matrix. Skipping.")
        return {}

    observed_aucs = evaluate_gene_set(
        X_sieve,
        y,
        fold_indices,
        classifier_name,
        seed,
        warn_context=f"{classifier_name} observed",
    )
    valid_observed = observed_aucs[~np.isnan(observed_aucs)]
    if valid_observed.size == 0:
        print("  ERROR: All observed folds failed. Skipping.")
        return {}

    observed_mean = float(np.nanmean(observed_aucs))
    observed_std = float(np.nanstd(observed_aucs))
    print(f"  Observed AUC: {observed_mean:.4f} +/- {observed_std:.4f}")

    print(f"  Running {n_permutations} permutations ({n_jobs} jobs)...")
    null_aucs = run_permutations(
        burden_values=burden_values,
        k_effective=k_effective,
        y=y,
        fold_indices=fold_indices,
        classifier_name=classifier_name,
        n_permutations=n_permutations,
        seed=seed,
        n_jobs=n_jobs,
    )

    valid_null_aucs = null_aucs[~np.isnan(null_aucs)]
    if valid_null_aucs.size == 0:
        print("  ERROR: All null permutations failed. Skipping.")
        return {}

    if valid_null_aucs.size != len(null_aucs):
        print(
            f"  WARNING: {len(null_aucs) - len(valid_null_aucs)} null permutations produced NaN "
            "and were excluded from summary statistics"
        )

    empirical_p = float(
        (np.sum(valid_null_aucs >= observed_mean) + 1) / (len(valid_null_aucs) + 1)
    )
    percentile_rank = float(np.mean(valid_null_aucs < observed_mean) * 100)
    print(f"  Empirical p = {empirical_p:.4f}, percentile = {percentile_rank:.1f}%")

    n_cases = int(np.sum(y == 1))
    n_controls = int(np.sum(y == 0))
    tag = f"{level}_topK{top_k_requested}"

    return {
        "level": level,
        "top_k": int(top_k_requested),
        "top_k_used": int(feature_info["top_k_used"]),
        "k_effective": int(k_effective),
        "classifier": classifier_name,
        "matched_genes": matched_genes,
        "missing_genes": missing_genes,
        "n_samples": int(len(y)),
        "n_cases": n_cases,
        "n_controls": n_controls,
        "consequence": consequence,
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "observed_auc": float(observed_mean),
        "observed_std": float(observed_std),
        "observed_aucs": observed_aucs,
        "null_aucs": valid_null_aucs,
        "empirical_p": float(empirical_p),
        "percentile_rank": float(percentile_rank),
        "X_sieve": X_sieve,
        "y": y,
        "sample_ids": sample_ids,
        "yaml_path": output_dir / f"nonlinear_validation_{tag}{output_suffix}.yaml",
        "npz_path": output_dir / f"null_aucs_{tag}{output_suffix}.npz",
        "plot_path": output_dir / f"validation_plot_{tag}{output_suffix}.png",
        "csv_path": output_dir / "csv" / f"feature_matrix_{consequence}_{level}_top{top_k_requested}.csv",
    }


def result_to_summary_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a validation result into one summary-table row."""
    return {
        "level": result["level"],
        "top_k": int(result["top_k"]),
        "k_effective": int(result["k_effective"]),
        "classifier": result["classifier"],
        "observed_auc": round(float(result["observed_auc"]), 6),
        "observed_std": round(float(result["observed_std"]), 6),
        "null_mean_auc": round(float(np.mean(result["null_aucs"])), 6),
        "null_std_auc": round(float(np.std(result["null_aucs"])), 6),
        "empirical_p": round(float(result["empirical_p"]), 6),
        "percentile_rank": round(float(result["percentile_rank"]), 2),
    }


def choose_primary_classifier(classifiers_to_run: Sequence[str]) -> str:
    """Prefer RF as the primary output when present."""
    for classifier_name in CLASSIFIER_ORDER:
        if classifier_name in classifiers_to_run:
            return classifier_name
    raise ValueError("No classifiers requested")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    burden_matrix_path = resolve_burden_matrix_path(args.burden_matrix, args.consequence)
    print(f"Loading burden matrix from {burden_matrix_path}...")
    burden_matrix = load_burden_matrix(burden_matrix_path).fillna(0.0)
    print(f"  Shape: {burden_matrix.shape}")

    print("Loading phenotypes...")
    phenotype_map = {str(sample_id): label for sample_id, label in load_phenotypes(args.phenotypes).items()}
    print(f"  Loaded {len(phenotype_map)} samples")

    common_samples = [sample_id for sample_id in burden_matrix.index if sample_id in phenotype_map]
    if not common_samples:
        print("ERROR: No overlapping samples between burden matrix and phenotypes.")
        sys.exit(1)

    burden_matrix = burden_matrix.loc[common_samples]
    burden_values = burden_matrix.to_numpy(dtype=np.float64, copy=True)
    sample_ids = np.asarray(common_samples, dtype=object)
    y = np.asarray([phenotype_map[sample_id] for sample_id in common_samples], dtype=np.int64)

    n_cases = int(np.sum(y == 1))
    n_controls = int(np.sum(y == 0))
    minority_fraction = min(n_cases, n_controls) / len(y)
    print(f"  Intersected: {len(common_samples)} samples ({n_cases} cases, {n_controls} controls)")
    if n_cases == 0 or n_controls == 0:
        print(
            f"ERROR: Only one class present ({n_cases} cases, {n_controls} controls). "
            "Cannot run stratified cross-validation."
        )
        sys.exit(1)

    if min(n_cases, n_controls) < args.cv_folds:
        print(
            f"ERROR: Minority class has {min(n_cases, n_controls)} samples, "
            f"but --cv-folds requires at least {args.cv_folds}. "
            "Reduce --cv-folds or provide more samples."
        )
        sys.exit(1)

    if minority_fraction < 0.20:
        print(
            "  WARNING: Minority class fraction is below 20%; "
            "class_weight='balanced' is enabled for all classifiers"
        )

    gene_files = detect_sieve_gene_files(args.sieve_genes)
    sorted_levels = sorted(gene_files.keys(), key=level_sort_key)
    print(f"  Detected levels: {sorted_levels}")

    if args.classifiers == "both":
        classifiers_to_run = ["rf", "lr"]
    else:
        classifiers_to_run = [args.classifiers]
    primary_classifier = choose_primary_classifier(classifiers_to_run)

    fold_indices = generate_fixed_folds(y, args.cv_folds, args.seed)
    print(
        f"  CV: {args.cv_folds} folds x {CV_REPEATS} repeats = "
        f"{len(fold_indices)} splits"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[Dict[str, Any]] = []
    for level in sorted_levels:
        gene_file = gene_files[level]
        print(f"\n{'=' * 60}")
        print(f"Level: {level}")
        print(f"{'=' * 60}")

        sieve_genes_df = load_sieve_genes(gene_file)
        n_available = len(sieve_genes_df)
        print(f"  SIEVE genes available: {n_available}")

        for requested_top_k in args.top_k:
            top_k_used = min(requested_top_k, n_available)
            if requested_top_k > n_available:
                print(
                    f"\n  top-{requested_top_k}: only {n_available} genes available; "
                    f"using all available genes"
                )

            print(f"\n  Selected top-{top_k_used} genes for requested top-{requested_top_k}")
            level_results: Dict[str, Dict[str, Any]] = {}

            for classifier_name in classifiers_to_run:
                print(f"\n  --- {level} top-{requested_top_k} ({classifier_name}) ---")
                output_suffix = "" if classifier_name == primary_classifier else f"_{classifier_name}"
                result = run_validation(
                    burden_matrix=burden_matrix,
                    burden_values=burden_values,
                    sieve_genes_df=sieve_genes_df,
                    y=y,
                    sample_ids=sample_ids,
                    level=level,
                    top_k_requested=requested_top_k,
                    top_k_used=top_k_used,
                    classifier_name=classifier_name,
                    fold_indices=fold_indices,
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                    n_jobs=args.n_jobs,
                    output_dir=args.output_dir,
                    consequence=args.consequence,
                    output_suffix=output_suffix,
                )
                if not result:
                    continue

                level_results[classifier_name] = result
                summary_rows.append(result_to_summary_row(result))

            if primary_classifier not in level_results:
                continue

            linear_baseline = None
            if primary_classifier == "rf" and "lr" in level_results:
                linear_baseline = level_results["lr"]

            write_validation_outputs(
                level_results[primary_classifier],
                output_dir=args.output_dir,
                cv_folds=args.cv_folds,
                linear_baseline=linear_baseline,
                also_export_csv=args.also_export_csv,
            )

            for classifier_name, result in level_results.items():
                if classifier_name == primary_classifier:
                    continue
                write_validation_outputs(
                    result,
                    output_dir=args.output_dir,
                    cv_folds=args.cv_folds,
                    also_export_csv=False,
                )

    if not summary_rows:
        print("\nNo results produced.")
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows)
    n_tests = len(summary_df)
    bonferroni_threshold = 0.05 / n_tests
    summary_df["bonferroni_significant"] = summary_df["empirical_p"].apply(
        lambda p_value: "yes" if p_value < bonferroni_threshold else "no"
    )

    summary_path = args.output_dir / "nonlinear_validation_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"\nSummary written to {summary_path}")

    heatmap_path = args.output_dir / "nonlinear_validation_heatmap.png"
    plot_summary_heatmap(summary_df, heatmap_path)
    print(f"Heatmap written to {heatmap_path}")

    report_path = args.output_dir / "nonlinear_validation_report.md"
    generate_report(summary_df, report_path)
    print(f"Report written to {report_path}")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Bonferroni threshold: {bonferroni_threshold:.6f} ({n_tests} tests)")
    print()
    for _, row in summary_df.iterrows():
        significance = ""
        if row["bonferroni_significant"] == "yes":
            significance = " **"
        elif row["empirical_p"] < 0.05:
            significance = " *"

        print(
            f"  {row['level']:>4s} top-{int(row['top_k']):<5d} {row['classifier']:>3s}: "
            f"AUC={row['observed_auc']:.4f} "
            f"(null={row['null_mean_auc']:.4f}), "
            f"p={row['empirical_p']:.4f}{significance}"
        )
    print()


if __name__ == "__main__":
    main()
