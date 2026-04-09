#!/usr/bin/env python3
"""
Non-linear classifier validation of SIEVE gene sets.

This script validates corrected SIEVE gene rankings from one or more
annotation levels against an independent validation cohort burden matrix.
For each ``(top_k, classifier)`` pair it draws one shared null distribution
of random gene sets, reuses that null across all requested annotation levels,
and computes empirical p-values with the ``(k + 1) / (N + 1)`` convention
recommended by Phipson and Smyth (2010), "Permutation P-values Should Never
Be Zero".

The output TSV contains one row per ``level x top_k x classifier``
combination. Benjamini-Hochberg FDR is then computed across the entire output
grid. Interpret ``fdr_bh`` in light of the grid you selected via
``--levels``, ``--top-k``, and ``--classifiers``.

Usage:
    python scripts/validate_nonlinear_classifier.py \
        --real-rankings-dir /path/to/corrected_rankings \
        --burden-matrix /path/to/gene_burden_matrix.parquet \
        --labels /path/to/phenotypes.tsv \
        --output-tsv /path/to/nonlinear_validation_summary.tsv \
        --top-k 100,500,1000,2000 \
        --classifiers rf,lr
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.stats.multitest import multipletests

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False

try:
    from threadpoolctl import threadpool_info
except ImportError:  # pragma: no cover
    threadpool_info = None

from src.data.vcf_parser import load_phenotypes


LEVEL_ORDER = ["L0", "L1", "L2", "L3"]
CLASSIFIER_ORDER = ["rf", "lr"]
CLASSIFIER_LABELS = {
    "rf": "random_forest",
    "lr": "logistic_regression",
}
SUMMARY_COLUMNS = [
    "level",
    "top_k",
    "k_effective",
    "classifier",
    "observed_auc",
    "observed_std",
    "null_mean_auc",
    "null_std_auc",
    "empirical_p",
    "z_score",
    "fdr_bh",
]
GENE_COLUMN_CANDIDATES = ["gene_name", "gene_symbol", "gene_id", "gene"]
LEVEL_RANKING_CANDIDATES = [
    "corrected_gene_rankings_with_significance.csv",
    "corrected_gene_rankings.csv",
    "corrected/corrected_gene_rankings_with_significance.csv",
    "corrected/corrected_gene_rankings.csv",
    "results/attribution_comparison_corrected/corrected_gene_rankings_with_significance.csv",
    "results/explainability/corrected/corrected_gene_rankings.csv",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate SIEVE gene sets with shared random-gene nulls.",
    )
    parser.add_argument(
        "--real-rankings-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing one subdirectory per annotation level "
            "with corrected gene rankings"
        ),
    )
    parser.add_argument(
        "--burden-matrix",
        type=Path,
        required=True,
        help="Validation-cohort gene burden matrix parquet file",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Validation-cohort phenotype/label TSV",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        required=True,
        help="Summary TSV to write",
    )
    parser.add_argument(
        "--top-k",
        required=True,
        help="Comma-separated list of top-k values, e.g. 100,500,1000,2000",
    )
    parser.add_argument(
        "--classifiers",
        required=True,
        help="Comma-separated classifier list from {rf,lr}",
    )
    parser.add_argument(
        "--levels",
        default="L0,L1,L2,L3",
        help="Comma-separated annotation levels to include (default: L0,L1,L2,L3)",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of random-gene null permutations",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=-1,
        help="Number of outer-loop cores for permutation evaluation (-1 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed for random gene-set draws",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds",
    )
    parser.add_argument(
        "--score-column",
        default="z_attribution",
        help=(
            "Ranking column to use for selecting real genes. "
            "Use z_attribution (maps to gene_z_score in gene rankings) "
            "or fdr_gene for significance-driven ranking."
        ),
    )
    return parser.parse_args(argv)


def parse_int_csv(value: str, argument_name: str) -> list[int]:
    """Parse a comma-separated integer list."""
    values = [token.strip() for token in value.split(",") if token.strip()]
    if not values:
        raise ValueError(f"{argument_name} must not be empty")
    parsed = [int(token) for token in values]
    if any(number <= 0 for number in parsed):
        raise ValueError(f"{argument_name} must contain positive integers")
    return parsed


def parse_choice_csv(
    value: str,
    argument_name: str,
    allowed: Sequence[str],
) -> list[str]:
    """Parse a comma-separated string list restricted to *allowed* values."""
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{argument_name} must not be empty")
    invalid = sorted(set(tokens) - set(allowed))
    if invalid:
        raise ValueError(
            f"{argument_name} contains unsupported values: {invalid}. "
            f"Allowed: {sorted(allowed)}"
        )
    deduplicated: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in seen:
            deduplicated.append(token)
            seen.add(token)
    return deduplicated


def level_sort_key(level: str) -> tuple[int, str]:
    """Sort known ablation levels in canonical order."""
    if level in LEVEL_ORDER:
        return LEVEL_ORDER.index(level), level
    return len(LEVEL_ORDER), level


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


def resolve_score_column(df: pd.DataFrame, requested_score_column: str) -> str:
    """Map requested score aliases onto the actual gene-ranking columns present."""
    aliases = [requested_score_column]
    if requested_score_column == "z_attribution":
        aliases = ["gene_z_score", "z_attribution", "mean_z_score", "gene_score"]

    for candidate in aliases:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Requested score column '{requested_score_column}' not found. "
        f"Available columns: {list(df.columns)}"
    )


def score_column_is_ascending(score_column: str) -> bool:
    """Return True when lower values mean higher priority."""
    lowered = score_column.lower()
    return lowered.startswith("fdr") or lowered.startswith("empirical_p")


def find_level_rankings_path(level_dir: Path) -> Path:
    """Locate the corrected gene rankings file for one annotation level."""
    for relative_path in LEVEL_RANKING_CANDIDATES:
        candidate = level_dir / relative_path
        if candidate.exists():
            return candidate

    discovered = sorted(level_dir.rglob("corrected_gene_rankings*.csv"))
    if len(discovered) == 1:
        return discovered[0]
    if len(discovered) > 1:
        raise ValueError(
            f"Multiple corrected gene ranking files found under {level_dir}: "
            f"{[str(path) for path in discovered]}"
        )

    raise FileNotFoundError(
        f"No corrected gene rankings found under {level_dir}. "
        f"Expected one of {LEVEL_RANKING_CANDIDATES}"
    )


def detect_level_ranking_files(
    real_rankings_dir: Path,
    requested_levels: Sequence[str],
) -> dict[str, Path]:
    """Resolve corrected gene-ranking files for the requested levels."""
    level_paths: dict[str, Path] = {}
    for level in requested_levels:
        level_dir = real_rankings_dir / level
        if not level_dir.exists():
            raise FileNotFoundError(
                f"Requested level directory does not exist: {level_dir}"
            )
        level_paths[level] = find_level_rankings_path(level_dir)
    return level_paths


def load_gene_rankings(
    rankings_path: Path,
    requested_score_column: str,
) -> pd.DataFrame:
    """Load and standardise one gene-ranking file."""
    df = pd.read_csv(rankings_path)

    gene_column = next(
        (column for column in GENE_COLUMN_CANDIDATES if column in df.columns),
        None,
    )
    if gene_column is None:
        raise ValueError(
            f"No gene identifier column found in {rankings_path}. "
            f"Expected one of {GENE_COLUMN_CANDIDATES}"
        )

    actual_score_column = resolve_score_column(df, requested_score_column)
    ascending = score_column_is_ascending(requested_score_column)

    ranked = (
        df[[gene_column, actual_score_column]]
        .copy()
        .rename(
            columns={
                gene_column: "gene_name",
                actual_score_column: "ranking_score",
            }
        )
    )
    ranked["gene_name"] = ranked["gene_name"].astype(str)
    ranked["ranking_score"] = pd.to_numeric(ranked["ranking_score"], errors="coerce")
    ranked = ranked.dropna(subset=["gene_name", "ranking_score"])
    ranked = ranked.drop_duplicates(subset=["gene_name"], keep="first")
    ranked = ranked.sort_values(
        "ranking_score",
        ascending=ascending,
        kind="mergesort",
    ).reset_index(drop=True)
    return ranked


def load_burden_matrix(path: Path) -> pd.DataFrame:
    """Load gene-burden matrix from parquet and normalise sample IDs."""
    burden_df = pd.read_parquet(path)
    if "sample_id" in burden_df.columns:
        burden_df = burden_df.set_index("sample_id")
    burden_df.index = burden_df.index.astype(str)
    burden_df.columns = [str(column) for column in burden_df.columns]
    return burden_df


def load_labels(path: Path) -> dict[str, int]:
    """Load SIEVE phenotype labels as a sample_id -> 0/1 mapping."""
    return {
        str(sample_id): int(label)
        for sample_id, label in load_phenotypes(path).items()
    }


def generate_folds(
    labels: np.ndarray,
    cv_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic stratified folds."""
    splitter = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(np.zeros(len(labels), dtype=np.float64), labels))


def build_classifier(classifier_name: str, seed: int) -> Pipeline | RandomForestClassifier:
    """Construct one classifier instance with deterministic random state."""
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


def evaluate_auc_for_feature_indices(
    burden_values: np.ndarray,
    labels: np.ndarray,
    feature_indices: np.ndarray,
    fold_indices: Sequence[tuple[np.ndarray, np.ndarray]],
    classifier_name: str,
    model_seed: int,
) -> np.ndarray:
    """Evaluate one gene set and return per-fold AUCs."""
    X = burden_values[:, feature_indices]
    aucs: list[float] = []

    for train_idx, test_idx in fold_indices:
        classifier = build_classifier(classifier_name, model_seed)
        classifier.fit(X[train_idx], labels[train_idx])
        probabilities = classifier.predict_proba(X[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(labels[test_idx], probabilities)))

    return np.asarray(aucs, dtype=np.float64)


def evaluate_random_gene_set_mean_auc(
    burden_values: np.ndarray,
    labels: np.ndarray,
    feature_indices: np.ndarray,
    fold_indices: Sequence[tuple[np.ndarray, np.ndarray]],
    classifier_name: str,
    model_seed: int,
) -> float:
    """Evaluate one random gene set and return its mean AUC."""
    aucs = evaluate_auc_for_feature_indices(
        burden_values=burden_values,
        labels=labels,
        feature_indices=feature_indices,
        fold_indices=fold_indices,
        classifier_name=classifier_name,
        model_seed=model_seed,
    )
    return float(np.mean(aucs))


def draw_random_gene_sets(
    gene_universe_size: int,
    set_size: int,
    n_permutations: int,
    seed: int,
) -> list[np.ndarray]:
    """Draw random burden-matrix column index sets of equal size."""
    if set_size > gene_universe_size:
        raise ValueError(
            f"Requested top_k={set_size} exceeds the burden-matrix gene universe "
            f"({gene_universe_size})"
        )

    rng = np.random.default_rng(seed)
    return [
        np.asarray(
            rng.choice(gene_universe_size, size=set_size, replace=False),
            dtype=np.int64,
        )
        for _ in range(n_permutations)
    ]


def derive_seed(top_k: int, classifier_name: str, master_seed: int) -> int:
    """Derive a stable child seed from top-k, classifier, and master seed."""
    classifier_offset = {"rf": 17, "lr": 29}[classifier_name]
    return int(master_seed + top_k * 1009 + classifier_offset * 100_003)


def run_shared_null_distribution(
    burden_values: np.ndarray,
    labels: np.ndarray,
    random_gene_sets: Sequence[np.ndarray],
    fold_indices: Sequence[tuple[np.ndarray, np.ndarray]],
    classifier_name: str,
    seed: int,
    n_cores: int,
) -> np.ndarray:
    """Evaluate the shared random-gene null distribution in parallel."""
    start = time.perf_counter()
    null_aucs = Parallel(
        n_jobs=n_cores,
        backend="loky",
        verbose=10,
    )(
        delayed(evaluate_random_gene_set_mean_auc)(
            burden_values=burden_values,
            labels=labels,
            feature_indices=feature_indices,
            fold_indices=fold_indices,
            classifier_name=classifier_name,
            model_seed=seed + permutation_index + 1,
        )
        for permutation_index, feature_indices in enumerate(random_gene_sets)
    )
    elapsed = time.perf_counter() - start
    print(
        f"  Shared null complete in {format_duration(elapsed)} "
        f"for {len(random_gene_sets)} permutations"
    )
    return np.asarray(null_aucs, dtype=np.float64)


def select_observed_gene_set(
    ranked_genes: pd.DataFrame,
    burden_columns: Sequence[str],
    top_k: int,
) -> dict[str, Any]:
    """Select the observed top-k genes and intersect them with the burden matrix."""
    selected_gene_names = ranked_genes["gene_name"].head(top_k).tolist()
    burden_lookup = {
        str(column).upper(): index for index, column in enumerate(burden_columns)
    }

    matched_indices: list[int] = []
    matched_genes: list[str] = []
    missing_genes: list[str] = []
    seen: set[str] = set()

    for gene_name in selected_gene_names:
        gene_upper = gene_name.upper()
        if gene_upper in burden_lookup and gene_upper not in seen:
            matched_indices.append(burden_lookup[gene_upper])
            matched_genes.append(str(burden_columns[burden_lookup[gene_upper]]))
            seen.add(gene_upper)
        elif gene_upper not in burden_lookup:
            missing_genes.append(gene_name)

    return {
        "selected_gene_names": selected_gene_names,
        "matched_genes": matched_genes,
        "missing_genes": missing_genes,
        "feature_indices": np.asarray(matched_indices, dtype=np.int64),
        "k_effective": int(len(matched_indices)),
    }


def compute_empirical_p(observed_value: float, null_distribution: np.ndarray) -> float:
    """
    Compute an empirical p-value using the (k + 1) / (N + 1) convention.

    ``k`` is the number of null values greater than or equal to the observed
    value. The +1 adjustment follows Phipson and Smyth (2010), ensuring the
    empirical p-value is never exactly zero.
    """
    k = int(np.sum(null_distribution >= observed_value))
    return float((k + 1) / (len(null_distribution) + 1))


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    p_array = np.asarray(p_values, dtype=np.float64)
    if _HAS_STATSMODELS:
        _, adjusted, _, _ = multipletests(p_array, method="fdr_bh")
        return adjusted

    n = len(p_array)
    order = np.argsort(p_array)
    sorted_p = p_array[order]
    adjusted_sorted = sorted_p * n / np.arange(1, n + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted


def log_thread_configuration() -> None:
    """Log BLAS/OpenMP thread settings for debugging."""
    print("Thread configuration:")
    env_names = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for name in env_names:
        print(f"  {name}={os.environ.get(name, '<unset>')}")

    if threadpool_info is None:
        print("  threadpoolctl not available; reporting environment variables only")
        return

    info = threadpool_info()
    if not info:
        print("  threadpoolctl reported no active pools yet")
        return

    for entry in info:
        internal_api = entry.get("internal_api", "unknown")
        num_threads = entry.get("num_threads", "unknown")
        prefix = entry.get("prefix", "unknown")
        print(
            f"  pool={prefix} internal_api={internal_api} "
            f"num_threads={num_threads}"
        )


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
        f"Shared null - {level} top-{top_k} ({classifier_label})\n"
        f"empirical p = {empirical_p:.4f}"
    )
    axes[0].legend()

    box = axes[1].boxplot(
        [null_aucs, observed_aucs],
        tick_labels=["Null (means)", "Observed (folds)"],
        patch_artist=True,
    )
    box["boxes"][0].set_facecolor("steelblue")
    box["boxes"][0].set_alpha(0.5)
    box["boxes"][1].set_facecolor("salmon")
    box["boxes"][1].set_alpha(0.7)
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Observed per-fold AUCs vs shared null")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot observed AUC across levels and top-k values."""
    classifiers = [
        name for name in CLASSIFIER_ORDER if name in summary_df["classifier"].unique()
    ]
    if not classifiers:
        return

    levels = sorted(summary_df["level"].unique(), key=level_sort_key)
    top_ks = sorted(summary_df["top_k"].unique())
    fig, axes = plt.subplots(
        1,
        len(classifiers),
        figsize=(max(6, len(top_ks) * 1.7 * len(classifiers)), max(4, len(levels) * 0.9 + 2)),
        squeeze=False,
    )

    matrices: list[np.ndarray] = []
    fdr_matrices: list[np.ndarray] = []
    for classifier_name in classifiers:
        subset = summary_df[summary_df["classifier"] == classifier_name]
        matrix = np.full((len(levels), len(top_ks)), np.nan, dtype=np.float64)
        fdr_matrix = np.full((len(levels), len(top_ks)), np.nan, dtype=np.float64)

        for i, level in enumerate(levels):
            for j, top_k in enumerate(top_ks):
                row = subset[(subset["level"] == level) & (subset["top_k"] == top_k)]
                if not row.empty:
                    matrix[i, j] = float(row["observed_auc"].iloc[0])
                    fdr_matrix[i, j] = float(row["fdr_bh"].iloc[0])

        matrices.append(matrix)
        fdr_matrices.append(fdr_matrix)

    finite_arrays = [matrix[np.isfinite(matrix)] for matrix in matrices if np.isfinite(matrix).any()]
    finite_values = np.concatenate(finite_arrays) if finite_arrays else np.empty(0, dtype=np.float64)
    vmin = max(0.40, float(np.min(finite_values)) - 0.02) if finite_values.size else 0.45
    vmax = min(0.90, float(np.max(finite_values)) + 0.02) if finite_values.size else 0.70
    if np.isclose(vmin, vmax):
        vmax = vmin + 0.05

    image = None
    for axis, classifier_name, matrix, fdr_matrix in zip(
        axes.ravel(),
        classifiers,
        matrices,
        fdr_matrices,
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
                fdr_value = fdr_matrix[i, j]
                if not np.isnan(fdr_value) and fdr_value < 0.05:
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
    fig.subplots_adjust(top=0.84, wspace=0.30)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_report(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Write a concise markdown summary of the validation results."""
    lines = [
        "# Non-linear classifier validation report",
        "",
        (
            "Empirical p-values use the (k + 1) / (N + 1) convention. "
            "Benjamini-Hochberg FDR is computed across the full result grid."
        ),
        "",
    ]

    significant = summary_df[summary_df["fdr_bh"] < 0.05]
    if significant.empty:
        lines.extend(["## FDR-significant results", "", "None.", ""])
    else:
        lines.extend(["## FDR-significant results", ""])
        for _, row in significant.sort_values(["fdr_bh", "empirical_p"]).iterrows():
            lines.append(
                f"- {row['level']} top-{row['top_k']} ({row['classifier']}): "
                f"AUC {row['observed_auc']:.3f}, null mean {row['null_mean_auc']:.3f}, "
                f"empirical p {row['empirical_p']:.4f}, fdr_bh {row['fdr_bh']:.4f}"
            )
        lines.append("")

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

    output_path.write_text("\n".join(lines), encoding="utf-8")


def choose_primary_classifier(classifiers_to_run: Sequence[str]) -> str:
    """Prefer RF as the primary classifier when present."""
    for classifier_name in CLASSIFIER_ORDER:
        if classifier_name in classifiers_to_run:
            return classifier_name
    raise ValueError("No classifiers requested")


def build_yaml_payload(
    result: dict[str, Any],
    cv_folds: int,
    linear_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one YAML payload for a validation result."""
    payload: dict[str, Any] = {
        "parameters": {
            "ablation_level": result["level"],
            "top_k": int(result["top_k"]),
            "k_effective": int(result["k_effective"]),
            "missing_genes": [str(gene) for gene in result["missing_genes"]],
            "n_samples": int(result["n_samples"]),
            "n_cases": int(result["n_cases"]),
            "n_controls": int(result["n_controls"]),
            "classifier": CLASSIFIER_LABELS[result["classifier"]],
            "cv_folds": int(cv_folds),
            "n_permutations": int(result["n_permutations"]),
            "seed": int(result["seed"]),
            "score_column": str(result["score_column"]),
        },
        "observed": {
            "mean_auc": float(round(result["observed_auc"], 6)),
            "std_auc": float(round(result["observed_std"], 6)),
            "per_fold_aucs": [float(round(value, 6)) for value in result["observed_aucs"]],
        },
        "null_distribution": {
            "mean": float(round(result["null_mean_auc"], 6)),
            "std": float(round(result["null_std_auc"], 6)),
            "median": float(round(float(np.median(result["null_aucs"])), 6)),
            "p5": float(round(float(np.percentile(result["null_aucs"], 5)), 6)),
            "p95": float(round(float(np.percentile(result["null_aucs"], 95)), 6)),
        },
        "empirical_p": float(round(result["empirical_p"], 6)),
        "z_score": float(round(result["z_score"], 6))
        if pd.notna(result["z_score"])
        else None,
        "fdr_bh": float(round(result["fdr_bh"], 6)),
    }

    if linear_baseline is not None:
        payload["linear_baseline"] = {
            "mean_auc": float(round(linear_baseline["observed_auc"], 6)),
            "std_auc": float(round(linear_baseline["observed_std"], 6)),
            "empirical_p": float(round(linear_baseline["empirical_p"], 6)),
            "fdr_bh": float(round(linear_baseline["fdr_bh"], 6)),
            "rf_minus_lr_auc": float(
                round(result["observed_auc"] - linear_baseline["observed_auc"], 6)
            ),
        }

    return payload


def write_auxiliary_outputs(
    results: Sequence[dict[str, Any]],
    output_dir: Path,
    cv_folds: int,
    classifiers_to_run: Sequence[str],
) -> None:
    """Write YAML, NPZ, plots, heatmap, and report outputs."""
    primary_classifier = choose_primary_classifier(classifiers_to_run)
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault((result["level"], result["top_k"]), {})[result["classifier"]] = result

    for (level, top_k), classifier_results in grouped.items():
        primary_result = classifier_results.get(primary_classifier)
        if primary_result is None:
            continue

        linear_baseline = classifier_results.get("lr") if primary_classifier == "rf" else None
        for classifier_name, result in classifier_results.items():
            suffix = "" if classifier_name == primary_classifier else f"_{classifier_name}"
            tag = f"{level}_topK{top_k}"

            yaml_path = output_dir / f"nonlinear_validation_{tag}{suffix}.yaml"
            npz_path = output_dir / f"null_aucs_{tag}{suffix}.npz"
            plot_path = output_dir / f"validation_plot_{tag}{suffix}.png"

            yaml_payload = build_yaml_payload(
                result,
                cv_folds=cv_folds,
                linear_baseline=linear_baseline if classifier_name == primary_classifier else None,
            )
            with open(yaml_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(yaml_payload, handle, default_flow_style=False, sort_keys=False)

            np.savez_compressed(npz_path, null_aucs=result["null_aucs"])
            plot_validation(
                observed_aucs=result["observed_aucs"],
                null_aucs=result["null_aucs"],
                observed_mean=result["observed_auc"],
                empirical_p=result["empirical_p"],
                level=result["level"],
                top_k=result["top_k"],
                classifier_name=result["classifier"],
                output_path=plot_path,
            )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        top_k_values = parse_int_csv(args.top_k, "--top-k")
        classifiers_to_run = parse_choice_csv(
            args.classifiers,
            "--classifiers",
            CLASSIFIER_ORDER,
        )
        requested_levels = parse_choice_csv(
            args.levels,
            "--levels",
            LEVEL_ORDER,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    log_thread_configuration()
    print("")

    print(f"Loading burden matrix from {args.burden_matrix}...")
    burden_matrix = load_burden_matrix(args.burden_matrix).fillna(0.0)
    print(f"  Shape: {burden_matrix.shape}")

    print(f"Loading labels from {args.labels}...")
    label_map = load_labels(args.labels)
    print(f"  Loaded {len(label_map)} labelled samples")

    common_samples = [
        sample_id for sample_id in burden_matrix.index if sample_id in label_map
    ]
    if not common_samples:
        print(
            "ERROR: No overlapping samples between burden matrix and labels.",
            file=sys.stderr,
        )
        sys.exit(1)

    burden_matrix = burden_matrix.loc[common_samples]
    burden_values = burden_matrix.to_numpy(dtype=np.float64, copy=True)
    labels = np.asarray([label_map[sample_id] for sample_id in common_samples], dtype=np.int64)

    n_cases = int(np.sum(labels == 1))
    n_controls = int(np.sum(labels == 0))
    print(
        f"  Intersected samples: {len(common_samples)} "
        f"({n_cases} cases, {n_controls} controls)"
    )

    if n_cases == 0 or n_controls == 0:
        print(
            "ERROR: Both case and control samples are required.",
            file=sys.stderr,
        )
        sys.exit(1)

    if min(n_cases, n_controls) < args.cv_folds:
        print(
            f"ERROR: Minority class has {min(n_cases, n_controls)} samples, "
            f"but --cv-folds={args.cv_folds} requires at least that many.",
            file=sys.stderr,
        )
        sys.exit(1)

    level_ranking_files = detect_level_ranking_files(
        real_rankings_dir=args.real_rankings_dir,
        requested_levels=requested_levels,
    )
    level_rankings = {
        level: load_gene_rankings(path, args.score_column)
        for level, path in level_ranking_files.items()
    }

    print("Resolved corrected gene rankings:")
    for level in sorted(level_ranking_files, key=level_sort_key):
        print(f"  {level}: {level_ranking_files[level]}")

    fold_indices = generate_folds(labels, args.cv_folds, args.seed)
    print(f"Using {args.cv_folds} stratified CV folds")

    output_dir = args.output_tsv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    gene_universe_size = burden_values.shape[1]

    for top_k in top_k_values:
        print(f"\n{'=' * 72}")
        print(f"top_k = {top_k}")
        print(f"{'=' * 72}")

        null_set_size = top_k
        if null_set_size > gene_universe_size:
            raise ValueError(
                f"Requested top_k={top_k} exceeds burden matrix width "
                f"({gene_universe_size} genes)"
            )

        for classifier_name in classifiers_to_run:
            print(f"\nShared null for classifier={classifier_name}")
            permutation_seed = derive_seed(top_k, classifier_name, args.seed)
            random_gene_sets = draw_random_gene_sets(
                gene_universe_size=gene_universe_size,
                set_size=null_set_size,
                n_permutations=args.n_permutations,
                seed=permutation_seed,
            )
            null_aucs = run_shared_null_distribution(
                burden_values=burden_values,
                labels=labels,
                random_gene_sets=random_gene_sets,
                fold_indices=fold_indices,
                classifier_name=classifier_name,
                seed=permutation_seed,
                n_cores=args.n_cores,
            )
            null_mean_auc = float(np.mean(null_aucs))
            null_std_auc = float(np.std(null_aucs))
            print(
                f"  Shared null mean AUC = {null_mean_auc:.4f}, "
                f"std = {null_std_auc:.4f}"
            )

            for level in requested_levels:
                ranked_genes = level_rankings[level]
                observed_gene_set = select_observed_gene_set(
                    ranked_genes=ranked_genes,
                    burden_columns=burden_matrix.columns,
                    top_k=top_k,
                )
                k_effective = observed_gene_set["k_effective"]
                print(
                    f"  {level}: matched {k_effective}/{top_k} genes "
                    f"({len(observed_gene_set['missing_genes'])} missing)"
                )

                if k_effective == 0:
                    raise ValueError(
                        f"No ranked genes for {level} matched the burden matrix "
                        f"for top_k={top_k}"
                    )

                observed_aucs = evaluate_auc_for_feature_indices(
                    burden_values=burden_values,
                    labels=labels,
                    feature_indices=observed_gene_set["feature_indices"],
                    fold_indices=fold_indices,
                    classifier_name=classifier_name,
                    model_seed=derive_seed(top_k, classifier_name, args.seed) + 1_000_000,
                )
                observed_auc = float(np.mean(observed_aucs))
                observed_std = float(np.std(observed_aucs))
                empirical_p = compute_empirical_p(observed_auc, null_aucs)
                if null_std_auc > 0:
                    z_score = float((observed_auc - null_mean_auc) / null_std_auc)
                else:
                    z_score = float("nan")

                results.append(
                    {
                        "level": level,
                        "top_k": int(top_k),
                        "k_effective": int(k_effective),
                        "classifier": classifier_name,
                        "observed_auc": observed_auc,
                        "observed_std": observed_std,
                        "null_mean_auc": null_mean_auc,
                        "null_std_auc": null_std_auc,
                        "empirical_p": empirical_p,
                        "z_score": z_score,
                        "observed_aucs": observed_aucs,
                        "null_aucs": null_aucs,
                        "missing_genes": observed_gene_set["missing_genes"],
                        "matched_genes": observed_gene_set["matched_genes"],
                        "n_samples": int(len(labels)),
                        "n_cases": int(n_cases),
                        "n_controls": int(n_controls),
                        "n_permutations": int(args.n_permutations),
                        "seed": int(args.seed),
                        "score_column": args.score_column,
                    }
                )
                print(
                    f"    observed_auc={observed_auc:.4f} "
                    f"empirical_p={empirical_p:.4f}"
                )

    if not results:
        print("ERROR: No results were produced.", file=sys.stderr)
        sys.exit(1)

    summary_df = pd.DataFrame(results)
    summary_df["fdr_bh"] = bh_fdr(summary_df["empirical_p"].to_numpy())

    summary_df = summary_df.sort_values(
        by=["level", "top_k", "classifier"],
        key=lambda series: (
            series.map(lambda value: level_sort_key(value)[0])
            if series.name == "level"
            else series.map(CLASSIFIER_ORDER.index)
            if series.name == "classifier"
            else series
        ),
        kind="mergesort",
    ).reset_index(drop=True)

    for (top_k, classifier_name), group in summary_df.groupby(["top_k", "classifier"]):
        if group["null_mean_auc"].nunique() != 1:
            raise AssertionError(
                f"Null mean AUC differs across levels for top_k={top_k}, "
                f"classifier={classifier_name}"
            )
        if group["null_std_auc"].nunique() != 1:
            raise AssertionError(
                f"Null std AUC differs across levels for top_k={top_k}, "
                f"classifier={classifier_name}"
            )

    sorted_fdr = summary_df.sort_values("empirical_p")["fdr_bh"].to_numpy()
    if np.any(np.diff(sorted_fdr) < -1e-12):
        raise AssertionError("fdr_bh is not monotone non-decreasing when sorted by empirical_p")

    summary_df = summary_df[SUMMARY_COLUMNS]
    summary_df.to_csv(
        args.output_tsv,
        sep="\t",
        index=False,
        float_format="%.10g",
    )
    print(f"\nSummary written to {args.output_tsv}")

    result_lookup = {
        (result["level"], result["top_k"], result["classifier"]): result
        for result in results
    }
    for _, row in summary_df.iterrows():
        result = result_lookup[(row["level"], row["top_k"], row["classifier"])]
        result["fdr_bh"] = float(row["fdr_bh"])

    write_auxiliary_outputs(
        results=results,
        output_dir=output_dir,
        cv_folds=args.cv_folds,
        classifiers_to_run=classifiers_to_run,
    )

    heatmap_path = output_dir / "nonlinear_validation_heatmap.png"
    plot_summary_heatmap(summary_df, heatmap_path)
    print(f"Heatmap written to {heatmap_path}")

    report_path = output_dir / "nonlinear_validation_report.md"
    generate_report(summary_df, report_path)
    print(f"Report written to {report_path}")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    for _, row in summary_df.iterrows():
        significance = " *" if row["fdr_bh"] < 0.05 else ""
        print(
            f"  {row['level']:>4s} top-{int(row['top_k']):<5d} {row['classifier']:>3s}: "
            f"AUC={row['observed_auc']:.4f} "
            f"(null={row['null_mean_auc']:.4f}), "
            f"p={row['empirical_p']:.4f}, "
            f"fdr={row['fdr_bh']:.4f}{significance}"
        )
    print("")


if __name__ == "__main__":
    main()
