#!/usr/bin/env python3
"""
Post-hoc power analysis for epistasis detection in the SIEVE project.

Given the noise level in synergy estimates, this script computes the minimum
epistatic effect size detectable at a given significance threshold. It converts
"we found nothing" into "we can exclude epistatic effects larger than X."

Usage:
    python scripts/epistasis_power_analysis.py \
        --cooccurrence results/cooccurrence_per_pair.csv \
        --cooccurrence-summary results/cooccurrence_by_maf_bin.csv \
        --real-attributions-npz results/real/attributions.npz \
        --null-attributions-npz results/null/attributions.npz \
        --output-dir results/power_analysis

Author: Francesco Lescai
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------

def estimate_noise_from_null_attributions(
    null_npz_path: str,
) -> Tuple[float, str]:
    """
    Estimate synergy noise from null-model attribution scores.

    Parameters
    ----------
    null_npz_path : str
        Path to ``attributions.npz`` produced by a null (permuted-label) model.

    Returns
    -------
    sigma_synergy : float
        Estimated standard deviation of synergy scores under the null.
    method : str
        Label describing the estimation method used.
    """
    data = np.load(null_npz_path, allow_pickle=True)
    variant_scores = data["variant_scores"]

    all_scores = []
    for sample_scores in variant_scores:
        if sample_scores is not None and len(sample_scores) > 0:
            all_scores.append(sample_scores.ravel())

    if len(all_scores) == 0:
        raise ValueError("Null attributions contain no variant scores.")

    combined = np.concatenate(all_scores)
    sigma_null = float(np.std(combined))
    # Synergy is the difference of sums; variance propagates as ~3x single-variant variance
    sigma_synergy = float(np.sqrt(3.0) * sigma_null)
    logger.info(
        "Noise estimated from null attributions: sigma_null=%.6f, sigma_synergy=%.6f",
        sigma_null,
        sigma_synergy,
    )
    return sigma_synergy, "null_attributions"


def estimate_noise_from_real_attributions(
    real_npz_path: str,
) -> Tuple[float, str]:
    """
    Estimate synergy noise from inter-sample variance of real attribution scores.

    Parameters
    ----------
    real_npz_path : str
        Path to ``attributions.npz`` from the real model.

    Returns
    -------
    sigma_synergy : float
    method : str
    """
    data = np.load(real_npz_path, allow_pickle=True)
    variant_scores = data["variant_scores"]
    metadata = data["metadata"]

    # Collect scores keyed by (chrom, position) across samples
    position_scores: Dict[Tuple, list] = {}
    for sample_scores, sample_meta in zip(variant_scores, metadata):
        if sample_scores is None or len(sample_scores) == 0:
            continue
        meta = sample_meta if isinstance(sample_meta, dict) else sample_meta.item()
        positions = meta.get("positions", [])
        chromosomes = meta.get("chromosomes", [])
        for i, score in enumerate(sample_scores.ravel()):
            chrom = str(chromosomes[i]) if i < len(chromosomes) else "?"
            pos = int(positions[i]) if i < len(positions) else i
            key = (chrom, pos)
            position_scores.setdefault(key, []).append(float(score))

    # Compute per-position variance, then average
    variances = []
    for scores_list in position_scores.values():
        if len(scores_list) >= 2:
            variances.append(np.var(scores_list, ddof=1))

    if len(variances) == 0:
        raise ValueError(
            "Cannot estimate noise: not enough multi-sample variants."
        )

    mean_var = float(np.mean(variances))
    sigma_attr = float(np.sqrt(mean_var))
    sigma_synergy = float(np.sqrt(3.0) * sigma_attr)
    logger.info(
        "Noise estimated from real inter-sample variance: sigma_attr=%.6f, sigma_synergy=%.6f",
        sigma_attr,
        sigma_synergy,
    )
    return sigma_synergy, "real_inter_sample_variance"


def estimate_noise_from_epistasis_results(
    epistasis_csv_path: str,
) -> Tuple[float, str]:
    """
    Estimate synergy noise directly from observed synergy scores.

    Parameters
    ----------
    epistasis_csv_path : str
        Path to ``epistasis_validation.csv``.

    Returns
    -------
    sigma_synergy : float
    method : str
    """
    df = pd.read_csv(epistasis_csv_path)
    if "synergy" not in df.columns:
        raise ValueError("epistasis_validation.csv missing 'synergy' column.")
    if len(df) < 2:
        raise ValueError("Too few synergy scores to estimate variance.")

    sigma_synergy = float(df["synergy"].std(ddof=1))
    logger.info(
        "Noise estimated from observed synergy distribution: sigma_synergy=%.6f",
        sigma_synergy,
    )
    return sigma_synergy, "observed_synergy_distribution"


def estimate_sigma_synergy(
    real_npz: Optional[str],
    null_npz: Optional[str],
    epistasis_csv: Optional[str],
) -> Tuple[float, str]:
    """
    Estimate synergy noise using the best available data source.

    Tries sources in priority order:
    1. Null attributions (preferred)
    2. Epistasis validation CSV
    3. Real attributions only
    4. Default constant (with warning)

    Parameters
    ----------
    real_npz : str or None
    null_npz : str or None
    epistasis_csv : str or None

    Returns
    -------
    sigma_synergy : float
    method : str
    """
    # 1. Null attributions (preferred)
    if null_npz and Path(null_npz).exists():
        try:
            return estimate_noise_from_null_attributions(null_npz)
        except Exception as exc:
            logger.warning("Failed to estimate from null attributions: %s", exc)

    # 2. Epistasis validation CSV
    if epistasis_csv and Path(epistasis_csv).exists():
        try:
            return estimate_noise_from_epistasis_results(epistasis_csv)
        except Exception as exc:
            logger.warning("Failed to estimate from epistasis CSV: %s", exc)

    # 3. Real attributions only
    if real_npz and Path(real_npz).exists():
        try:
            return estimate_noise_from_real_attributions(real_npz)
        except Exception as exc:
            logger.warning("Failed to estimate from real attributions: %s", exc)

    # 4. Fallback
    default_sigma = 0.05
    logger.warning(
        "WARNING: No empirical noise source available. "
        "Using default sigma_synergy=%.4f. "
        "Results should be interpreted with caution.",
        default_sigma,
    )
    print(
        "\n" + "=" * 60
        + "\n  WARNING: Using default sigma_synergy=0.05\n"
        "  No null baseline, real attributions, or epistasis results found.\n"
        "  Power estimates may be inaccurate.\n"
        + "=" * 60 + "\n"
    )
    return default_sigma, "default_fallback"


# ---------------------------------------------------------------------------
# Power calculations
# ---------------------------------------------------------------------------

def compute_mde(
    n_effective: np.ndarray,
    sigma_synergy: float,
    alpha_corrected: float,
    power: float = 0.8,
) -> np.ndarray:
    """
    Compute minimum detectable effect (MDE) for a two-sided z-test.

    Parameters
    ----------
    n_effective : np.ndarray
        Effective sample sizes for interaction testing.
    sigma_synergy : float
        Standard deviation of synergy scores under the null.
    alpha_corrected : float
        Per-test significance level after multiple-testing correction.
    power : float
        Desired statistical power (default 0.8).

    Returns
    -------
    np.ndarray
        Minimum detectable effect sizes, same shape as *n_effective*.
    """
    z_alpha = stats.norm.ppf(1.0 - alpha_corrected / 2.0)
    z_power = stats.norm.ppf(power)

    n = np.asarray(n_effective, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mde = (z_alpha + z_power) * sigma_synergy / np.sqrt(n)
    # Where n <= 0, MDE is undefined
    mde = np.where(n > 0, mde, np.inf)
    return mde


def compute_effective_n_from_counts(
    n11: np.ndarray,
    n10: np.ndarray,
    n01: np.ndarray,
    n00: np.ndarray,
) -> np.ndarray:
    """
    Approximate the effective sample size for a 2x2 interaction contrast.

    Uses the harmonic-style combination of the four contingency-table cells.
    If any cell is empty, the interaction contrast is not estimable and the
    effective sample size is reported as 0.
    """
    n11 = np.asarray(n11, dtype=float)
    n10 = np.asarray(n10, dtype=float)
    n01 = np.asarray(n01, dtype=float)
    n00 = np.asarray(n00, dtype=float)

    positive = (n11 > 0) & (n10 > 0) & (n01 > 0) & (n00 > 0)
    effective_n = np.zeros_like(n11, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        effective_n[positive] = 1.0 / (
            (1.0 / n11[positive])
            + (1.0 / n10[positive])
            + (1.0 / n01[positive])
            + (1.0 / n00[positive])
        )
    return effective_n


def compute_corrected_alpha(
    alpha: float,
    n_tests: int,
    method: str = "bonferroni",
) -> float:
    """
    Compute corrected significance threshold.

    Parameters
    ----------
    alpha : float
        Family-wise error rate.
    n_tests : int
        Number of tests.
    method : str
        ``'bonferroni'`` or ``'fdr'``.

    Returns
    -------
    float
        Corrected per-test alpha.
    """
    if n_tests <= 0:
        return alpha
    if method == "bonferroni":
        return alpha / n_tests
    elif method == "fdr":
        # Approximate: use the most conservative BH threshold (rank 1)
        return alpha / n_tests
    else:
        raise ValueError(f"Unknown correction method: {method}")


def minimum_effective_n_for_effect(
    effect_size: float,
    sigma_synergy: float,
    alpha_corrected: float,
    power: float = 0.8,
) -> int:
    """
    Compute minimum co-occurrence count needed to detect a given effect size.

    Parameters
    ----------
    effect_size : float
    sigma_synergy : float
    alpha_corrected : float
    power : float

    Returns
    -------
    int
        Minimum sample size (co-occurrence count).
    """
    if effect_size <= 0:
        return np.iinfo(np.int64).max
    z_alpha = stats.norm.ppf(1.0 - alpha_corrected / 2.0)
    z_power = stats.norm.ppf(power)
    n = ((z_alpha + z_power) * sigma_synergy / effect_size) ** 2
    return int(np.ceil(n))


def add_pair_contingency_metrics(
    cooccur_df: pd.DataFrame,
    total_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add 2x2 contingency counts and effective sample size to a co-occurrence table.

    Prefers exact columns written by ``audit_cooccurrence.py``. For legacy files,
    falls back to reconstructing carrier counts from frequencies and cohort size.
    """
    df = cooccur_df.copy()

    if total_samples is None:
        total_samples = infer_total_samples_from_cooccurrence(df)

    if "carrier_count_a" not in df.columns:
        if total_samples is None or "freq_a" not in df.columns:
            raise ValueError(
                "Cannot derive carrier_count_a. Provide a cooccurrence file generated "
                "by the updated audit script or one containing freq_a plus inferable "
                "total sample size."
            )
        df["carrier_count_a"] = np.rint(df["freq_a"] * total_samples).astype(int)

    if "carrier_count_b" not in df.columns:
        if total_samples is None or "freq_b" not in df.columns:
            raise ValueError(
                "Cannot derive carrier_count_b. Provide a cooccurrence file generated "
                "by the updated audit script or one containing freq_b plus inferable "
                "total sample size."
            )
        df["carrier_count_b"] = np.rint(df["freq_b"] * total_samples).astype(int)

    if "n_only_a" not in df.columns:
        df["n_only_a"] = (
            df["carrier_count_a"].astype(int) - df["n_cooccur"].astype(int)
        ).clip(lower=0)
    if "n_only_b" not in df.columns:
        df["n_only_b"] = (
            df["carrier_count_b"].astype(int) - df["n_cooccur"].astype(int)
        ).clip(lower=0)
    if "n_neither" not in df.columns:
        if total_samples is None:
            raise ValueError(
                "Cannot derive n_neither without exact contingency columns or an "
                "inferable total sample size."
            )
        df["n_neither"] = (
            total_samples
            - df["n_cooccur"].astype(int)
            - df["n_only_a"].astype(int)
            - df["n_only_b"].astype(int)
        ).clip(lower=0)

    df["min_cell_count"] = df[
        ["n_cooccur", "n_only_a", "n_only_b", "n_neither"]
    ].min(axis=1)
    df["n_effective"] = compute_effective_n_from_counts(
        df["n_cooccur"].values,
        df["n_only_a"].values,
        df["n_only_b"].values,
        df["n_neither"].values,
    )
    if total_samples is not None and total_samples > 0:
        df["cooccurrence_rate"] = df["n_cooccur"] / float(total_samples)
        df["effective_fraction"] = df["n_effective"] / float(total_samples)
    return df


def infer_total_samples_from_cooccurrence(
    cooccur_df: pd.DataFrame,
) -> Optional[int]:
    """Infer the cohort size from expected co-occurrence values."""
    required = {"freq_a", "freq_b", "expected_cooccur"}
    if not required.issubset(cooccur_df.columns):
        return None

    freq_product = cooccur_df["freq_a"] * cooccur_df["freq_b"]
    valid = freq_product > 0
    if not valid.any():
        return None

    estimates = cooccur_df.loc[valid, "expected_cooccur"] / freq_product.loc[valid]
    finite = estimates[np.isfinite(estimates)]
    if finite.empty:
        return None
    return int(round(float(finite.median())))


def summarise_power_by_maf_bin(
    per_pair_df: pd.DataFrame,
    effective_col: str = "n_effective",
    threshold: float = 0.1,
) -> pd.DataFrame:
    """Summarise co-occurrence and MDE by MAF-bin combination."""
    if not {"maf_bin_a", "maf_bin_b"}.issubset(per_pair_df.columns):
        return pd.DataFrame(
            {
                "median_n_cooccur": [float(per_pair_df["n_cooccur"].median())],
                "median_min_cell_count": [float(per_pair_df["min_cell_count"].median())],
                "median_n_effective": [float(per_pair_df[effective_col].median())],
                "median_mde": [float(per_pair_df["mde"].replace(np.inf, np.nan).median())],
                "mean_mde": [float(per_pair_df["mde"].replace(np.inf, np.nan).mean())],
                "n_pairs": [int(len(per_pair_df))],
                "n_testable_pairs": [int((per_pair_df["min_cell_count"] >= 5).sum())],
                "proportion_mde_lt_threshold": [float((per_pair_df["mde"] < threshold).mean())],
            }
        )

    summary = (
        per_pair_df.groupby(["maf_bin_a", "maf_bin_b"], dropna=False)
        .agg(
            n_pairs=(effective_col, "count"),
            median_n_cooccur=("n_cooccur", "median"),
            mean_n_cooccur=("n_cooccur", "mean"),
            median_min_cell_count=("min_cell_count", "median"),
            mean_min_cell_count=("min_cell_count", "mean"),
            median_n_effective=(effective_col, "median"),
            mean_n_effective=(effective_col, "mean"),
            n_testable_pairs=("min_cell_count", lambda values: int((values >= 5).sum())),
            median_mde=("mde", lambda values: float(np.nanmedian(np.where(np.isfinite(values), values, np.nan)))),
            mean_mde=("mde", lambda values: float(np.nanmean(np.where(np.isfinite(values), values, np.nan)))),
            proportion_mde_lt_threshold=("mde", lambda values: float(np.mean(values < threshold))),
        )
        .reset_index()
    )
    return summary


def estimate_common_pair_effective_fraction(
    per_pair_df: pd.DataFrame,
    total_samples: Optional[int],
) -> Optional[float]:
    """Estimate the effective-information fraction for common-common pairs."""
    if total_samples is None or total_samples <= 0:
        return None
    if not {"freq_a", "freq_b", "n_effective"}.issubset(per_pair_df.columns):
        return None

    common_pairs = per_pair_df[
        (per_pair_df["freq_a"] >= 0.05)
        & (per_pair_df["freq_b"] >= 0.05)
    ]
    if common_pairs.empty:
        return None

    rates = common_pairs["n_effective"] / float(total_samples)
    rates = rates[np.isfinite(rates) & (rates > 0)]
    if rates.empty:
        return None
    return float(rates.median())


def estimate_common_pair_cooccurrence_rate(
    per_pair_df: pd.DataFrame,
    total_samples: Optional[int],
) -> Optional[float]:
    """Estimate the joint-carrier fraction for common-common pairs."""
    if total_samples is None or total_samples <= 0:
        return None
    if not {"freq_a", "freq_b", "n_cooccur"}.issubset(per_pair_df.columns):
        return None

    common_pairs = per_pair_df[
        (per_pair_df["freq_a"] >= 0.05)
        & (per_pair_df["freq_b"] >= 0.05)
    ]
    if common_pairs.empty:
        return None

    rates = common_pairs["n_cooccur"] / float(total_samples)
    rates = rates[np.isfinite(rates) & (rates > 0)]
    if rates.empty:
        return None
    return float(rates.median())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_power_plot(
    summary_df: pd.DataFrame,
    output_path: Path,
    bio_threshold: float = 0.1,
) -> None:
    """
    Create a heatmap / scatter of MAF bin vs MDE.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Power analysis summary by MAF bin combination.
    output_path : Path
        Where to save the PNG.
    bio_threshold : float
        Biologically meaningful effect size threshold for reference line.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    maf_cols = None
    if {"maf_bin_a", "maf_bin_b"}.issubset(summary_df.columns):
        maf_cols = ("maf_bin_a", "maf_bin_b")
    elif {"maf_bin_1", "maf_bin_2"}.issubset(summary_df.columns):
        maf_cols = ("maf_bin_1", "maf_bin_2")

    # If we have maf_bin columns, create a grouped bar chart
    if maf_cols is not None:
        summary_df = summary_df.copy()
        summary_df["bin_label"] = (
            summary_df[maf_cols[0]].astype(str)
            + " x "
            + summary_df[maf_cols[1]].astype(str)
        )
        # Cap MDE for display
        plot_mde = summary_df["median_mde"].fillna(np.inf).clip(upper=2.0)
        colours = [
            "#2ecc71" if m < bio_threshold else "#e74c3c"
            for m in summary_df["median_mde"]
        ]
        ax.barh(summary_df["bin_label"], plot_mde, color=colours, edgecolor="grey")
        ax.axvline(bio_threshold, color="black", linestyle="--", linewidth=1.2,
                    label=f"Bio. threshold ({bio_threshold})")
        ax.set_xlabel("Median MDE (synergy effect size)")
        ax.set_ylabel("MAF bin combination")
        ax.set_title("Minimum Detectable Epistatic Effect by MAF Bin")
    else:
        # Fallback: scatter of n_cooccur vs MDE
        if "median_n_cooccur" in summary_df.columns:
            x_values = (
                summary_df["median_n_effective"]
                if "median_n_effective" in summary_df.columns
                else summary_df["median_n_cooccur"]
            )
            ax.scatter(
                x_values,
                summary_df["median_mde"].clip(upper=2.0),
                alpha=0.7,
                edgecolors="grey",
            )
            ax.axhline(bio_threshold, color="black", linestyle="--", linewidth=1.2,
                        label=f"Bio. threshold ({bio_threshold})")
            ax.set_xlabel("Median effective sample size")
            ax.set_ylabel("Median MDE")
            ax.set_title("Minimum Detectable Epistatic Effect vs Effective Sample Size")
        else:
            ax.text(0.5, 0.5, "Insufficient data for plot",
                    ha="center", va="center", transform=ax.transAxes)

    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Power analysis plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc power analysis for epistasis detection. "
            "Computes the minimum epistatic effect size detectable at a "
            "given significance threshold."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--cooccurrence", type=str, required=True,
        help="Path to cooccurrence_per_pair.csv",
    )
    parser.add_argument(
        "--cooccurrence-summary", type=str, required=True,
        help="Path to cooccurrence_by_maf_bin.csv",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for power analysis results",
    )

    # Optional noise sources
    parser.add_argument(
        "--real-attributions-npz", type=str, default=None,
        help="Path to attributions.npz from real model",
    )
    parser.add_argument(
        "--null-attributions-npz", type=str, default=None,
        help="Path to attributions.npz from null (permuted-label) model",
    )
    parser.add_argument(
        "--epistasis-results", type=str, default=None,
        help="Path to epistasis_validation.csv",
    )

    # Statistical parameters
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Family-wise significance level",
    )
    parser.add_argument(
        "--correction", type=str, default="bonferroni",
        choices=["bonferroni", "fdr"],
        help="Multiple-testing correction method",
    )

    return parser.parse_args()


def main() -> None:
    """Run epistasis power analysis."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Epistasis Power Analysis")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Estimate noise
    # ------------------------------------------------------------------
    sigma_synergy, noise_method = estimate_sigma_synergy(
        real_npz=args.real_attributions_npz,
        null_npz=args.null_attributions_npz,
        epistasis_csv=args.epistasis_results,
    )
    print(f"\nNoise estimation method: {noise_method}")
    print(f"Estimated sigma_synergy: {sigma_synergy:.6f}")

    # ------------------------------------------------------------------
    # 2. Load co-occurrence data
    # ------------------------------------------------------------------
    print("\nLoading co-occurrence data...")
    cooccur_df = pd.read_csv(args.cooccurrence)
    pd.read_csv(args.cooccurrence_summary)
    print(f"  Total variant pairs: {len(cooccur_df):,}")

    # Determine co-occurrence column name
    cooccur_col = "n_cooccur"
    if cooccur_col not in cooccur_df.columns:
        # Try alternative names
        for candidate in ["n_cooccurrence", "cooccurrence", "count"]:
            if candidate in cooccur_df.columns:
                cooccur_col = candidate
                break
        else:
            logger.error(
                "Cannot find co-occurrence count column. "
                "Available columns: %s",
                list(cooccur_df.columns),
            )
            sys.exit(1)

    estimated_total_samples = infer_total_samples_from_cooccurrence(cooccur_df)
    per_pair_df = add_pair_contingency_metrics(
        cooccur_df,
        total_samples=estimated_total_samples,
    )

    # Testable pairs: all cells in the 2x2 table have support
    testable_mask = per_pair_df["min_cell_count"] >= 5
    n_testable = int(testable_mask.sum())
    print(f"  Testable pairs (all four cells >= 5): {n_testable:,}")

    # ------------------------------------------------------------------
    # 3. Corrected alpha
    # ------------------------------------------------------------------
    if n_testable == 0:
        logger.warning(
            "No pairs met the all-cells >= 5 threshold. Using uncorrected alpha for "
            "descriptive MDE calculations."
        )
    alpha_corrected = compute_corrected_alpha(
        args.alpha, max(n_testable, 1), args.correction,
    )
    print(f"\nSignificance level: alpha={args.alpha}")
    print(f"Correction method: {args.correction}")
    print(f"Corrected alpha: {alpha_corrected:.2e}")

    # ------------------------------------------------------------------
    # 4. Per-pair MDE
    # ------------------------------------------------------------------
    print("\nComputing per-pair MDE...")
    per_pair_df["mde"] = compute_mde(
        per_pair_df["n_effective"].values,
        sigma_synergy,
        alpha_corrected,
    )

    per_pair_path = output_dir / "power_analysis_per_pair.csv"
    per_pair_df.to_csv(per_pair_path, index=False)
    print(f"  Saved per-pair results to {per_pair_path}")

    # ------------------------------------------------------------------
    # 5. MAF-bin stratified summary
    # ------------------------------------------------------------------
    print("\nComputing MAF-bin stratified summary...")
    maf_summary = summarise_power_by_maf_bin(per_pair_df, effective_col="n_effective")

    maf_bin_path = output_dir / "power_analysis_by_maf_bin.csv"
    maf_summary.to_csv(maf_bin_path, index=False)
    print(f"  Saved MAF-bin summary to {maf_bin_path}")

    # ------------------------------------------------------------------
    # 6. Headline statistics
    # ------------------------------------------------------------------
    finite_mde = per_pair_df["mde"][np.isfinite(per_pair_df["mde"])]
    prop_detectable = float((finite_mde < 0.1).mean()) if len(finite_mde) > 0 else 0.0

    # Minimum effective sample sizes needed for representative common-common pairs
    min_effective_n_01 = minimum_effective_n_for_effect(
        effect_size=0.1,
        sigma_synergy=sigma_synergy,
        alpha_corrected=alpha_corrected,
    )
    min_effective_n_005 = minimum_effective_n_for_effect(
        effect_size=0.05,
        sigma_synergy=sigma_synergy,
        alpha_corrected=alpha_corrected,
    )

    common_common_cooccurrence_rate = estimate_common_pair_cooccurrence_rate(
        per_pair_df,
        total_samples=estimated_total_samples,
    )
    common_common_effective_fraction = estimate_common_pair_effective_fraction(
        per_pair_df,
        total_samples=estimated_total_samples,
    )
    min_cohort_effect_01 = (
        int(np.ceil(min_effective_n_01 / common_common_effective_fraction))
        if common_common_effective_fraction
        else None
    )
    min_cohort_effect_005 = (
        int(np.ceil(min_effective_n_005 / common_common_effective_fraction))
        if common_common_effective_fraction
        else None
    )

    summary_dict = {
        "noise_estimation_method": noise_method,
        "sigma_synergy": float(sigma_synergy),
        "alpha": float(args.alpha),
        "correction_method": args.correction,
        "alpha_corrected": float(alpha_corrected),
        "testable_pair_definition": "all four cells in the 2x2 carrier contingency table have >= 5 samples",
        "estimated_total_samples": estimated_total_samples,
        "n_total_pairs": int(len(per_pair_df)),
        "n_testable_pairs": n_testable,
        "median_min_cell_count": float(per_pair_df["min_cell_count"].median()),
        "median_effective_n": float(per_pair_df["n_effective"].median()),
        "median_mde": float(finite_mde.median()) if len(finite_mde) > 0 else None,
        "mean_mde": float(finite_mde.mean()) if len(finite_mde) > 0 else None,
        "proportion_pairs_mde_lt_0.1": float(prop_detectable),
        "median_common_common_cooccurrence_rate": common_common_cooccurrence_rate,
        "median_common_common_effective_fraction": common_common_effective_fraction,
        "min_effective_n_for_effect_0.1": min_effective_n_01,
        "min_effective_n_for_effect_0.05": min_effective_n_005,
        "min_cohort_for_common_common_effect_0.1": min_cohort_effect_01,
        "min_cohort_for_common_common_effect_0.05": min_cohort_effect_005,
    }

    summary_path = output_dir / "power_analysis_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary_dict, f, default_flow_style=False, sort_keys=False)
    print(f"\n  Summary saved to {summary_path}")

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    print("\nGenerating power analysis plot...")
    plot_path = output_dir / "power_analysis_plot.png"
    create_power_plot(maf_summary, plot_path)

    # ------------------------------------------------------------------
    # 8. Print headline results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Power Analysis Summary")
    print("=" * 60)
    print(f"  Noise method:          {noise_method}")
    print(f"  sigma_synergy:         {sigma_synergy:.6f}")
    print(f"  Testable pairs:        {n_testable:,}")
    print(f"  Median effective N:    {per_pair_df['n_effective'].median():.2f}")
    print(f"  Corrected alpha:       {alpha_corrected:.2e}")
    if len(finite_mde) > 0:
        print(f"  Median MDE:            {finite_mde.median():.4f}")
        print(f"  Pairs with MDE < 0.1:  {prop_detectable:.1%}")
    if min_cohort_effect_01 is not None and min_cohort_effect_005 is not None:
        print(f"  Min cohort for d=0.10: {min_cohort_effect_01:,}")
        print(f"  Min cohort for d=0.05: {min_cohort_effect_005:,}")
    else:
        print("  Min cohort estimates:  unavailable (no common-common co-occurrence basis)")
    print("=" * 60)


if __name__ == "__main__":
    main()
