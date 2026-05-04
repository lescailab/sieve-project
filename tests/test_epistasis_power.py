import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scripts.epistasis_power_analysis import (
    add_pair_contingency_metrics,
    build_summary_metric_definitions,
    compute_alpha_threshold,
    compute_effective_n_from_counts,
    compute_mde,
    estimate_sigma_synergy,
    parse_args,
    summarise_mde_detection_rates,
    summarise_power_by_maf_bin,
)


def test_compute_mde_matches_closed_form_solution():
    sigma_synergy = 0.1
    n_cooccur = np.array([100.0])
    alpha_corrected = 0.05 / 1000.0

    observed = compute_mde(n_cooccur, sigma_synergy, alpha_corrected)[0]
    expected = (
        stats.norm.ppf(1.0 - alpha_corrected / 2.0) + stats.norm.ppf(0.8)
    ) * sigma_synergy / np.sqrt(100.0)

    assert observed == pytest.approx(expected)


def test_compute_alpha_threshold_bonferroni():
    assert compute_alpha_threshold(0.05, 1000, "bonferroni") == pytest.approx(0.00005)


def test_compute_alpha_threshold_fdr_bh_returns_alpha():
    assert compute_alpha_threshold(0.05, 1000, "fdr_bh") == pytest.approx(0.05)


def test_old_fdr_value_errors_out(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "epistasis_power_analysis.py",
            "--cooccurrence",
            "cooccurrence_per_pair.csv",
            "--cooccurrence-summary",
            "cooccurrence_by_maf_bin.csv",
            "--output-dir",
            "power_analysis",
            "--correction",
            "fdr",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        parse_args()
    assert "fdr_bh" in capsys.readouterr().err


def test_compute_effective_n_requires_all_four_cells():
    values = compute_effective_n_from_counts(
        n11=np.array([10, 10]),
        n10=np.array([10, 0]),
        n01=np.array([10, 10]),
        n00=np.array([10, 10]),
    )

    assert values[0] == pytest.approx(2.5)
    assert values[1] == 0.0


def test_add_pair_contingency_metrics_derives_full_2x2_counts():
    df = pd.DataFrame(
        {
            "freq_a": [0.5],
            "freq_b": [0.4],
            "expected_cooccur": [4.0],
            "n_cooccur": [3],
        }
    )

    result = add_pair_contingency_metrics(df, total_samples=20)

    assert result.loc[0, "carrier_count_a"] == 10
    assert result.loc[0, "carrier_count_b"] == 8
    assert result.loc[0, "n_only_a"] == 7
    assert result.loc[0, "n_only_b"] == 5
    assert result.loc[0, "n_neither"] == 5
    assert result.loc[0, "min_cell_count"] == 3


def test_summarise_power_by_maf_bin_groups_rows_correctly():
    per_pair_df = pd.DataFrame(
        {
            "maf_bin_a": ["1-5%", "1-5%", "5-10%"],
            "maf_bin_b": ["5-10%", "5-10%", "5-10%"],
            "n_cooccur": [4, 8, 10],
            "min_cell_count": [2, 5, 10],
            "n_effective": [1.0, 4.0, 8.0],
            "mde": [0.2, 0.1, 0.05],
        }
    )

    summary = summarise_power_by_maf_bin(per_pair_df)
    grouped = summary.set_index(["maf_bin_a", "maf_bin_b"])

    assert grouped.loc[("1-5%", "5-10%"), "n_pairs"] == 2
    assert grouped.loc[("1-5%", "5-10%"), "median_n_cooccur"] == pytest.approx(6.0)
    assert grouped.loc[("1-5%", "5-10%"), "median_n_effective"] == pytest.approx(2.5)
    assert grouped.loc[("1-5%", "5-10%"), "n_testable_pairs"] == 1


def test_zero_effective_sample_size_reports_infinite_mde():
    values = compute_mde(np.array([0, 5]), sigma_synergy=0.1, alpha_corrected=0.05)
    assert np.isinf(values[0])
    assert np.isfinite(values[1])


def test_summarise_mde_detection_rates_distinguishes_all_vs_finite_pairs():
    summary = summarise_mde_detection_rates(np.array([0.05, np.inf, 0.2]), threshold=0.1)

    assert summary["n_pairs_with_finite_mde"] == 2
    assert summary["proportion_testable_pairs_mde_lt_threshold"] == pytest.approx(0.5)
    assert summary["proportion_all_pairs_mde_lt_threshold"] == pytest.approx(1 / 3)
    assert summary["median_mde_finite_pairs"] == pytest.approx(0.125)


def test_build_summary_metric_definitions_explains_mde():
    definitions = build_summary_metric_definitions()

    assert "n_pairs_with_finite_mde" in definitions
    assert "minimum detectable effect" in definitions["median_mde_finite_pairs"].lower()


def test_estimate_sigma_synergy_uses_null_attributions_without_real_npz(tmp_path):
    variant_scores = np.empty(2, dtype=object)
    variant_scores[0] = np.array([0.1, -0.1])
    variant_scores[1] = np.array([0.2, -0.2])

    npz_path = tmp_path / "null_attributions.npz"
    np.savez(npz_path, variant_scores=variant_scores)

    sigma_synergy, method = estimate_sigma_synergy(
        real_npz=None,
        null_npz=str(npz_path),
        epistasis_csv=None,
    )

    assert method == "null_attributions"
    assert sigma_synergy == pytest.approx(np.sqrt(3.0) * np.std(np.array([0.1, -0.1, 0.2, -0.2])))
