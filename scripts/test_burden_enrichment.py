"""
Test whether SIEVE-derived gene sets show significantly stronger case-control
burden differences than random gene sets of the same size.

Uses a pre-computed gene-level burden matrix (from extract_validation_burden.py)
to avoid re-parsing the VCF for each permutation.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.data.vcf_parser import load_phenotypes

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test burden enrichment of SIEVE gene sets vs random permutations.",
    )
    parser.add_argument(
        "--burden-dir",
        type=Path,
        required=True,
        help="Directory with burden TSVs from extract_validation_burden.py",
    )
    parser.add_argument(
        "--gene-matrix",
        type=Path,
        default=None,
        help="Path to gene_burden_matrix.parquet (default: <burden-dir>/gene_burden_matrix.parquet)",
    )
    parser.add_argument(
        "--background-genes",
        type=Path,
        default=None,
        help=(
            "Text file listing all gene symbols in the validation cohort (one per line). "
            "If not provided, background genes are derived from the gene burden matrix."
        ),
    )
    parser.add_argument(
        "--phenotypes",
        type=Path,
        default=None,
        help="Phenotype file (only needed if burden TSVs are not available in --burden-dir)",
    )
    parser.add_argument(
        "--sieve-genes",
        type=Path,
        required=True,
        help="SIEVE gene list TSV (same as extract_validation_burden input)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for enrichment results",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of random gene set permutations (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="Gene set sizes to test (default: 50 100 200)",
    )
    parser.add_argument(
        "--covariates",
        type=Path,
        default=None,
        help="Optional TSV with covariates for logistic regression",
    )
    parser.add_argument(
        "--consequence-types",
        nargs="+",
        default=["total"],
        help="Burden types to test (default: total). Options: total, missense, lof, synonymous",
    )
    parser.add_argument(
        "--correction",
        choices=["bonferroni", "fdr_bh"],
        default="fdr_bh",
        help=(
            "Multiple-testing correction for the multi-top-K summary in the "
            "report (default fdr_bh)."
        ),
    )
    return parser.parse_args(argv)


def logistic_regression_z(
    burden: np.ndarray,
    labels: np.ndarray,
    covariates: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Fit logistic regression and return z-statistic and p-value for burden.

    Parameters
    ----------
    burden : np.ndarray
        Per-sample burden counts, shape (n_samples,).
    labels : np.ndarray
        Binary labels (0/1), shape (n_samples,).
    covariates : np.ndarray or None
        Optional covariate matrix, shape (n_samples, n_covariates).

    Returns
    -------
    z_stat : float
        Z-statistic for the burden coefficient.
    p_value : float
        Two-sided p-value.
    """
    import warnings

    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    if covariates is not None:
        X = np.column_stack([burden, covariates])
    else:
        X = burden.reshape(-1, 1)

    X = sm.add_constant(X)

    try:
        model = sm.Logit(labels, X)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings(
                "ignore", message=".*Perfect separation.*"
            )
            warnings.filterwarnings(
                "ignore", message=".*Maximum Likelihood optimization failed.*"
            )
            result = model.fit(disp=0, maxiter=100)
        z_stat = result.tvalues[1]
        p_value = result.pvalues[1]
        if np.isnan(z_stat) or np.isnan(p_value):
            z_stat = 0.0
            p_value = 1.0
    except (PerfectSeparationError, np.linalg.LinAlgError):
        z_stat = 0.0
        p_value = 1.0

    return float(z_stat), float(p_value)


def mannwhitney_test(
    burden: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """
    Mann-Whitney U test comparing burden between cases and controls.

    Returns
    -------
    U_stat : float
    p_value : float
    """
    cases = burden[labels == 1]
    controls = burden[labels == 0]

    if len(cases) == 0 or len(controls) == 0:
        return 0.0, 1.0

    try:
        u_stat, p_value = stats.mannwhitneyu(cases, controls, alternative="two-sided")
    except ValueError:
        u_stat, p_value = 0.0, 1.0

    return float(u_stat), float(p_value)


def compute_burden_for_gene_set(
    gene_matrix: pd.DataFrame,
    gene_set: set[str],
) -> np.ndarray:
    """
    Sum burden across columns matching the gene set.

    Uses Index.get_indexer_for for O(k) lookup instead of scanning all columns.

    Parameters
    ----------
    gene_matrix : pd.DataFrame
        Gene burden matrix (samples × genes).
    gene_set : Set[str]
        Set of gene names (upper-cased).

    Returns
    -------
    np.ndarray
        Per-sample total burden, shape (n_samples,).
    """
    if not gene_set:
        return np.zeros(len(gene_matrix), dtype=np.float64)

    idx = gene_matrix.columns.get_indexer_for(list(gene_set))
    valid_idx = idx[idx != -1]
    if valid_idx.size == 0:
        return np.zeros(len(gene_matrix), dtype=np.float64)

    return gene_matrix.iloc[:, valid_idx].sum(axis=1).values


def run_enrichment_test(
    gene_matrix: pd.DataFrame,
    labels: np.ndarray,
    sieve_genes: set[str],
    background_genes: list[str],
    n_permutations: int = 10000,
    seed: int = 42,
    covariates: np.ndarray | None = None,
) -> dict:
    """
    Run permutation-based enrichment test for a single gene set.

    Parameters
    ----------
    gene_matrix : pd.DataFrame
        Gene burden matrix (samples × genes).
    labels : np.ndarray
        Binary phenotype labels.
    sieve_genes : Set[str]
        Target gene set (upper-cased).
    background_genes : List[str]
        All gene symbols available for permutation (upper-cased).
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.
    covariates : np.ndarray or None
        Optional covariates.

    Returns
    -------
    dict
        Results including observed and null statistics.
    """
    rng = np.random.default_rng(seed)
    k = len(sieve_genes)

    # Observed burden
    observed_burden = compute_burden_for_gene_set(gene_matrix, sieve_genes)
    obs_z, obs_p = logistic_regression_z(observed_burden, labels, covariates)
    obs_u, obs_u_p = mannwhitney_test(observed_burden, labels)

    cases_mask = labels == 1
    controls_mask = labels == 0

    result = {
        "observed": {
            "mean_cases": float(observed_burden[cases_mask].mean()) if cases_mask.any() else 0.0,
            "mean_controls": float(observed_burden[controls_mask].mean()) if controls_mask.any() else 0.0,
            "logistic_z": obs_z,
            "logistic_p": obs_p,
            "mannwhitney_U": obs_u,
            "mannwhitney_p": obs_u_p,
        },
    }

    # Null distribution
    null_z_values = np.zeros(n_permutations)
    bg_array = np.array(background_genes)

    for i in range(n_permutations):
        perm_indices = rng.choice(len(bg_array), size=k, replace=False)
        perm_genes = set(bg_array[perm_indices])
        perm_burden = compute_burden_for_gene_set(gene_matrix, perm_genes)
        null_z, _ = logistic_regression_z(perm_burden, labels, covariates)
        null_z_values[i] = null_z

    # Empirical p-value (using absolute z for two-sided test)
    n_extreme = np.sum(np.abs(null_z_values) >= np.abs(obs_z))
    empirical_p = (n_extreme + 1) / (n_permutations + 1)

    # Percentile rank of observed |z| in null |z| distribution
    percentile_rank = float(
        np.sum(np.abs(null_z_values) < np.abs(obs_z)) / n_permutations * 100
    )

    result["permutation"] = {
        "n_permutations": n_permutations,
        "seed": seed,
        "empirical_p": float(empirical_p),
        "percentile_rank": percentile_rank,
        "null_mean_z": float(null_z_values.mean()),
        "null_std_z": float(null_z_values.std()),
    }
    result["null_z_values"] = null_z_values

    return result


def plot_null_distribution(
    null_z_values: np.ndarray,
    observed_z: float,
    empirical_p: float,
    output_path: Path,
    title: str = "Burden Enrichment Test",
) -> None:
    """Plot histogram of null z-statistics with observed value marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_z_values, bins=50, alpha=0.7, color="steelblue", edgecolor="white", label="Null distribution")
    ax.axvline(observed_z, color="red", linewidth=2, linestyle="--", label=f"Observed z = {observed_z:.2f}")
    ax.set_xlabel("Logistic regression z-statistic")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}\n(empirical p = {empirical_p:.4f})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_report(
    all_results: dict[int, dict],
    output_path: Path,
    cohort_name: str = "validation",
    correction: str = "fdr_bh",
) -> None:
    """Generate markdown summary report."""
    from statsmodels.stats.multitest import multipletests

    lines = [
        f"# Cross-Cohort Burden Validation Report — {cohort_name}",
        "",
        "## Summary",
        "",
    ]

    for k, res in sorted(all_results.items()):
        obs = res["observed"]
        perm = res["permutation"]
        lines.extend([
            f"### Top-{k} genes",
            "",
            f"- **Genes tested**: {res.get('n_sieve_genes', k)} "
            f"({res.get('n_sieve_genes_found', '?')} found in VCF)",
            f"- **Mean burden** — cases: {obs['mean_cases']:.2f}, controls: {obs['mean_controls']:.2f}",
            f"- **Logistic regression z**: {obs['logistic_z']:.3f} (p = {obs['logistic_p']:.4f})",
            f"- **Mann-Whitney U**: {obs['mannwhitney_U']:.0f} (p = {obs['mannwhitney_p']:.4f})",
            f"- **Empirical p** (vs {perm['n_permutations']} permutations): **{perm['empirical_p']:.4f}**",
            f"- **Percentile rank**: {perm['percentile_rank']:.1f}%",
            "",
        ])

    # Multiple testing
    n_tests = len(all_results)
    if n_tests > 1:
        pvals = np.array([res["permutation"]["empirical_p"] for _, res in sorted(all_results.items())])
        reject, padj, _, _ = multipletests(pvals, alpha=0.05, method=correction)
        method_label = {"bonferroni": "Bonferroni", "fdr_bh": "Benjamini-Hochberg FDR"}[correction]
        lines.extend([
            f"## Multiple Testing Correction ({method_label})",
            "",
            f"- Number of tests: {n_tests}",
            f"- Method: {correction}",
            "",
        ])
        for (k, res), pad, rej in zip(
            sorted(all_results.items()), padj, reject, strict=False
        ):
            status = "significant" if rej else "not significant"
            lines.append(
                f"- Top-{k}: empirical p = {res['permutation']['empirical_p']:.4f}, "
                f"adjusted p = {pad:.4f} — **{status}**"
            )
        lines.append("")

    output_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load gene burden matrix
    matrix_path = args.gene_matrix or (args.burden_dir / "gene_burden_matrix.parquet")
    if not matrix_path.exists():
        print(
            f"ERROR: Gene burden matrix not found at {matrix_path}. "
            "Run extract_validation_burden.py with --compute-full-gene-matrix first."
        )
        sys.exit(1)

    print(f"Loading gene burden matrix from {matrix_path}")
    gene_matrix = pd.read_parquet(matrix_path)
    print(f"  Matrix shape: {gene_matrix.shape} (samples × genes)")

    # Load phenotypes from burden files or phenotype file
    # Try loading from burden TSV first
    burden_file = None
    for k in args.top_k:
        candidate = args.burden_dir / f"burden_topK{k}.tsv"
        if candidate.exists():
            burden_file = candidate
            break

    if burden_file is not None:
        burden_df = pd.read_csv(burden_file, sep="\t")
        sample_labels = dict(
            zip(burden_df["sample_id"], burden_df["phenotype"], strict=False)
        )
    elif args.phenotypes is not None:
        sample_labels = load_phenotypes(args.phenotypes)
    else:
        print("ERROR: No phenotype source. Provide --phenotypes or ensure burden files exist.")
        sys.exit(1)

    # Align matrix rows with phenotype labels
    common_samples = [s for s in gene_matrix.index if s in sample_labels]
    if not common_samples:
        print("ERROR: No overlapping samples between matrix and phenotypes.")
        sys.exit(1)

    gene_matrix = gene_matrix.loc[common_samples]
    labels = np.array([sample_labels[s] for s in common_samples])
    print(f"  Samples with phenotypes: {len(common_samples)} "
          f"({(labels == 1).sum()} cases, {(labels == 0).sum()} controls)")

    # Normalise matrix gene names for matching
    gene_matrix.columns = [c.upper() for c in gene_matrix.columns]

    # Background gene universe — must be restricted to genes actually in the matrix
    matrix_gene_set: set[str] = set(gene_matrix.columns)
    if args.background_genes and args.background_genes.exists():
        raw_bg = [
            g.strip().upper()
            for g in args.background_genes.read_text().splitlines()
            if g.strip()
        ]
        bg_genes = [g for g in raw_bg if g in matrix_gene_set]
        dropped = len(raw_bg) - len(bg_genes)
        if dropped > 0:
            print(
                f"  WARNING: {dropped} background genes not found in burden matrix "
                f"and will be ignored."
            )
    else:
        bg_genes = list(gene_matrix.columns)

    if not bg_genes:
        print("ERROR: No background genes overlap with burden matrix columns.")
        sys.exit(1)

    print(f"  Background gene universe: {len(bg_genes)} genes")

    # Load SIEVE gene list
    sep = "\t"
    with open(args.sieve_genes) as f:
        first_line = f.readline()
        if "," in first_line and "\t" not in first_line:
            sep = ","
    sieve_df = pd.read_csv(args.sieve_genes, sep=sep)
    sieve_df = sieve_df.sort_values("gene_rank")

    # Load covariates if provided
    covariates = None
    if args.covariates and args.covariates.exists():
        cov_df = pd.read_csv(args.covariates, sep="\t", index_col=0)
        cov_df = cov_df.loc[common_samples]
        covariates = cov_df.values
        print(f"  Covariates: {cov_df.shape[1]} columns")

    # Run enrichment for each top-k
    all_results: dict[int, dict] = {}

    for k in args.top_k:
        actual_k = min(k, len(sieve_df))
        sieve_genes_all = set(sieve_df.head(actual_k)["gene_name"].str.upper())

        # Restrict to genes actually present in the burden matrix so that
        # observed and permuted gene sets are comparable
        found_in_matrix = sieve_genes_all & matrix_gene_set
        missing = sieve_genes_all - found_in_matrix
        sieve_genes = found_in_matrix

        print(f"\n--- Top-{k} ({len(sieve_genes)} of {len(sieve_genes_all)} genes found in matrix) ---")
        if missing:
            print(f"  Missing genes: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
        if not sieve_genes:
            print("  Skipping: no SIEVE genes present in the burden matrix.")
            continue

        # Run each consequence type
        for csq_type in args.consequence_types:
            if csq_type == "total":
                matrix_to_use = gene_matrix
            else:
                csq_matrix_path = args.burden_dir / f"gene_burden_matrix_{csq_type}.parquet"
                if not csq_matrix_path.exists():
                    print(f"  Skipping {csq_type}: matrix not found at {csq_matrix_path}")
                    continue
                matrix_to_use = pd.read_parquet(csq_matrix_path)
                matrix_to_use = matrix_to_use.loc[common_samples]
                matrix_to_use.columns = [c.upper() for c in matrix_to_use.columns]

            print(f"  Testing {csq_type} burden...")
            result = run_enrichment_test(
                gene_matrix=matrix_to_use,
                labels=labels,
                sieve_genes=sieve_genes,
                background_genes=bg_genes,
                n_permutations=args.n_permutations,
                seed=args.seed,
                covariates=covariates,
            )

            result["n_sieve_genes"] = len(sieve_genes)
            result["n_sieve_genes_found"] = len(found_in_matrix)
            result["n_sieve_genes_missing"] = len(missing)
            result["missing_genes"] = sorted(missing)

            # Save null z-values
            null_z = result.pop("null_z_values")
            suffix = f"_topK{k}" if csq_type == "total" else f"_topK{k}_{csq_type}"
            np.savez_compressed(
                args.output_dir / f"null_distribution{suffix}.npz",
                null_z=null_z,
            )

            # Plot
            plot_null_distribution(
                null_z,
                result["observed"]["logistic_z"],
                result["permutation"]["empirical_p"],
                args.output_dir / f"enrichment_plot{suffix}.png",
                title=f"SIEVE Top-{k} Genes — {csq_type} burden",
            )

            # Save YAML
            with open(args.output_dir / f"enrichment{suffix}.yaml", "w") as f:
                yaml.dump(result, f, default_flow_style=False, sort_keys=False)

            print(f"    Observed z = {result['observed']['logistic_z']:.3f}, "
                  f"empirical p = {result['permutation']['empirical_p']:.4f}")

            if csq_type == "total":
                all_results[k] = result

    # Multiple testing correction across all tests
    all_p_values = []
    for _k, res in all_results.items():
        all_p_values.append(res["permutation"]["empirical_p"])

    n_tests = len(all_p_values)
    if n_tests > 0:
        bonferroni_threshold = 0.05 / n_tests
        summary = {
            "n_tests": n_tests,
            "bonferroni_threshold": float(bonferroni_threshold),
            "top_k_values": args.top_k,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
            "results": {},
        }
        for k, res in all_results.items():
            emp_p = res["permutation"]["empirical_p"]
            summary["results"][f"top_{k}"] = {
                "empirical_p": emp_p,
                "significant_after_bonferroni": emp_p < bonferroni_threshold,
                "logistic_z": res["observed"]["logistic_z"],
                "logistic_p": res["observed"]["logistic_p"],
            }

        with open(args.output_dir / "cross_cohort_validation_summary.yaml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    # Generate report
    generate_report(
        all_results,
        args.output_dir / "validation_report.md",
        correction=args.correction,
    )
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
