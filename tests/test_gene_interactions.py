import math
from dataclasses import dataclass

import pandas as pd
import pytest

import scripts.aggregate_gene_interactions as interactions_mod
from scripts.aggregate_gene_interactions import (
    _normalise_gene_pair,
    annotate_with_null_significance,
    build_carrier_indices,
    build_network_outputs,
    compute_gene_pair_cooccurrence,
    load_gene_rankings,
    standardise_variant_rankings,
)


@dataclass
class DummyVariant:
    chrom: str
    pos: int
    gene: str


@dataclass
class DummySample:
    sample_id: str
    label: int
    variants: list[DummyVariant]


def build_gene_samples() -> list[DummySample]:
    sample_gene_sets = [
        ("S0", 1, ["GENEA", "GENEB"]),
        ("S1", 1, ["GENEA", "GENEB"]),
        ("S2", 1, ["GENEA", "GENEB"]),
        ("S3", 1, ["GENEA", "GENEB"]),
        ("S4", 1, ["GENEA", "GENEC"]),
        ("S5", 0, ["GENEA", "GENEC"]),
        ("S6", 0, ["GENEA", "GENEC"]),
        ("S7", 0, ["GENEB", "GENEC"]),
        ("S8", 0, ["GENEB", "GENEC"]),
        ("S9", 0, ["GENEA", "GENEB", "GENEC"]),
    ]
    chrom_map = {"GENEA": "1", "GENEB": "1", "GENEC": "2"}
    pos_map = {"GENEA": 100, "GENEB": 200, "GENEC": 300}

    samples = []
    for sample_id, label, genes in sample_gene_sets:
        variants = [
            DummyVariant(chrom=chrom_map[gene], pos=pos_map[gene], gene=gene)
            for gene in genes
        ]
        samples.append(DummySample(sample_id=sample_id, label=label, variants=variants))
    return samples


def test_compute_gene_pair_cooccurrence_counts_known_overlaps():
    samples = build_gene_samples()
    gene_to_samples, sample_labels, gene_to_chrom, total_samples = build_carrier_indices(samples)
    gene_rankings = pd.DataFrame(
        {
            "gene_symbol": ["GENEA", "GENEB", "GENEC"],
            "gene_score": [4.0, 9.0, 1.0],
            "gene_rank": [2, 1, 3],
        }
    )

    pairs_df = compute_gene_pair_cooccurrence(
        top_genes=["GENEA", "GENEB", "GENEC"],
        gene_to_samples=gene_to_samples,
        sample_labels=sample_labels,
        total_samples=total_samples,
        gene_to_chrom=gene_to_chrom,
        gene_rankings_df=gene_rankings,
    )
    pairs = pairs_df.set_index(["gene_a", "gene_b"])

    assert pairs.loc[("GENEA", "GENEB"), "n_cooccur"] == 5
    assert pairs.loc[("GENEA", "GENEC"), "n_cooccur"] == 4
    assert pairs.loc[("GENEB", "GENEC"), "n_cooccur"] == 3
    assert bool(pairs.loc[("GENEA", "GENEB"), "same_chrom"]) is True
    assert bool(pairs.loc[("GENEA", "GENEC"), "same_chrom"]) is False


def test_gene_pair_order_is_normalised():
    assert _normalise_gene_pair("GENEB", "GENEA") == ("GENEA", "GENEB")


def test_interaction_score_matches_rank_quantile_formula():
    samples = build_gene_samples()
    gene_to_samples, sample_labels, gene_to_chrom, total_samples = build_carrier_indices(samples)
    gene_rankings = pd.DataFrame(
        {
            "gene_symbol": ["GENEA", "GENEB"],
            "gene_score": [4.0, 9.0],
            "gene_rank": [2, 1],
        }
    )

    pairs_df = compute_gene_pair_cooccurrence(
        top_genes=["GENEA", "GENEB"],
        gene_to_samples=gene_to_samples,
        sample_labels=sample_labels,
        total_samples=total_samples,
        gene_to_chrom=gene_to_chrom,
        gene_rankings_df=gene_rankings,
    )

    expected = math.sqrt(0.5 * 1.0) * math.log1p(5)
    assert pairs_df.iloc[0]["interaction_score"] == pytest.approx(round(expected, 6))
    assert pairs_df.iloc[0]["gene_score_a"] == pytest.approx(4.0)
    assert pairs_df.iloc[0]["gene_score_b"] == pytest.approx(9.0)
    assert pairs_df.iloc[0]["gene_score_quantile_a"] == pytest.approx(0.5)
    assert pairs_df.iloc[0]["gene_score_quantile_b"] == pytest.approx(1.0)


def test_filtering_by_min_cooccurrence_behaves_as_expected():
    samples = build_gene_samples()
    gene_to_samples, sample_labels, gene_to_chrom, total_samples = build_carrier_indices(samples)
    gene_rankings = pd.DataFrame(
        {
            "gene_symbol": ["GENEA", "GENEB", "GENEC"],
            "gene_score": [4.0, 9.0, 1.0],
            "gene_rank": [2, 1, 3],
        }
    )
    pairs_df = compute_gene_pair_cooccurrence(
        top_genes=["GENEA", "GENEB", "GENEC"],
        gene_to_samples=gene_to_samples,
        sample_labels=sample_labels,
        total_samples=total_samples,
        gene_to_chrom=gene_to_chrom,
        gene_rankings_df=gene_rankings,
    )

    filtered = pairs_df[pairs_df["n_cooccur"] >= 4]
    assert set(zip(filtered["gene_a"], filtered["gene_b"], strict=False)) == {
        ("GENEA", "GENEB"),
        ("GENEA", "GENEC"),
    }


def test_min_cooccur_filter_skips_fisher_for_discarded_pairs(monkeypatch):
    samples = build_gene_samples()
    gene_to_samples, sample_labels, gene_to_chrom, total_samples = build_carrier_indices(samples)
    gene_rankings = pd.DataFrame(
        {
            "gene_symbol": ["GENEA", "GENEB", "GENEC"],
            "gene_score": [4.0, 9.0, 1.0],
            "gene_rank": [2, 1, 3],
        }
    )

    def fail_fisher(*_args, **_kwargs):
        raise AssertionError("fisher_exact called for a discarded pair")

    monkeypatch.setattr(interactions_mod, "fisher_exact", fail_fisher)
    pairs_df = compute_gene_pair_cooccurrence(
        top_genes=["GENEA", "GENEB", "GENEC"],
        gene_to_samples=gene_to_samples,
        sample_labels=sample_labels,
        total_samples=total_samples,
        gene_to_chrom=gene_to_chrom,
        gene_rankings_df=gene_rankings,
        min_cooccur_samples=99,
    )

    assert pairs_df.empty


def test_degree_centrality_counts_unique_partners():
    pairs_df = pd.DataFrame(
        {
            "gene_a": ["GENEA", "GENEA", "GENEA"],
            "gene_b": ["GENEB", "GENEC", "GENED"],
            "interaction_score": [1.0, 2.0, 3.0],
            "n_cooccur": [5, 5, 5],
        }
    )
    gene_rankings = pd.DataFrame(
        {
            "gene_symbol": ["GENEA", "GENEB", "GENEC", "GENED"],
            "gene_score": [5.0, 4.0, 3.0, 2.0],
            "gene_rank": [1, 2, 3, 4],
        }
    )

    _, nodes_df = build_network_outputs(pairs_df, gene_rankings)
    nodes = nodes_df.set_index("gene")

    assert nodes.loc["GENEA", "degree"] == 3
    assert nodes.loc["GENEA", "n_partners"] == 3


def test_gene_frequency_one_behaves_correctly():
    gene_to_samples = {
        "GENEA": set(range(10)),
        "GENEB": {0, 1, 2, 3, 4},
    }
    sample_labels = {idx: int(idx < 5) for idx in range(10)}
    gene_to_chrom = {"GENEA": "1", "GENEB": "2"}
    gene_rankings = pd.DataFrame(
        {
            "gene_symbol": ["GENEA", "GENEB"],
            "gene_score": [3.0, 2.0],
            "gene_rank": [1, 2],
        }
    )

    pairs_df = compute_gene_pair_cooccurrence(
        top_genes=["GENEA", "GENEB"],
        gene_to_samples=gene_to_samples,
        sample_labels=sample_labels,
        total_samples=10,
        gene_to_chrom=gene_to_chrom,
        gene_rankings_df=gene_rankings,
    )

    row = pairs_df.iloc[0]
    assert row["freq_gene_a"] == pytest.approx(1.0)
    assert row["n_cooccur"] == 5
    assert row["expected_cooccur"] == pytest.approx(5.0)


def test_load_gene_rankings_resolves_gene_symbols_from_variant_rankings(tmp_path):
    gene_rankings_path = tmp_path / "gene_rankings.csv"
    variant_rankings_path = tmp_path / "variant_rankings.csv"

    pd.DataFrame(
        {
            "gene_id": [101, 202],
            "gene_score": [2.0, 1.0],
            "gene_rank": [1, 2],
        }
    ).to_csv(gene_rankings_path, index=False)
    pd.DataFrame(
        {
            "gene_id": [101, 202],
            "gene_name": ["GENEA", "GENEB"],
            "chromosome": ["1", "2"],
            "position": [100, 200],
            "mean_attribution": [0.2, 0.1],
        }
    ).to_csv(variant_rankings_path, index=False)

    variant_df = standardise_variant_rankings(pd.read_csv(variant_rankings_path))
    loaded, metadata = load_gene_rankings(
        gene_rankings_path=gene_rankings_path,
        variant_rankings_df=variant_df,
        top_k=10,
        min_score=0.0,
        significance_threshold="p_0.01",
        min_significant_variants=1,
        allow_nonsignificant_genes=True,
    )

    assert loaded["gene_symbol"].tolist() == ["GENEA", "GENEB"]
    assert metadata["uses_null_significance"] is False


def test_load_gene_rankings_accepts_corrected_gene_z_scores(tmp_path):
    gene_rankings_path = tmp_path / "corrected_gene_rankings.csv"
    variant_rankings_path = tmp_path / "corrected_variant_rankings.csv"

    pd.DataFrame(
        {
            "gene_name": ["GENEA", "GENEB"],
            "gene_z_score": [3.2, 1.5],
            "gene_rank": [1, 2],
        }
    ).to_csv(gene_rankings_path, index=False)
    pd.DataFrame(
        {
            "gene_name": ["GENEA", "GENEB"],
            "chromosome": ["1", "2"],
            "position": [100, 200],
            "z_attribution": [3.2, 1.5],
        }
    ).to_csv(variant_rankings_path, index=False)

    variant_df = standardise_variant_rankings(pd.read_csv(variant_rankings_path))
    loaded, _ = load_gene_rankings(
        gene_rankings_path=gene_rankings_path,
        variant_rankings_df=variant_df,
        top_k=10,
        min_score=0.0,
        significance_threshold="p_0.01",
        min_significant_variants=1,
        allow_nonsignificant_genes=True,
    )

    assert loaded["gene_symbol"].tolist() == ["GENEA", "GENEB"]
    assert loaded["gene_score"].tolist() == pytest.approx([3.2, 1.5])


def test_null_significance_is_recomputed_in_corrected_score_space(tmp_path):
    real_path = tmp_path / "corrected_variant_rankings.csv"
    null_path = tmp_path / "null_variant_rankings.csv"

    pd.DataFrame(
        {
            "gene_name": ["GENEA", "GENEB"],
            "chromosome": ["1", "2"],
            "position": [100, 200],
            "z_attribution": [3.5, 0.2],
        }
    ).to_csv(real_path, index=False)
    pd.DataFrame(
        {
            "gene_name": ["GENEA", "GENEB", "GENEC", "GENED"],
            "chromosome": ["1", "1", "2", "2"],
            "position": [100, 110, 200, 210],
            "mean_attribution": [0.0, 1.0, 0.0, 1.0],
        }
    ).to_csv(null_path, index=False)

    real_df = standardise_variant_rankings(pd.read_csv(real_path))
    annotated, metadata = annotate_with_null_significance(
        real_df,
        null_rankings_path=null_path,
        significance_threshold="p_0.01",
    )

    assert metadata["significance_source"] == "null_rankings_recomputed"
    assert bool(annotated.loc[annotated["gene_symbol"] == "GENEA", "exceeds_null_p_0.01"].iloc[0]) is True
    assert bool(annotated.loc[annotated["gene_symbol"] == "GENEB", "exceeds_null_p_0.01"].iloc[0]) is False


def test_load_gene_rankings_filters_to_null_significant_genes(tmp_path):
    gene_rankings_path = tmp_path / "corrected_gene_rankings.csv"
    variant_rankings_path = tmp_path / "corrected_variant_rankings.csv"
    null_path = tmp_path / "null_variant_rankings.csv"

    pd.DataFrame(
        {
            "gene_name": ["GENEA", "GENEB"],
            "gene_z_score": [3.5, 2.0],
            "gene_rank": [1, 2],
        }
    ).to_csv(gene_rankings_path, index=False)
    pd.DataFrame(
        {
            "gene_name": ["GENEA", "GENEB"],
            "chromosome": ["1", "2"],
            "position": [100, 200],
            "z_attribution": [3.5, 0.2],
        }
    ).to_csv(variant_rankings_path, index=False)
    pd.DataFrame(
        {
            "gene_name": ["GENEA", "GENEB", "GENEC", "GENED"],
            "chromosome": ["1", "1", "2", "2"],
            "position": [100, 110, 200, 210],
            "mean_attribution": [0.0, 1.0, 0.0, 1.0],
        }
    ).to_csv(null_path, index=False)

    variant_df = standardise_variant_rankings(pd.read_csv(variant_rankings_path))
    variant_df, metadata = annotate_with_null_significance(
        variant_df,
        null_rankings_path=null_path,
        significance_threshold="p_0.01",
    )

    loaded, gene_metadata = load_gene_rankings(
        gene_rankings_path=gene_rankings_path,
        variant_rankings_df=variant_df,
        top_k=10,
        min_score=0.0,
        significance_threshold="p_0.01",
        min_significant_variants=1,
        allow_nonsignificant_genes=False,
    )

    assert metadata["significance_source"] == "null_rankings_recomputed"
    assert gene_metadata["uses_null_significance"] is True
    assert loaded["gene_symbol"].tolist() == ["GENEA"]
    assert loaded.iloc[0]["significant_variant_count"] == 1
