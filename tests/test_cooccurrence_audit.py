from dataclasses import dataclass

import pytest

from scripts.audit_cooccurrence import (
    assign_maf_bin,
    build_variant_index,
    compute_cooccurrence,
    select_pairs,
)


@dataclass
class DummyVariant:
    chrom: str
    pos: int
    gene: str
    genotype: int = 1
    consequence: str = "missense_variant"


@dataclass
class DummySample:
    sample_id: str
    label: int
    variants: list[DummyVariant]
    sex: str | None = None


def build_samples() -> list[DummySample]:
    return [
        DummySample(
            sample_id="S1",
            label=1,
            variants=[
                DummyVariant("1", 100, "GENE1"),
                DummyVariant("1", 100, "GENE1"),
                DummyVariant("1", 200, "GENE1"),
                DummyVariant("2", 100, "GENE2"),
            ],
        ),
        DummySample(
            sample_id="S2",
            label=1,
            variants=[
                DummyVariant("1", 100, "GENE1"),
                DummyVariant("2", 100, "GENE2"),
                DummyVariant("2", 300, "GENE3"),
            ],
        ),
        DummySample(
            sample_id="S3",
            label=0,
            variants=[
                DummyVariant("1", 100, "GENE1"),
                DummyVariant("1", 200, "GENE1"),
                DummyVariant("1", 400, "GENE3"),
            ],
        ),
        DummySample(
            sample_id="S4",
            label=0,
            variants=[
                DummyVariant("1", 200, "GENE1"),
                DummyVariant("2", 300, "GENE3"),
                DummyVariant("3", 100, "GENE4"),
            ],
        ),
        DummySample(
            sample_id="S5",
            label=1,
            variants=[
                DummyVariant("1", 200, "GENE1"),
                DummyVariant("2", 100, "GENE2"),
                DummyVariant("3", 100, "GENE4"),
            ],
        ),
    ]


def test_build_variant_index_counts_unique_carriers_per_sample():
    samples = build_samples()
    variant_info, sample_variants, n_cases, n_controls = build_variant_index(samples)

    assert n_cases == 3
    assert n_controls == 2
    assert variant_info[("1", 100)]["carrier_count"] == 3
    assert variant_info[("1", 100)]["case_carriers"] == 2
    assert variant_info[("1", 100)]["control_carriers"] == 1
    assert sample_variants[0] == {("1", 100), ("1", 200), ("2", 100)}


def test_compute_cooccurrence_expected_and_flags():
    samples = build_samples()
    variant_info, sample_variants, _, _ = build_variant_index(samples)
    records = compute_cooccurrence(
        pairs=[(("1", 100), ("1", 200)), (("1", 100), ("3", 100))],
        sample_variants=sample_variants,
        variant_info=variant_info,
        all_samples=samples,
        total_samples=len(samples),
        bin_edges=[0.2, 0.4, 0.6],
    )

    first = records[0]
    assert first["carrier_count_a"] == 3
    assert first["carrier_count_b"] == 4
    assert first["n_cooccur"] == 2
    assert first["n_only_a"] == 1
    assert first["n_only_b"] == 2
    assert first["n_neither"] == 0
    assert first["min_cell_count"] == 0
    assert first["n_cooccur_cases"] == 1
    assert first["n_cooccur_controls"] == 1
    assert first["expected_cooccur"] == pytest.approx(2.4)
    assert first["obs_exp_ratio"] == pytest.approx(0.8333)
    assert first["same_gene"] is True
    assert first["same_chrom"] is True

    second = records[1]
    assert second["same_gene"] is False
    assert second["same_chrom"] is False


def test_assign_maf_bin_places_boundary_values_in_upper_bin():
    bin_edges = [0.2, 0.4, 0.6]
    assert assign_maf_bin(0.2, bin_edges) == "20.0-40.0%"
    assert assign_maf_bin(0.4, bin_edges) == "40.0-60.0%"


def test_select_pairs_respects_small_pair_budget():
    samples = build_samples()
    variant_info, _, _, _ = build_variant_index(samples)

    selected = select_pairs(
        variant_info=variant_info,
        top_k=4,
        max_pairs=1,
        bin_edges=[0.2, 0.4, 0.6],
        total_samples=len(samples),
        seed=42,
    )

    assert len(selected) == 1


def test_variants_with_same_position_on_different_chromosomes_are_distinct():
    samples = build_samples()
    variant_info, _, _, _ = build_variant_index(samples)

    assert variant_info[("1", 100)]["carrier_count"] == 3
    assert variant_info[("2", 100)]["carrier_count"] == 3
    assert variant_info[("3", 100)]["carrier_count"] == 2
