"""Tests for fixed and learned ALiBi relative-position execution.

Fixed and learned ALiBi share one genomic prior: slopes derived from the target
weight ratio, the characteristic distances (1, 10, 100, 5000) bp, and the
distance scale. Fixed keeps those slopes; learned starts from them and trains
them. Tests use four heads because the prior is defined only for four heads.
"""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass

import pytest
import torch
import torch.nn as nn

from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    DEFAULT_ALIBI_CHARACTERISTIC_DISTANCES_BP,
    DEFAULT_ALIBI_DISTANCE_SCALE,
    DEFAULT_ALIBI_TARGET_WEIGHT_RATIO,
    AbsolutePositionEncoding,
    AlibiDistanceFunction,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    resolve_position_encoding_config,
)
from src.encoding.position_layout import LearnedBinnedAbsolutePositionLayout
from src.models.chunked_sieve import ChunkedSIEVEModel
from src.models.position_runtime import (
    FixedAlibiRelativePositionRuntime,
    LearnedAlibiRelativePositionRuntime,
    build_alibi_initial_slope_logits,
    build_genomic_alibi_slopes,
    build_relative_position_runtime,
    validate_attention_runtime_support,
    validate_phase7_runtime_support,
)
from src.models.sieve import SIEVE

MODEL_KWARGS = {
    "latent_dim": 8,
    "hidden_dim": 10,
    "num_heads": 4,
    "num_attention_layers": 1,
    "classifier_hidden_dim": 12,
    "dropout": 0.0,
    "num_covariates": 0,
    "classifier_type": "flatten",
}

# Canonical genomic slopes m_h = -ln(0.75) / ln(1 + D_h / 1 bp) for
# D = (1, 10, 100, 5000) bp, written out so the tests do not reuse the
# implementation formula.
GENOMIC_FIXED_SLOPES = (
    0.4150374992788438,
    0.11997274264444947,
    0.06233468257264059,
    0.03377583571193253,
)
TEST_DISTANCES_BP = (1.0, 10.0, 100.0, 5000.0)


def _expected_slopes(*, ratio: float = 0.75, scale: float = 1.0) -> tuple[float, ...]:
    """Independent test-side derivation of m_h = -ln(r) / ln(1 + D_h / s)."""
    return tuple(-math.log(ratio) / math.log(1.0 + d / scale) for d in TEST_DISTANCES_BP)


# Many runtime tests use a 10 bp scale so hand-computed transforms stay simple.
SLOPES_S10 = _expected_slopes(scale=10.0)


def _resolve_custom(
    *,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    relative: RelativePositionEncoding = RelativePositionEncoding.ALIBI_FIXED,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    num_chromosomes: int = 3,
    position_dim: int = 4,
    position_bin_size: int = 10,
    alibi_distance_function: AlibiDistanceFunction = AlibiDistanceFunction.LOG1P,
    alibi_distance_scale: float = 10.0,
):
    request_kwargs = {}
    if relative in {
        RelativePositionEncoding.ALIBI_FIXED,
        RelativePositionEncoding.ALIBI_LEARNED,
    }:
        request_kwargs.update(
            {
                "alibi_distance_function": alibi_distance_function,
                "alibi_distance_scale": alibi_distance_scale,
            }
        )
    if absolute is AbsolutePositionEncoding.SINUSOIDAL:
        request_kwargs["position_dim"] = position_dim
    if absolute is AbsolutePositionEncoding.LEARNED_BINNED:
        request_kwargs["position_dim"] = position_dim
        request_kwargs["position_bin_size"] = position_bin_size
    return resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=absolute,
            relative_position_encoding=relative,
            chromosome_encoding=chromosome,
            cross_chromosome_policy=cross_policy,
            **request_kwargs,
        ),
        AnnotationLevel.L3,
        latent_dim=MODEL_KWARGS["latent_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_chromosomes=num_chromosomes,
    )


def _layout() -> LearnedBinnedAbsolutePositionLayout:
    return LearnedBinnedAbsolutePositionLayout(
        schema_version=1,
        coordinate_origin=1,
        layout="chromosome_local_contiguous",
        chromosome_lengths_bp=(20, 15, 12),
        bins_per_chromosome=(2, 2, 2),
        num_embeddings=6,
    )


def _model(config, *, layout=None) -> SIEVE:
    model = SIEVE(
        input_dim=config.input_dim,
        num_genes=5,
        latent_dim=MODEL_KWARGS["latent_dim"],
        hidden_dim=MODEL_KWARGS["hidden_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_attention_layers=MODEL_KWARGS["num_attention_layers"],
        classifier_hidden_dim=MODEL_KWARGS["classifier_hidden_dim"],
        dropout=MODEL_KWARGS["dropout"],
        num_chromosomes=config.chromosome.num_chromosomes,
        num_covariates=MODEL_KWARGS["num_covariates"],
        classifier_type=MODEL_KWARGS["classifier_type"],
        position_encoding=config,
        learned_binned_position_layout=layout,
    )
    model.eval()
    return model


def _batch(config):
    content = torch.tensor(
        [
            [
                [0.2, 1.0, 0.0, 0.0, 1.0, 0.3, 0.7],
                [0.8, 0.0, 1.0, 0.0, 0.0, 0.6, 0.1],
                [0.5, 0.0, 0.0, 1.0, 0.0, 0.4, 0.2],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    absolute_width = config.absolute.position_dim or 0
    absolute = torch.zeros(1, 4, absolute_width)
    positions = torch.tensor([[1, 6, 11, 0]], dtype=torch.long)
    gene_ids = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    mask = torch.tensor([[True, True, True, False]])
    chrom_ids = torch.tensor([[0, 1, 0, 0]], dtype=torch.long)
    return content, absolute, positions, gene_ids, mask, chrom_ids


def _fixed_runtime(
    *,
    num_heads: int = 4,
    distance_function: AlibiDistanceFunction = AlibiDistanceFunction.LINEAR,
    distance_scale: float = 10.0,
    target_weight_ratio: float = 0.75,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
) -> FixedAlibiRelativePositionRuntime:
    return FixedAlibiRelativePositionRuntime(
        num_heads=num_heads,
        distance_function=distance_function,
        distance_scale=distance_scale,
        target_weight_ratio=target_weight_ratio,
        cross_chromosome_policy=cross_policy,
    )


def _learned_runtime(
    *,
    num_heads: int = 4,
    distance_function: AlibiDistanceFunction = AlibiDistanceFunction.LINEAR,
    distance_scale: float = 10.0,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
) -> LearnedAlibiRelativePositionRuntime:
    return LearnedAlibiRelativePositionRuntime(
        num_heads=num_heads,
        distance_function=distance_function,
        distance_scale=distance_scale,
        cross_chromosome_policy=cross_policy,
    )


def _runtime(**kwargs) -> FixedAlibiRelativePositionRuntime:
    return _fixed_runtime(**kwargs)


def _initial_logits(
    num_heads: int = 4,
    dtype=torch.float64,
    *,
    ratio: float = 0.75,
    scale: float = 10.0,
) -> torch.Tensor:
    return torch.tensor(
        build_alibi_initial_slope_logits(
            num_heads,
            target_weight_ratio=ratio,
            distance_scale=scale,
        ),
        dtype=dtype,
    )


def _base_scores(dtype=torch.float64) -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]],
                [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ],
        dtype=dtype,
    )


def _unused_query_key(dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    query = torch.randn(1, 4, 3, 2, dtype=dtype)
    key = torch.randn(1, 4, 3, 2, dtype=dtype)
    return query, key


def _adjust(
    runtime: FixedAlibiRelativePositionRuntime | LearnedAlibiRelativePositionRuntime,
    base_scores: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    chrom_ids: torch.Tensor | None = None,
    cross_chromosome_bias: torch.Tensor | None = None,
    alibi_slope_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    query_key_dtype = base_scores.dtype if base_scores.dtype.is_floating_point else torch.float32
    query, key = _unused_query_key(query_key_dtype)
    return runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=(
            torch.tensor([[10, 30, 50]], dtype=torch.long) if positions is None else positions
        ),
        chrom_ids=(torch.tensor([[0, 0, 0]], dtype=torch.long) if chrom_ids is None else chrom_ids),
        position_bias=None,
        cross_chromosome_bias=cross_chromosome_bias,
        alibi_slope_logits=alibi_slope_logits,
    )


def _positional_state_keys(model: nn.Module) -> set[str]:
    tokens = (
        "position_bias.weight",
        "cross_chromosome_bias",
        "chrom_embedding.weight",
        "absolute_position_embedding.weight",
        "alibi_slope",
    )
    return {key for key in model.state_dict() if any(token in key for token in tokens)}


def _assert_parameterless_plain_runtime(runtime) -> None:
    assert not isinstance(runtime, nn.Module)
    assert not hasattr(runtime, "parameters")
    assert not hasattr(runtime, "buffers")
    assert is_dataclass(runtime)
    for field in fields(runtime):
        assert not isinstance(getattr(runtime, field.name), torch.Tensor)


def test_genomic_prior_defaults_and_distances_have_one_authority():
    assert DEFAULT_ALIBI_CHARACTERISTIC_DISTANCES_BP == TEST_DISTANCES_BP
    assert DEFAULT_ALIBI_TARGET_WEIGHT_RATIO == 0.75
    assert DEFAULT_ALIBI_DISTANCE_SCALE == 1.0


def test_genomic_alibi_slopes_match_canonical_coefficients():
    slopes = build_genomic_alibi_slopes(4, target_weight_ratio=0.75, distance_scale=1.0)

    assert len(slopes) == 4
    for observed, expected in zip(slopes, GENOMIC_FIXED_SLOPES, strict=True):
        assert math.isclose(observed, expected, rel_tol=1e-15, abs_tol=0.0)


def test_genomic_alibi_slopes_are_strictly_decreasing_and_positive():
    m0, m1, m2, m3 = build_genomic_alibi_slopes(4, target_weight_ratio=0.75, distance_scale=1.0)

    assert m0 > m1 > m2 > m3 > 0


@pytest.mark.parametrize("ratio", [0.75, 0.9, 0.5, 1e-6, 1 - 1e-9])
@pytest.mark.parametrize("scale", [1.0, 10.0, 37.5])
def test_genomic_alibi_slopes_hit_target_ratio_at_characteristic_distances(ratio, scale):
    # Non-0.75 ratios and non-1 scales are software test values only.
    slopes = build_genomic_alibi_slopes(4, target_weight_ratio=ratio, distance_scale=scale)

    for observed, expected in zip(slopes, _expected_slopes(ratio=ratio, scale=scale), strict=True):
        assert math.isclose(observed, expected, rel_tol=1e-12)
    for slope, distance_bp in zip(slopes, TEST_DISTANCES_BP, strict=True):
        relative_factor = math.exp(-slope * math.log1p(distance_bp / scale))
        assert math.isclose(relative_factor, ratio, rel_tol=1e-9)


def test_genomic_alibi_slopes_change_deterministically_with_ratio():
    slopes_075 = build_genomic_alibi_slopes(4, target_weight_ratio=0.75, distance_scale=1.0)
    slopes_090 = build_genomic_alibi_slopes(4, target_weight_ratio=0.9, distance_scale=1.0)

    assert slopes_090 == build_genomic_alibi_slopes(4, target_weight_ratio=0.9, distance_scale=1.0)
    # A weaker attenuation (larger r) means uniformly smaller slopes, by ln(r) ratio.
    for low, high in zip(slopes_075, slopes_090, strict=True):
        assert high < low
        assert math.isclose(high / low, math.log(0.9) / math.log(0.75), rel_tol=1e-12)


@pytest.mark.parametrize(
    ("head", "characteristic_distance_bp"),
    [(0, 1), (1, 10), (2, 100), (3, 5000)],
)
def test_canonical_characteristic_relative_factor_is_three_quarters(
    head,
    characteristic_distance_bp,
):
    slope = build_genomic_alibi_slopes(4, target_weight_ratio=0.75, distance_scale=1.0)[head]
    # Relative exponential factor exp(-m_h * log1p(d / 1 bp)) on the
    # unnormalised attention weight versus an identical zero-distance pair.
    relative_factor = math.exp(-slope * math.log1p(characteristic_distance_bp / 1.0))

    assert math.isclose(relative_factor, 0.75, rel_tol=1e-12)


# Relative exponential factors (1 + d) ** (-m_h) at r = 0.75 and a 1 bp scale.
# These multiply the unnormalised softmax numerator; they are not probabilities.
GENOMIC_RELATIVE_FACTOR_TABLE = [
    (1, (0.750, 0.920, 0.958, 0.977)),
    (5, (0.475, 0.807, 0.894, 0.941)),
    (10, (0.370, 0.750, 0.861, 0.922)),
    (100, (0.147, 0.575, 0.750, 0.856)),
    (1_000, (0.057, 0.437, 0.650, 0.792)),
    (5_000, (0.029, 0.360, 0.588, 0.750)),
    (10_000, (0.022, 0.331, 0.563, 0.733)),
]


@pytest.mark.parametrize(("distance_bp", "expected_factors"), GENOMIC_RELATIVE_FACTOR_TABLE)
def test_fixed_alibi_runtime_relative_factors_match_genomic_table(distance_bp, expected_factors):
    runtime = _fixed_runtime(distance_function=AlibiDistanceFunction.LOG1P, distance_scale=1.0)
    adjusted = _adjust(
        runtime,
        torch.zeros(1, 4, 2, 2, dtype=torch.float64),
        positions=torch.tensor([[1_000, 1_000 + distance_bp]], dtype=torch.long),
        chrom_ids=torch.tensor([[0, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float64),
    )
    observed = torch.exp(adjusted[0, :, 0, 1])

    torch.testing.assert_close(
        observed,
        torch.tensor(expected_factors, dtype=torch.float64),
        rtol=0.0,
        atol=5e-4,
    )
    # The score penalty is exactly m_h * log1p(distance_bp / distance_scale).
    torch.testing.assert_close(
        adjusted[0, :, 0, 1],
        torch.tensor(
            [-slope * math.log1p(distance_bp / 1.0) for slope in GENOMIC_FIXED_SLOPES],
            dtype=torch.float64,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fixed_alibi_score_decreases_monotonically_with_distance_for_every_head():
    runtime = _fixed_runtime(distance_function=AlibiDistanceFunction.LOG1P, distance_scale=1.0)
    distances = [0, 1, 5, 10, 100, 1_000, 5_000, 10_000, 100_000]
    positions = torch.tensor([[1_000 + distance for distance in distances]], dtype=torch.long)
    n = len(distances)
    adjusted = _adjust(
        runtime,
        torch.zeros(1, 4, n, n, dtype=torch.float64),
        positions=positions,
        chrom_ids=torch.zeros(1, n, dtype=torch.long),
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float64),
    )

    for head in range(4):
        row = adjusted[0, head, 0, :]
        assert torch.all(row[1:] < row[:-1])


@pytest.mark.parametrize("num_heads", [1, 2, 3, 6, 8, 16])
def test_genomic_alibi_rejects_non_four_head_architectures_for_fixed_and_learned(num_heads):
    with pytest.raises(ValueError, match="num_heads=4"):
        build_genomic_alibi_slopes(num_heads, target_weight_ratio=0.75, distance_scale=1.0)
    with pytest.raises(ValueError, match="num_heads=4"):
        build_alibi_initial_slope_logits(num_heads, target_weight_ratio=0.75, distance_scale=1.0)
    with pytest.raises(ValueError, match="num_heads=4"):
        _fixed_runtime(num_heads=num_heads)
    with pytest.raises(ValueError, match="num_heads=4"):
        _learned_runtime(num_heads=num_heads)


@pytest.mark.parametrize("bad_ratio", [0.0, 1.0, -0.5, 1.5, math.nan, math.inf, -math.inf, True])
def test_genomic_alibi_rejects_invalid_target_weight_ratio(bad_ratio):
    with pytest.raises(ValueError, match="alibi_target_weight_ratio"):
        build_genomic_alibi_slopes(4, target_weight_ratio=bad_ratio, distance_scale=1.0)
    with pytest.raises(ValueError, match="alibi_target_weight_ratio"):
        _fixed_runtime(target_weight_ratio=bad_ratio)


@pytest.mark.parametrize("ratio", [0.75, 0.9])
@pytest.mark.parametrize("scale", [1.0, 25.0])
def test_learned_alibi_initial_logits_softplus_to_genomic_fixed_slopes(ratio, scale):
    raw_logits = torch.tensor(
        build_alibi_initial_slope_logits(4, target_weight_ratio=ratio, distance_scale=scale),
        dtype=torch.float64,
    )
    fixed = _fixed_runtime(distance_scale=scale, target_weight_ratio=ratio)

    torch.testing.assert_close(
        torch.nn.functional.softplus(raw_logits),
        torch.tensor(fixed.fixed_slopes, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-15,
    )


def test_learned_alibi_initial_logits_are_stable_for_extreme_ratios():
    for ratio in (1e-300, 1 - 1e-12):
        raw_logits = torch.tensor(
            build_alibi_initial_slope_logits(4, target_weight_ratio=ratio, distance_scale=1.0),
            dtype=torch.float64,
        )
        assert torch.isfinite(raw_logits).all()
        torch.testing.assert_close(
            torch.nn.functional.softplus(raw_logits),
            torch.tensor(
                build_genomic_alibi_slopes(4, target_weight_ratio=ratio, distance_scale=1.0),
                dtype=torch.float64,
            ),
            rtol=1e-9,
            atol=0.0,
        )


@pytest.mark.parametrize("num_heads", [True, 0, -1])
def test_genomic_alibi_helpers_reject_invalid_head_counts(num_heads):
    with pytest.raises(ValueError, match="num_heads"):
        build_genomic_alibi_slopes(num_heads, target_weight_ratio=0.75, distance_scale=1.0)
    with pytest.raises(ValueError, match="num_heads"):
        build_alibi_initial_slope_logits(num_heads, target_weight_ratio=0.75, distance_scale=1.0)


def test_fixed_alibi_runtime_is_plain_and_stores_only_python_slope_tuple():
    runtime = _runtime(num_heads=4)

    _assert_parameterless_plain_runtime(runtime)
    assert runtime.fixed_slopes == build_genomic_alibi_slopes(
        4, target_weight_ratio=0.75, distance_scale=10.0
    )
    assert len(runtime.fixed_slopes) == 4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_heads": True}, "num_heads"),
        ({"num_heads": 0}, "num_heads"),
        ({"num_heads": 2}, "num_heads"),
        ({"distance_function": "linear"}, "distance_function"),
        ({"distance_scale": 0.0}, "distance_scale"),
        ({"distance_scale": math.inf}, "distance_scale"),
        ({"target_weight_ratio": 1.0}, "alibi_target_weight_ratio"),
        ({"cross_policy": "separate"}, "cross_chromosome_policy"),
    ],
)
def test_fixed_alibi_runtime_rejects_invalid_direct_construction(kwargs, message):
    valid = {
        "num_heads": 4,
        "distance_function": AlibiDistanceFunction.LINEAR,
        "distance_scale": 10.0,
        "cross_policy": CrossChromosomePolicy.SEPARATE,
    }
    valid.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _runtime(**valid)


def test_factory_builds_fixed_alibi_runtime_with_resolved_settings():
    config = _resolve_custom(
        alibi_distance_function=AlibiDistanceFunction.LOG1P,
        alibi_distance_scale=25.0,
    )

    with pytest.raises(ValueError, match="num_heads"):
        build_relative_position_runtime(config)
    with pytest.raises(ValueError, match="num_heads=4"):
        build_relative_position_runtime(config, num_heads=2)
    runtime = build_relative_position_runtime(config, num_heads=4)

    assert isinstance(runtime, FixedAlibiRelativePositionRuntime)
    assert runtime.fixed_slopes == build_genomic_alibi_slopes(
        4, target_weight_ratio=0.75, distance_scale=25.0
    )
    assert runtime.target_weight_ratio == 0.75
    assert runtime.distance_function is AlibiDistanceFunction.LOG1P
    assert runtime.distance_scale == 25.0
    assert runtime.cross_chromosome_policy is CrossChromosomePolicy.SEPARATE


def test_factory_builds_learned_alibi_runtime_with_resolved_settings():
    config = _resolve_custom(
        relative=RelativePositionEncoding.ALIBI_LEARNED,
        alibi_distance_function=AlibiDistanceFunction.LOG1P,
        alibi_distance_scale=25.0,
    )

    with pytest.raises(ValueError, match="num_heads"):
        build_relative_position_runtime(config)
    # Learned ALiBi shares the genomic prior, so it is also four-head only.
    with pytest.raises(ValueError, match="num_heads=4"):
        build_relative_position_runtime(config, num_heads=2)
    runtime = build_relative_position_runtime(config, num_heads=4)

    assert isinstance(runtime, LearnedAlibiRelativePositionRuntime)
    assert runtime.distance_function is AlibiDistanceFunction.LOG1P
    assert runtime.distance_scale == 25.0
    assert runtime.cross_chromosome_policy is CrossChromosomePolicy.SEPARATE
    _assert_parameterless_plain_runtime(runtime)


@pytest.mark.parametrize("ratio", [0.75, 0.9])
def test_fixed_and_learned_alibi_scores_are_identical_at_initialisation(ratio):
    fixed = _fixed_runtime(
        distance_function=AlibiDistanceFunction.LOG1P,
        distance_scale=25.0,
        target_weight_ratio=ratio,
    )
    learned = _learned_runtime(
        distance_function=AlibiDistanceFunction.LOG1P,
        distance_scale=25.0,
    )
    base_scores = _base_scores()
    positions = torch.tensor([[10, 30, 50]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1, 0]], dtype=torch.long)
    cross_bias = torch.zeros(4, dtype=torch.float64)

    fixed_scores = _adjust(
        fixed,
        base_scores,
        positions=positions,
        chrom_ids=chrom_ids,
        cross_chromosome_bias=cross_bias,
    )
    learned_scores = _adjust(
        learned,
        base_scores,
        positions=positions,
        chrom_ids=chrom_ids,
        cross_chromosome_bias=cross_bias,
        alibi_slope_logits=_initial_logits(ratio=ratio, scale=25.0),
    )

    torch.testing.assert_close(learned_scores, fixed_scores, rtol=1e-12, atol=1e-12)
    # Positions 10 and 50 share chromosome 0: distance 40, scale 25.
    transform = math.log1p(40.0 / 25.0)
    expected = _expected_slopes(ratio=ratio, scale=25.0)
    for head in range(4):
        torch.testing.assert_close(
            fixed_scores[0, head, 0, 2],
            base_scores[0, head, 0, 2] - expected[head] * transform,
            rtol=1e-12,
            atol=1e-12,
        )


def test_learned_alibi_effective_slopes_remain_positive_from_negative_raw_logits():
    runtime = _learned_runtime(distance_function=AlibiDistanceFunction.LINEAR)
    raw_logits = torch.tensor([-10.0, -2.0, -5.0, -1.0], dtype=torch.float64)
    effective = torch.nn.functional.softplus(raw_logits)
    adjusted = _adjust(
        runtime,
        torch.zeros(1, 4, 2, 2, dtype=torch.float64),
        positions=torch.tensor([[10, 20]], dtype=torch.long),
        chrom_ids=torch.tensor([[0, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float64),
        alibi_slope_logits=raw_logits,
    )

    assert torch.all(effective > 0)
    assert torch.all(adjusted[0, :, 0, 1] < 0)


def test_fixed_alibi_linear_zero_distance_and_symmetry_are_exact():
    runtime = _runtime(distance_function=AlibiDistanceFunction.LINEAR)
    adjusted = _adjust(runtime, _base_scores(), cross_chromosome_bias=torch.zeros(4))

    torch.testing.assert_close(adjusted[0, 0, 0, 0], _base_scores()[0, 0, 0, 0])
    torch.testing.assert_close(adjusted[0, 1, 1, 1], _base_scores()[0, 1, 1, 1])
    torch.testing.assert_close(
        _base_scores()[0, 0, 0, 1] - adjusted[0, 0, 0, 1],
        _base_scores()[0, 0, 1, 0] - adjusted[0, 0, 1, 0],
    )


def test_fixed_alibi_linear_transform_is_hand_computable():
    runtime = _runtime(distance_function=AlibiDistanceFunction.LINEAR)
    adjusted = _adjust(runtime, _base_scores(), cross_chromosome_bias=torch.zeros(4))

    # positions 10 and 30 have distance 20, scale 10, transformed distance 2.
    torch.testing.assert_close(
        adjusted[0, 0, 0, 1],
        torch.tensor(2.0 - 2 * SLOPES_S10[0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        adjusted[0, 1, 0, 1],
        torch.tensor(-2.0 - 2 * SLOPES_S10[1], dtype=torch.float64),
    )


def test_fixed_alibi_log1p_transform_and_scale_placement_are_hand_computable():
    runtime = _runtime(distance_function=AlibiDistanceFunction.LOG1P)
    adjusted = _adjust(runtime, _base_scores(), cross_chromosome_bias=torch.zeros(4))
    expected_transform = math.log1p(20.0 / 10.0)
    wrong_transform = math.log1p(20.0) / 10.0

    torch.testing.assert_close(
        adjusted[0, 0, 0, 1],
        torch.tensor(2.0 - SLOPES_S10[0] * expected_transform, dtype=torch.float64),
    )
    assert not math.isclose(expected_transform, wrong_transform)


def test_larger_same_chromosome_distance_has_stronger_negative_penalty():
    runtime = _runtime(distance_function=AlibiDistanceFunction.LINEAR)
    adjusted = _adjust(runtime, _base_scores(), cross_chromosome_bias=torch.zeros(4))
    penalty_near = _base_scores()[0, 0, 0, 1] - adjusted[0, 0, 0, 1]
    penalty_far = _base_scores()[0, 0, 0, 2] - adjusted[0, 0, 0, 2]

    assert penalty_far > penalty_near > 0


def test_fixed_alibi_multiple_heads_broadcast_distinct_slopes():
    runtime = _runtime(distance_function=AlibiDistanceFunction.LINEAR)
    adjusted = _adjust(runtime, _base_scores(), cross_chromosome_bias=torch.zeros(4))

    # positions 10 and 30: linear transform 20 / 10 = 2 for every head.
    torch.testing.assert_close(
        adjusted[0, :, 0, 1],
        _base_scores()[0, :, 0, 1] - 2.0 * torch.tensor(SLOPES_S10, dtype=torch.float64),
    )
    assert len(set(adjusted[0, :, 0, 1].tolist())) == 4


def test_fixed_alibi_separate_cross_pairs_use_bias_without_distance_penalty():
    runtime = _runtime(distance_function=AlibiDistanceFunction.LINEAR)
    base_scores = _base_scores()
    adjusted = _adjust(
        runtime,
        base_scores,
        chrom_ids=torch.tensor([[0, 1, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.tensor([0.25, -0.5, 0.125, -0.0625], dtype=torch.float64),
    )

    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1] + 0.25)
    torch.testing.assert_close(adjusted[0, 1, 0, 1], base_scores[0, 1, 0, 1] - 0.5)
    torch.testing.assert_close(adjusted[0, 2, 0, 1], base_scores[0, 2, 0, 1] + 0.125)
    torch.testing.assert_close(adjusted[0, 3, 0, 1], base_scores[0, 3, 0, 1] - 0.0625)
    assert adjusted[0, 0, 0, 2] != base_scores[0, 0, 0, 2]


def test_fixed_alibi_separate_zero_cross_bias_leaves_cross_score_as_base():
    runtime = _runtime(distance_function=AlibiDistanceFunction.LINEAR)
    base_scores = _base_scores()
    adjusted = _adjust(
        runtime,
        base_scores,
        chrom_ids=torch.tensor([[0, 1, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float64),
    )

    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1])


def test_fixed_alibi_mask_runtime_leaves_cross_pairs_for_outer_attention_mask():
    runtime = _runtime(cross_policy=CrossChromosomePolicy.MASK)
    base_scores = _base_scores()
    adjusted = _adjust(
        runtime,
        base_scores,
        chrom_ids=torch.tensor([[0, 1, 0]], dtype=torch.long),
    )

    torch.testing.assert_close(adjusted[..., 0, 1], base_scores[..., 0, 1])
    assert adjusted[0, 0, 0, 2] != base_scores[0, 0, 0, 2]


@pytest.mark.parametrize(
    ("cross_policy", "bias", "message"),
    [
        (CrossChromosomePolicy.SEPARATE, None, "cross_chromosome_bias"),
        (CrossChromosomePolicy.SEPARATE, torch.zeros(3), "shape"),
        (CrossChromosomePolicy.SEPARATE, torch.zeros(4, dtype=torch.long), "floating"),
        (CrossChromosomePolicy.MASK, torch.zeros(4), "must be None"),
    ],
)
def test_fixed_alibi_cross_bias_contract_rejects_invalid_inputs(cross_policy, bias, message):
    runtime = _runtime(cross_policy=cross_policy)

    with pytest.raises(ValueError, match=message):
        _adjust(
            runtime,
            _base_scores(),
            chrom_ids=torch.tensor([[0, 1, 0]], dtype=torch.long),
            cross_chromosome_bias=bias,
        )


@pytest.mark.parametrize(
    ("base_scores", "message"),
    [
        (torch.zeros(1, 4, 3, 3, dtype=torch.long), "floating"),
        (torch.zeros(1, 4, 3), "shape"),
        (torch.zeros(1, 3, 3, 3), "head"),
        (torch.zeros(1, 4, 3, 4), "square"),
    ],
)
def test_fixed_alibi_rejects_invalid_base_scores(base_scores, message):
    with pytest.raises(ValueError, match=message):
        _adjust(_runtime(), base_scores, cross_chromosome_bias=torch.zeros(4))


def test_fixed_alibi_rejects_float_positions_and_missing_chrom_ids():
    runtime = _runtime()
    base_scores = _base_scores()

    with pytest.raises(ValueError, match="integer"):
        _adjust(
            runtime,
            base_scores,
            positions=torch.tensor([[10.0, 30.0, 50.0]]),
            cross_chromosome_bias=torch.zeros(4),
        )
    query, key = _unused_query_key()
    with pytest.raises(ValueError, match="chrom_ids"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=torch.tensor([[10, 30, 50]], dtype=torch.long),
            chrom_ids=None,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(4),
        )


def test_fixed_alibi_accepts_narrow_integer_positions_without_overflowing_subtraction():
    runtime = _runtime(distance_function=AlibiDistanceFunction.LINEAR)
    base_scores = torch.zeros(1, 4, 2, 2, dtype=torch.float64)
    adjusted = _adjust(
        runtime,
        base_scores,
        positions=torch.tensor([[1, 100]], dtype=torch.int8),
        chrom_ids=torch.tensor([[0, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float64),
    )

    # distance 99, scale 10, linear transform 9.9.
    torch.testing.assert_close(
        adjusted[0, 0, 0, 1],
        torch.tensor(-9.9 * SLOPES_S10[0], dtype=torch.float64),
    )


def test_fixed_alibi_preserves_one_bp_distance_for_large_float32_coordinates():
    runtime = _runtime(
        distance_function=AlibiDistanceFunction.LINEAR,
        distance_scale=1.0,
    )
    adjusted = _adjust(
        runtime,
        torch.zeros(1, 4, 2, 2, dtype=torch.float32),
        positions=torch.tensor([[250_000_001, 250_000_002]], dtype=torch.long),
        chrom_ids=torch.tensor([[0, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float32),
    )

    for head, slope in enumerate(GENOMIC_FIXED_SLOPES):
        torch.testing.assert_close(adjusted[0, head, 0, 1], torch.tensor(-slope))
        torch.testing.assert_close(adjusted[0, head, 1, 0], torch.tensor(-slope))


def test_fixed_alibi_preserves_float64_slope_precision_without_float32_rounding():
    adjusted = _adjust(
        _runtime(
            distance_function=AlibiDistanceFunction.LINEAR,
            distance_scale=1.0,
        ),
        torch.zeros(1, 4, 2, 2, dtype=torch.float64),
        positions=torch.tensor([[10, 11]], dtype=torch.long),
        chrom_ids=torch.tensor([[0, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float64),
    )
    expected = torch.tensor(-0.4150374992788438, dtype=torch.float64)

    assert torch.equal(adjusted[0, 0, 0, 1], expected)
    assert torch.equal(adjusted[0, 0, 1, 0], expected)


def test_learned_alibi_preserves_one_bp_distance_for_large_float32_coordinates():
    adjusted = _adjust(
        _learned_runtime(
            distance_function=AlibiDistanceFunction.LINEAR,
            distance_scale=1.0,
        ),
        torch.zeros(1, 4, 2, 2, dtype=torch.float32),
        positions=torch.tensor([[250_000_001, 250_000_002]], dtype=torch.long),
        chrom_ids=torch.tensor([[0, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float32),
        alibi_slope_logits=_initial_logits(dtype=torch.float32, scale=1.0),
    )

    for head, slope in enumerate(GENOMIC_FIXED_SLOPES):
        torch.testing.assert_close(adjusted[0, head, 0, 1], torch.tensor(-slope))
        torch.testing.assert_close(adjusted[0, head, 1, 0], torch.tensor(-slope))


def test_fixed_alibi_runtime_allows_padded_zero_because_it_has_no_mask():
    runtime = _runtime(distance_function=AlibiDistanceFunction.LINEAR)

    adjusted = _adjust(
        runtime,
        torch.zeros(1, 4, 2, 2, dtype=torch.float64),
        positions=torch.tensor([[0, 10]], dtype=torch.long),
        chrom_ids=torch.tensor([[0, 0]], dtype=torch.long),
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float64),
    )

    assert torch.isfinite(adjusted).all()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
def test_fixed_alibi_returned_score_dtype_matches_base_scores(dtype):
    runtime = _runtime(distance_function=AlibiDistanceFunction.LOG1P)
    base_scores = _base_scores(dtype=dtype)

    adjusted = _adjust(
        runtime,
        base_scores,
        cross_chromosome_bias=torch.zeros(4, dtype=dtype),
    )

    assert adjusted.dtype is dtype


def test_fixed_alibi_output_does_not_depend_on_query_or_key_when_base_scores_are_fixed():
    runtime = _runtime(distance_function=AlibiDistanceFunction.LINEAR)
    base_scores = _base_scores()
    positions = torch.tensor([[10, 30, 50]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0, 0]], dtype=torch.long)
    query_a, key_a = _unused_query_key()
    query_b = query_a + 1000.0
    key_b = key_a - 1000.0

    adjusted_a = runtime.adjust_attention_scores(
        base_scores,
        query=query_a,
        key=key_a,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float64),
    )
    adjusted_b = runtime.adjust_attention_scores(
        base_scores,
        query=query_b,
        key=key_b,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float64),
    )

    torch.testing.assert_close(adjusted_a, adjusted_b)


def test_learned_alibi_output_does_not_depend_on_query_or_key_when_base_scores_are_fixed():
    runtime = _learned_runtime(distance_function=AlibiDistanceFunction.LINEAR)
    base_scores = _base_scores()
    positions = torch.tensor([[10, 30, 50]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0, 0]], dtype=torch.long)
    query_a, key_a = _unused_query_key()
    query_b = query_a + 1000.0
    key_b = key_a - 1000.0

    adjusted_a = runtime.adjust_attention_scores(
        base_scores,
        query=query_a,
        key=key_a,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float64),
        alibi_slope_logits=_initial_logits(),
    )
    adjusted_b = runtime.adjust_attention_scores(
        base_scores,
        query=query_b,
        key=key_b,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(4, dtype=torch.float64),
        alibi_slope_logits=_initial_logits(),
    )

    torch.testing.assert_close(adjusted_a, adjusted_b)


def test_fixed_alibi_rejects_position_bias_embedding():
    with pytest.raises(ValueError, match="position_bias"):
        _runtime().adjust_attention_scores(
            _base_scores(),
            query=_unused_query_key()[0],
            key=_unused_query_key()[1],
            positions=torch.tensor([[10, 30, 50]], dtype=torch.long),
            chrom_ids=torch.tensor([[0, 0, 0]], dtype=torch.long),
            position_bias=nn.Embedding(2, 2),
            cross_chromosome_bias=torch.zeros(4),
        )


@pytest.mark.parametrize(
    ("alibi_slope_logits", "message"),
    [
        (None, "alibi_slope_logits"),
        (torch.zeros(3), "shape"),
        (torch.zeros(4, dtype=torch.long), "floating"),
    ],
)
def test_learned_alibi_requires_valid_raw_slope_logits(alibi_slope_logits, message):
    with pytest.raises(ValueError, match=message):
        _adjust(
            _learned_runtime(),
            _base_scores(),
            cross_chromosome_bias=torch.zeros(4),
            alibi_slope_logits=alibi_slope_logits,
        )


def test_fixed_alibi_rejects_accidental_raw_slope_logits():
    with pytest.raises(ValueError, match="alibi_slope_logits"):
        _adjust(
            _fixed_runtime(),
            _base_scores(),
            cross_chromosome_bias=torch.zeros(4),
            alibi_slope_logits=torch.zeros(4),
        )


@pytest.mark.parametrize(
    ("cross_policy", "expected_keys"),
    [
        (CrossChromosomePolicy.MASK, set()),
        (
            CrossChromosomePolicy.SEPARATE,
            {"attention.attention_layers.0.cross_chromosome_bias"},
        ),
    ],
)
def test_fixed_alibi_state_key_sets_are_exact_for_mask_and_separate(
    cross_policy,
    expected_keys,
):
    config = _resolve_custom(cross_policy=cross_policy)
    model = _model(config)
    layer = model.attention.attention_layers[0]

    assert layer.position_bias is None
    assert _positional_state_keys(model) == expected_keys
    if cross_policy is CrossChromosomePolicy.SEPARATE:
        assert layer.cross_chromosome_bias.shape == (MODEL_KWARGS["num_heads"],)
        assert torch.equal(
            layer.cross_chromosome_bias,
            torch.zeros_like(layer.cross_chromosome_bias),
        )
    else:
        assert layer.cross_chromosome_bias is None


@pytest.mark.parametrize(
    ("cross_policy", "expected_keys"),
    [
        (
            CrossChromosomePolicy.MASK,
            {"attention.attention_layers.0.alibi_slope_logits"},
        ),
        (
            CrossChromosomePolicy.SEPARATE,
            {
                "attention.attention_layers.0.alibi_slope_logits",
                "attention.attention_layers.0.cross_chromosome_bias",
            },
        ),
    ],
)
def test_learned_alibi_state_key_sets_are_exact_for_mask_and_separate(
    cross_policy,
    expected_keys,
):
    config = _resolve_custom(
        relative=RelativePositionEncoding.ALIBI_LEARNED,
        cross_policy=cross_policy,
    )
    model = _model(config)
    layer = model.attention.attention_layers[0]

    assert layer.position_bias is None
    assert _positional_state_keys(model) == expected_keys
    assert layer.alibi_slope_logits.shape == (MODEL_KWARGS["num_heads"],)
    assert layer.alibi_slope_logits.dtype.is_floating_point
    assert layer.alibi_slope_logits.requires_grad is True
    assert torch.isfinite(layer.alibi_slope_logits).all()
    torch.testing.assert_close(
        torch.nn.functional.softplus(layer.alibi_slope_logits.detach()),
        # _resolve_custom uses r = 0.75 (default) and a 10 bp test scale.
        torch.tensor(SLOPES_S10, dtype=layer.alibi_slope_logits.dtype),
    )
    if cross_policy is CrossChromosomePolicy.SEPARATE:
        assert layer.cross_chromosome_bias.shape == (MODEL_KWARGS["num_heads"],)
        assert torch.equal(
            layer.cross_chromosome_bias,
            torch.zeros_like(layer.cross_chromosome_bias),
        )
    else:
        assert layer.cross_chromosome_bias is None


def test_fixed_alibi_chromosome_embedding_and_learned_binned_state_surfaces_are_independent():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        chromosome=ChromosomeEncoding.LEARNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
    )
    model = _model(config, layout=_layout())

    assert _positional_state_keys(model) == {
        "absolute_position_embedding.weight",
        "attention.attention_layers.0.chrom_embedding.weight",
        "attention.attention_layers.0.cross_chromosome_bias",
    }


def test_chunked_fixed_alibi_state_keys_use_base_model_prefix_naturally():
    config = _resolve_custom(cross_policy=CrossChromosomePolicy.SEPARATE)
    chunked = ChunkedSIEVEModel(_model(config))

    assert _positional_state_keys(chunked) == {
        "base_model.attention.attention_layers.0.cross_chromosome_bias"
    }


def test_chunked_learned_alibi_state_keys_use_base_model_prefix_naturally():
    config = _resolve_custom(
        relative=RelativePositionEncoding.ALIBI_LEARNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
    )
    chunked = ChunkedSIEVEModel(_model(config))

    assert _positional_state_keys(chunked) == {
        "base_model.attention.attention_layers.0.alibi_slope_logits",
        "base_model.attention.attention_layers.0.cross_chromosome_bias",
    }


@pytest.mark.parametrize(
    ("chromosome", "cross_policy"),
    [
        (ChromosomeEncoding.NONE, CrossChromosomePolicy.SEPARATE),
        (ChromosomeEncoding.LEARNED, CrossChromosomePolicy.SEPARATE),
        (ChromosomeEncoding.NONE, CrossChromosomePolicy.MASK),
    ],
)
def test_fixed_alibi_model_forward_backward_return_attention_and_padding(
    chromosome,
    cross_policy,
):
    config = _resolve_custom(chromosome=chromosome, cross_policy=cross_policy)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)
    content = content.clone().requires_grad_(True)

    logits, intermediates = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        return_attention=True,
        content_features=content,
        absolute_position_features=absolute,
    )
    logits.sum().backward()

    attention = intermediates["attention_weights"][0]
    assert logits.shape == (1, 1)
    assert attention.shape == (1, MODEL_KWARGS["num_heads"], 4, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(content.grad).all()
    assert not any("alibi_slope" in key for key in model.state_dict())
    if cross_policy is CrossChromosomePolicy.MASK:
        assert torch.all(attention[0, :, 0, 1] == 0)
        assert torch.all(attention[0, :, 0, 0] > 0)


def test_fixed_alibi_cross_bias_receives_gradient_with_cross_chromosome_pairs():
    config = _resolve_custom(cross_policy=CrossChromosomePolicy.SEPARATE)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)

    logits, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=absolute,
    )
    logits.sum().backward()

    grad = model.attention.attention_layers[0].cross_chromosome_bias.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.any(grad != 0)


@pytest.mark.parametrize(
    "cross_policy",
    [CrossChromosomePolicy.SEPARATE, CrossChromosomePolicy.MASK],
)
def test_learned_alibi_model_forward_backward_and_gradients(cross_policy):
    config = _resolve_custom(
        relative=RelativePositionEncoding.ALIBI_LEARNED,
        cross_policy=cross_policy,
    )
    model = _model(config)
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)
    positions = torch.tensor([[1, 6, 11, 0]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0, 1, 0]], dtype=torch.long)
    content = content.clone().requires_grad_(True)

    logits, intermediates = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        return_attention=True,
        content_features=content,
        absolute_position_features=absolute,
    )
    logits.sum().backward()

    layer = model.attention.attention_layers[0]
    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(content.grad).all()
    assert layer.alibi_slope_logits.grad is not None
    assert torch.isfinite(layer.alibi_slope_logits.grad).all()
    assert torch.any(layer.alibi_slope_logits.grad != 0)
    if cross_policy is CrossChromosomePolicy.SEPARATE:
        assert layer.cross_chromosome_bias.grad is not None
        assert torch.isfinite(layer.cross_chromosome_bias.grad).all()
        assert torch.any(layer.cross_chromosome_bias.grad != 0)
    else:
        assert layer.cross_chromosome_bias is None
        attention = intermediates["attention_weights"][0]
        assert torch.all(attention[0, :, 0, 2] == 0)


@pytest.mark.parametrize(
    "cross_policy", [CrossChromosomePolicy.SEPARATE, CrossChromosomePolicy.MASK]
)
def test_fixed_and_learned_models_match_at_initialisation_then_only_learned_moves(cross_policy):
    fixed_config = _resolve_custom(alibi_distance_scale=1.0, cross_policy=cross_policy)
    learned_config = _resolve_custom(
        relative=RelativePositionEncoding.ALIBI_LEARNED,
        alibi_distance_scale=1.0,
        cross_policy=cross_policy,
    )
    torch.manual_seed(0)
    fixed_model = _model(fixed_config)
    learned_model = _model(learned_config)
    # Share every weight; learned keeps only its own alibi_slope_logits.
    missing, unexpected = learned_model.load_state_dict(fixed_model.state_dict(), strict=False)
    assert missing == ["attention.attention_layers.0.alibi_slope_logits"]
    assert unexpected == []
    layer = learned_model.attention.attention_layers[0]
    fixed_runtime = fixed_model.attention.attention_layers[0]._relative_position_runtime
    torch.testing.assert_close(
        torch.nn.functional.softplus(layer.alibi_slope_logits.detach().double()),
        torch.tensor(fixed_runtime.fixed_slopes, dtype=torch.float64),
        rtol=1e-6,
        atol=0.0,
    )

    content, absolute, _, gene_ids, mask, _ = _batch(fixed_config)
    positions = torch.tensor([[1, 6, 11, 0]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0, 1, 0]], dtype=torch.long)

    def _forward(model):
        return model(
            None,
            positions,
            gene_ids,
            mask,
            chrom_ids=chrom_ids,
            return_attention=True,
            content_features=content,
            absolute_position_features=absolute,
        )

    fixed_logits, fixed_intermediates = _forward(fixed_model)
    learned_logits, learned_intermediates = _forward(learned_model)
    torch.testing.assert_close(learned_logits, fixed_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        learned_intermediates["attention_weights"][0],
        fixed_intermediates["attention_weights"][0],
        rtol=1e-5,
        atol=1e-6,
    )

    # One optimiser step moves the learned slopes; fixed ALiBi has none to move.
    learned_model.train()
    optimiser = torch.optim.SGD([layer.alibi_slope_logits], lr=0.1)
    before = layer.alibi_slope_logits.detach().clone()
    _forward(learned_model)[0].sum().backward()
    optimiser.step()
    assert layer.alibi_slope_logits.requires_grad is True
    assert not torch.equal(layer.alibi_slope_logits.detach(), before)
    assert not any(isinstance(p, torch.Tensor) for p in fixed_runtime.fixed_slopes)


def test_fixed_alibi_rejects_real_position_zero_but_allows_masked_zero():
    config = _resolve_custom(cross_policy=CrossChromosomePolicy.SEPARATE)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)

    model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=absolute,
    )
    bad_positions = positions.clone()
    bad_positions[0, 1] = 0
    with pytest.raises(ValueError, match="ALiBi positions"):
        model(
            None,
            bad_positions,
            gene_ids,
            mask,
            chrom_ids=chrom_ids,
            content_features=content,
            absolute_position_features=absolute,
        )


def test_learned_alibi_rejects_real_position_zero_but_allows_masked_zero():
    config = _resolve_custom(relative=RelativePositionEncoding.ALIBI_LEARNED)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)

    model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=absolute,
    )
    bad_positions = positions.clone()
    bad_positions[0, 1] = 0
    with pytest.raises(ValueError, match="ALiBi positions"):
        model(
            None,
            bad_positions,
            gene_ids,
            mask,
            chrom_ids=chrom_ids,
            content_features=content,
            absolute_position_features=absolute,
        )


def test_fixed_alibi_requires_chrom_ids_in_model_even_without_chromosome_embedding():
    config = _resolve_custom(chromosome=ChromosomeEncoding.NONE)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, _chrom_ids = _batch(config)

    with pytest.raises(ValueError, match="chrom_ids"):
        model(
            None,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=absolute,
        )


def test_learned_alibi_requires_chrom_ids_in_model_even_without_chromosome_embedding():
    config = _resolve_custom(
        relative=RelativePositionEncoding.ALIBI_LEARNED,
        chromosome=ChromosomeEncoding.NONE,
    )
    model = _model(config)
    content, absolute, positions, gene_ids, mask, _chrom_ids = _batch(config)

    with pytest.raises(ValueError, match="chrom_ids"):
        model(
            None,
            positions,
            gene_ids,
            mask,
            content_features=content,
            absolute_position_features=absolute,
        )


def test_fixed_alibi_position_sensitivity_changes_same_chromosome_attention():
    config = _resolve_custom(cross_policy=CrossChromosomePolicy.SEPARATE)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, _chrom_ids = _batch(config)
    same_chrom_ids = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    positions_b = positions.clone()
    positions_b[0, 1] = 30

    _, intermediates_a = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=same_chrom_ids,
        return_attention=True,
        content_features=content,
        absolute_position_features=absolute,
    )
    _, intermediates_b = model(
        None,
        positions_b,
        gene_ids,
        mask,
        chrom_ids=same_chrom_ids,
        return_attention=True,
        content_features=content,
        absolute_position_features=absolute,
    )

    assert not torch.equal(
        intermediates_a["attention_weights"][0],
        intermediates_b["attention_weights"][0],
    )


def test_learned_alibi_position_sensitivity_changes_same_chromosome_attention():
    config = _resolve_custom(
        relative=RelativePositionEncoding.ALIBI_LEARNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
    )
    model = _model(config)
    content, absolute, positions, gene_ids, mask, _chrom_ids = _batch(config)
    same_chrom_ids = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    positions_b = positions.clone()
    positions_b[0, 1] = 35

    _, intermediates_a = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=same_chrom_ids,
        return_attention=True,
        content_features=content,
        absolute_position_features=absolute,
    )
    _, intermediates_b = model(
        None,
        positions_b,
        gene_ids,
        mask,
        chrom_ids=same_chrom_ids,
        return_attention=True,
        content_features=content,
        absolute_position_features=absolute,
    )

    assert not torch.equal(
        intermediates_a["attention_weights"][0],
        intermediates_b["attention_weights"][0],
    )


def test_fixed_alibi_with_learned_binned_absolute_forward_backward_succeeds():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
    )
    model = _model(config, layout=_layout())
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)
    content = content.clone().requires_grad_(True)

    logits, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=absolute,
    )
    logits.sum().backward()

    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all()
    assert model.absolute_position_embedding is not None
    assert torch.isfinite(content.grad).all()
    assert torch.isfinite(model.attention.attention_layers[0].cross_chromosome_bias.grad).all()


@pytest.mark.parametrize(
    "cross_policy",
    [CrossChromosomePolicy.SEPARATE, CrossChromosomePolicy.MASK],
)
def test_learned_alibi_with_learned_binned_absolute_forward_backward_succeeds(cross_policy):
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        relative=RelativePositionEncoding.ALIBI_LEARNED,
        cross_policy=cross_policy,
    )
    model = _model(config, layout=_layout())
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)
    content = content.clone().requires_grad_(True)

    logits, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=chrom_ids,
        content_features=content,
        absolute_position_features=absolute,
    )
    logits.sum().backward()

    layer = model.attention.attention_layers[0]
    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all()
    assert model.absolute_position_embedding is not None
    assert torch.isfinite(content.grad).all()
    assert torch.isfinite(layer.alibi_slope_logits.grad).all()
    if cross_policy is CrossChromosomePolicy.SEPARATE:
        assert layer.cross_chromosome_bias is not None
        assert torch.isfinite(layer.cross_chromosome_bias.grad).all()
    else:
        assert layer.cross_chromosome_bias is None


def test_attention_runtime_support_allows_fixed_and_learned_alibi():
    fixed = _resolve_custom(relative=RelativePositionEncoding.ALIBI_FIXED)
    learned = _resolve_custom(relative=RelativePositionEncoding.ALIBI_LEARNED)

    validate_attention_runtime_support(fixed)
    validate_attention_runtime_support(learned)


@pytest.mark.parametrize(
    "relative",
    [RelativePositionEncoding.ALIBI_FIXED, RelativePositionEncoding.ALIBI_LEARNED],
)
def test_phase7_gate_still_rejects_alibi_after_attention_support_exists(relative):
    config = _resolve_custom(relative=relative)

    with pytest.raises(NotImplementedError, match=relative.value):
        validate_phase7_runtime_support(config)
