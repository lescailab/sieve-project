"""Tests for Phase 9B RoPE relative-position runtime execution."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass

import pytest
import torch
import torch.nn as nn

from src.encoding.levels import AnnotationLevel
from src.encoding.position_config import (
    AbsolutePositionEncoding,
    ChromosomeEncoding,
    CrossChromosomePolicy,
    PositionEncodingRequest,
    PositionPreset,
    RelativePositionEncoding,
    resolve_position_encoding_config,
)
from src.encoding.position_layout import LearnedBinnedAbsolutePositionLayout
from src.models.position_runtime import (
    NoRelativePositionRuntime,
    RopeRelativePositionRuntime,
    T5RelativePositionRuntime,
    build_relative_position_runtime,
)
from src.models.sieve import SIEVE

MODEL_KWARGS = {
    "latent_dim": 8,
    "hidden_dim": 10,
    "num_heads": 2,
    "num_attention_layers": 1,
    "classifier_hidden_dim": 12,
    "dropout": 0.0,
    "num_covariates": 0,
    "classifier_type": "flatten",
}


def _resolve_custom(
    *,
    absolute: AbsolutePositionEncoding = AbsolutePositionEncoding.NONE,
    relative: RelativePositionEncoding = RelativePositionEncoding.ROPE,
    chromosome: ChromosomeEncoding = ChromosomeEncoding.NONE,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
    num_chromosomes: int = 3,
    position_dim: int = 4,
    position_bin_size: int = 10,
    **kwargs,
):
    request_kwargs = {}
    if relative is RelativePositionEncoding.ROPE:
        request_kwargs.update(
            {
                "rope_coordinate_scale": kwargs.pop("rope_coordinate_scale", 10.0),
                "rope_base": kwargs.pop("rope_base", 100.0),
            }
        )
    if absolute is AbsolutePositionEncoding.SINUSOIDAL:
        request_kwargs["position_dim"] = position_dim
    if absolute is AbsolutePositionEncoding.LEARNED_BINNED:
        request_kwargs["position_dim"] = position_dim
        request_kwargs["position_bin_size"] = position_bin_size
    request_kwargs.update(kwargs)
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


def _model(config, *, layout=None, layers: int = 1) -> SIEVE:
    model = SIEVE(
        input_dim=config.input_dim,
        num_genes=5,
        latent_dim=MODEL_KWARGS["latent_dim"],
        hidden_dim=MODEL_KWARGS["hidden_dim"],
        num_heads=MODEL_KWARGS["num_heads"],
        num_attention_layers=layers,
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


def _base_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return torch.matmul(query, key.transpose(-2, -1)) / (query.shape[-1] ** 0.5)


def _positional_state_keys(model: nn.Module) -> set[str]:
    tokens = (
        "position_bias.weight",
        "cross_chromosome_bias",
        "chrom_embedding.weight",
        "absolute_position_embedding.weight",
    )
    return {key for key in model.state_dict() if any(token in key for token in tokens)}


def _assert_parameterless_plain_runtime(runtime) -> None:
    assert not isinstance(runtime, nn.Module)
    assert not hasattr(runtime, "parameters")
    assert not hasattr(runtime, "buffers")
    assert is_dataclass(runtime)
    for field in fields(runtime):
        assert not isinstance(getattr(runtime, field.name), torch.Tensor)


def _runtime(
    *,
    head_dim: int = 4,
    coordinate_scale: float = 1.0,
    rope_base: float = 100.0,
    cross_policy: CrossChromosomePolicy = CrossChromosomePolicy.SEPARATE,
) -> RopeRelativePositionRuntime:
    return RopeRelativePositionRuntime(
        head_dim=head_dim,
        coordinate_scale=coordinate_scale,
        rope_base=rope_base,
        cross_chromosome_policy=cross_policy,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"head_dim": True}, "head_dim"),
        ({"head_dim": 0}, "head_dim"),
        ({"head_dim": 3}, "even"),
        ({"coordinate_scale": 0.0}, "coordinate_scale"),
        ({"coordinate_scale": math.inf}, "coordinate_scale"),
        ({"rope_base": 0.0}, "rope_base"),
        ({"rope_base": math.nan}, "rope_base"),
        ({"cross_policy": "separate"}, "CrossChromosomePolicy"),
    ],
)
def test_rope_runtime_rejects_invalid_direct_settings(kwargs, message):
    settings = {
        "head_dim": 4,
        "coordinate_scale": 10.0,
        "rope_base": 100.0,
        "cross_policy": CrossChromosomePolicy.SEPARATE,
    }
    settings.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _runtime(**settings)


def test_rope_factory_preserves_resolved_settings_and_requires_head_dim():
    config = _resolve_custom(rope_coordinate_scale=25.0, rope_base=50000.0)

    with pytest.raises(ValueError, match="head_dim"):
        build_relative_position_runtime(config)

    runtime = build_relative_position_runtime(config, head_dim=4)

    assert isinstance(runtime, RopeRelativePositionRuntime)
    assert runtime.head_dim == 4
    assert runtime.coordinate_scale == 25.0
    assert runtime.rope_base == 50000.0
    assert runtime.cross_chromosome_policy is CrossChromosomePolicy.SEPARATE
    _assert_parameterless_plain_runtime(runtime)


def test_rope_adjacent_pair_rotation_and_raw_position_convention():
    runtime = _runtime(head_dim=4, coordinate_scale=2.0, rope_base=16.0)
    values = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1]], dtype=torch.long)

    rotated = runtime.rotate(values, positions)
    first_angle = 0.5
    second_angle = 0.125
    expected = torch.tensor(
        [
            [
                [
                    [
                        1.0 * math.cos(first_angle) - 2.0 * math.sin(first_angle),
                        1.0 * math.sin(first_angle) + 2.0 * math.cos(first_angle),
                        3.0 * math.cos(second_angle) - 4.0 * math.sin(second_angle),
                        3.0 * math.sin(second_angle) + 4.0 * math.cos(second_angle),
                    ]
                ]
            ]
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(rotated, expected, rtol=0, atol=1e-12)


def test_rope_zero_position_helper_rotation_is_identity_reference_only():
    runtime = _runtime()
    values = torch.randn(1, 2, 3, 4, dtype=torch.float64)
    positions = torch.zeros(1, 3, dtype=torch.long)

    torch.testing.assert_close(runtime.rotate(values, positions), values, rtol=0, atol=0)


def test_equal_positions_preserve_query_key_dot_product():
    runtime = _runtime(coordinate_scale=3.0, rope_base=1000.0)
    query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.5, -1.0, 2.5, 0.25]]]], dtype=torch.float64)
    positions = torch.tensor([[17]], dtype=torch.long)

    rotated_query = runtime.rotate(query, positions)
    rotated_key = runtime.rotate(key, positions)

    torch.testing.assert_close(
        torch.sum(rotated_query * rotated_key, dim=-1),
        torch.sum(query * key, dim=-1),
        rtol=0,
        atol=1e-12,
    )


def test_unequal_positions_lock_relative_rotation_direction_and_score():
    runtime = _runtime(head_dim=2, coordinate_scale=1.0, rope_base=100.0)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1, 3]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0]], dtype=torch.long)
    base_scores = _base_scores(query, key)

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(1, dtype=torch.float64),
    )
    expected_01 = math.cos(1.0 - 3.0) / math.sqrt(2.0)
    expected_10 = math.cos(3.0 - 1.0) / math.sqrt(2.0)

    torch.testing.assert_close(
        adjusted[0, 0, 0, 1],
        torch.tensor(expected_01, dtype=torch.float64),
    )
    torch.testing.assert_close(
        adjusted[0, 0, 1, 0],
        torch.tensor(expected_10, dtype=torch.float64),
    )


def test_coordinate_scale_affects_angle_exactly_once():
    values = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[4]], dtype=torch.long)

    rotated_scale_1 = _runtime(head_dim=2, coordinate_scale=1.0).rotate(values, positions)
    rotated_scale_2 = _runtime(head_dim=2, coordinate_scale=2.0).rotate(values, positions)

    torch.testing.assert_close(
        rotated_scale_1[0, 0, 0],
        torch.tensor([math.cos(4.0), math.sin(4.0)], dtype=torch.float64),
    )
    torch.testing.assert_close(
        rotated_scale_2[0, 0, 0],
        torch.tensor([math.cos(2.0), math.sin(2.0)], dtype=torch.float64),
    )


def test_rope_base_controls_second_adjacent_pair_frequency():
    values = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[4]], dtype=torch.long)

    base_16 = _runtime(head_dim=4, rope_base=16.0).rotate(values, positions)
    base_10000 = _runtime(head_dim=4, rope_base=10000.0).rotate(values, positions)

    torch.testing.assert_close(
        base_16[0, 0, 0, 2:],
        torch.tensor([math.cos(1.0), math.sin(1.0)], dtype=torch.float64),
    )
    torch.testing.assert_close(
        base_10000[0, 0, 0, 2:],
        torch.tensor([math.cos(0.04), math.sin(0.04)], dtype=torch.float64),
    )


def test_rope_separate_cross_chromosome_uses_unrotated_base_plus_bias():
    runtime = _runtime(head_dim=2, coordinate_scale=1.0)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1, 3]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    base_scores = _base_scores(query, key)
    bias = torch.tensor([0.25], dtype=torch.float64)

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=bias,
    )

    rotated_cross = _base_scores(runtime.rotate(query, positions), runtime.rotate(key, positions))
    assert not torch.isclose(rotated_cross[0, 0, 0, 1], base_scores[0, 0, 0, 1])
    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1] + 0.25)
    torch.testing.assert_close(adjusted[0, 0, 1, 0], base_scores[0, 0, 1, 0] + 0.25)


def test_zero_cross_bias_makes_cross_score_equal_unrotated_base_score():
    runtime = _runtime(head_dim=2)
    query = torch.tensor([[[[1.0, 0.0], [0.5, 2.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.25, 1.0], [2.0, -0.5]]]], dtype=torch.float64)
    positions = torch.tensor([[5, 9]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    base_scores = _base_scores(query, key)

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(1, dtype=torch.float64),
    )

    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1])


def test_nonzero_cross_bias_broadcasts_independently_by_head():
    runtime = _runtime(head_dim=2)
    query = torch.ones(1, 2, 2, 2, dtype=torch.float64)
    key = torch.ones(1, 2, 2, 2, dtype=torch.float64)
    positions = torch.tensor([[1, 2]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    base_scores = _base_scores(query, key)
    bias = torch.tensor([0.25, -0.5], dtype=torch.float64)

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=bias,
    )

    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1] + 0.25)
    torch.testing.assert_close(adjusted[0, 1, 0, 1], base_scores[0, 1, 0, 1] - 0.5)


def test_same_chromosome_score_ignores_cross_bias():
    runtime = _runtime(head_dim=2)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1, 3]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 0]], dtype=torch.long)
    base_scores = _base_scores(query, key)

    adjusted_a = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.tensor([0.0], dtype=torch.float64),
    )
    adjusted_b = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.tensor([100.0], dtype=torch.float64),
    )

    torch.testing.assert_close(adjusted_a, adjusted_b)


def test_rope_mask_routes_same_chromosome_scores_without_cross_bias():
    runtime = _runtime(head_dim=2, cross_policy=CrossChromosomePolicy.MASK)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1, 3]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    base_scores = _base_scores(query, key)

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
    )

    rotated_scores = _base_scores(runtime.rotate(query, positions), runtime.rotate(key, positions))
    torch.testing.assert_close(adjusted[0, 0, 0, 0], rotated_scores[0, 0, 0, 0])
    torch.testing.assert_close(adjusted[0, 0, 0, 1], base_scores[0, 0, 0, 1])
    with pytest.raises(ValueError, match="must be None"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(1),
        )


def test_rope_runtime_rejects_missing_chrom_ids_bad_bias_and_position_float_dtype():
    runtime = _runtime()
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    base_scores = _base_scores(query, key)
    positions = torch.tensor([[1, 2, 0]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1, 999]], dtype=torch.long)

    with pytest.raises(ValueError, match="chrom_ids"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=None,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(2),
        )
    with pytest.raises(ValueError, match="shape"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(3),
        )
    with pytest.raises(ValueError, match="integer dtype"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions.float(),
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(2),
        )


def test_rope_runtime_rejects_integer_base_scores_before_score_routing():
    runtime = _runtime()
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    positions = torch.tensor([[1, 2, 3]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1, 0]], dtype=torch.long)
    base_scores = torch.zeros(1, 2, 3, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="base_scores must use a floating dtype"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(2),
        )


def test_rope_runtime_rejects_position_bias_and_nonfloating_cross_bias():
    runtime = _runtime()
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 2, 2, 4)
    base_scores = _base_scores(query, key)
    positions = torch.tensor([[1, 2]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="position_bias"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=nn.Embedding(2, 2),
            cross_chromosome_bias=torch.zeros(2),
        )
    with pytest.raises(ValueError, match="floating"):
        runtime.adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(2, dtype=torch.long),
        )


def test_non_rope_runtimes_reject_accidental_cross_bias_wiring():
    base_scores = torch.randn(1, 2, 2, 2)
    query = torch.randn(1, 2, 2, 4)
    key = torch.randn(1, 2, 2, 4)
    positions = torch.tensor([[1, 2]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)

    with pytest.raises(ValueError, match="cross_chromosome_bias"):
        NoRelativePositionRuntime().adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=None,
            cross_chromosome_bias=torch.zeros(2),
        )
    with pytest.raises(ValueError, match="cross_chromosome_bias"):
        T5RelativePositionRuntime(8, 1000, CrossChromosomePolicy.MASK).adjust_attention_scores(
            base_scores,
            query=query,
            key=key,
            positions=positions,
            chrom_ids=chrom_ids,
            position_bias=nn.Embedding(8, 2),
            cross_chromosome_bias=torch.zeros(2),
        )


def test_float64_input_uses_float64_rotation_and_matching_reference():
    runtime = _runtime(head_dim=2, coordinate_scale=2.0)
    values = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    positions = torch.tensor([[1]], dtype=torch.long)

    rotated = runtime.rotate(values, positions)

    assert rotated.dtype is torch.float64
    torch.testing.assert_close(
        rotated[0, 0, 0],
        torch.tensor([math.cos(0.5), math.sin(0.5)], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_inputs_promote_rope_rotation_to_float32(dtype):
    values = torch.tensor([[[[1.0, 0.0]]]], dtype=dtype)
    positions = torch.tensor([[1]], dtype=torch.long)

    rotated = _runtime(head_dim=2).rotate(values, positions)

    assert rotated.dtype is torch.float32


def test_low_precision_adjusted_score_dtype_matches_base_scores_when_supported():
    runtime = _runtime(head_dim=2)
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float16)
    key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float16)
    positions = torch.tensor([[1, 2]], dtype=torch.long)
    chrom_ids = torch.tensor([[0, 1]], dtype=torch.long)
    try:
        base_scores = _base_scores(query, key)
    except RuntimeError as exc:
        pytest.skip(f"CPU float16 matmul unsupported: {exc}")

    adjusted = runtime.adjust_attention_scores(
        base_scores,
        query=query,
        key=key,
        positions=positions,
        chrom_ids=chrom_ids,
        position_bias=None,
        cross_chromosome_bias=torch.zeros(1, dtype=torch.float16),
    )

    assert adjusted.dtype is base_scores.dtype


@pytest.mark.parametrize(
    ("relative", "expected_keys"),
    [
        (RelativePositionEncoding.NONE, set()),
        (
            RelativePositionEncoding.T5_BUCKET,
            {"attention.attention_layers.0.position_bias.weight"},
        ),
    ],
)
def test_representative_non_rope_state_key_sets_are_exact(relative, expected_keys):
    config = _resolve_custom(
        relative=relative,
        num_chromosomes=0 if relative is RelativePositionEncoding.NONE else 3,
        num_position_buckets=8 if relative is RelativePositionEncoding.T5_BUCKET else None,
        max_position_distance=1000 if relative is RelativePositionEncoding.T5_BUCKET else None,
    )
    model = _model(config)

    assert _positional_state_keys(model) == expected_keys


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
def test_rope_state_key_sets_are_exact_for_mask_and_separate(cross_policy, expected_keys):
    config = _resolve_custom(cross_policy=cross_policy)
    model = _model(config)
    layer = model.attention.attention_layers[0]

    assert layer.position_bias is None
    assert _positional_state_keys(model) == expected_keys
    if cross_policy is CrossChromosomePolicy.SEPARATE:
        assert layer.cross_chromosome_bias.shape == (MODEL_KWARGS["num_heads"],)
        assert torch.equal(
            layer.cross_chromosome_bias, torch.zeros_like(layer.cross_chromosome_bias)
        )
    else:
        assert layer.cross_chromosome_bias is None


def test_rope_chromosome_embedding_and_learned_binned_state_surfaces_are_independent():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        chromosome=ChromosomeEncoding.LEARNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
    )
    model = _model(config, layout=_layout())

    assert _positional_state_keys(model) == {
        "absolute_position_embedding.weight",
        "attention.attention_layers.0.cross_chromosome_bias",
        "attention.attention_layers.0.chrom_embedding.weight",
    }


def test_learned_binned_absolute_and_rope_relative_compose_in_model_forward_backward():
    config = _resolve_custom(
        absolute=AbsolutePositionEncoding.LEARNED_BINNED,
        cross_policy=CrossChromosomePolicy.SEPARATE,
    )
    model = _model(config, layout=_layout())
    content, absolute, positions, gene_ids, mask, chrom_ids = _batch(config)
    content = content.clone().requires_grad_(True)
    layer = model.attention.attention_layers[0]

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
    assert layer.cross_chromosome_bias is not None
    assert layer.cross_chromosome_bias.shape == (MODEL_KWARGS["num_heads"],)
    assert torch.isfinite(content.grad).all()
    assert torch.isfinite(layer.cross_chromosome_bias.grad).all()


def test_chunked_rope_state_keys_use_base_model_prefix_naturally():
    from src.models.chunked_sieve import ChunkedSIEVEModel

    config = _resolve_custom(cross_policy=CrossChromosomePolicy.SEPARATE)
    chunked = ChunkedSIEVEModel(_model(config))

    assert _positional_state_keys(chunked) == {
        "base_model.attention.attention_layers.0.cross_chromosome_bias"
    }


@pytest.mark.parametrize(
    ("chromosome", "cross_policy"),
    [
        (ChromosomeEncoding.NONE, CrossChromosomePolicy.SEPARATE),
        (ChromosomeEncoding.LEARNED, CrossChromosomePolicy.SEPARATE),
        (ChromosomeEncoding.NONE, CrossChromosomePolicy.MASK),
    ],
)
def test_rope_model_forward_backward_return_attention_and_padding(chromosome, cross_policy):
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
    if cross_policy is CrossChromosomePolicy.MASK:
        assert torch.all(attention[0, :, 0, 1] == 0)
        assert torch.all(attention[0, :, 0, 0] > 0)


def test_rope_attention_rejects_real_position_zero_but_allows_padded_zero():
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
    with pytest.raises(ValueError, match="real RoPE positions"):
        model(
            None,
            bad_positions,
            gene_ids,
            mask,
            chrom_ids=chrom_ids,
            content_features=content,
            absolute_position_features=absolute,
        )


def test_rope_requires_chrom_ids_in_model_even_without_chromosome_embedding():
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


def test_rope_cross_bias_receives_gradient_with_cross_chromosome_pairs():
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
    assert torch.any(grad != 0)


def test_rope_cross_bias_has_no_nonzero_gradient_for_same_chromosome_batch():
    config = _resolve_custom(cross_policy=CrossChromosomePolicy.SEPARATE)
    model = _model(config)
    content, absolute, positions, gene_ids, mask, _chrom_ids = _batch(config)
    same_chrom_ids = torch.tensor([[0, 0, 0, 999]], dtype=torch.long)

    logits, _ = model(
        None,
        positions,
        gene_ids,
        mask,
        chrom_ids=same_chrom_ids,
        content_features=content,
        absolute_position_features=absolute,
    )
    logits.sum().backward()

    grad = model.attention.attention_layers[0].cross_chromosome_bias.grad
    if grad is not None:
        assert torch.all(grad == 0)


def test_rope_runtime_api_has_no_value_tensor_argument():
    parameter_names = RopeRelativePositionRuntime.adjust_attention_scores.__code__.co_varnames

    assert "value" not in parameter_names


# Benchmark architecture: latent_dim=32 with 4 heads gives head_dim=8, i.e.
# four rotary pairs. The quoted RoPE periods only hold for this head_dim.
_BENCHMARK_LATENT_DIM = 32
_BENCHMARK_NUM_HEADS = 4


def _benchmark_rope_runtime(
    coordinate_scale: float | None = None, rope_base: float | None = None
) -> RopeRelativePositionRuntime:
    """Build the RoPE runtime of the benchmark architecture through the resolver."""
    config = resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=AbsolutePositionEncoding.NONE,
            relative_position_encoding=RelativePositionEncoding.ROPE,
            chromosome_encoding=ChromosomeEncoding.NONE,
            cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
            rope_coordinate_scale=coordinate_scale,
            rope_base=rope_base,
        ),
        AnnotationLevel.L3,
        latent_dim=_BENCHMARK_LATENT_DIM,
        num_heads=_BENCHMARK_NUM_HEADS,
        num_chromosomes=3,
    )
    runtime = build_relative_position_runtime(
        config,
        head_dim=_BENCHMARK_LATENT_DIM // _BENCHMARK_NUM_HEADS,
        num_heads=_BENCHMARK_NUM_HEADS,
    )
    assert isinstance(runtime, RopeRelativePositionRuntime)
    return runtime


def _rotary_frequencies_rad_per_bp(runtime: RopeRelativePositionRuntime) -> list[float]:
    """Read each pair's angular frequency from the executable rotation.

    One unit vector per pair is rotated to position 1 bp; the resulting angle
    is that pair's frequency in rad/bp. No formula is re-implemented here.
    """
    num_pairs = runtime.head_dim // 2
    values = torch.zeros(1, 1, num_pairs, runtime.head_dim, dtype=torch.float64)
    for pair in range(num_pairs):
        values[0, 0, pair, 2 * pair] = 1.0
    rotated = runtime.rotate(values, torch.ones(1, num_pairs, dtype=torch.long))
    return [
        math.atan2(rotated[0, 0, pair, 2 * pair + 1].item(), rotated[0, 0, pair, 2 * pair].item())
        for pair in range(num_pairs)
    ]


def test_historical_architecture_derives_four_rotary_pairs_from_defaults():
    # Historical selected architecture: latent_dim=32, num_heads=4. The head
    # dimension (and so the number of rotary pairs) follows from the model, not
    # from any RoPE-specific constant.
    latent_dim, num_heads = _BENCHMARK_LATENT_DIM, _BENCHMARK_NUM_HEADS
    config = resolve_position_encoding_config(
        PositionEncodingRequest(
            preset=PositionPreset.CUSTOM,
            absolute_position_encoding=AbsolutePositionEncoding.NONE,
            relative_position_encoding=RelativePositionEncoding.ROPE,
            chromosome_encoding=ChromosomeEncoding.NONE,
            cross_chromosome_policy=CrossChromosomePolicy.SEPARATE,
        ),
        AnnotationLevel.L3,
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_chromosomes=3,
    )
    model = SIEVE(
        input_dim=config.input_dim,
        num_genes=5,
        latent_dim=latent_dim,
        hidden_dim=64,
        num_heads=num_heads,
        num_attention_layers=1,
        num_chromosomes=3,
        position_encoding=config,
    )
    runtime = model.attention.attention_layers[0]._relative_position_runtime

    assert config.relative.rope_coordinate_scale == 1.35
    assert config.relative.rope_base == 34000.0
    assert isinstance(runtime, RopeRelativePositionRuntime)
    assert runtime.head_dim == 8
    assert runtime.head_dim // 2 == 4

    angles = _rotary_frequencies_rad_per_bp(runtime)
    # One position unit is 1.35 bp, so the angle at 1 bp is the angular
    # frequency in rad/bp: (1 / 1.35) * 34000 ** (-pair / 4).
    expected_frequencies = [0.7407407407, 0.0545502195, 0.0040172307, 0.0002958401]
    for observed, expected in zip(angles, expected_frequencies, strict=True):
        assert math.isclose(observed, expected, rel_tol=0, abs_tol=1e-10)
    periods = [2 * math.pi / frequency for frequency in angles]
    assert [round(period, 3) for period in periods] == [8.482, 115.182, 1564.059, 21238.45]
    # Adjacent pairs differ by 34000 ** (1 / 4), not by an exact integer.
    for faster, slower in zip(angles, angles[1:], strict=False):
        assert math.isclose(faster / slower, 34000.0**0.25, rel_tol=1e-9)


# Diagnostic near-recurrence level. With equal per-pair contributions the
# normalised positional similarity is the mean of cos(phase) over the pairs, so
# 0.95 means every pair is simultaneously close to its zero-distance phase.
# The current defaults DO have local peaks above this level (four rotary pairs
# cannot avoid them); they are printed, not asserted away.
_ROPE_NEAR_RECURRENCE_LEVEL = 0.95
# Engineering regression guard, not a theoretical or biological threshold. It
# sits just above the strongest recurrence of the current defaults in the
# scanned interval (about 0.9622 at 119 bp) so that a future parameter change
# cannot silently make aliasing substantially worse.
_ROPE_RECURRENCE_REGRESSION_BOUND = 0.965
# Benchmark target interval, scanned at every integer separation.
_ROPE_SCAN_MIN_BP = 1
_ROPE_SCAN_MAX_BP = 20_000


def _rope_spectrum_scan(
    runtime: RopeRelativePositionRuntime,
    min_bp: int = _ROPE_SCAN_MIN_BP,
    max_bp: int = _ROPE_SCAN_MAX_BP,
) -> dict:
    """Scan the equal-pair positional similarity at every integer separation.

    ``S(d) = mean_k cos(omega_k * d)`` is a diagnostic of the RoPE positional
    spectrum: the score RoPE gives two identical, equal-per-pair Q/K vectors at
    separation ``d``, relative to separation 0. It is not a learned attention
    score; trained Q/K weight the pairs unequally and add phase offsets.

    Near zero, S is high simply because of local continuity. A recurrence is
    therefore only counted after the curve has left that neighbourhood: every
    local maximum after the first local minimum is a recurrence peak.
    """
    frequencies = _rotary_frequencies_rad_per_bp(runtime)
    distances = torch.arange(min_bp, max_bp + 1, dtype=torch.float64)
    phases = distances[:, None] * torch.tensor(frequencies, dtype=torch.float64)[None, :]
    similarity = torch.cos(phases).mean(dim=1)

    values = similarity.tolist()
    first_minimum = next(
        index
        for index in range(1, len(values) - 1)
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]
    )
    peaks = [
        (min_bp + index, values[index])
        for index in range(first_minimum + 1, len(values))
        if values[index] >= values[index - 1]
        and (index == len(values) - 1 or values[index] >= values[index + 1])
    ]
    strongest_distance, strongest_similarity = max(peaks, key=lambda peak: peak[1])
    return {
        "coordinate_scale": runtime.coordinate_scale,
        "rope_base": runtime.rope_base,
        "head_dim": runtime.head_dim,
        "num_pairs": runtime.head_dim // 2,
        "range_bp": (min_bp, max_bp),
        "frequencies": frequencies,
        "periods": [2 * math.pi / frequency for frequency in frequencies],
        "first_minimum_bp": min_bp + first_minimum,
        "similarity": {min_bp + index: value for index, value in enumerate(values)},
        "peaks": peaks,
        "strongest_distance": strongest_distance,
        "strongest_similarity": strongest_similarity,
    }


def _format_rope_spectrum_report(report: dict) -> str:
    lines = [
        "RoPE positional-spectrum diagnostic (equal-pair similarity, not an attention score)",
        f"  coordinate_scale = {report['coordinate_scale']} bp",
        f"  rope_base        = {report['rope_base']}",
        f"  head_dim         = {report['head_dim']} ({report['num_pairs']} rotary pairs)",
        f"  scanned range    = {report['range_bp'][0]}-{report['range_bp'][1]} bp "
        "(every integer separation)",
        "  pair | omega (rad/bp) | period (bp)",
    ]
    for pair, (frequency, period) in enumerate(
        zip(report["frequencies"], report["periods"], strict=True)
    ):
        lines.append(f"  {pair:>4} | {frequency:14.10f} | {period:12.3f}")
    lines.append(f"  first local minimum at {report['first_minimum_bp']} bp")
    lines.append(
        f"  strongest recurrence = {report['strongest_similarity']:.5f} "
        f"at {report['strongest_distance']} bp"
    )
    near = [peak for peak in report["peaks"] if peak[1] >= _ROPE_NEAR_RECURRENCE_LEVEL]
    lines.append(f"  recurrence peaks >= {_ROPE_NEAR_RECURRENCE_LEVEL}: {len(near)}")
    for distance, value in near:
        lines.append(f"    {distance:>6} bp  S = {value:.5f}")
    return "\n".join(lines)


def _runtime_equal_pair_similarity(
    runtime: RopeRelativePositionRuntime, distances: list[int]
) -> list[float]:
    """Equal-pair similarity through ``adjust_attention_scores`` itself.

    With Q = K = all ones every rotary pair contributes ``2 cos(omega * d)``,
    so the score at separation d divided by the score at separation 0 is the
    mean cosine over pairs.
    """
    positions = torch.tensor([[1, *[1 + distance for distance in distances]]], dtype=torch.long)
    values = torch.ones(1, 1, positions.shape[1], runtime.head_dim, dtype=torch.float64)
    scores = runtime.adjust_attention_scores(
        _base_scores(values, values),
        query=values,
        key=values,
        positions=positions,
        chrom_ids=torch.zeros_like(positions),
        position_bias=None,
        cross_chromosome_bias=torch.zeros(1, dtype=torch.float64),
    )[0, 0, 0]
    return (scores[1:] / scores[0]).tolist()


def test_rope_default_spectrum_scan_reports_and_bounds_recurrences():
    runtime = _benchmark_rope_runtime()
    report = _rope_spectrum_scan(runtime)
    print()
    print(_format_rope_spectrum_report(report))

    similarity = report["similarity"]
    # Known values of the current defaults; the 10358 bp recurrence of the
    # previous 1.5 / 12000 spectrum is gone, but peaks above 0.95 remain.
    expected = {
        8: 0.96065,
        110: 0.96095,
        119: 0.96222,
        1493: 0.95794,
        1501: 0.95368,
        1612: 0.95839,
        10358: -0.03432,
        18771: 0.92401,
    }
    for distance, value in expected.items():
        assert similarity[distance] == pytest.approx(value, abs=5e-5), distance
    assert report["strongest_distance"] == 119
    assert [peak[0] for peak in report["peaks"] if peak[1] >= _ROPE_NEAR_RECURRENCE_LEVEL] == [
        8,
        110,
        119,
        1493,
        1501,
        1612,
    ]

    # Cross-check the scan through the real score path.
    probes = [8, 119, 1612, 10358, 18771]
    through_runtime = _runtime_equal_pair_similarity(runtime, probes)
    for distance, observed in zip(probes, through_runtime, strict=True):
        assert observed == pytest.approx(similarity[distance], abs=1e-12), distance

    assert (
        report["strongest_similarity"] < _ROPE_RECURRENCE_REGRESSION_BOUND
    ), _format_rope_spectrum_report(report)


@pytest.mark.parametrize(
    ("coordinate_scale", "rope_base", "worst_distance"),
    [
        # Previous provisional spectrum: near-recurrence 0.9846 at 10358 bp.
        pytest.param(1.5, 12000.0, 10358, id="scale1.5-base12000"),
        # Historical harmonic spectrum: exact frequency ratio 10, joint return
        # to zero phase near 2 * pi * 1000 bp (0.9957 at 6283 bp).
        pytest.param(1.0, 10000.0, 6283, id="harmonic-scale1-base10000"),
    ],
)
def test_rope_recurrence_regression_bound_rejects_previous_spectra(
    coordinate_scale, rope_base, worst_distance
):
    runtime = _benchmark_rope_runtime(coordinate_scale, rope_base)
    report = _rope_spectrum_scan(runtime)
    print()
    print(_format_rope_spectrum_report(report))

    assert report["strongest_distance"] == worst_distance
    assert report["strongest_similarity"] >= _ROPE_RECURRENCE_REGRESSION_BOUND
    [through_runtime] = _runtime_equal_pair_similarity(runtime, [worst_distance])
    assert through_runtime == pytest.approx(report["strongest_similarity"], abs=1e-12)
