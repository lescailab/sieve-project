"""
Tests for the two correctness fixes in SIEVE:

1. ``EfficientGeneAggregator`` max-pool no longer leaks padded zeros into the
   gene-0 (or any padded gene-id) embedding.
2. Position-bias and chromosome embedding are chromosome-aware: same-coordinate
   variants on different chromosomes are no longer treated as "0 bp apart" by
   the relative-position bucketing, and cross-chromosome attention itself is
   still allowed (no -inf masking).

Author: Francesco Lescai
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.encoding.positional import relative_position_bucket
from src.models.aggregation import EfficientGeneAggregator, GeneAggregator
from src.models.attention import PositionAwareSparseAttention
from src.models.sieve import SIEVE


# ----- Aggregator regression --------------------------------------------------


def test_efficient_aggregator_max_does_not_leak_padded_zero_into_gene_zero():
    """
    With aggregation='max' and padded gene_id=0, a padded zero embedding used
    to win the elementwise max against gene 0's real (negative) embeddings,
    silently clipping that gene's embedding to >= 0. The fix replaces masked
    rows with -inf so they lose every comparison.
    """
    torch.manual_seed(0)

    num_genes = 3
    latent_dim = 4
    aggregator = EfficientGeneAggregator(num_genes=num_genes, latent_dim=latent_dim, aggregation='max')

    # Gene 0 has two real, all-negative variants; gene 1 has one real positive
    # variant; the rest of the row is padding (mask=False, gene_id=0,
    # embedding=0). The padded rows must NOT win the max for gene 0.
    real_neg = torch.tensor([
        [-1.5, -2.0, -3.0, -0.5],
        [-0.7, -1.1, -2.5, -0.9],
    ])
    real_pos_gene1 = torch.tensor([[1.0, 0.5, 0.7, 0.2]])
    pad_zero = torch.zeros(2, latent_dim)

    variant_embeddings = torch.cat([real_neg, real_pos_gene1, pad_zero], dim=0).unsqueeze(0)
    gene_ids = torch.tensor([[0, 0, 1, 0, 0]])
    mask = torch.tensor([[True, True, True, False, False]])

    gene_embeddings = aggregator(variant_embeddings, gene_ids, mask)

    # Gene 0's max along each latent dim must come from the real (negative)
    # rows, NOT from the padded zeros — otherwise the embedding would be 0.
    expected_gene0 = torch.maximum(real_neg[0], real_neg[1])
    assert torch.allclose(gene_embeddings[0, 0], expected_gene0), (
        f"Gene 0 max-pool leaked padded zero. Got {gene_embeddings[0, 0]}, "
        f"expected {expected_gene0}"
    )
    # Sanity: gene 0's max must indeed be strictly negative on every dim.
    assert torch.all(gene_embeddings[0, 0] < 0), (
        "Gene 0's max-pooled embedding should be strictly negative; "
        "non-negative values indicate the bug is back."
    )

    # Gene 2 has no real variants — the empty-gene branch should map -inf to 0.
    assert torch.all(gene_embeddings[0, 2] == 0)


def test_efficient_aggregator_max_matches_loop_aggregator_when_no_leak_possible():
    """
    Cross-check the corrected EfficientGeneAggregator against the loop-based
    GeneAggregator (which has always filtered correctly by ``gene_mask & mask``).
    """
    torch.manual_seed(1)

    num_genes = 4
    latent_dim = 6
    batch = 2
    num_variants = 9

    variant_embeddings = torch.randn(batch, num_variants, latent_dim) * 2 - 1
    gene_ids = torch.tensor([
        [0, 0, 1, 1, 2, 2, 3, 0, 0],
        [3, 3, 2, 2, 1, 1, 0, 0, 0],
    ])
    mask = torch.tensor([
        [True, True, True, True, True, False, False, False, False],
        [True, True, True, True, True, True, False, False, False],
    ])

    eff = EfficientGeneAggregator(num_genes=num_genes, latent_dim=latent_dim, aggregation='max')
    loop = GeneAggregator(num_genes=num_genes, latent_dim=latent_dim, aggregation='max')

    eff_out = eff(variant_embeddings, gene_ids, mask)
    loop_out = loop(variant_embeddings, gene_ids, mask)

    assert torch.allclose(eff_out, loop_out, atol=1e-6), (
        "Efficient and loop max aggregators disagree after the padding fix."
    )


def test_efficient_aggregator_mean_and_sum_unchanged():
    """The fix only affects the 'max' branch; mean/sum must be untouched."""
    torch.manual_seed(2)

    num_genes = 2
    latent_dim = 3
    variant_embeddings = torch.randn(1, 4, latent_dim)
    gene_ids = torch.tensor([[0, 1, 0, 1]])
    mask = torch.tensor([[True, True, True, False]])

    for agg in ('mean', 'sum'):
        eff = EfficientGeneAggregator(num_genes=num_genes, latent_dim=latent_dim, aggregation=agg)
        loop = GeneAggregator(num_genes=num_genes, latent_dim=latent_dim, aggregation=agg)
        assert torch.allclose(
            eff(variant_embeddings, gene_ids, mask),
            loop(variant_embeddings, gene_ids, mask),
            atol=1e-6,
        ), f"Aggregator parity broken for aggregation={agg}"


# ----- Position bucketing -----------------------------------------------------


def test_relative_position_bucket_backward_compatible_without_chroms():
    """When chrom tensors are omitted, behaviour must be identical to legacy."""
    qp = torch.tensor([100, 200, 300, 1_000_000])
    kp = torch.tensor([100, 150, 250, 999_000])
    buckets = relative_position_bucket(qp, kp, num_buckets=32)
    assert buckets.shape == (4, 4)
    assert torch.all((buckets >= 0) & (buckets < 32))


def test_relative_position_bucket_routes_cross_chrom_to_dedicated_bucket():
    """
    When chromosome tensors are supplied, every cross-chromosome (q, k) pair
    must land in the dedicated bucket index ``num_buckets``, while same-chrom
    pairs match the legacy bucketing exactly.
    """
    qp = torch.tensor([100, 200, 300])
    kp = torch.tensor([100, 200, 300])
    qc = torch.tensor([0, 0, 1])
    kc = torch.tensor([0, 1, 1])

    legacy = relative_position_bucket(qp, kp, num_buckets=32)
    chrom_aware = relative_position_bucket(qp, kp, num_buckets=32, query_chroms=qc, key_chroms=kc)

    same_chrom = qc[:, None] == kc[None, :]
    # Cross-chromosome pairs => dedicated bucket index = num_buckets
    cross_buckets = chrom_aware[~same_chrom]
    assert torch.all(cross_buckets == 32), (
        f"Expected cross-chrom bucket to be 32, got {cross_buckets.tolist()}"
    )
    # Same-chromosome pairs match legacy
    assert torch.all(chrom_aware[same_chrom] == legacy[same_chrom])


def test_relative_position_bucket_same_coord_different_chrom_does_not_get_zero_bucket():
    """
    Two variants at identical coordinates on different chromosomes must NOT
    share the legacy "0 bp apart" bucket. They must go to the dedicated
    cross-chromosome bucket instead.
    """
    qp = torch.tensor([12345])
    kp = torch.tensor([12345])
    qc = torch.tensor([0])
    kc = torch.tensor([1])

    bucket = relative_position_bucket(qp, kp, num_buckets=32, query_chroms=qc, key_chroms=kc)
    same_pos_same_chrom = relative_position_bucket(qp, kp, num_buckets=32)

    assert bucket.item() != same_pos_same_chrom.item(), (
        "Same coordinate on different chromosomes was bucketed as 0 bp apart "
        "— this is the bug the fix is meant to remove."
    )
    assert bucket.item() == 32


def test_relative_position_bucket_requires_both_chrom_tensors():
    qp = torch.tensor([100, 200])
    kp = torch.tensor([100, 200])
    qc = torch.tensor([0, 0])
    try:
        relative_position_bucket(qp, kp, query_chroms=qc, key_chroms=None)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when only one chrom tensor is supplied.")


# ----- Attention regression ---------------------------------------------------


def test_position_aware_attention_allows_cross_chromosome_attention():
    """
    The fix MUST NOT mask cross-chromosome attention. Q·K stays free; only the
    position-bias scalar uses one shared learned offset for cross-chrom pairs.
    Verify by feeding two variants on different chromosomes and confirming the
    softmaxed attention probabilities are finite and non-zero in both
    directions.
    """
    torch.manual_seed(3)
    latent_dim = 16
    num_heads = 2
    num_chromosomes = 3

    attention = PositionAwareSparseAttention(
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_chromosomes=num_chromosomes,
    )
    attention.eval()

    # Two variants — one on chrom 0, one on chrom 1, at the same coordinate.
    x = torch.randn(1, 2, latent_dim)
    positions = torch.tensor([[12345, 12345]])
    chrom_ids = torch.tensor([[0, 1]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    out, attn = attention(x, positions, mask=mask, chrom_ids=chrom_ids, return_attention=True)
    assert out.shape == x.shape
    assert torch.isfinite(attn).all()
    # Both off-diagonal probabilities must be > 0 (cross-chrom attention is
    # NOT masked; it is allowed to be small but never zero by construction).
    assert (attn[0, :, 0, 1] > 0).all()
    assert (attn[0, :, 1, 0] > 0).all()


def test_position_aware_attention_chrom_embedding_disambiguates_same_coord():
    """
    With ``num_chromosomes > 0`` the attention module adds a learnable
    chromosome embedding to the input. After we monkey-patch that embedding to
    non-zero values, two variants with identical features and identical
    positions but different chromosome ids must produce different attention
    outputs — i.e. the chromosome label changes the representation.
    """
    torch.manual_seed(4)
    latent_dim = 16
    num_heads = 2
    num_chromosomes = 3

    attention = PositionAwareSparseAttention(
        latent_dim=latent_dim,
        num_heads=num_heads,
        num_chromosomes=num_chromosomes,
    )
    # Force the chrom embedding away from its zero init so it actually
    # discriminates chromosomes for this test.
    with torch.no_grad():
        attention.chrom_embedding.weight.copy_(torch.randn_like(attention.chrom_embedding.weight))
    attention.eval()

    x_single = torch.randn(1, 1, latent_dim)
    positions = torch.tensor([[12345]])
    mask = torch.ones(1, 1, dtype=torch.bool)

    out_chrom0, _ = attention(x_single, positions, mask=mask, chrom_ids=torch.tensor([[0]]))
    out_chrom1, _ = attention(x_single, positions, mask=mask, chrom_ids=torch.tensor([[1]]))

    assert not torch.allclose(out_chrom0, out_chrom1, atol=1e-6), (
        "Chromosome embedding did not disambiguate two variants at the same "
        "coordinate on different chromosomes."
    )


# ----- End-to-end model -------------------------------------------------------


def test_sieve_forward_with_chrom_ids_runs_and_produces_gradients():
    """
    Smoke-test: SIEVE.forward accepts chrom_ids, returns finite logits with the
    expected shape, and gradients flow back into the chrom_embedding.
    """
    torch.manual_seed(5)
    input_dim = 8
    num_genes = 4
    num_chromosomes = 5

    model = SIEVE(
        input_dim=input_dim,
        num_genes=num_genes,
        latent_dim=16,
        hidden_dim=16,
        num_heads=2,
        num_attention_layers=1,
        classifier_hidden_dim=16,
        num_chromosomes=num_chromosomes,
    )
    model.train()

    batch, num_variants = 2, 6
    features = torch.randn(batch, num_variants, input_dim, requires_grad=False)
    positions = torch.randint(1, 100_000, (batch, num_variants))
    gene_ids = torch.randint(0, num_genes, (batch, num_variants))
    chrom_ids = torch.randint(0, num_chromosomes, (batch, num_variants))
    mask = torch.ones(batch, num_variants, dtype=torch.bool)

    logits, _ = model(features, positions, gene_ids, mask, chrom_ids=chrom_ids)
    assert logits.shape == (batch, 1)
    assert torch.isfinite(logits).all()

    # Confirm chrom_embedding receives gradient.
    loss = logits.sum()
    loss.backward()
    chrom_emb = model.attention.attention_layers[0].chrom_embedding
    assert chrom_emb is not None
    assert chrom_emb.weight.grad is not None
    assert torch.any(chrom_emb.weight.grad != 0)


def test_sieve_forward_without_chrom_ids_still_works():
    """Backward compatibility: chrom_ids is optional."""
    torch.manual_seed(6)
    model = SIEVE(
        input_dim=8,
        num_genes=4,
        latent_dim=16,
        hidden_dim=16,
        num_heads=2,
        num_attention_layers=1,
        classifier_hidden_dim=16,
        num_chromosomes=0,
    )
    model.eval()

    features = torch.randn(1, 5, 8)
    positions = torch.randint(1, 100_000, (1, 5))
    gene_ids = torch.randint(0, 4, (1, 5))
    mask = torch.ones(1, 5, dtype=torch.bool)

    logits, _ = model(features, positions, gene_ids, mask)
    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all()


if __name__ == '__main__':
    test_efficient_aggregator_max_does_not_leak_padded_zero_into_gene_zero()
    test_efficient_aggregator_max_matches_loop_aggregator_when_no_leak_possible()
    test_efficient_aggregator_mean_and_sum_unchanged()
    test_relative_position_bucket_backward_compatible_without_chroms()
    test_relative_position_bucket_routes_cross_chrom_to_dedicated_bucket()
    test_relative_position_bucket_same_coord_different_chrom_does_not_get_zero_bucket()
    test_relative_position_bucket_requires_both_chrom_tensors()
    test_position_aware_attention_allows_cross_chromosome_attention()
    test_position_aware_attention_chrom_embedding_disambiguates_same_coord()
    test_sieve_forward_with_chrom_ids_runs_and_produces_gradients()
    test_sieve_forward_without_chrom_ids_still_works()
    print("All chrom-aware attention tests passed.")
