import numpy as np
import pytest
import torch

from src.explain.attention_analysis import AttentionAnalyzer


def test_find_top_interactions_supports_percentile_thresholding():
    analyzer = AttentionAnalyzer(
        torch.nn.Identity(),
        device="cpu",
        threshold_mode="percentile",
        attention_percentile=80.0,
    )

    attention_weights = [
        torch.tensor(
            [[[[0.0, 0.1, 0.2], [0.1, 0.0, 0.9], [0.2, 0.9, 0.0]]]],
            dtype=torch.float32,
        )
    ]
    positions = torch.tensor([[100, 200, 300]], dtype=torch.long)
    gene_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.tensor([[True, True, True]])
    sample_indices = torch.tensor([17], dtype=torch.long)
    chunk_indices = torch.tensor([4], dtype=torch.long)

    interactions = analyzer.find_top_interactions(
        attention_weights=attention_weights,
        positions=positions,
        gene_ids=gene_ids,
        mask=mask,
        top_k=10,
        sample_indices=sample_indices,
        chunk_indices=chunk_indices,
    )

    assert len(interactions) == 1
    interaction = interactions[0]
    assert interaction["sample_idx"] == 17
    assert interaction["chunk_idx"] == 4
    assert interaction["variant1_pos"] == 200
    assert interaction["variant2_pos"] == 300
    assert interaction["attention_threshold_mode"] == "percentile"
    assert interaction["attention_percentile"] == pytest.approx(80.0)
    assert interaction["attention_threshold_value"] == pytest.approx(
        np.percentile(np.array([0.1, 0.2, 0.9]), 80.0)
    )


def test_aggregate_interactions_counts_distinct_samples_not_occurrences():
    analyzer = AttentionAnalyzer(torch.nn.Identity(), device="cpu")

    aggregated = analyzer.aggregate_interactions_across_samples(
        all_sample_interactions=[
            [
                {
                    "sample_idx": 7,
                    "variant1_pos": 100,
                    "variant1_gene": 1,
                    "variant2_pos": 200,
                    "variant2_gene": 2,
                    "attention_score": 0.9,
                    "same_gene": False,
                    "distance": 100,
                },
                {
                    "sample_idx": 7,
                    "variant1_pos": 100,
                    "variant1_gene": 1,
                    "variant2_pos": 200,
                    "variant2_gene": 2,
                    "attention_score": 0.8,
                    "same_gene": False,
                    "distance": 100,
                },
            ],
            [
                {
                    "sample_idx": 9,
                    "variant1_pos": 200,
                    "variant1_gene": 2,
                    "variant2_pos": 100,
                    "variant2_gene": 1,
                    "attention_score": 0.7,
                    "same_gene": False,
                    "distance": 100,
                }
            ],
        ],
        min_samples=2,
    )

    assert len(aggregated) == 1
    interaction = aggregated[0]
    assert interaction["num_samples"] == 2
    assert interaction["n_occurrences"] == 3
    assert interaction["mean_attention"] == pytest.approx((0.9 + 0.8 + 0.7) / 3.0)
