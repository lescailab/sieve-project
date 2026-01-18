#!/usr/bin/env python3
"""
Integration test for Phase 1C: Model Architecture

This script tests the complete SIEVE model architecture on real data:
1. Load and encode data (Phase 1A + 1B)
2. Test individual model components
3. Test full SIEVE model forward pass
4. Test attention weight extraction
5. Test model summary and parameter counts
6. Test gradient flow

Author: Lescai Lab
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import build_sample_variants
from src.encoding import AnnotationLevel, VariantDataset, collate_samples
from src.models import (
    VariantEncoder,
    PositionAwareSparseAttention,
    EfficientGeneAggregator,
    PhenotypeClassifier,
    SIEVE,
)


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_variant_encoder():
    """Test VariantEncoder component."""
    print_section("Testing VariantEncoder")

    encoder = VariantEncoder(input_dim=71, hidden_dim=128, latent_dim=64)

    # Test forward pass
    features = torch.randn(2, 10, 71)  # [batch, variants, features]
    encoded = encoder(features)

    print(f"Input shape: {features.shape}")
    print(f"Output shape: {encoded.shape}")
    print(f"Expected shape: [2, 10, 64]")

    assert encoded.shape == (2, 10, 64), "Wrong output shape"
    assert not torch.isnan(encoded).any(), "NaN in output"
    assert not torch.isinf(encoded).any(), "Inf in output"

    print("✓ VariantEncoder works correctly")


def test_attention():
    """Test PositionAwareSparseAttention component."""
    print_section("Testing Position-Aware Sparse Attention")

    attention = PositionAwareSparseAttention(
        latent_dim=64,
        num_heads=4,
        num_position_buckets=32
    )

    # Test forward pass
    x = torch.randn(2, 10, 64)  # [batch, variants, latent_dim]
    positions = torch.randint(100, 10000, (2, 10))  # [batch, variants]
    mask = torch.ones(2, 10, dtype=torch.bool)

    output, attn_weights = attention(x, positions, mask, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Expected attention shape: [2, 4, 10, 10] (batch, heads, queries, keys)")

    assert output.shape == x.shape, "Output shape should match input"
    assert attn_weights.shape == (2, 4, 10, 10), "Wrong attention shape"

    # Check attention weights sum to 1
    attn_sum = attn_weights.sum(dim=-1)
    print(f"Attention weights sum (should be ~1.0): {attn_sum[0, 0, 0]:.6f}")
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "Attention doesn't sum to 1"

    print("✓ Attention mechanism works correctly")


def test_gene_aggregator():
    """Test GeneAggregator component."""
    print_section("Testing Gene Aggregator")

    aggregator = EfficientGeneAggregator(
        num_genes=50,
        latent_dim=64,
        aggregation='max'
    )

    # Test forward pass
    variant_emb = torch.randn(2, 20, 64)  # [batch, variants, latent_dim]
    gene_ids = torch.randint(0, 50, (2, 20))  # [batch, variants]
    mask = torch.ones(2, 20, dtype=torch.bool)

    gene_emb = aggregator(variant_emb, gene_ids, mask)

    print(f"Variant embeddings shape: {variant_emb.shape}")
    print(f"Gene embeddings shape: {gene_emb.shape}")
    print(f"Expected shape: [2, 50, 64]")

    assert gene_emb.shape == (2, 50, 64), "Wrong gene embedding shape"

    # Check that aggregation worked
    genes_with_variants = len(torch.unique(gene_ids[0]))
    print(f"Number of genes with variants in sample 0: {genes_with_variants}")

    print("✓ Gene aggregator works correctly")


def test_classifier():
    """Test PhenotypeClassifier component."""
    print_section("Testing Phenotype Classifier")

    classifier = PhenotypeClassifier(
        num_genes=50,
        latent_dim=64,
        hidden_dim=256
    )

    # Test forward pass
    gene_emb = torch.randn(2, 50, 64)  # [batch, genes, latent_dim]
    logits = classifier(gene_emb)

    print(f"Gene embeddings shape: {gene_emb.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected shape: [2, 1]")

    assert logits.shape == (2, 1), "Wrong logits shape"

    print("✓ Classifier works correctly")


def test_full_model():
    """Test complete SIEVE model."""
    print_section("Testing Complete SIEVE Model")

    model = SIEVE(
        input_dim=71,
        num_genes=50,
        latent_dim=64,
        hidden_dim=128,
        num_heads=4,
        num_attention_layers=2,
        classifier_hidden_dim=256,
    )

    # Test forward pass
    features = torch.randn(2, 20, 71)  # [batch, variants, features]
    positions = torch.randint(100, 10000, (2, 20))
    gene_ids = torch.randint(0, 50, (2, 20))
    mask = torch.ones(2, 20, dtype=torch.bool)

    logits, intermediates = model(
        features,
        positions,
        gene_ids,
        mask,
        return_attention=True,
        return_intermediate=True
    )

    print(f"Input shape: {features.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [2, 1]")

    assert logits.shape == (2, 1), "Wrong output shape"

    # Check intermediates
    print("\nIntermediate representations:")
    print(f"  Variant embeddings: {intermediates['variant_embeddings'].shape}")
    print(f"  Attended embeddings: {intermediates['attended_embeddings'].shape}")
    print(f"  Gene embeddings: {intermediates['gene_embeddings'].shape}")
    print(f"  Attention layers: {len(intermediates['attention_weights'])}")

    print("✓ Full SIEVE model works correctly")


def test_model_on_real_data():
    """Test model on actual VCF data."""
    print_section("Testing on Real Data")

    # Load test data
    vcf_path = Path("test_data/small/test_data.vcf.gz")
    pheno_path = Path("test_data/small/test_data_phenotypes.tsv")

    print(f"Loading data from {vcf_path}...")
    samples = build_sample_variants(vcf_path, pheno_path)

    # Create dataset at L3
    dataset = VariantDataset(samples, AnnotationLevel.L3)
    num_genes = dataset.num_genes

    print(f"Dataset: {len(dataset)} samples, {num_genes} genes")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collate_samples,
        shuffle=False
    )

    # Create model
    model = SIEVE(
        input_dim=71,  # L3 dimension
        num_genes=num_genes,
        latent_dim=64,
        num_heads=4,
        num_attention_layers=2,
    )

    # Get model summary
    summary = model.get_model_summary()
    print("\nModel Summary:")
    print(f"  Input dimension: {summary['input_dim']}")
    print(f"  Number of genes: {summary['num_genes']}")
    print(f"  Latent dimension: {summary['latent_dim']}")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Trainable parameters: {summary['trainable_parameters']:,}")

    print("\nParameter breakdown:")
    print(f"  Encoder: {summary['encoder_params']:,}")
    print(f"  Attention: {summary['attention_params']:,}")
    print(f"  Aggregator: {summary['aggregator_params']:,}")
    print(f"  Classifier: {summary['classifier_params']:,}")

    # Test forward pass
    print("\nTesting forward pass on real data...")
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            features = batch['features']
            positions = batch['positions']
            gene_ids = batch['gene_ids']
            mask = batch['mask']
            labels = batch['labels']

            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Features: {features.shape}")
            print(f"  Positions: {positions.shape}")
            print(f"  Gene IDs: {gene_ids.shape}")
            print(f"  Mask: {mask.shape}")
            print(f"  Labels: {labels.shape}")

            # Forward pass
            logits, intermediates = model(
                features,
                positions,
                gene_ids,
                mask,
                return_attention=True,
                return_intermediate=True
            )

            print(f"  Logits: {logits.shape}")
            print(f"  Logits values: {logits.squeeze().tolist()}")

            # Convert to probabilities
            probs = torch.sigmoid(logits).squeeze()
            print(f"  Probabilities: {probs.tolist()}")
            print(f"  True labels: {labels.tolist()}")

            # Check attention patterns
            attn_weights = intermediates['attention_weights'][0]  # First layer
            print(f"  Attention shape: {attn_weights.shape}")

            # Average attention weights
            avg_attn = attn_weights.mean(dim=1)  # Average over heads
            print(f"  Average attention shape: {avg_attn.shape}")

            # Check for NaN or Inf
            assert not torch.isnan(logits).any(), "NaN in logits"
            assert not torch.isinf(logits).any(), "Inf in logits"

            print(f"  ✓ Batch {batch_idx + 1} processed successfully")

            # Only test first 2 batches
            if batch_idx >= 1:
                break

    print("\n✓ Model works correctly on real data")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print_section("Testing Gradient Flow")

    model = SIEVE(
        input_dim=71,
        num_genes=50,
        latent_dim=64,
        num_heads=4,
        num_attention_layers=2,
    )

    # Create dummy data
    features = torch.randn(2, 20, 71, requires_grad=True)
    positions = torch.randint(100, 10000, (2, 20))
    gene_ids = torch.randint(0, 50, (2, 20))
    mask = torch.ones(2, 20, dtype=torch.bool)
    labels = torch.tensor([0, 1], dtype=torch.float32).unsqueeze(1)

    # Forward pass
    logits, _ = model(features, positions, gene_ids, mask)

    # Compute loss
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients
    has_gradients = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())

    print(f"Parameters with gradients: {has_gradients}/{total_params}")

    # Check gradient magnitudes
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"Gradient norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
    print(f"Mean gradient norm: {sum(grad_norms) / len(grad_norms):.6f}")

    assert has_gradients == total_params, "Not all parameters have gradients"
    assert all(g < 1000 for g in grad_norms), "Gradient explosion detected"

    print("✓ Gradients flow correctly through the model")


def main():
    """Run all model architecture tests."""
    print("=" * 80)
    print("Phase 1C Integration Test: Model Architecture")
    print("=" * 80)

    # Test individual components
    test_variant_encoder()
    test_attention()
    test_gene_aggregator()
    test_classifier()

    # Test full model
    test_full_model()

    # Test on real data
    test_model_on_real_data()

    # Test gradient flow
    test_gradient_flow()

    # Final summary
    print_section("Test Summary")
    print("✅ All Phase 1C tests passed successfully!")
    print("\nTested components:")
    print("  ✓ VariantEncoder (feature projection)")
    print("  ✓ PositionAwareSparseAttention (core innovation)")
    print("  ✓ GeneAggregator (variant → gene pooling)")
    print("  ✓ PhenotypeClassifier (binary classification)")
    print("  ✓ Complete SIEVE model (end-to-end)")
    print("  ✓ Forward pass on real VCF data")
    print("  ✓ Gradient flow and backpropagation")
    print("\nPhase 1C implementation is validated and ready for Phase 1D (Training)!")


if __name__ == '__main__':
    main()
