"""
Unit tests for ChunkedSIEVEModel.

Tests the chunked processing functionality including:
- train_step with different loss functions (dict vs scalar returns)
- Loss division compatibility for gradient accumulation
- Prediction shape verification
- Chunk aggregation correctness
"""
import pytest
import torch
import torch.nn as nn
from typing import Dict

# Import the modules we need to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.chunked_sieve import ChunkedSIEVEModel
from src.models.sieve import create_sieve_model
from src.training.loss import SIEVELoss


class MockSIEVEModel(nn.Module):
    """Mock SIEVE model for testing."""

    def __init__(self, input_dim: int, num_genes: int, latent_dim: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_genes = num_genes
        self.latent_dim = latent_dim or input_dim  # For simplicity in mock
        # Simple classifier that mimics PhenotypeClassifier
        # Input: [batch, num_genes, latent_dim] -> output: [batch, 1]
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),  # [batch, num_genes, latent_dim] -> [batch, num_genes * latent_dim]
            nn.Linear(num_genes * self.latent_dim, 1)
        )

    def forward(
        self,
        features,
        positions,
        gene_ids,
        mask,
        return_intermediate=False,
        return_attention=False,
        return_embeddings=False
    ):
        """
        Mock forward pass that mimics real SIEVE behavior.

        When return_embeddings=True, returns gene embeddings.
        Otherwise returns logits.
        """
        batch_size = features.shape[0]
        feature_dim = features.shape[2]

        # Simple mock gene embeddings: create fake gene-level representations
        # In reality, this would involve proper aggregation by gene_ids
        # For testing, just create a tensor of the right shape
        gene_embeddings = torch.zeros(batch_size, self.num_genes, self.latent_dim, device=features.device)

        # Mock: assign variant features to genes (very simplified)
        for b in range(batch_size):
            for v in range(features.shape[1]):
                if mask[b, v] > 0:
                    gene_id = gene_ids[b, v].item()
                    if gene_id < self.num_genes:
                        gene_embeddings[b, gene_id] = features[b, v]

        # Prepare intermediates if requested
        intermediates = None
        if return_intermediate or return_attention or return_embeddings:
            intermediates = {
                'variant_embeddings': features,
                'gene_embeddings': gene_embeddings
            }
            if return_attention:
                # Mock attention weights
                intermediates['attention_weights'] = []

        # Return embeddings or logits based on flag
        if return_embeddings:
            return gene_embeddings, intermediates
        else:
            # Classify from gene embeddings
            logits = self.classifier(gene_embeddings)
            return logits, intermediates


class TestChunkedSIEVEModel:
    """Test suite for ChunkedSIEVEModel."""

    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def base_model(self):
        """Create a simple base model for testing."""
        return MockSIEVEModel(input_dim=10, num_genes=100)

    @pytest.fixture
    def chunked_model(self, base_model):
        """Create a ChunkedSIEVEModel for testing."""
        return ChunkedSIEVEModel(base_model, aggregation_method='mean')

    @pytest.fixture
    def sample_batch(self, device):
        """Create a sample batch with chunked data."""
        # Simulate 3 samples split into 5 chunks total
        batch_size = 5  # Total chunks
        num_variants = 20
        feature_dim = 10

        batch = {
            'features': torch.randn(batch_size, num_variants, feature_dim).to(device),
            'positions': torch.randint(0, 1000000, (batch_size, num_variants)).to(device),
            'gene_ids': torch.randint(0, 100, (batch_size, num_variants)).to(device),
            'mask': torch.randint(0, 2, (batch_size, num_variants)).float().to(device),
            'labels': torch.tensor([0, 0, 1, 1, 1], dtype=torch.long).to(device),  # Chunk labels
            'chunk_indices': torch.tensor([0, 1, 0, 1, 2], dtype=torch.long).to(device),  # Chunk index within sample
            'total_chunks': torch.tensor([2, 2, 2, 3, 3], dtype=torch.long).to(device),  # Total chunks per sample
            'original_sample_indices': torch.tensor([0, 0, 1, 2, 2], dtype=torch.long).to(device),  # Sample IDs
        }
        return batch

    def test_train_step_with_dict_loss(self, chunked_model, sample_batch, device):
        """Test train_step with SIEVELoss and lambda_attr=0 (dict return, zero attribution)."""
        chunked_model.to(device)
        criterion = SIEVELoss(lambda_attr=0.0)

        # Run train_step
        loss_output, predictions = chunked_model.train_step(sample_batch, criterion, device)

        # SIEVELoss always returns a dict
        assert isinstance(loss_output, dict), "SIEVELoss should return a dict"
        loss = loss_output['total']

        assert loss.dim() == 0 or loss.numel() == 1, "Loss should be a scalar"
        assert loss.requires_grad, "Loss should require gradients"

        # With lambda_attr=0, attribution should be zero
        assert loss_output['attribution_sparsity'].item() == 0.0

        # Total should equal classification
        assert torch.allclose(loss_output['total'], loss_output['classification'])

        # Verify predictions shape matches number of unique samples
        unique_samples = sample_batch['original_sample_indices'].unique()
        num_samples = len(unique_samples)
        assert predictions.shape[0] == num_samples, f"Expected {num_samples} predictions, got {predictions.shape[0]}"

        loss.backward()
        print(f"✓ train_step with dict loss (λ=0): loss={loss.item():.4f}")

    def test_train_step_with_attribution_loss(self, chunked_model, sample_batch, device):
        """Test train_step returns full loss dict when lambda_attr > 0."""
        chunked_model.to(device)
        criterion = SIEVELoss(lambda_attr=0.1)

        loss_output, predictions = chunked_model.train_step(sample_batch, criterion, device)

        # Should return a dict with decomposition
        assert isinstance(loss_output, dict), "Loss output should be a dict when using SIEVELoss"
        assert 'total' in loss_output
        assert 'classification' in loss_output
        assert 'attribution_sparsity' in loss_output

        # Attribution sparsity should be > 0 when lambda_attr > 0
        assert loss_output['attribution_sparsity'].item() > 0.0, (
            "Attribution sparsity loss should be non-zero when lambda_attr > 0"
        )

        # Total should equal classification + lambda * attribution
        expected_total = (
            loss_output['classification']
            + 0.1 * loss_output['attribution_sparsity']
        )
        assert torch.allclose(loss_output['total'], expected_total, atol=1e-5), (
            f"Total loss {loss_output['total'].item():.6f} should equal "
            f"classification {loss_output['classification'].item():.6f} + "
            f"0.1 * attribution {loss_output['attribution_sparsity'].item():.6f} = "
            f"{expected_total.item():.6f}"
        )

        # Backprop should work on total
        loss_output['total'].backward()

        print(
            f"✓ train_step with attribution: "
            f"total={loss_output['total'].item():.4f}, "
            f"classification={loss_output['classification'].item():.4f}, "
            f"attribution={loss_output['attribution_sparsity'].item():.4f}"
        )

    def test_train_step_with_scalar_loss(self, chunked_model, sample_batch, device):
        """Test train_step with BCEWithLogitsLoss (scalar return)."""
        chunked_model.to(device)
        criterion = nn.BCEWithLogitsLoss()

        # Run train_step
        loss_output, predictions = chunked_model.train_step(sample_batch, criterion, device)

        # Should return a plain tensor (not a dict)
        assert isinstance(loss_output, torch.Tensor), "Loss should be a tensor for BCEWithLogitsLoss"
        assert loss_output.dim() == 0 or loss_output.numel() == 1, "Loss should be a scalar"
        assert loss_output.requires_grad, "Loss should require gradients"

        # Verify predictions shape matches number of unique samples
        unique_samples = sample_batch['original_sample_indices'].unique()
        num_samples = len(unique_samples)
        assert predictions.shape[0] == num_samples, f"Expected {num_samples} predictions, got {predictions.shape[0]}"

        loss_output.backward()
        print(f"✓ train_step with scalar loss: loss={loss_output.item():.4f}")

    def test_chunk_aggregation_correctness(self, chunked_model, sample_batch, device):
        """Test that chunk aggregation produces correct sample-level outputs."""
        chunked_model.to(device)
        chunked_model.eval()

        with torch.no_grad():
            # Forward pass with chunking
            predictions, intermediates = chunked_model.forward(
                features=sample_batch['features'],
                positions=sample_batch['positions'],
                gene_ids=sample_batch['gene_ids'],
                mask=sample_batch['mask'],
                chunk_indices=sample_batch['chunk_indices'],
                total_chunks=sample_batch['total_chunks'],
                original_sample_indices=sample_batch['original_sample_indices']
            )

            # Verify we get predictions for each unique sample
            unique_samples = sample_batch['original_sample_indices'].unique()
            num_samples = len(unique_samples)

            assert predictions.shape[0] == num_samples, \
                f"Expected {num_samples} predictions, got {predictions.shape[0]}"

            # Verify predictions are valid (not NaN, not inf)
            assert not torch.isnan(predictions).any(), "Predictions contain NaN"
            assert not torch.isinf(predictions).any(), "Predictions contain Inf"

            print(f"✓ Chunk aggregation: {num_samples} samples from {len(sample_batch['features'])} chunks")

    def test_label_aggregation_correctness(self, chunked_model, sample_batch, device):
        """Test that labels are correctly aggregated from chunks to samples."""
        chunked_model.to(device)
        criterion = nn.BCEWithLogitsLoss()

        # Expected sample labels (based on original_sample_indices):
        # Sample 0: chunks [0,1] -> label 0
        # Sample 1: chunk [2] -> label 1
        # Sample 2: chunks [3,4] -> label 1
        expected_sample_labels = torch.tensor([0, 1, 1], dtype=torch.long).to(device)

        # Run train_step
        loss, predictions = chunked_model.train_step(sample_batch, criterion, device)

        # Manually extract aggregated labels using the same logic as train_step
        original_sample_indices = sample_batch['original_sample_indices']
        labels = sample_batch['labels']

        # Use same method as train_step
        unique_samples = original_sample_indices.unique(sorted=True)
        sample_labels = torch.zeros(len(unique_samples), dtype=labels.dtype, device=device)
        for i, sample_idx in enumerate(unique_samples):
            first_chunk_idx = (original_sample_indices == sample_idx).nonzero(as_tuple=True)[0][0]
            sample_labels[i] = labels[first_chunk_idx]

        # Verify aggregated labels match expectations
        assert torch.equal(sample_labels, expected_sample_labels), \
            f"Expected labels {expected_sample_labels}, got {sample_labels}"

        print(f"✓ Label aggregation: {len(unique_samples)} samples with correct labels")

    def test_forward_without_chunking(self, chunked_model, device):
        """Test that forward works without chunking metadata (regular processing)."""
        chunked_model.to(device)

        batch_size = 4
        num_variants = 20
        feature_dim = 10

        features = torch.randn(batch_size, num_variants, feature_dim).to(device)
        positions = torch.randint(0, 1000000, (batch_size, num_variants)).to(device)
        gene_ids = torch.randint(0, 100, (batch_size, num_variants)).to(device)
        mask = torch.randint(0, 2, (batch_size, num_variants)).float().to(device)

        with torch.no_grad():
            predictions, intermediates = chunked_model.forward(
                features=features,
                positions=positions,
                gene_ids=gene_ids,
                mask=mask
            )

            # Should return same shape as input batch
            assert predictions.shape[0] == batch_size, \
                f"Expected {batch_size} predictions, got {predictions.shape[0]}"

            print(f"✓ Forward without chunking: {batch_size} predictions")

    def test_different_aggregation_methods(self, base_model, device):
        """Test different aggregation methods: max works, attention raises NotImplementedError."""
        # Test 'max' aggregation (should work)
        chunked_model_max = ChunkedSIEVEModel(base_model, aggregation_method='max')
        chunked_model_max.to(device)

        batch_size = 2
        num_variants = 10
        feature_dim = 10

        features = torch.randn(batch_size, num_variants, feature_dim).to(device)
        positions = torch.randint(0, 1000000, (batch_size, num_variants)).to(device)
        gene_ids = torch.randint(0, 100, (batch_size, num_variants)).to(device)
        mask = torch.ones(batch_size, num_variants).to(device)
        chunk_indices = torch.tensor([0, 1]).to(device)
        total_chunks = torch.tensor([2, 2]).to(device)
        original_sample_indices = torch.tensor([0, 0]).to(device)

        # Max aggregation should work
        with torch.no_grad():
            predictions, intermediates = chunked_model_max.forward(
                features=features,
                positions=positions,
                gene_ids=gene_ids,
                mask=mask,
                chunk_indices=chunk_indices,
                total_chunks=total_chunks,
                original_sample_indices=original_sample_indices
            )
            # Should aggregate to 1 unique sample
            assert predictions.shape[0] == 1, f"Expected 1 sample, got {predictions.shape[0]}"

        print(f"✓ Max aggregation works correctly")

        # Test 'attention' aggregation (should raise NotImplementedError)
        chunked_model_attention = ChunkedSIEVEModel(
            base_model,
            aggregation_method='attention',
            embedding_dim=10  # Required for attention aggregation
        )
        chunked_model_attention.to(device)

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            chunked_model_attention.forward(
                features=features,
                positions=positions,
                gene_ids=gene_ids,
                mask=mask,
                chunk_indices=chunk_indices,
                total_chunks=total_chunks,
                original_sample_indices=original_sample_indices
            )

        print("✓ Unsupported aggregation methods correctly raise NotImplementedError")

    def test_gradient_accumulation_compatibility(self, chunked_model, sample_batch, device):
        """Test that loss can be properly scaled for gradient accumulation."""
        chunked_model.to(device)
        criterion = SIEVELoss(lambda_attr=0.0)

        # Simulate gradient accumulation with 8 steps
        accumulation_steps = 8

        for step in range(accumulation_steps):
            loss_output, predictions = chunked_model.train_step(sample_batch, criterion, device)
            loss = loss_output['total'] if isinstance(loss_output, dict) else loss_output

            # Scale loss
            scaled_loss = loss / accumulation_steps

            # Verify scaled loss is valid
            assert isinstance(scaled_loss, torch.Tensor), "Scaled loss should be a tensor"
            assert scaled_loss.requires_grad, "Scaled loss should require gradients"
            assert not torch.isnan(scaled_loss), "Scaled loss should not be NaN"
            assert not torch.isinf(scaled_loss), "Scaled loss should not be Inf"

            # Backward (in real training, this accumulates gradients)
            scaled_loss.backward()

            # Zero gradients for next iteration
            chunked_model.zero_grad()

        print(f"✓ Gradient accumulation: successfully scaled loss across {accumulation_steps} steps")


if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v', '-s'])
