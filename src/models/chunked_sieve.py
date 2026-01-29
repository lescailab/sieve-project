"""
Chunked SIEVE model wrapper for whole-genome processing.

Wraps the base SIEVE model to handle chunked inputs and aggregate
chunk-level outputs into sample-level predictions.

Author: Lescai Lab
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


class ChunkedSIEVEModel(nn.Module):
    """
    Wrapper around SIEVE model to handle chunked variant processing.

    Processes each chunk through the base SIEVE model, then aggregates
    chunk embeddings or logits to produce sample-level predictions.

    Parameters
    ----------
    base_model : nn.Module
        The base SIEVE model
    aggregation_method : str
        How to aggregate chunks: 'mean', 'max', 'attention', 'logit_mean'
        - 'mean': Average chunk embeddings before final classification
        - 'max': Max-pool chunk embeddings before final classification
        - 'attention': Weighted average of chunk embeddings using learned attention
        - 'logit_mean': Average chunk logits (predictions) directly
    embedding_dim : Optional[int]
        Dimension of chunk embeddings (required for attention aggregation)

    Examples
    --------
    >>> base_model = create_sieve_model(config, num_genes=1000)
    >>> chunked_model = ChunkedSIEVEModel(base_model, aggregation_method='mean')
    >>> # Training processes chunks, aggregates automatically
    """

    def __init__(
        self,
        base_model: nn.Module,
        aggregation_method: str = 'mean',
        embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.aggregation_method = aggregation_method

        if aggregation_method == 'attention':
            if embedding_dim is None:
                raise ValueError("embedding_dim required for attention aggregation")
            # Learned attention weights over chunks
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

    def forward(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        gene_ids: torch.Tensor,
        mask: torch.Tensor,
        chunk_indices: Optional[torch.Tensor] = None,
        total_chunks: Optional[torch.Tensor] = None,
        original_sample_indices: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with automatic chunk aggregation.

        If chunk metadata is provided, groups chunks by original sample
        and aggregates. Otherwise, processes as regular batch.

        Parameters
        ----------
        features : torch.Tensor
            [batch_size, max_variants, feature_dim]
        positions : torch.Tensor
            [batch_size, max_variants]
        gene_ids : torch.Tensor
            [batch_size, max_variants]
        mask : torch.Tensor
            [batch_size, max_variants]
        chunk_indices : Optional[torch.Tensor]
            [batch_size] - which chunk within the sample
        total_chunks : Optional[torch.Tensor]
            [batch_size] - total chunks for each sample
        original_sample_indices : Optional[torch.Tensor]
            [batch_size] - which original sample this chunk belongs to
        return_embeddings : bool
            If True, return embeddings instead of logits

        Returns
        -------
        torch.Tensor
            Sample-level predictions [num_samples] or [num_samples, num_classes]
        """
        # Process all chunks through base model
        chunk_outputs, intermediates = self.base_model(
            features, positions, gene_ids, mask, return_intermediate=True
        )

        # If no chunk metadata, return as-is (regular processing)
        if original_sample_indices is None:
            return chunk_outputs

        # Aggregate chunks by original sample
        device = chunk_outputs.device

        # Get unique samples and their chunk counts
        unique_samples = original_sample_indices.unique()
        num_samples = len(unique_samples)

        # Aggregate chunk outputs into sample-level predictions
        # NOTE: Currently only 'logit_mean' and 'mean' are supported
        # Other aggregation methods require architectural changes to expose
        # sample-level embeddings before the final classification layer

        # Map each chunk to its sample index
        sample_mapping = torch.searchsorted(
            unique_samples, original_sample_indices
        )

        if self.aggregation_method == 'logit_mean' or self.aggregation_method == 'mean':
            # Aggregate logits directly by averaging chunk predictions
            aggregated = torch.zeros(num_samples, dtype=chunk_outputs.dtype, device=device)
            counts = torch.zeros(num_samples, dtype=torch.float32, device=device)

            # Sum logits per sample
            chunk_outputs_flat = chunk_outputs.squeeze(-1) if chunk_outputs.dim() > 1 else chunk_outputs
            aggregated.scatter_add_(0, sample_mapping, chunk_outputs_flat)
            counts.scatter_add_(0, sample_mapping, torch.ones_like(chunk_outputs_flat, dtype=torch.float32))

            # Average
            aggregated = aggregated / counts.clamp(min=1)

        elif self.aggregation_method in ['max', 'attention']:
            # These methods are not currently supported
            # They would require the base model to expose sample-level embeddings
            # before classification, which the current architecture doesn't do
            raise NotImplementedError(
                f"Aggregation method '{self.aggregation_method}' is not yet implemented. "
                f"Use 'logit_mean' or 'mean' instead. See ChunkedSIEVEModel docstring."
            )

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        return aggregated

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training step that handles chunk aggregation.

        NOTE: Attribution regularization (lambda_attr > 0) is not currently
        supported with chunked processing. Use lambda_attr=0.0 when training
        chunked models.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch from ChunkedVariantDataset
        criterion : nn.Module
            Loss function (should be BCEWithLogitsLoss or similar)
        device : torch.device
            Device to use

        Returns
        -------
        loss : torch.Tensor
            Loss value
        predictions : torch.Tensor
            Sample-level predictions
        """
        # Move to device
        features = batch['features'].to(device)
        positions = batch['positions'].to(device)
        gene_ids = batch['gene_ids'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)

        chunk_indices = batch.get('chunk_indices')
        total_chunks = batch.get('total_chunks')
        original_sample_indices = batch.get('original_sample_indices')

        if chunk_indices is not None:
            chunk_indices = chunk_indices.to(device)
            total_chunks = total_chunks.to(device)
            original_sample_indices = original_sample_indices.to(device)

            # Aggregate chunk labels to sample labels (vectorized)
            # All chunks from same sample have same label
            # Strategy: sort by sample index, then take first of each group
            sorted_indices = torch.argsort(original_sample_indices)
            sorted_sample_ids = original_sample_indices[sorted_indices]

            # Find where sample ID changes (these are the first chunks of each sample)
            # Prepend True to always include first element
            is_first_chunk = torch.cat([
                torch.tensor([True], device=device),
                sorted_sample_ids[1:] != sorted_sample_ids[:-1]
            ])

            # Extract labels at these positions
            sample_labels = labels[sorted_indices[is_first_chunk]]
        else:
            sample_labels = labels

        # Forward pass (aggregates chunks automatically)
        predictions = self.forward(
            features, positions, gene_ids, mask,
            chunk_indices, total_chunks, original_sample_indices
        ).squeeze()

        # Compute loss at sample level
        loss = criterion(predictions, sample_labels.float())

        return loss, predictions
