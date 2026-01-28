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
        chunk_outputs = self.base_model(features, positions, gene_ids, mask)

        # If no chunk metadata, return as-is (regular processing)
        if original_sample_indices is None:
            return chunk_outputs

        # Aggregate chunks by original sample
        device = chunk_outputs.device
        batch_size = chunk_outputs.shape[0]

        # Get unique samples and their chunk counts
        unique_samples = original_sample_indices.unique()
        num_samples = len(unique_samples)

        # Initialize aggregated outputs
        if self.aggregation_method == 'logit_mean':
            # Aggregate logits directly
            aggregated = torch.zeros(num_samples, device=device)

            for i, sample_idx in enumerate(unique_samples):
                # Find all chunks for this sample
                chunk_mask = (original_sample_indices == sample_idx)
                sample_chunks = chunk_outputs[chunk_mask]

                # Average logits
                aggregated[i] = sample_chunks.mean()

        else:
            # Need embeddings for other aggregation methods
            # For binary classification, chunk_outputs are logits [batch, 1]
            # We need to get embeddings from base model
            # This requires base model to have a get_embeddings method

            if not hasattr(self.base_model, 'get_embeddings'):
                # Fallback: use logit aggregation
                aggregated = torch.zeros(num_samples, device=device)
                for i, sample_idx in enumerate(unique_samples):
                    chunk_mask = (original_sample_indices == sample_idx)
                    sample_chunks = chunk_outputs[chunk_mask]
                    aggregated[i] = sample_chunks.mean()
            else:
                # Get embeddings for all chunks
                chunk_embeddings = self.base_model.get_embeddings(
                    features, positions, gene_ids, mask
                )
                embedding_dim = chunk_embeddings.shape[1]

                if self.aggregation_method == 'mean':
                    # Mean pooling over chunks
                    aggregated_embeddings = torch.zeros(
                        num_samples, embedding_dim, device=device
                    )
                    for i, sample_idx in enumerate(unique_samples):
                        chunk_mask = (original_sample_indices == sample_idx)
                        sample_chunks = chunk_embeddings[chunk_mask]
                        aggregated_embeddings[i] = sample_chunks.mean(dim=0)

                elif self.aggregation_method == 'max':
                    # Max pooling over chunks
                    aggregated_embeddings = torch.zeros(
                        num_samples, embedding_dim, device=device
                    )
                    for i, sample_idx in enumerate(unique_samples):
                        chunk_mask = (original_sample_indices == sample_idx)
                        sample_chunks = chunk_embeddings[chunk_mask]
                        aggregated_embeddings[i] = sample_chunks.max(dim=0)[0]

                elif self.aggregation_method == 'attention':
                    # Attention-weighted pooling
                    aggregated_embeddings = torch.zeros(
                        num_samples, embedding_dim, device=device
                    )
                    for i, sample_idx in enumerate(unique_samples):
                        chunk_mask = (original_sample_indices == sample_idx)
                        sample_chunks = chunk_embeddings[chunk_mask]

                        # Compute attention weights
                        attention_scores = self.attention(sample_chunks)  # [num_chunks, 1]
                        attention_weights = torch.softmax(attention_scores, dim=0)

                        # Weighted sum
                        aggregated_embeddings[i] = (sample_chunks * attention_weights).sum(dim=0)

                else:
                    raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

                if return_embeddings:
                    return aggregated_embeddings

                # Apply final classifier to aggregated embeddings
                if hasattr(self.base_model, 'classifier'):
                    aggregated = self.base_model.classifier(aggregated_embeddings)
                else:
                    # Fallback: average chunk logits
                    aggregated = torch.zeros(num_samples, device=device)
                    for i, sample_idx in enumerate(unique_samples):
                        chunk_mask = (original_sample_indices == sample_idx)
                        sample_chunks = chunk_outputs[chunk_mask]
                        aggregated[i] = sample_chunks.mean()

        return aggregated

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training step that handles chunk aggregation.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch from ChunkedVariantDataset
        criterion : nn.Module
            Loss function
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

            # Get unique sample labels (aggregate from chunks)
            unique_samples = original_sample_indices.unique()
            sample_labels = torch.zeros(len(unique_samples), dtype=torch.long, device=device)
            for i, sample_idx in enumerate(unique_samples):
                chunk_mask = (original_sample_indices == sample_idx)
                # All chunks from same sample have same label
                sample_labels[i] = labels[chunk_mask][0]
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
