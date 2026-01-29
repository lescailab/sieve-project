"""
Chunked SIEVE model wrapper for whole-genome processing.

Wraps the base SIEVE model to handle chunked inputs and aggregate
chunk-level embeddings into sample-level predictions while preserving
interpretability for explainability analysis.

Key features:
- Aggregates gene embeddings across chunks (not logits)
- Preserves gene embeddings for integrated gradients
- Supports gene-level attribution regularization
- Provides chunk-wise attention patterns for epistasis detection

Author: Lescai Lab
"""

from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn


class ChunkedSIEVEModel(nn.Module):
    """
    Wrapper around SIEVE model to handle chunked variant processing.

    Processes each chunk through the base SIEVE model to get gene embeddings,
    aggregates embeddings across chunks, then applies classification.
    This preserves interpretability for explainability analysis.

    Parameters
    ----------
    base_model : nn.Module
        The base SIEVE model
    aggregation_method : str
        How to aggregate gene embeddings across chunks:
        - 'mean' or 'logit_mean': Average gene embeddings (default)
        - 'max': Element-wise max of gene embeddings
        - 'attention': Learned weighted average (not yet implemented)
    embedding_dim : Optional[int]
        Dimension of embeddings (required for attention aggregation)

    Key Methods
    -----------
    forward() : Returns logits and intermediates (including gene embeddings)
    get_gene_embeddings() : Extract aggregated gene embeddings for explainability
    get_attention_patterns() : Extract chunk-wise attention patterns
    train_step() : Training with support for attribution regularization

    Examples
    --------
    >>> base_model = create_sieve_model(config, num_genes=1000)
    >>> chunked_model = ChunkedSIEVEModel(base_model, aggregation_method='mean')
    >>> # Training processes chunks, aggregates embeddings, preserves interpretability
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
        return_attention: bool = False,
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with automatic chunk aggregation.

        If chunk metadata is provided, aggregates gene embeddings across chunks
        before classification. This preserves interpretability for explainability.

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
        return_attention : bool
            If True, collect attention weights from chunks
        return_intermediate : bool
            If True, return intermediate embeddings

        Returns
        -------
        logits : torch.Tensor
            Sample-level predictions [num_samples] or [num_samples, 1]
        intermediates : Optional[Dict]
            Dictionary containing:
            - 'gene_embeddings': Aggregated gene embeddings
            - 'attention_weights': List of attention weights per chunk (if return_attention=True)
            - 'chunk_metadata': Mapping from chunks to samples
        """
        device = features.device

        # If no chunk metadata, process as regular batch
        # NOTE: Delegates directly to base_model. The base model should be on the
        # same device as ChunkedSIEVEModel to ensure output tensors match input device.
        if original_sample_indices is None:
            return self.base_model(
                features, positions, gene_ids, mask,
                return_attention=return_attention,
                return_intermediate=return_intermediate
            )

        # Process all chunks through base model to get gene embeddings
        chunk_gene_embeddings, chunk_intermediates = self.base_model(
            features, positions, gene_ids, mask,
            return_embeddings=True,  # Get gene embeddings, not logits
            return_attention=return_attention,
            return_intermediate=return_intermediate
        )
        # chunk_gene_embeddings: [num_chunks, num_genes, latent_dim]

        # Get unique samples and map chunks to samples
        unique_samples = original_sample_indices.unique(sorted=True)
        num_samples = len(unique_samples)
        sample_mapping = torch.searchsorted(unique_samples, original_sample_indices)

        # Get dimensions
        num_genes = chunk_gene_embeddings.shape[1]
        latent_dim = chunk_gene_embeddings.shape[2]

        # Aggregate gene embeddings across chunks
        if self.aggregation_method in ['mean', 'logit_mean']:
            # Average gene embeddings across chunks per sample
            # For each gene, average the embeddings from chunks containing that gene

            # Initialize aggregated embeddings
            aggregated_embeddings = torch.zeros(
                num_samples, num_genes, latent_dim,
                dtype=chunk_gene_embeddings.dtype,
                device=device
            )

            # Count how many chunks contribute to each gene per sample
            counts = torch.zeros(
                num_samples, num_genes,
                dtype=torch.float32,
                device=device
            )

            # Vectorized accumulation of embeddings per sample
            # Sum embeddings from all chunks into their corresponding samples
            aggregated_embeddings.index_add_(0, sample_mapping, chunk_gene_embeddings)

            # Compute per-chunk gene presence mask: genes with non-zero embeddings
            # Use L2 norm for robustness to sign cancellations
            gene_has_variants = (chunk_gene_embeddings.pow(2).sum(dim=-1) > 1e-9).float()

            # Accumulate counts of contributing chunks per gene and sample
            counts.index_add_(0, sample_mapping, gene_has_variants)

            # Average, explicitly avoiding division by zero:
            # Only divide where at least one chunk contributed to a gene,
            # and leave zero embeddings unchanged when there are no variants.
            counts_expanded = counts.unsqueeze(-1)
            nonzero_mask = counts_expanded > 0
            safe_counts_expanded = torch.where(
                nonzero_mask,
                counts_expanded,
                torch.ones_like(counts_expanded)
            )
            aggregated_embeddings = torch.where(
                nonzero_mask,
                aggregated_embeddings / safe_counts_expanded,
                aggregated_embeddings
            )

        elif self.aggregation_method == 'max':
            # Max-pool gene embeddings across chunks per sample
            aggregated_embeddings = torch.zeros(
                num_samples, num_genes, latent_dim,
                dtype=chunk_gene_embeddings.dtype,
                device=device
            )

            # Vectorized element-wise max across chunks per sample using scatter_reduce
            if chunk_gene_embeddings.numel() > 0:
                # sample_mapping: [num_chunks] -> expand to match chunk_gene_embeddings
                index = sample_mapping.view(-1, 1, 1).expand(-1, num_genes, latent_dim)
                aggregated_embeddings.scatter_reduce_(
                    dim=0,
                    index=index,
                    src=chunk_gene_embeddings,
                    reduce='amax',
                    include_self=True  # Keep initial zeros, matching previous behavior
                )

        elif self.aggregation_method == 'attention':
            # Learned attention-weighted aggregation
            if not hasattr(self, 'attention'):
                raise ValueError("Attention aggregation requires embedding_dim parameter")

            # Reshape for attention: [num_samples, max_chunks_per_sample, num_genes, latent_dim]
            # This is complex - for now, fall back to mean
            raise NotImplementedError(
                "Attention aggregation not yet implemented. Use 'mean' or 'max'."
            )
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Apply classifier on aggregated embeddings
        # NOTE: This intentionally bypasses base_model.forward() and assumes
        # that the base model exposes a callable 'classifier' attribute that
        # can operate directly on aggregated gene embeddings of shape
        # [num_samples, num_genes, latent_dim].
        # If using a different base model, it must conform to this interface.
        logits = self.base_model.classifier(aggregated_embeddings)

        # Prepare intermediates if requested
        intermediates = None
        if return_intermediate or return_attention:
            intermediates = {
                'gene_embeddings': aggregated_embeddings,
                'chunk_metadata': {
                    'sample_mapping': sample_mapping,
                    'unique_samples': unique_samples
                }
            }
            if return_attention and chunk_intermediates is not None:
                intermediates['attention_weights'] = chunk_intermediates.get('attention_weights', [])

        return logits, intermediates

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training step that handles chunk aggregation.

        Supports attribution regularization (lambda_attr > 0) by computing
        gene-level sparsity on aggregated embeddings.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch from ChunkedVariantDataset
        criterion : nn.Module
            Loss function. Supported types:
            - SIEVELoss: Returns dict {'total': scalar, ...} when lambda_attr > 0
            - BCEWithLogitsLoss: Returns scalar tensor
            Note: For attribution regularization support, loss function must have
            a 'lambda_attr' attribute that can be checked via hasattr().
            Custom loss functions should follow this interface convention.
        device : torch.device
            Device to use

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value (extracted from dict if criterion returns dict)
        predictions : torch.Tensor
            Sample-level predictions [num_samples] or [num_samples, 1]

        Notes
        -----
        Attribution regularization uses gene-level sparsity (not variant-level)
        since chunked processing operates on aggregated gene embeddings.
        This encourages the model to rely on fewer genes rather than fewer variants.
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
            # All chunks from same sample have same label.
            # Get unique samples in sorted order (matching forward method)
            unique_samples = original_sample_indices.unique(sorted=True)

            # For each unique sample, find its first occurrence in original_sample_indices
            # and extract the label at that position
            sample_labels = torch.zeros(len(unique_samples), dtype=labels.dtype, device=device)
            for i, sample_idx in enumerate(unique_samples):
                # Find first chunk belonging to this sample
                first_chunk_idx = (original_sample_indices == sample_idx).nonzero(as_tuple=True)[0][0]
                sample_labels[i] = labels[first_chunk_idx]
        else:
            sample_labels = labels

        # Forward pass (aggregates chunks automatically)
        # Get intermediates for attribution regularization if needed
        need_embeddings = hasattr(criterion, 'lambda_attr') and criterion.lambda_attr > 0

        predictions, intermediates = self.forward(
            features, positions, gene_ids, mask,
            chunk_indices, total_chunks, original_sample_indices,
            return_intermediate=need_embeddings
        )
        # Ensure 1D tensor for loss computation
        if predictions.dim() > 1:
            predictions = predictions.view(-1)

        # Compute loss at sample level
        if need_embeddings and intermediates is not None:
            # Pass gene embeddings for attribution regularization
            loss_output = criterion(
                predictions, sample_labels.float(),
                gene_embeddings=intermediates['gene_embeddings']
            )
        else:
            # Standard classification loss only
            loss_output = criterion(predictions, sample_labels.float())

        # Handle both dict (SIEVELoss) and scalar (BCEWithLogitsLoss) returns
        if isinstance(loss_output, dict):
            loss = loss_output['total']
        else:
            loss = loss_output

        return loss, predictions

    def get_gene_embeddings(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        gene_ids: torch.Tensor,
        mask: torch.Tensor,
        chunk_indices: Optional[torch.Tensor] = None,
        total_chunks: Optional[torch.Tensor] = None,
        original_sample_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get aggregated gene embeddings for explainability.

        Parameters
        ----------
        features, positions, gene_ids, mask : torch.Tensor
            Variant data
        chunk_indices, total_chunks, original_sample_indices : Optional[torch.Tensor]
            Chunking metadata

        Returns
        -------
        torch.Tensor
            Aggregated gene embeddings [num_samples, num_genes, latent_dim]
        """
        _, intermediates = self.forward(
            features, positions, gene_ids, mask,
            chunk_indices, total_chunks, original_sample_indices,
            return_intermediate=True
        )

        if intermediates is None:
            raise RuntimeError(
                "ChunkedSIEVEModel.forward did not return intermediates "
                "despite return_intermediate=True when calling get_gene_embeddings."
            )

        if 'gene_embeddings' not in intermediates:
            raise KeyError(
                "Intermediates returned by ChunkedSIEVEModel.forward do not contain "
                "'gene_embeddings'. Ensure the base model is configured to produce "
                "gene embeddings when return_intermediate=True."
            )

        return intermediates['gene_embeddings']

    def get_attention_patterns(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        gene_ids: torch.Tensor,
        mask: torch.Tensor,
        chunk_indices: Optional[torch.Tensor] = None,
        total_chunks: Optional[torch.Tensor] = None,
        original_sample_indices: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Get attention patterns for explainability.

        NOTE: For chunked data, returns attention weights from each chunk.
        These are within-chunk attention patterns, not full sample attention.

        Parameters
        ----------
        features, positions, gene_ids, mask : torch.Tensor
            Variant data
        chunk_indices, total_chunks, original_sample_indices : Optional[torch.Tensor]
            Chunking metadata

        Returns
        -------
        List[torch.Tensor]
            Attention weights per chunk. Returns empty list if:
            - Base model does not support attention (intermediates is None)
            - Base model did not return attention weights
            - return_attention=True was not honored by base model
        """
        _, intermediates = self.forward(
            features, positions, gene_ids, mask,
            chunk_indices, total_chunks, original_sample_indices,
            return_attention=True
        )

        if intermediates is None or 'attention_weights' not in intermediates:
            return []

        return intermediates['attention_weights']
