"""
Gene-level aggregation for SIEVE.

This module implements aggregation of variant-level embeddings to gene-level
representations. This is crucial for reducing dimensionality from thousands
of variants to hundreds of genes.

Author: Lescai Lab
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor


class GeneAggregator(nn.Module):
    """
    Aggregate variant embeddings to gene-level representations.

    This module pools multiple variants belonging to the same gene into a
    single gene-level embedding. The aggregation must be permutation-invariant
    (variant order shouldn't matter within a gene).

    Aggregation methods:
    - 'max': Element-wise maximum (default)
    - 'mean': Element-wise mean
    - 'sum': Element-wise sum

    Parameters
    ----------
    num_genes : int
        Number of unique genes in the dataset
    latent_dim : int
        Dimension of variant embeddings
    aggregation : Literal['max', 'mean', 'sum']
        Aggregation method (default: 'max')

    Attributes
    ----------
    num_genes : int
        Number of genes
    latent_dim : int
        Embedding dimension
    aggregation : str
        Aggregation method

    Examples
    --------
    >>> aggregator = GeneAggregator(num_genes=100, latent_dim=64, aggregation='max')
    >>> variant_emb = torch.randn(2, 50, 64)  # [batch, variants, latent_dim]
    >>> gene_ids = torch.randint(0, 100, (2, 50))  # [batch, variants]
    >>> mask = torch.ones(2, 50, dtype=torch.bool)
    >>> gene_emb = aggregator(variant_emb, gene_ids, mask)
    >>> gene_emb.shape
    torch.Size([2, 100, 64])
    """

    def __init__(
        self,
        num_genes: int,
        latent_dim: int,
        aggregation: Literal['max', 'mean', 'sum'] = 'max',
    ):
        super().__init__()

        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.aggregation = aggregation

    def forward(
        self,
        variant_embeddings: Tensor,
        gene_ids: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Aggregate variants to genes.

        Parameters
        ----------
        variant_embeddings : Tensor
            Variant embeddings, shape (batch, num_variants, latent_dim)
        gene_ids : Tensor
            Gene assignments, shape (batch, num_variants)
            Values should be in range [0, num_genes-1]
        mask : Optional[Tensor]
            Validity mask, shape (batch, num_variants)
            True = real variant, False = padding

        Returns
        -------
        Tensor
            Gene embeddings, shape (batch, num_genes, latent_dim)

        Notes
        -----
        Genes with no variants will have zero embeddings.
        Padding positions (mask=False) are ignored.
        """
        batch_size, num_variants, latent_dim = variant_embeddings.shape
        device = variant_embeddings.device

        # Initialize output
        gene_embeddings = torch.zeros(
            batch_size,
            self.num_genes,
            latent_dim,
            device=device,
            dtype=variant_embeddings.dtype
        )

        # Apply mask if provided
        if mask is not None:
            # Mask out padding positions
            variant_embeddings = variant_embeddings * mask.unsqueeze(-1).float()

        if self.aggregation == 'max':
            # Max pooling
            # For each gene, take element-wise maximum of all variants
            for b in range(batch_size):
                for gene_id in range(self.num_genes):
                    # Find variants belonging to this gene
                    gene_mask = gene_ids[b] == gene_id

                    if mask is not None:
                        gene_mask = gene_mask & mask[b]

                    if gene_mask.any():
                        # Get variants for this gene
                        gene_variants = variant_embeddings[b, gene_mask]  # (n_variants_in_gene, latent_dim)

                        # Element-wise max
                        gene_embeddings[b, gene_id] = gene_variants.max(dim=0).values

        elif self.aggregation == 'mean':
            # Mean pooling
            for b in range(batch_size):
                for gene_id in range(self.num_genes):
                    gene_mask = gene_ids[b] == gene_id

                    if mask is not None:
                        gene_mask = gene_mask & mask[b]

                    if gene_mask.any():
                        gene_variants = variant_embeddings[b, gene_mask]
                        gene_embeddings[b, gene_id] = gene_variants.mean(dim=0)

        elif self.aggregation == 'sum':
            # Sum pooling
            for b in range(batch_size):
                for gene_id in range(self.num_genes):
                    gene_mask = gene_ids[b] == gene_id

                    if mask is not None:
                        gene_mask = gene_mask & mask[b]

                    if gene_mask.any():
                        gene_variants = variant_embeddings[b, gene_mask]
                        gene_embeddings[b, gene_id] = gene_variants.sum(dim=0)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        return gene_embeddings


class EfficientGeneAggregator(nn.Module):
    """
    More efficient implementation using scatter operations.

    This is faster than the loop-based implementation above, especially
    for large batches.

    Parameters
    ----------
    num_genes : int
        Number of unique genes
    latent_dim : int
        Embedding dimension
    aggregation : Literal['max', 'mean', 'sum']
        Aggregation method

    Examples
    --------
    >>> aggregator = EfficientGeneAggregator(num_genes=100, latent_dim=64)
    >>> variant_emb = torch.randn(2, 50, 64)
    >>> gene_ids = torch.randint(0, 100, (2, 50))
    >>> mask = torch.ones(2, 50, dtype=torch.bool)
    >>> gene_emb = aggregator(variant_emb, gene_ids, mask)
    >>> gene_emb.shape
    torch.Size([2, 100, 64])
    """

    def __init__(
        self,
        num_genes: int,
        latent_dim: int,
        aggregation: Literal['max', 'mean', 'sum'] = 'max',
    ):
        super().__init__()

        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.aggregation = aggregation

    def forward(
        self,
        variant_embeddings: Tensor,
        gene_ids: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Aggregate variants to genes using scatter operations.

        Parameters
        ----------
        variant_embeddings : Tensor
            Variant embeddings, shape (batch, num_variants, latent_dim)
        gene_ids : Tensor
            Gene assignments, shape (batch, num_variants)
        mask : Optional[Tensor]
            Validity mask, shape (batch, num_variants)

        Returns
        -------
        Tensor
            Gene embeddings, shape (batch, num_genes, latent_dim)
        """
        batch_size, num_variants, latent_dim = variant_embeddings.shape
        device = variant_embeddings.device

        # Apply mask
        if mask is not None:
            variant_embeddings = variant_embeddings * mask.unsqueeze(-1).float()

        # Initialize output
        if self.aggregation == 'max':
            # For max, initialize with very negative values
            gene_embeddings = torch.full(
                (batch_size, self.num_genes, latent_dim),
                fill_value=float('-inf'),
                device=device,
                dtype=variant_embeddings.dtype
            )
        else:
            gene_embeddings = torch.zeros(
                batch_size,
                self.num_genes,
                latent_dim,
                device=device,
                dtype=variant_embeddings.dtype
            )

        # Expand gene_ids for broadcasting
        # Shape: (batch, num_variants, latent_dim)
        gene_ids_expanded = gene_ids.unsqueeze(-1).expand(-1, -1, latent_dim)

        # Scatter operation
        if self.aggregation == 'max':
            gene_embeddings.scatter_reduce_(
                dim=1,
                index=gene_ids_expanded,
                src=variant_embeddings,
                reduce='amax',
                include_self=False
            )
            # Replace -inf with 0 for genes with no variants
            gene_embeddings = torch.where(
                torch.isinf(gene_embeddings),
                torch.zeros_like(gene_embeddings),
                gene_embeddings
            )

        elif self.aggregation == 'sum':
            gene_embeddings.scatter_reduce_(
                dim=1,
                index=gene_ids_expanded,
                src=variant_embeddings,
                reduce='sum'
            )

        elif self.aggregation == 'mean':
            # For mean, first sum then divide by count
            gene_embeddings.scatter_reduce_(
                dim=1,
                index=gene_ids_expanded,
                src=variant_embeddings,
                reduce='sum'
            )

            # Count variants per gene
            ones = torch.ones_like(variant_embeddings)
            if mask is not None:
                ones = ones * mask.unsqueeze(-1).float()

            gene_counts = torch.zeros(
                batch_size,
                self.num_genes,
                latent_dim,
                device=device,
                dtype=variant_embeddings.dtype
            )

            gene_counts.scatter_reduce_(
                dim=1,
                index=gene_ids_expanded,
                src=ones,
                reduce='sum'
            )

            # Avoid division by zero
            gene_counts = torch.clamp(gene_counts, min=1.0)

            # Compute mean
            gene_embeddings = gene_embeddings / gene_counts

        return gene_embeddings
