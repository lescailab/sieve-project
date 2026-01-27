"""
Integrated Gradients for variant attribution.

Uses Captum's IntegratedGradients to compute variant-level importance scores.
This allows us to identify which variants most strongly influence the model's
predictions for each sample.

Author: Lescai Lab
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from captum.attr import IntegratedGradients


class IntegratedGradientsExplainer:
    """
    Compute variant attributions using Integrated Gradients.

    This class wraps Captum's IntegratedGradients to work with SIEVE's
    multi-input architecture (features, positions, gene_ids, mask).

    Parameters
    ----------
    model : nn.Module
        Trained SIEVE model
    device : str
        Device to run computations ('cuda' or 'cpu')
    n_steps : int
        Number of integration steps (default: 50)

    Attributes
    ----------
    model : nn.Module
        The SIEVE model
    device : str
        Computation device
    ig : IntegratedGradients
        Captum's IntegratedGradients instance

    Examples
    --------
    >>> explainer = IntegratedGradientsExplainer(model, device='cuda')
    >>> attributions = explainer.attribute(features, positions, gene_ids, mask)
    >>> # attributions shape: (batch, num_variants, input_dim)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        n_steps: int = 50
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.n_steps = n_steps

        # Wrap model to work with Captum
        self.model_wrapper = SIEVEWrapper(model)

        # Create IntegratedGradients instance
        self.ig = IntegratedGradients(self.model_wrapper)

    def attribute(
        self,
        variant_features: Tensor,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor,
        target: Optional[int] = None,
        baseline: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute integrated gradients attributions for variants.

        Parameters
        ----------
        variant_features : Tensor
            Variant features, shape (batch, num_variants, input_dim)
        positions : Tensor
            Genomic positions, shape (batch, num_variants)
        gene_ids : Tensor
            Gene assignments, shape (batch, num_variants)
        mask : Tensor
            Validity mask, shape (batch, num_variants)
        target : Optional[int]
            Target class (0 or 1). If None, uses predicted class
        baseline : Optional[Tensor]
            Baseline input for integration. If None, uses zeros

        Returns
        -------
        attributions : Tensor
            Variant attributions, shape (batch, num_variants, input_dim)
        """
        # Move inputs to device
        variant_features = variant_features.to(self.device)
        positions = positions.to(self.device)
        gene_ids = gene_ids.to(self.device)
        mask = mask.to(self.device)

        # Create baseline (zero features)
        if baseline is None:
            baseline = torch.zeros_like(variant_features)

        # Compute attributions
        attributions = self.ig.attribute(
            inputs=variant_features,
            baselines=baseline,
            target=target,
            additional_forward_args=(positions, gene_ids, mask),
            n_steps=self.n_steps
        )

        return attributions

    def attribute_batch(
        self,
        dataloader,
        aggregate: str = 'l2'
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Compute attributions for a full dataset.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader yielding batches of samples
        aggregate : str
            How to aggregate feature attributions to variant-level scores:
            - 'l2': L2 norm across features (default)
            - 'l1': L1 norm across features
            - 'sum': Sum across features
            - 'mean': Mean across features

        Returns
        -------
        all_attributions : List[np.ndarray]
            List of attribution arrays, one per sample
        all_variant_scores : List[np.ndarray]
            List of aggregated variant scores, one per sample
        all_metadata : List[Dict]
            List of metadata dicts containing positions, genes, etc.
        """
        all_attributions = []
        all_variant_scores = []
        all_metadata = []

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                # Extract batch data
                features = batch['features']
                positions = batch['positions']
                gene_ids = batch['gene_ids']
                mask = batch['mask']

                # Compute attributions
                attributions = self.attribute(
                    features, positions, gene_ids, mask
                )

                # Convert to numpy
                attributions_np = attributions.cpu().numpy()
                mask_np = mask.cpu().numpy()

                # Aggregate feature attributions to variant scores
                if aggregate == 'l2':
                    variant_scores = np.linalg.norm(attributions_np, ord=2, axis=2)
                elif aggregate == 'l1':
                    variant_scores = np.linalg.norm(attributions_np, ord=1, axis=2)
                elif aggregate == 'sum':
                    variant_scores = np.sum(attributions_np, axis=2)
                elif aggregate == 'mean':
                    variant_scores = np.mean(attributions_np, axis=2)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregate}")

                # Store per-sample results
                for i in range(features.shape[0]):
                    # Get valid variants only
                    valid_mask = mask_np[i]

                    all_attributions.append(attributions_np[i][valid_mask])
                    all_variant_scores.append(variant_scores[i][valid_mask])

                    # Store metadata
                    metadata = {
                        'positions': positions[i][mask[i]].cpu().numpy(),
                        'gene_ids': gene_ids[i][mask[i]].cpu().numpy(),
                        'sample_idx': len(all_metadata),
                    }
                    if 'sample_id' in batch:
                        metadata['sample_id'] = batch['sample_id'][i]
                    if 'labels' in batch:
                        metadata['label'] = batch['labels'][i].cpu().item()

                    all_metadata.append(metadata)

        return all_attributions, all_variant_scores, all_metadata

    def get_top_variants(
        self,
        variant_scores: np.ndarray,
        metadata: Dict,
        top_k: int = 100
    ) -> List[Tuple[int, int, float]]:
        """
        Get top K variants by attribution score.

        Parameters
        ----------
        variant_scores : np.ndarray
            Variant attribution scores, shape (num_variants,)
        metadata : Dict
            Metadata dict with 'positions' and 'gene_ids'
        top_k : int
            Number of top variants to return

        Returns
        -------
        top_variants : List[Tuple[int, int, float]]
            List of (position, gene_id, score) tuples for top variants
        """
        # Get absolute scores (importance regardless of direction)
        abs_scores = np.abs(variant_scores)

        # Get top K indices
        top_indices = np.argsort(abs_scores)[::-1][:top_k]

        # Collect results
        top_variants = []
        for idx in top_indices:
            position = int(metadata['positions'][idx])
            gene_id = int(metadata['gene_ids'][idx])
            score = float(variant_scores[idx])
            top_variants.append((position, gene_id, score))

        return top_variants


class SIEVEWrapper(nn.Module):
    """
    Wrapper for SIEVE model to work with Captum.

    Captum expects a model that takes a single input tensor.
    This wrapper takes variant_features as input and passes
    other arguments through.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        variant_features: Tensor,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor
    ) -> Tensor:
        """Forward pass returning only logits."""
        logits, _ = self.model(
            variant_features,
            positions,
            gene_ids,
            mask,
            return_attention=False,
            return_intermediate=False
        )
        return logits
