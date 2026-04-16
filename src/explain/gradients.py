"""
Integrated Gradients for variant attribution.

Uses Captum's IntegratedGradients to compute variant-level importance scores.
This allows us to identify which variants most strongly influence the model's
predictions for each sample.

Author: Francesco Lescai
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
        n_steps: int = 50,
        max_variants: int = 2000
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.n_steps = n_steps
        self.max_variants = max_variants

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
        covariates: Optional[Tensor] = None,
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
        covariates : Optional[Tensor]
            Sample-level covariates, shape (batch, num_covariates).
            Must be provided when the model was trained with covariates
            (``num_covariates > 0``).  Omitting covariates for a model that
            expects them will explain a different function than was trained.

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
        if covariates is not None:
            covariates = covariates.to(self.device)

        # Create baseline (zero features)
        if baseline is None:
            baseline = torch.zeros_like(variant_features)

        # Build additional_forward_args — covariates always included so the
        # wrapper signature stays stable; None is passed when unused.
        additional = (positions, gene_ids, mask, covariates)

        # Compute attributions
        attributions = self.ig.attribute(
            inputs=variant_features,
            baselines=baseline,
            target=target,
            additional_forward_args=additional,
            n_steps=self.n_steps
        )

        return attributions

    def attribute_batch(
        self,
        dataloader,
        aggregate: str = 'l2',
        num_covariates: int = 0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Compute attributions for a full dataset.

        IMPORTANT: Processes samples one at a time to avoid OOM errors.
        Integrated gradients requires storing all intermediate activations
        for gradient computation. With attention mechanisms over thousands
        of variants, batch processing exceeds GPU memory.

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
        num_covariates : int
            Number of covariates expected by the model (default 0).
            When > 0, a 'sex' tensor must be present in each batch.

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
        # NOTE: Do NOT use torch.no_grad() - we need gradients for integrated gradients!

        # Calculate total samples for progress tracking
        total_samples = len(dataloader.dataset)
        print(f"Processing {total_samples} samples individually (required for memory efficiency)...")

        for batch_idx, batch in enumerate(dataloader):
            # Extract batch data
            features = batch['features']
            positions = batch['positions']
            gene_ids = batch['gene_ids']
            mask = batch['mask']
            batch_size = features.shape[0]

            # --- Covariate handling ---
            # Build the per-sample covariate vector using the same logic as
            # ChunkedSIEVEModel.train_step (via build_sample_covariates).
            batch_sex = batch.get('sex')
            batch_covariates = batch.get('covariates')
            if num_covariates > 0:
                if batch_sex is None and batch_covariates is None:
                    raise ValueError(
                        f"Model has num_covariates={num_covariates} but the batch "
                        "contains no covariate tensor. Cannot build covariate vector."
                    )
                # Import here to avoid circular dependency at module level
                from src.models.chunked_sieve import build_sample_covariates
                batch_covariates_full = build_sample_covariates(
                    batch_sex, num_covariates, batch_size,
                    torch.device(self.device),
                    batch_covariates=batch_covariates,
                )
            elif (batch_sex is not None or batch_covariates is not None) and num_covariates == 0:
                raise ValueError(
                    "A covariate tensor is present in the batch but num_covariates=0. "
                    "Either set num_covariates to the correct value or remove the "
                    "covariates from the batch."
                )
            else:
                batch_covariates_full = None

            # CRITICAL: Process each sample individually to avoid OOM
            # Integrated gradients requires storing all intermediate activations,
            # which for attention mechanisms with many variants becomes huge
            for i in range(batch_size):
                # Progress update
                sample_num = len(all_metadata) + 1
                if sample_num % 10 == 0 or sample_num == total_samples:
                    print(f"  Processed {sample_num}/{total_samples} samples...", flush=True)

                # Extract single sample (keep batch dimension)
                sample_features = features[i:i+1]
                sample_positions = positions[i:i+1]
                sample_gene_ids = gene_ids[i:i+1]
                sample_mask = mask[i:i+1]

                # Per-sample covariate slice
                sample_covariates = (
                    batch_covariates_full[i:i+1] if batch_covariates_full is not None else None
                )

                # CRITICAL: Limit variants to avoid OOM
                # Count valid variants for this sample
                num_valid_variants = sample_mask[0].sum().item()

                if num_valid_variants > self.max_variants:
                    # Too many variants - need to subsample
                    valid_indices = torch.where(sample_mask[0])[0]

                    # Random sampling of variant indices
                    selected_indices = valid_indices[torch.randperm(len(valid_indices))[:self.max_variants]]
                    selected_indices = selected_indices.sort()[0]  # Keep sorted for locality

                    # Truncate to selected variants
                    sample_features_truncated = sample_features[:, selected_indices, :]
                    sample_positions_truncated = sample_positions[:, selected_indices]
                    sample_gene_ids_truncated = sample_gene_ids[:, selected_indices]
                    sample_mask_truncated = sample_mask[:, selected_indices]

                    # Track original indices for metadata
                    original_indices = selected_indices
                else:
                    # Use all variants
                    sample_features_truncated = sample_features
                    sample_positions_truncated = sample_positions
                    sample_gene_ids_truncated = sample_gene_ids
                    sample_mask_truncated = sample_mask
                    original_indices = None

                # Compute attributions for this single sample (possibly truncated)
                sample_attributions = self.attribute(
                    sample_features_truncated, sample_positions_truncated,
                    sample_gene_ids_truncated, sample_mask_truncated,
                    covariates=sample_covariates,
                )

                # Convert to numpy
                attributions_np = sample_attributions[0].cpu().numpy()
                mask_np_truncated = sample_mask_truncated[0].cpu().numpy()

                # Aggregate feature attributions to variant scores
                if aggregate == 'l2':
                    variant_scores = np.linalg.norm(attributions_np, ord=2, axis=1)
                elif aggregate == 'l1':
                    variant_scores = np.linalg.norm(attributions_np, ord=1, axis=1)
                elif aggregate == 'sum':
                    variant_scores = np.sum(attributions_np, axis=1)
                elif aggregate == 'mean':
                    variant_scores = np.mean(attributions_np, axis=1)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregate}")

                # Get valid variants only
                valid_mask = mask_np_truncated

                all_attributions.append(attributions_np[valid_mask])
                all_variant_scores.append(variant_scores[valid_mask])

                # Store metadata (use truncated positions/genes if applicable)
                if original_indices is not None:
                    # Was truncated - use the selected subset
                    metadata = {
                        'positions': sample_positions_truncated[0][sample_mask_truncated[0]].cpu().numpy(),
                        'gene_ids': sample_gene_ids_truncated[0][sample_mask_truncated[0]].cpu().numpy(),
                        'sample_idx': len(all_metadata),
                        'num_variants_original': num_valid_variants,
                        'num_variants_analyzed': self.max_variants,
                        'truncated': True,
                    }
                else:
                    # Not truncated - use all
                    metadata = {
                        'positions': positions[i][mask[i]].cpu().numpy(),
                        'gene_ids': gene_ids[i][mask[i]].cpu().numpy(),
                        'sample_idx': len(all_metadata),
                        'num_variants_original': num_valid_variants,
                        'num_variants_analyzed': num_valid_variants,
                        'truncated': False,
                    }
                # Fix: use sample_ids (plural) not sample_id
                if 'sample_ids' in batch:
                    metadata['sample_id'] = batch['sample_ids'][i]
                if 'labels' in batch:
                    metadata['label'] = batch['labels'][i].cpu().item()

                all_metadata.append(metadata)

                # Clear GPU cache after each sample
                if self.device == 'cuda':
                    torch.cuda.empty_cache()

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
    This wrapper takes variant_features as the differentiable input and
    passes all other arguments (positions, gene_ids, mask, and optionally
    covariates) through ``additional_forward_args``.

    Covariates are passed as the optional last positional argument so that
    the function signature is identical whether or not the model uses them.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        variant_features: Tensor,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor,
        covariates: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass returning only logits."""
        logits, _ = self.model(
            variant_features,
            positions,
            gene_ids,
            mask,
            covariates=covariates,
            return_attention=False,
            return_intermediate=False
        )
        return logits
