"""
SHAP-based epistasis detection for variant interactions.

Uses SHAP (SHapley Additive exPlanations) interaction values to quantify
variant-variant interactions more rigorously than attention weights alone.

Author: Lescai Lab
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import shap


class SHAPEpistasisDetector:
    """
    Detect epistatic interactions using SHAP interaction values.

    SHAP interaction values quantify how the presence of one variant
    modifies the importance of another variant for the prediction.

    Parameters
    ----------
    model : nn.Module
        Trained SIEVE model
    device : str
        Device to run computations ('cuda' or 'cpu')
    background_samples : int
        Number of background samples for SHAP (default: 50)

    Examples
    --------
    >>> detector = SHAPEpistasisDetector(model, device='cuda')
    >>> interactions = detector.compute_interactions(
    ...     features, positions, gene_ids, mask
    ... )
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        background_samples: int = 50
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.background_samples = background_samples

    def create_background_data(
        self,
        dataloader,
        max_samples: int = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Create background dataset for SHAP.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for creating background
        max_samples : int
            Maximum number of samples to use

        Returns
        -------
        background : Tuple of Tensors
            (features, positions, gene_ids, mask)
        """
        all_features = []
        all_positions = []
        all_gene_ids = []
        all_masks = []

        n_samples = min(max_samples or self.background_samples, len(dataloader.dataset))

        for batch in dataloader:
            all_features.append(batch['features'])
            all_positions.append(batch['positions'])
            all_gene_ids.append(batch['gene_ids'])
            all_masks.append(batch['mask'])

            if len(all_features) * batch['features'].shape[0] >= n_samples:
                break

        # Concatenate and truncate to desired size
        features = torch.cat(all_features, dim=0)[:n_samples]
        positions = torch.cat(all_positions, dim=0)[:n_samples]
        gene_ids = torch.cat(all_gene_ids, dim=0)[:n_samples]
        masks = torch.cat(all_masks, dim=0)[:n_samples]

        return features, positions, gene_ids, masks

    def compute_variant_shap_values(
        self,
        features: Tensor,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor,
        background: Tuple[Tensor, Tensor, Tensor, Tensor],
        max_variants: int = 100
    ) -> np.ndarray:
        """
        Compute SHAP values for variants.

        Due to computational cost, this computes SHAP for the top K variants
        by integrated gradients score.

        Parameters
        ----------
        features : Tensor
            Variant features to explain
        positions : Tensor
            Genomic positions
        gene_ids : Tensor
            Gene IDs
        mask : Tensor
            Validity mask
        background : Tuple
            Background data for SHAP
        max_variants : int
            Maximum number of variants to analyze

        Returns
        -------
        shap_values : np.ndarray
            SHAP values for variants
        """
        # Create wrapper model
        def model_fn(variant_features_numpy):
            """Wrapper for SHAP."""
            variant_features = torch.tensor(
                variant_features_numpy,
                dtype=torch.float32,
                device=self.device
            )

            with torch.no_grad():
                logits, _ = self.model(
                    variant_features,
                    positions.to(self.device),
                    gene_ids.to(self.device),
                    mask.to(self.device),
                    return_attention=False
                )

            return torch.sigmoid(logits).cpu().numpy()

        # Prepare data
        features_np = features.cpu().numpy()
        background_features = background[0].cpu().numpy()

        # Create explainer
        explainer = shap.KernelExplainer(
            model_fn,
            background_features[:self.background_samples]
        )

        # Compute SHAP values (this is slow!)
        print(f"Computing SHAP values (this may take several minutes)...")
        shap_values = explainer.shap_values(
            features_np,
            nsamples=100  # Number of coalitions to sample
        )

        return shap_values

    def estimate_interactions_from_attention(
        self,
        attention_weights: List[Tensor],
        attributions: np.ndarray,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor,
        top_k: int = 50
    ) -> List[Dict]:
        """
        Fast approximation of interactions using attention × attribution.

        This combines attention patterns (which variants attend to each other)
        with attribution scores (which variants are important) to prioritize
        likely epistatic interactions for full SHAP analysis.

        Parameters
        ----------
        attention_weights : List[Tensor]
            Attention from each layer
        attributions : np.ndarray
            Variant attribution scores
        positions : Tensor
            Genomic positions
        gene_ids : Tensor
            Gene IDs
        mask : Tensor
            Validity mask
        top_k : int
            Number of top interactions to return

        Returns
        -------
        interactions : List[Dict]
            Prioritized interactions for SHAP validation
        """
        # Average attention across layers and heads
        attn = torch.mean(torch.stack(attention_weights), dim=0)  # (batch, heads, vars, vars)
        attn = attn.mean(dim=1)  # (batch, vars, vars)

        interactions = []

        for b in range(attn.shape[0]):
            # Get valid variants
            valid_mask = mask[b].cpu().numpy()
            n_valid = valid_mask.sum()

            if n_valid < 2:
                continue

            # Get attention and attributions for valid variants
            attn_matrix = attn[b].cpu().numpy()[:n_valid, :n_valid]
            attr_scores = np.abs(attributions[b][:n_valid])

            # Interaction score = attention_weight × (importance_i + importance_j)
            # This prioritizes:
            # - High mutual attention (from model)
            # - High individual importance (from IG)
            for i in range(n_valid):
                for j in range(i + 1, n_valid):
                    interaction_score = attn_matrix[i, j] * (attr_scores[i] + attr_scores[j])

                    if interaction_score > 0.01:  # Threshold
                        interactions.append({
                            'sample_idx': b,
                            'variant1_idx': i,
                            'variant2_idx': j,
                            'variant1_pos': int(positions[b][i].item()),
                            'variant2_pos': int(positions[b][j].item()),
                            'variant1_gene': int(gene_ids[b][i].item()),
                            'variant2_gene': int(gene_ids[b][j].item()),
                            'interaction_score': float(interaction_score),
                            'attention': float(attn_matrix[i, j]),
                            'variant1_importance': float(attr_scores[i]),
                            'variant2_importance': float(attr_scores[j]),
                        })

        # Sort by interaction score
        interactions.sort(key=lambda x: x['interaction_score'], reverse=True)

        return interactions[:top_k]

    def validate_interaction_with_perturbation(
        self,
        features: Tensor,
        positions: Tensor,
        gene_ids: Tensor,
        mask: Tensor,
        variant1_idx: int,
        variant2_idx: int
    ) -> Dict:
        """
        Validate interaction using counterfactual perturbation.

        Tests if the effect of variant1 depends on the presence of variant2
        by comparing predictions in four conditions:
        1. Both present
        2. Only variant1
        3. Only variant2
        4. Neither present

        Parameters
        ----------
        features : Tensor
            Variant features (single sample)
        positions : Tensor
            Positions (single sample)
        gene_ids : Tensor
            Gene IDs (single sample)
        mask : Tensor
            Mask (single sample)
        variant1_idx : int
            Index of first variant
        variant2_idx : int
            Index of second variant

        Returns
        -------
        validation : Dict
            Interaction validation results with:
            - predictions for each condition
            - synergy score (interaction strength)
            - interaction_type (synergistic/antagonistic)
        """
        self.model.eval()

        # Ensure single sample
        if features.dim() == 2:
            features = features.unsqueeze(0)
            positions = positions.unsqueeze(0)
            gene_ids = gene_ids.unsqueeze(0)
            mask = mask.unsqueeze(0)

        features = features.to(self.device)
        positions = positions.to(self.device)
        gene_ids = gene_ids.to(self.device)
        mask = mask.to(self.device)

        with torch.no_grad():
            # Condition 1: Both present (original)
            logits_both, _ = self.model(features, positions, gene_ids, mask)
            pred_both = torch.sigmoid(logits_both).item()

            # Condition 2: Only variant1 (ablate variant2)
            features_v1 = features.clone()
            features_v1[0, variant2_idx, :] = 0
            mask_v1 = mask.clone()
            mask_v1[0, variant2_idx] = False

            logits_v1, _ = self.model(features_v1, positions, gene_ids, mask_v1)
            pred_v1 = torch.sigmoid(logits_v1).item()

            # Condition 3: Only variant2 (ablate variant1)
            features_v2 = features.clone()
            features_v2[0, variant1_idx, :] = 0
            mask_v2 = mask.clone()
            mask_v2[0, variant1_idx] = False

            logits_v2, _ = self.model(features_v2, positions, gene_ids, mask_v2)
            pred_v2 = torch.sigmoid(logits_v2).item()

            # Condition 4: Neither present (ablate both)
            features_neither = features.clone()
            features_neither[0, variant1_idx, :] = 0
            features_neither[0, variant2_idx, :] = 0
            mask_neither = mask.clone()
            mask_neither[0, variant1_idx] = False
            mask_neither[0, variant2_idx] = False

            logits_neither, _ = self.model(features_neither, positions, gene_ids, mask_neither)
            pred_neither = torch.sigmoid(logits_neither).item()

        # Compute synergy
        # Synergy = f(v1, v2) - f(v1, ~v2) - f(~v1, v2) + f(~v1, ~v2)
        # Positive synergy = synergistic (combined effect > sum of individual)
        # Negative synergy = antagonistic (combined effect < sum of individual)
        individual_v1 = pred_v1 - pred_neither
        individual_v2 = pred_v2 - pred_neither
        combined = pred_both - pred_neither
        synergy = combined - individual_v1 - individual_v2

        # Classify interaction type
        if abs(synergy) <= 0.01:  # Small epsilon for near-zero
            interaction_type = 'independent'
        elif synergy > 0:
            interaction_type = 'synergistic'
        else:
            interaction_type = 'antagonistic'

        return {
            'pred_both': pred_both,
            'pred_variant1_only': pred_v1,
            'pred_variant2_only': pred_v2,
            'pred_neither': pred_neither,
            'effect_variant1': individual_v1,
            'effect_variant2': individual_v2,
            'effect_combined': combined,
            'synergy': synergy,
            'interaction_type': interaction_type,
            'is_significant': abs(synergy) > 0.05  # Threshold
        }
