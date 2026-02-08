"""
Variant encoder for SIEVE.

This module implements the variant encoder that projects variant features
from the input dimension (which varies by annotation level) to a fixed
latent dimension for processing by the attention layers.

Author: Francesco Lescai
"""

import torch
import torch.nn as nn
from torch import Tensor


class VariantEncoder(nn.Module):
    """
    Encode variant features to fixed latent dimension.

    This is a simple MLP that projects variant features from the input
    dimension (which varies by annotation level L0-L4) to a fixed latent
    dimension for downstream processing.

    Architecture:
        Linear(input_dim → hidden_dim)
        → ReLU
        → LayerNorm
        → Dropout
        → Linear(hidden_dim → latent_dim)

    Parameters
    ----------
    input_dim : int
        Input feature dimension (depends on annotation level)
    hidden_dim : int
        Hidden layer dimension (default: 128)
    latent_dim : int
        Output latent dimension (default: 64)
    dropout : float
        Dropout probability (default: 0.1)

    Attributes
    ----------
    encoder : nn.Sequential
        The encoding layers

    Examples
    --------
    >>> encoder = VariantEncoder(input_dim=71, hidden_dim=128, latent_dim=64)
    >>> features = torch.randn(10, 100, 71)  # [batch, variants, features]
    >>> encoded = encoder(features)
    >>> encoded.shape
    torch.Size([10, 100, 64])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode variant features.

        Parameters
        ----------
        x : Tensor
            Variant features, shape (batch, num_variants, input_dim)

        Returns
        -------
        Tensor
            Encoded features, shape (batch, num_variants, latent_dim)
        """
        return self.encoder(x)

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.latent_dim
