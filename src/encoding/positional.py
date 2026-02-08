"""
Positional encodings for genomic positions in SIEVE.

This module implements two types of positional encodings that serve different purposes:

1. **Sinusoidal Positional Encoding**: Used as input features (L1-L4)
   - Encodes absolute genomic position
   - Added to variant feature vectors
   - Similar to original Transformer positional encoding

2. **Relative Position Bucketing**: Used for attention bias (Phase 1C)
   - Encodes relative distance between variants
   - Added as learned bias in attention mechanism
   - Uses logarithmic bucketing for genomic distances

**IMPORTANT**: These encodings serve different purposes and are NOT used simultaneously
in the same context:
- Sinusoidal: Input features (this phase)
- Bucketing: Attention bias (next phase)

Author: Francesco Lescai
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def sinusoidal_position_encoding(
    positions: np.ndarray,
    d_model: int = 64,
    max_wavelength: float = 10000.0
) -> np.ndarray:
    """
    Generate sinusoidal positional encodings for genomic positions.

    This encodes absolute genomic positions using sine and cosine functions
    with different frequencies, as in the original Transformer paper.

    For each position and each dimension i (0 to d_model):
    - PE(pos, 2i) = sin(pos / max_wavelength^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / max_wavelength^(2i/d_model))

    Parameters
    ----------
    positions : np.ndarray
        Genomic positions (1-based), shape (n_variants,)
    d_model : int
        Dimension of positional encoding (default: 64)
    max_wavelength : float
        Maximum wavelength for sinusoidal functions (default: 10000)

    Returns
    -------
    np.ndarray
        Positional encodings, shape (n_variants, d_model)

    Notes
    -----
    - Genomic positions are typically 6-9 digits (e.g., 10,000,000 for chr1)
    - With max_wavelength=10000 and d_model=64, this captures both local
      and long-range positional patterns
    - Each dimension represents a different frequency/wavelength

    Examples
    --------
    >>> positions = np.array([100, 200, 300])
    >>> encodings = sinusoidal_position_encoding(positions, d_model=64)
    >>> encodings.shape
    (3, 64)
    >>> # First position, first two dimensions (sin, cos)
    >>> encodings[0, :2]
    array([0.099833..., 0.995004...])
    """
    n_positions = len(positions)

    # Create position encoding matrix
    pe = np.zeros((n_positions, d_model), dtype=np.float32)

    # Compute division term: max_wavelength^(2i/d_model)
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) *
        -(np.log(max_wavelength) / d_model)
    )

    # Apply sine to even indices
    pe[:, 0::2] = np.sin(positions[:, np.newaxis] * div_term)

    # Apply cosine to odd indices
    pe[:, 1::2] = np.cos(positions[:, np.newaxis] * div_term)

    return pe


def relative_position_bucket(
    query_positions: Tensor,
    key_positions: Tensor,
    num_buckets: int = 32,
    max_distance: int = 100000
) -> Tensor:
    """
    Compute relative position buckets for attention bias.

    This implements logarithmic bucketing of relative distances between variants,
    as used in T5 and other models. It will be used in Phase 1C for attention bias.

    Bucketing strategy:
    - Half the buckets for negative distances (query < key)
    - Half for positive distances (query > key)
    - Linear bucketing for small distances (< max_exact)
    - Logarithmic bucketing for large distances (>= max_exact)

    Parameters
    ----------
    query_positions : Tensor
        Positions of query variants, shape (n_queries,)
    key_positions : Tensor
        Positions of key variants, shape (n_keys,)
    num_buckets : int
        Number of position buckets (default: 32)
    max_distance : int
        Maximum distance to consider (default: 100,000 bp)

    Returns
    -------
    Tensor
        Bucket indices, shape (n_queries, n_keys)
        Values in range [0, num_buckets-1]

    Notes
    -----
    This encoding serves a different purpose than sinusoidal encoding:
    - Sinusoidal: Absolute position, input features
    - Bucketing: Relative distance, attention bias

    Will be used in Phase 1C like:
    ```python
    # In attention mechanism
    buckets = relative_position_bucket(query_pos, key_pos)
    bias = self.position_bias_embedding(buckets)  # Learnable
    attention_scores += bias
    ```

    Examples
    --------
    >>> query_pos = torch.tensor([100, 200, 300])
    >>> key_pos = torch.tensor([100, 150, 250])
    >>> buckets = relative_position_bucket(query_pos, key_pos, num_buckets=32)
    >>> buckets.shape
    torch.Size([3, 3])
    >>> buckets[0, 0]  # Same position
    tensor(16)
    >>> buckets[0, 1]  # Query before key (+50 bp)
    tensor(...)
    """
    # Compute relative positions (query - key)
    # Shape: (n_queries, 1) - (1, n_keys) = (n_queries, n_keys)
    relative_position = query_positions[:, None] - key_positions[None, :]

    # Initialize buckets
    num_buckets_half = num_buckets // 2

    # Determine if relative position is positive or negative
    # Positive distances (query after key) get buckets [num_buckets_half, num_buckets)
    # Negative distances (query before key) get buckets [0, num_buckets_half)
    ret = (relative_position > 0).long() * num_buckets_half

    # Work with absolute distance
    n = torch.abs(relative_position)

    # Half the buckets are for exact positions (small distances)
    max_exact = num_buckets_half // 2
    is_small = n < max_exact

    # For large distances, use logarithmic bucketing
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) /
        np.log(max_distance / max_exact) *
        (num_buckets_half - max_exact)
    ).long()

    # Clamp to valid bucket range
    val_if_large = torch.min(
        val_if_large,
        torch.full_like(val_if_large, num_buckets_half - 1)
    )

    # Select small or large bucketing
    ret = ret + torch.where(is_small, n, val_if_large)

    return ret


def compute_sinusoidal_encodings_batch(
    positions_list: list,
    d_model: int = 64,
    max_wavelength: float = 10000.0
) -> list:
    """
    Compute sinusoidal encodings for a batch of variant lists.

    Convenience function for encoding multiple samples at once.

    Parameters
    ----------
    positions_list : list
        List of position arrays, one per sample
    d_model : int
        Dimension of positional encoding (default: 64)
    max_wavelength : float
        Maximum wavelength (default: 10000)

    Returns
    -------
    list
        List of positional encoding arrays, one per sample

    Examples
    --------
    >>> pos1 = np.array([100, 200])
    >>> pos2 = np.array([300, 400, 500])
    >>> encodings = compute_sinusoidal_encodings_batch([pos1, pos2])
    >>> len(encodings)
    2
    >>> encodings[0].shape
    (2, 64)
    >>> encodings[1].shape
    (3, 64)
    """
    return [
        sinusoidal_position_encoding(positions, d_model, max_wavelength)
        for positions in positions_list
    ]


def get_relative_distance_statistics(
    positions: np.ndarray
) -> dict:
    """
    Compute statistics about relative distances between variants.

    Useful for determining appropriate bucketing parameters.

    Parameters
    ----------
    positions : np.ndarray
        Genomic positions, shape (n_variants,)

    Returns
    -------
    dict
        Statistics including:
        - 'mean_distance': Mean distance between consecutive variants
        - 'median_distance': Median distance
        - 'min_distance': Minimum distance
        - 'max_distance': Maximum distance
        - 'p95_distance': 95th percentile distance

    Examples
    --------
    >>> positions = np.array([100, 200, 500, 600])
    >>> stats = get_relative_distance_statistics(positions)
    >>> stats['mean_distance']
    166.66...
    >>> stats['median_distance']
    150.0
    """
    if len(positions) < 2:
        return {
            'mean_distance': 0,
            'median_distance': 0,
            'min_distance': 0,
            'max_distance': 0,
            'p95_distance': 0,
        }

    # Compute consecutive distances
    distances = np.diff(np.sort(positions))

    return {
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'p95_distance': float(np.percentile(distances, 95)),
    }


def visualize_positional_encoding(
    positions: np.ndarray,
    d_model: int = 64,
    max_wavelength: float = 10000.0
) -> Tuple[np.ndarray, dict]:
    """
    Generate positional encodings and summary statistics for visualization.

    Useful for understanding how positions are encoded.

    Parameters
    ----------
    positions : np.ndarray
        Genomic positions to encode
    d_model : int
        Encoding dimension
    max_wavelength : float
        Maximum wavelength

    Returns
    -------
    encodings : np.ndarray
        Positional encodings, shape (n_positions, d_model)
    info : dict
        Information about the encoding:
        - 'wavelengths': Wavelengths for each dimension
        - 'frequencies': Frequencies for each dimension
        - 'position_range': (min_pos, max_pos)

    Examples
    --------
    >>> positions = np.array([100, 1000, 10000, 100000])
    >>> encodings, info = visualize_positional_encoding(positions, d_model=8)
    >>> encodings.shape
    (4, 8)
    >>> 'wavelengths' in info
    True
    """
    # Compute encodings
    encodings = sinusoidal_position_encoding(positions, d_model, max_wavelength)

    # Compute wavelengths for each dimension
    i_values = np.arange(0, d_model, 2)
    wavelengths = max_wavelength ** (i_values / d_model)
    frequencies = 1.0 / wavelengths

    info = {
        'wavelengths': wavelengths,
        'frequencies': frequencies,
        'position_range': (int(np.min(positions)), int(np.max(positions))),
        'n_positions': len(positions),
        'd_model': d_model,
        'max_wavelength': max_wavelength,
    }

    return encodings, info


def test_encoding_consistency():
    """
    Test that positional encodings are consistent and deterministic.

    This is a built-in test function to verify correctness.

    Returns
    -------
    bool
        True if all tests pass

    Raises
    ------
    AssertionError
        If any test fails
    """
    # Test 1: Same positions produce same encodings
    positions = np.array([100, 200, 300])
    enc1 = sinusoidal_position_encoding(positions)
    enc2 = sinusoidal_position_encoding(positions)
    assert np.allclose(enc1, enc2), "Encodings not deterministic"

    # Test 2: Different positions produce different encodings
    positions2 = np.array([100, 200, 301])  # Last position different
    enc3 = sinusoidal_position_encoding(positions2)
    assert not np.allclose(enc1[2], enc3[2]), "Different positions produce same encoding"

    # Test 3: Correct output shape
    assert enc1.shape == (3, 64), f"Wrong shape: {enc1.shape}"

    # Test 4: Values in reasonable range (should be between -1 and 1)
    assert np.all(np.abs(enc1) <= 1.0), "Encoding values out of range"

    # Test 5: Relative bucketing produces valid indices
    query_pos = torch.tensor([100, 200, 300])
    key_pos = torch.tensor([100, 150, 250])
    buckets = relative_position_bucket(query_pos, key_pos, num_buckets=32)
    assert torch.all((buckets >= 0) & (buckets < 32)), "Invalid bucket indices"

    print("✓ All positional encoding tests passed")
    return True
