"""Utility functions for RQ-VAE."""

import math

import torch


def laplace_smoothing(counts: torch.Tensor, n_categories: int, epsilon: float = 1e-5) -> torch.Tensor:
    """Apply Laplace smoothing to counts for stable averaging.

    Adds a small constant to each count to avoid division by zero when
    computing averages. The smoothing is normalized so the total sum is preserved.

    Formula: smoothed = (counts + epsilon) / (n + n_categories * epsilon) * n
    where n = counts.sum()

    Args:
        counts: Count tensor of shape (n_categories,)
        n_categories: Number of categories (used for normalization)
        epsilon: Small constant for smoothing

    Returns:
        Smoothed counts tensor of shape (n_categories,)
    """
    n = counts.sum()
    return (counts + epsilon) / (n + n_categories * epsilon) * n


def log2_int(x: int) -> int:
    """Compute log base 2 of an integer, ensuring x is a power of 2.

    Args:
        x: Integer that must be a power of 2

    Returns:
        Integer log base 2 of x

    Raises:
        AssertionError: If x is not a power of 2
    """
    result = int(math.log2(x))
    assert 2**result == x, f"Value must be power of 2, got {x}"
    return result
