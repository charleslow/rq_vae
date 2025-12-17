"""Utility functions for RQ-VAE."""

import math


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
