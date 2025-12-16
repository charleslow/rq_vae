"""Residual Quantizer for RQ-VAE.

Implements residual quantization where each position is represented by D codebook
vectors summed together. This achieves K^D representational capacity with only D*K
parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResidualQuantizer(nn.Module):
    """Residual Vector Quantizer with D depth levels and K codebook entries.

    For each spatial position, recursively:
    1. Find nearest codebook vector
    2. Subtract it to get residual
    3. Repeat for D levels

    The final quantized representation is the sum of all selected codebook vectors.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 512,
        num_quantizers: int = 8,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        """
        Args:
            dim: Dimension of input vectors and codebook entries
            codebook_size: Number of entries per codebook (K)
            num_quantizers: Number of residual quantization levels (D)
            commitment_weight: Weight for commitment loss
            ema_decay: Decay rate for EMA codebook updates
            epsilon: Small constant for numerical stability
            threshold_ema_dead_code: Threshold for replacing dead codes
        """
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # Initialize codebooks for each depth level
        # Shape: (num_quantizers, codebook_size, dim)
        self.codebooks = nn.Parameter(
            torch.randn(num_quantizers, codebook_size, dim) * 0.02
        )

        # EMA cluster sizes and sums for codebook update
        self.register_buffer(
            "ema_cluster_size",
            torch.zeros(num_quantizers, codebook_size)
        )
        self.register_buffer(
            "ema_embed_sum",
            torch.zeros(num_quantizers, codebook_size, dim)
        )
        self.register_buffer("initialized", torch.tensor([False]))

    def _init_codebook(self, x: torch.Tensor):
        """Initialize codebook from first batch of data."""
        if self.initialized.item():
            return

        # x: (batch * seq, dim)
        # Sample random vectors from input to initialize codebook
        n = x.shape[0]
        for d in range(self.num_quantizers):
            indices = torch.randperm(n, device=x.device)[:self.codebook_size]
            if indices.shape[0] < self.codebook_size:
                # If not enough samples, repeat
                indices = indices.repeat(self.codebook_size // indices.shape[0] + 1)[:self.codebook_size]
            self.codebooks.data[d] = x[indices]

        self.initialized.fill_(True)

    def _quantize_level(
        self,
        residual: torch.Tensor,
        level: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize at a single depth level.

        Args:
            residual: Input residual vectors (batch * seq, dim)
            level: Current depth level

        Returns:
            quantized: Quantized vectors
            indices: Codebook indices
            distances: Squared distances to nearest codes
        """
        codebook = self.codebooks[level]  # (codebook_size, dim)

        # Compute squared distances: ||x - c||^2 = ||x||^2 - 2*x@c^T + ||c||^2
        distances = (
            residual.pow(2).sum(dim=-1, keepdim=True)
            - 2 * residual @ codebook.t()
            + codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )

        # Find nearest codebook entry
        indices = distances.argmin(dim=-1)  # (batch * seq,)

        # Get quantized vectors
        quantized = F.embedding(indices, codebook)  # (batch * seq, dim)

        return quantized, indices, distances

    def _update_codebook_ema(
        self,
        residual: torch.Tensor,
        indices: torch.Tensor,
        level: int
    ):
        """Update codebook using exponential moving average.

        Args:
            residual: Input residual vectors (batch * seq, dim)
            indices: Selected codebook indices (batch * seq,)
            level: Current depth level
        """
        if not self.training:
            return

        # Count assignments to each code
        one_hot = F.one_hot(indices, self.codebook_size).float()  # (batch * seq, codebook_size)
        cluster_size = one_hot.sum(dim=0)  # (codebook_size,)
        embed_sum = one_hot.t() @ residual  # (codebook_size, dim)

        # EMA update
        self.ema_cluster_size[level] = (
            self.ema_decay * self.ema_cluster_size[level]
            + (1 - self.ema_decay) * cluster_size
        )
        self.ema_embed_sum[level] = (
            self.ema_decay * self.ema_embed_sum[level]
            + (1 - self.ema_decay) * embed_sum
        )

        # Laplace smoothing to avoid division by zero
        n = self.ema_cluster_size[level].sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size[level] + self.epsilon)
            / (n + self.codebook_size * self.epsilon) * n
        )

        # Update codebook
        self.codebooks.data[level] = (
            self.ema_embed_sum[level] / cluster_size_smoothed.unsqueeze(-1)
        )

        # Replace dead codes
        dead_mask = self.ema_cluster_size[level] < self.threshold_ema_dead_code
        if dead_mask.any():
            # Replace with random vectors from input
            n_dead = dead_mask.sum().item()
            random_indices = torch.randperm(residual.shape[0], device=residual.device)[:n_dead]
            self.codebooks.data[level, dead_mask] = residual[random_indices]
            self.ema_cluster_size[level, dead_mask] = 1.0
            self.ema_embed_sum[level, dead_mask] = residual[random_indices]

    def forward(
        self,
        x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Forward pass with residual quantization.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Dictionary containing:
                - quantized: Quantized output (batch, seq_len, dim)
                - indices: Codebook indices (batch, seq_len, num_quantizers)
                - commitment_loss: Commitment loss scalar
                - codebook_loss: Codebook loss scalar (if not using EMA)
                - perplexity: Average codebook perplexity across levels
        """
        batch, seq_len, dim = x.shape

        # Flatten for processing
        x_flat = rearrange(x, "b s d -> (b s) d")

        # Initialize codebook on first forward pass
        self._init_codebook(x_flat)

        # Residual quantization
        residual = x_flat.clone()
        quantized_sum = torch.zeros_like(x_flat)
        all_indices = []
        total_commitment_loss = 0.0
        total_perplexity = 0.0

        for level in range(self.num_quantizers):
            # Quantize current residual
            quantized, indices, _ = self._quantize_level(residual, level)

            # Update codebook with EMA
            self._update_codebook_ema(residual, indices, level)

            # Commitment loss: encourage encoder output to stay close to codes
            commitment_loss = F.mse_loss(residual, quantized.detach())
            total_commitment_loss = total_commitment_loss + commitment_loss

            # Compute perplexity (codebook utilization)
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.codebook_size).float()
                avg_probs = one_hot.mean(dim=0)
                perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
                total_perplexity = total_perplexity + perplexity

            # Accumulate quantized vectors
            quantized_sum = quantized_sum + quantized

            # Update residual for next level
            residual = residual - quantized

            # Store indices
            all_indices.append(indices)

        # Stack indices: (batch * seq, num_quantizers) -> (batch, seq_len, num_quantizers)
        indices = torch.stack(all_indices, dim=-1)
        indices = rearrange(indices, "(b s) d -> b s d", b=batch, s=seq_len)

        # Reshape quantized output
        quantized_out = rearrange(quantized_sum, "(b s) d -> b s d", b=batch, s=seq_len)

        # Straight-through estimator: gradients flow through quantized
        quantized_out = x + (quantized_out - x).detach()

        return {
            "quantized": quantized_out,
            "indices": indices,
            "commitment_loss": total_commitment_loss * self.commitment_weight,
            "perplexity": total_perplexity / self.num_quantizers,
        }

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices back to continuous vectors.

        Args:
            indices: Codebook indices (batch, seq_len, num_quantizers)

        Returns:
            Decoded vectors (batch, seq_len, dim)
        """
        batch, seq_len, num_q = indices.shape

        # Sum up codebook vectors across depth levels
        quantized_sum = torch.zeros(
            batch, seq_len, self.dim,
            device=indices.device,
            dtype=self.codebooks.dtype
        )

        for level in range(num_q):
            level_indices = indices[:, :, level]  # (batch, seq_len)
            codebook = self.codebooks[level]  # (codebook_size, dim)
            quantized = F.embedding(level_indices, codebook)  # (batch, seq_len, dim)
            quantized_sum = quantized_sum + quantized

        return quantized_sum

    def get_codebook_usage(self) -> torch.Tensor:
        """Get usage statistics for each codebook level.

        Returns:
            Tensor of shape (num_quantizers,) with fraction of codes used
        """
        usage = (self.ema_cluster_size > self.threshold_ema_dead_code).float().mean(dim=-1)
        return usage
