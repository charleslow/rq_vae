"""Residual Quantizer for RQ-VAE.

Implements residual quantization where each position is represented by D codebook
vectors summed together. This achieves K^D representational capacity with only D*K
parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils import laplace_smoothing


class Quantizer(nn.Module):
    """Single-level Vector Quantizer with K codebook entries.

    Finds the nearest codebook vector for each input vector.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 512,
        ema_decay: float = 0.999,
        epsilon: float = 1e-8,
        threshold_ema_dead_code: int = 1,
    ):
        """
        Args:
            dim: Dimension of input vectors and codebook entries
            codebook_size: Number of entries in the codebook (K)
            ema_decay: Decay rate for EMA codebook updates
            epsilon: Small constant for numerical stability
            threshold_ema_dead_code: Threshold for replacing dead codes
        """
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # Initialize codebook
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim) * 0.02)

        # EMA cluster sizes and sums for codebook update
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", torch.zeros(codebook_size, dim))
        self.register_buffer("initialized", torch.tensor([False]))

    def init_codebook(self, x: torch.Tensor):
        """Initialize codebook from data.

        Args:
            x: Input vectors (n, dim)
        """
        if self.initialized.item():
            return

        n = x.shape[0]
        indices = torch.randperm(n, device=x.device)[: self.codebook_size]
        if indices.shape[0] < self.codebook_size:
            indices = indices.repeat(self.codebook_size // indices.shape[0] + 1)[: self.codebook_size]
        self.codebook.data = x[indices]
        self.initialized.fill_(True)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Find nearest codebook vectors.

        Args:
            x: Input vectors (n, dim)

        Returns:
            quantized: Quantized vectors (n, dim)
            indices: Codebook indices (n,)
        """
        # Use torch.cdist for nearest neighbor search
        distances = torch.cdist(x, self.codebook)  # (n, codebook_size)

        # Find nearest codebook entry
        indices = distances.argmin(dim=-1)  # (n,)

        # Get quantized vectors
        quantized = F.embedding(indices, self.codebook)  # (n, dim)

        return quantized, indices

    def update_codebook_ema(self, x: torch.Tensor, indices: torch.Tensor) -> int:
        """Update codebook using exponential moving average.

        Args:
            x: Input vectors (n, dim)
            indices: Selected codebook indices (n,)

        Returns:
            Number of dead codes that were reinitialized
        """
        if not self.training:
            return 0

        # Count assignments to each code
        one_hot = F.one_hot(indices, self.codebook_size).float()  # (n, codebook_size)
        cluster_size = one_hot.sum(dim=0)  # (codebook_size,)
        embed_sum = one_hot.t() @ x  # (codebook_size, dim)

        # EMA update to codebook
        d = self.ema_decay
        self.ema_cluster_size = d * self.ema_cluster_size + (1 - d) * cluster_size
        self.ema_embed_sum = d * self.ema_embed_sum + (1 - d) * embed_sum

        # Laplace smoothing to avoid division by zero
        cluster_size_smoothed = laplace_smoothing(self.ema_cluster_size, self.codebook_size, self.epsilon)
        self.codebook.data = self.ema_embed_sum / cluster_size_smoothed.unsqueeze(-1)

        # Replace dead codes
        dead_mask = self.ema_cluster_size < self.threshold_ema_dead_code
        n_dead = dead_mask.sum().item()
        if n_dead > 0:
            n_samples = x.shape[0]
            if n_dead <= n_samples:
                random_indices = torch.randperm(n_samples, device=x.device)[:n_dead]
            else:
                random_indices = torch.randint(0, n_samples, (n_dead,), device=x.device)
            self.codebook.data[dead_mask] = x[random_indices]
            self.ema_cluster_size[dead_mask] = 1.0
            self.ema_embed_sum[dead_mask] = x[random_indices]

        return n_dead

    def compute_perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute perplexity (codebook utilization).

        Computes exp(entropy) = exp(-sum(p * log(p)))
        - Bounded between 1 and codebook_size.
        - We want this to be high for good codebook usage.

        Args:
            indices: Codebook indices (n,)

        Returns:
            Perplexity scalar
        """
        one_hot = F.one_hot(indices, self.codebook_size).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity


class ResidualQuantizer(nn.Module):
    """Residual Vector Quantizer with multiple codebook levels.

    For each spatial position, recursively:
    1. Find nearest codebook vector
    2. Subtract it to get residual
    3. Repeat for all levels

    The final quantized representation is the sum of all selected codebook vectors.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 512,
        codebook_levels: int = 8,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.999,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        """
        Args:
            dim: Dimension of input vectors and codebook entries
            codebook_size: Number of entries per codebook (K)
            codebook_levels: Number of residual quantization levels (D)
            commitment_weight: Weight for commitment loss
            ema_decay: Decay rate for EMA codebook updates
            epsilon: Small constant for numerical stability
            threshold_ema_dead_code: Threshold for replacing dead codes
        """
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_levels = codebook_levels
        self.commitment_weight = commitment_weight

        # Create a stack of quantizers
        self.quantizers = nn.ModuleList(
            [
                Quantizer(
                    dim=dim,
                    codebook_size=codebook_size,
                    ema_decay=ema_decay,
                    epsilon=epsilon,
                    threshold_ema_dead_code=threshold_ema_dead_code,
                )
                for _ in range(codebook_levels)
            ]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass with residual quantization.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Dictionary containing:
                - quantized: Quantized output (batch, seq_len, dim)
                - indices: Codebook indices (batch, seq_len, codebook_levels)
                - commitment_loss: Commitment loss scalar
                - perplexities: Per-level codebook perplexity (codebook_levels,)
                - dead_code_replacements: Per-level count of reinitialized codes (codebook_levels,)
        """
        batch, seq_len, dim = x.shape

        # Flatten for processing
        x_flat = rearrange(x, "b s d -> (b s) d")

        # Initialize codebooks on first forward pass
        for quantizer in self.quantizers:
            quantizer.init_codebook(x_flat)

        # Residual quantization
        residual = x_flat.clone()
        quantized_sum = torch.zeros_like(x_flat)
        all_indices = []
        total_commitment_loss = 0.0
        perplexities = []
        dead_code_replacements = []

        for quantizer in self.quantizers:
            # Quantize current residual
            quantized, indices = quantizer.quantize(residual)

            # Update codebook with EMA and track dead code replacements
            n_replaced = quantizer.update_codebook_ema(residual, indices)
            dead_code_replacements.append(n_replaced)

            # Commitment loss: encourage encoder output to stay close to codes
            commitment_loss = F.mse_loss(residual, quantized.detach())
            total_commitment_loss = total_commitment_loss + commitment_loss

            # Compute perplexity (codebook utilization)
            with torch.no_grad():
                perplexity = quantizer.compute_perplexity(indices)
                perplexities.append(perplexity)

            # Accumulate quantized vectors
            quantized_sum = quantized_sum + quantized

            # Update residual for next level
            residual = residual - quantized

            # Store indices
            all_indices.append(indices)

        # Stack indices: (batch * seq, codebook_levels) -> (batch, seq_len, codebook_levels)
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
            "perplexities": torch.stack(perplexities),
            "dead_code_replacements": dead_code_replacements,
        }

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices back to continuous vectors.

        Args:
            indices: Codebook indices (batch, seq_len, codebook_levels)

        Returns:
            Decoded vectors (batch, seq_len, dim)
        """
        batch, seq_len, num_levels = indices.shape
        dtype = self.quantizers[0].codebook.dtype
        device = indices.device

        # Sum up codebook vectors across depth levels
        quantized_sum = torch.zeros(batch, seq_len, self.dim, device=device, dtype=dtype)

        for level, quantizer in enumerate(self.quantizers[:num_levels]):
            level_indices = indices[:, :, level]  # (batch, seq_len)
            quantized = F.embedding(level_indices, quantizer.codebook)  # (batch, seq_len, dim)
            quantized_sum = quantized_sum + quantized

        return quantized_sum

    def get_codebook_usage(self) -> torch.Tensor:
        """Get usage statistics for each codebook level.

        Returns:
            Tensor of shape (codebook_levels,) with fraction of codes used
        """
        usage = torch.stack([(q.ema_cluster_size > q.threshold_ema_dead_code).float().mean() for q in self.quantizers])
        return usage
