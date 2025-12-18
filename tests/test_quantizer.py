"""Tests for ResidualQuantizer module."""

import pytest
import torch

from src.model.quantizer import ResidualQuantizer


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestResidualQuantizer:
    """Essential tests for ResidualQuantizer."""

    @pytest.fixture
    def quantizer(self):
        """Create a quantizer instance for testing."""
        return ResidualQuantizer(
            dim=64,
            codebook_size=32,
            codebook_levels=4,
            commitment_weight=0.25,
        )

    def test_forward_shape(self, quantizer):
        """Test forward pass output shapes."""
        device = get_device()
        quantizer = quantizer.to(device)

        batch, seq_len, dim = 2, 16, 64
        x = torch.randn(batch, seq_len, dim, device=device)

        output = quantizer(x)

        assert output["quantized"].shape == (batch, seq_len, dim)
        assert output["indices"].shape == (batch, seq_len, 4)
        assert output["commitment_loss"].dim() == 0
        assert output["perplexities"].shape == (4,)  # One perplexity per codebook level

    def test_decode_indices(self, quantizer):
        """Test decoding indices back to continuous vectors."""
        device = get_device()
        quantizer = quantizer.to(device)

        x = torch.randn(2, 16, 64, device=device)
        output = quantizer(x)

        decoded = quantizer.decode_indices(output["indices"])
        assert decoded.shape == (2, 16, 64)

    def test_straight_through_estimator(self, quantizer):
        """Test that gradients flow through quantized values."""
        device = get_device()
        quantizer = quantizer.to(device)

        x = torch.randn(2, 16, 64, device=device, requires_grad=True)
        output = quantizer(x)
        output["quantized"].sum().backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_ema_update_training_vs_eval(self, quantizer):
        """Test that EMA updates only happen in training mode."""
        device = get_device()
        quantizer = quantizer.to(device)

        x = torch.randn(8, 32, 64, device=device)

        # Training mode - EMA should update
        quantizer.train()
        quantizer(x)

        # Eval mode - EMA should not update
        quantizer.eval()
        ema_before_eval = torch.stack([q.ema_cluster_size.clone() for q in quantizer.quantizers])
        quantizer(x)
        ema_after_eval = torch.stack([q.ema_cluster_size for q in quantizer.quantizers])

        assert torch.allclose(ema_before_eval, ema_after_eval)

    @pytest.mark.parametrize("codebook_size", [16, 64, 128])
    def test_indices_in_valid_range(self, codebook_size):
        """Test that indices are within valid codebook range."""
        device = get_device()
        quantizer = ResidualQuantizer(
            dim=64, codebook_size=codebook_size, codebook_levels=4
        ).to(device)

        x = torch.randn(2, 16, 64, device=device)
        output = quantizer(x)

        assert (output["indices"] >= 0).all()
        assert (output["indices"] < codebook_size).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
