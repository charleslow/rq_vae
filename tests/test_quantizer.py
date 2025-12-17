"""Tests for ResidualQuantizer module."""

import pytest
import torch
import torch.nn as nn

from src.model.quantizer import ResidualQuantizer


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestResidualQuantizer:
    """Test suite for ResidualQuantizer."""

    @pytest.fixture
    def quantizer(self):
        """Create a quantizer instance for testing."""
        return ResidualQuantizer(
            dim=64,
            codebook_size=32,
            num_quantizers=4,
            commitment_weight=0.25,
        )

    def test_initialization(self, quantizer):
        """Test quantizer initialization."""
        assert quantizer.dim == 64
        assert quantizer.codebook_size == 32
        assert quantizer.num_quantizers == 4
        assert quantizer.codebooks.shape == (4, 32, 64)
        assert not quantizer.initialized.item()

    def test_forward_shape(self, quantizer):
        """Test forward pass output shapes."""
        device = get_device()
        quantizer = quantizer.to(device)

        batch, seq_len, dim = 2, 16, 64
        x = torch.randn(batch, seq_len, dim, device=device)

        output = quantizer(x)

        assert "quantized" in output
        assert "indices" in output
        assert "commitment_loss" in output
        assert "perplexity" in output

        assert output["quantized"].shape == (batch, seq_len, dim)
        assert output["indices"].shape == (batch, seq_len, 4)
        assert output["commitment_loss"].dim() == 0  # scalar
        assert output["perplexity"].dim() == 0  # scalar

    def test_codebook_initialization(self, quantizer):
        """Test codebook initialization from first batch."""
        device = get_device()
        quantizer = quantizer.to(device)

        x = torch.randn(2, 16, 64, device=device)

        assert not quantizer.initialized.item()
        quantizer(x)
        assert quantizer.initialized.item()

    def test_decode_indices(self, quantizer):
        """Test decoding indices back to continuous vectors."""
        device = get_device()
        quantizer = quantizer.to(device)

        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, 64, device=device)

        output = quantizer(x)
        indices = output["indices"]

        decoded = quantizer.decode_indices(indices)

        assert decoded.shape == (batch, seq_len, 64)
        # Decoded should be close to quantized (modulo straight-through estimator)
        assert torch.allclose(decoded, output["quantized"], atol=1e-5)

    def test_straight_through_estimator(self, quantizer):
        """Test that gradients flow through quantized values."""
        device = get_device()
        quantizer = quantizer.to(device)

        x = torch.randn(2, 16, 64, device=device, requires_grad=True)

        output = quantizer(x)
        loss = output["quantized"].sum()
        loss.backward()

        # Gradients should flow to input
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_commitment_loss(self, quantizer):
        """Test commitment loss computation."""
        device = get_device()
        quantizer = quantizer.to(device)

        x = torch.randn(2, 16, 64, device=device)

        output = quantizer(x)

        # Commitment loss should be positive
        assert output["commitment_loss"] > 0
        # Should be scaled by commitment_weight
        assert output["commitment_loss"] < 100  # reasonable magnitude

    def test_ema_update_training_mode(self, quantizer):
        """Test that EMA updates only happen in training mode."""
        device = get_device()
        quantizer = quantizer.to(device)

        x = torch.randn(2, 16, 64, device=device)

        # Training mode - EMA should update
        quantizer.train()
        quantizer(x)  # Initialize
        ema_before = quantizer.ema_cluster_size.clone()

        quantizer(x)  # Second pass should update EMA
        ema_after = quantizer.ema_cluster_size

        # In training, EMA should change
        assert not torch.allclose(ema_before, ema_after)

        # Eval mode - EMA should not update
        quantizer.eval()
        ema_before = quantizer.ema_cluster_size.clone()
        quantizer(x)
        ema_after = quantizer.ema_cluster_size

        assert torch.allclose(ema_before, ema_after)

    def test_codebook_usage(self, quantizer):
        """Test codebook usage statistics."""
        device = get_device()
        quantizer = quantizer.to(device)

        x = torch.randn(4, 32, 64, device=device)
        quantizer(x)

        usage = quantizer.get_codebook_usage()

        assert usage.shape == (4,)  # One per quantizer
        assert (usage >= 0).all() and (usage <= 1).all()

    def test_different_batch_sizes(self, quantizer):
        """Test that quantizer works with different batch sizes."""
        device = get_device()
        quantizer = quantizer.to(device)

        # Initialize with first batch
        x1 = torch.randn(2, 16, 64, device=device)
        output1 = quantizer(x1)

        # Try different batch size
        x2 = torch.randn(4, 16, 64, device=device)
        output2 = quantizer(x2)

        assert output1["quantized"].shape == (2, 16, 64)
        assert output2["quantized"].shape == (4, 16, 64)

    def test_different_sequence_lengths(self, quantizer):
        """Test that quantizer works with different sequence lengths."""
        device = get_device()
        quantizer = quantizer.to(device)

        # Initialize with first sequence
        x1 = torch.randn(2, 16, 64, device=device)
        output1 = quantizer(x1)

        # Try different sequence length
        x2 = torch.randn(2, 32, 64, device=device)
        output2 = quantizer(x2)

        assert output1["quantized"].shape == (2, 16, 64)
        assert output2["quantized"].shape == (2, 32, 64)

    @pytest.mark.parametrize("num_quantizers", [1, 2, 4, 8])
    def test_different_quantizer_depths(self, num_quantizers):
        """Test different numbers of quantization levels."""
        device = get_device()

        quantizer = ResidualQuantizer(
            dim=64,
            codebook_size=32,
            num_quantizers=num_quantizers,
        ).to(device)

        x = torch.randn(2, 16, 64, device=device)
        output = quantizer(x)

        assert output["indices"].shape == (2, 16, num_quantizers)

    @pytest.mark.parametrize("codebook_size", [16, 32, 64, 128])
    def test_different_codebook_sizes(self, codebook_size):
        """Test different codebook sizes."""
        device = get_device()

        quantizer = ResidualQuantizer(
            dim=64,
            codebook_size=codebook_size,
            num_quantizers=4,
        ).to(device)

        x = torch.randn(2, 16, 64, device=device)
        output = quantizer(x)

        # Indices should be in valid range
        assert (output["indices"] >= 0).all()
        assert (output["indices"] < codebook_size).all()

    def test_cpu_cuda_consistency(self, quantizer):
        """Test that results are consistent between CPU and CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Set seed for reproducibility
        torch.manual_seed(42)
        x_cpu = torch.randn(2, 16, 64)

        # CPU version
        quantizer_cpu = quantizer
        torch.manual_seed(42)
        quantizer_cpu.codebooks.data = torch.randn(4, 32, 64) * 0.02
        output_cpu = quantizer_cpu(x_cpu)

        # CUDA version
        quantizer_cuda = ResidualQuantizer(
            dim=64,
            codebook_size=32,
            num_quantizers=4,
            commitment_weight=0.25,
        ).cuda()
        quantizer_cuda.initialized.fill_(False)
        torch.manual_seed(42)
        quantizer_cuda.codebooks.data = torch.randn(4, 32, 64, device="cuda") * 0.02
        output_cuda = quantizer_cuda(x_cpu.cuda())

        # Indices should match
        assert torch.equal(output_cpu["indices"], output_cuda["indices"].cpu())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
