"""Tests for RQVAE model."""

import pytest
import torch

from tests.mocks import create_mock_rqvae


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestRQVAE:
    """Essential tests for RQVAE model."""

    @pytest.fixture
    def model(self):
        """Create a mock RQVAE model for fast testing."""
        return create_mock_rqvae(
            latent_dim=128,
            compression_factor=4,
            codebook_size=32,
            codebook_levels=4,
            hidden_size=256,
            vocab_size=1000,
            freeze_backbone=True,
        )

    def test_forward_shape(self, model):
        """Test forward pass output shapes."""
        device = get_device()
        model = model.to(device)

        batch, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)

        output = model(input_ids)

        assert output["logits"].shape == (batch, seq_len, 1000)
        assert output["indices"].shape == (batch, seq_len // 4, 4)
        assert output["reconstruction_loss"] > 0
        assert output["total_loss"] >= output["reconstruction_loss"]

    def test_encode_decode_cycle(self, model):
        """Test full encode-decode cycle."""
        device = get_device()
        model = model.to(device)

        batch, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)

        enc_output = model.encode(input_ids)
        logits = model.decode(enc_output["quantized"], target_len=seq_len)

        assert logits.shape == (batch, seq_len, model.decoder.vocab_size)
        assert enc_output["indices"].shape == (batch, seq_len // 4, 4)

    def test_gradient_flow(self, model):
        """Test gradient flow through the model."""
        device = get_device()
        model = model.to(device)

        input_ids = torch.randint(0, 1000, (2, 64), device=device)
        output = model(input_ids)
        output["total_loss"].backward()

        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "No gradients found in model"

    def test_decode_indices(self, model):
        """Test decoding from discrete indices."""
        device = get_device()
        model = model.to(device)

        indices = torch.randint(0, 32, (2, 16, 4), device=device)
        decoded = model.decode_indices(indices)

        assert decoded.shape == (2, 16, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
