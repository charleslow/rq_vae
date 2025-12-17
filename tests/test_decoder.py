"""Tests for TextDecoder module."""

import pytest
import torch

from tests.mocks import create_mock_decoder


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTextDecoder:
    """Essential tests for TextDecoder."""

    @pytest.fixture
    def decoder(self):
        """Create a mock decoder for fast testing."""
        return create_mock_decoder(
            latent_dim=128,
            compression_factor=4,
            hidden_size=256,
            vocab_size=1000,
            num_latent_layers=1,
            freeze_backbone=True,
        )

    def test_forward_shape(self, decoder):
        """Test forward pass output shape with upsampling."""
        device = get_device()
        decoder = decoder.to(device)

        batch, compressed_len, latent_dim = 2, 16, 128
        latent = torch.randn(batch, compressed_len, latent_dim, device=device)

        output = decoder(latent)

        expected_len = compressed_len * 4
        assert output.shape == (batch, expected_len, 1000)

    def test_with_target_len(self, decoder):
        """Test decoder with explicit target length."""
        device = get_device()
        decoder = decoder.to(device)

        latent = torch.randn(2, 16, 128, device=device)
        output = decoder(latent, target_len=50)

        assert output.shape[1] == 50

    def test_gradient_flow(self):
        """Test gradient flow through decoder."""
        device = get_device()
        decoder = create_mock_decoder(
            latent_dim=128,
            compression_factor=4,
            num_latent_layers=1,
            freeze_backbone=False,
        ).to(device)

        latent = torch.randn(2, 16, 128, device=device, requires_grad=True)
        output = decoder(latent)
        output.sum().backward()

        assert latent.grad is not None
        assert not torch.allclose(latent.grad, torch.zeros_like(latent.grad))

    def test_freeze_unfreeze_backbone(self, decoder):
        """Test freezing and unfreezing backbone."""
        assert all(not p.requires_grad for p in decoder.backbone.parameters())

        decoder.unfreeze_backbone()
        assert all(p.requires_grad for p in decoder.backbone.parameters())

        decoder.freeze_backbone()
        assert all(not p.requires_grad for p in decoder.backbone.parameters())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
