"""Tests for TextEncoder module."""

import pytest
import torch

from tests.mocks import create_mock_encoder


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTextEncoder:
    """Essential tests for TextEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create a mock encoder for fast testing."""
        return create_mock_encoder(
            latent_dim=128,
            compression_factor=4,
            hidden_size=256,
            vocab_size=1000,
            num_latent_layers=1,
            freeze_backbone=True,
        )

    def test_forward_shape(self, encoder):
        """Test forward pass output shape with compression."""
        device = get_device()
        encoder = encoder.to(device)

        batch, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)

        output = encoder(input_ids)

        expected_len = seq_len // 4
        assert output.shape == (batch, expected_len, 128)

    def test_gradient_flow(self):
        """Test gradient flow when backbone is unfrozen."""
        device = get_device()
        encoder = create_mock_encoder(
            latent_dim=128,
            compression_factor=4,
            num_latent_layers=1,
            freeze_backbone=False,
        ).to(device)

        input_ids = torch.randint(0, 1000, (2, 64), device=device)
        output = encoder(input_ids)
        output.sum().backward()

        has_grad = any(
            p.grad is not None for p in encoder.parameters() if p.requires_grad
        )
        assert has_grad, "No gradients found in encoder"

    def test_freeze_unfreeze_backbone(self, encoder):
        """Test freezing and unfreezing backbone."""
        # Initially frozen
        assert all(not p.requires_grad for p in encoder.backbone.parameters())

        encoder.unfreeze_backbone()
        assert all(p.requires_grad for p in encoder.backbone.parameters())

        encoder.freeze_backbone()
        assert all(not p.requires_grad for p in encoder.backbone.parameters())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
