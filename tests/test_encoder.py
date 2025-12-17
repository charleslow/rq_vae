"""Tests for TextEncoder module."""

import pytest
import torch

from src.model.encoder import TextEncoder


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTextEncoder:
    """Test suite for TextEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create a minimal encoder for testing.

        Note: Uses a small model to reduce memory and runtime.
        """
        # Use smallest Qwen model for testing
        return TextEncoder(
            model_name="Qwen/Qwen2.5-0.5B",
            latent_dim=128,
            compression_factor=4,
            freeze_backbone=True,
            num_latent_layers=1,
        )

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.latent_dim == 128
        assert encoder.compression_factor == 4
        assert encoder.num_latent_layers == 1

        # Check backbone is frozen
        for param in encoder.backbone.parameters():
            assert not param.requires_grad

    def test_forward_shape(self, encoder):
        """Test forward pass output shape."""
        device = get_device()
        encoder = encoder.to(device)

        batch, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)
        attention_mask = torch.ones(batch, seq_len, device=device)

        output = encoder(input_ids, attention_mask)

        # Should compress by factor of 4
        expected_len = seq_len // 4
        assert output.shape == (batch, expected_len, 128)

    def test_compression_factor(self):
        """Test different compression factors."""
        device = get_device()

        for compression_factor in [2, 4, 8]:
            encoder = TextEncoder(
                model_name="Qwen/Qwen2.5-0.5B",
                latent_dim=128,
                compression_factor=compression_factor,
                freeze_backbone=True,
                num_latent_layers=0,
            ).to(device)

            batch, seq_len = 2, 64
            input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)

            output = encoder(input_ids)

            expected_len = seq_len // compression_factor
            assert output.shape == (batch, expected_len, 128)

    def test_without_attention_mask(self, encoder):
        """Test forward pass without attention mask."""
        device = get_device()
        encoder = encoder.to(device)

        batch, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)

        # Should work without attention mask
        output = encoder(input_ids, attention_mask=None)

        expected_len = seq_len // 4
        assert output.shape == (batch, expected_len, 128)

    def test_gradient_flow_unfrozen(self):
        """Test gradient flow when backbone is unfrozen."""
        device = get_device()

        encoder = TextEncoder(
            model_name="Qwen/Qwen2.5-0.5B",
            latent_dim=128,
            compression_factor=4,
            freeze_backbone=False,  # Unfreeze for this test
            num_latent_layers=1,
        ).to(device)

        input_ids = torch.randint(0, 1000, (2, 64), device=device)

        output = encoder(input_ids)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for unfrozen parameters
        has_grad = False
        for name, param in encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break

        assert has_grad, "No gradients found in encoder"

    def test_freeze_unfreeze_backbone(self, encoder):
        """Test freezing and unfreezing backbone."""
        # Initially frozen
        for param in encoder.backbone.parameters():
            assert not param.requires_grad

        # Unfreeze
        encoder.unfreeze_backbone()
        for param in encoder.backbone.parameters():
            assert param.requires_grad

        # Freeze again
        encoder.freeze_backbone()
        for param in encoder.backbone.parameters():
            assert not param.requires_grad

    def test_latent_refinement_layers(self):
        """Test encoder with different numbers of latent refinement layers."""
        device = get_device()

        for num_layers in [0, 1, 2]:
            encoder = TextEncoder(
                model_name="Qwen/Qwen2.5-0.5B",
                latent_dim=128,
                compression_factor=4,
                freeze_backbone=True,
                num_latent_layers=num_layers,
            ).to(device)

            assert len(encoder.latent_refinement) == num_layers

            input_ids = torch.randint(0, 1000, (2, 64), device=device)
            output = encoder(input_ids)

            assert output.shape == (2, 16, 128)

    def test_different_sequence_lengths(self, encoder):
        """Test with different input sequence lengths."""
        device = get_device()
        encoder = encoder.to(device)

        for seq_len in [32, 64, 128]:
            input_ids = torch.randint(0, 1000, (2, seq_len), device=device)
            output = encoder(input_ids)

            expected_len = seq_len // 4
            assert output.shape == (2, expected_len, 128)

    def test_different_latent_dims(self):
        """Test encoder with different latent dimensions."""
        device = get_device()

        for latent_dim in [64, 128, 256]:
            encoder = TextEncoder(
                model_name="Qwen/Qwen2.5-0.5B",
                latent_dim=latent_dim,
                compression_factor=4,
                freeze_backbone=True,
                num_latent_layers=1,
            ).to(device)

            input_ids = torch.randint(0, 1000, (2, 64), device=device)
            output = encoder(input_ids)

            assert output.shape == (2, 16, latent_dim)

    def test_deterministic_in_eval_mode(self, encoder):
        """Test that encoder is deterministic in eval mode."""
        device = get_device()
        encoder = encoder.to(device)
        encoder.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, 1000, (2, 64), device=device)

        with torch.no_grad():
            output1 = encoder(input_ids)
            output2 = encoder(input_ids)

        assert torch.allclose(output1, output2)

    def test_compression_factor_must_be_power_of_2(self):
        """Test that non-power-of-2 compression factors raise error."""
        with pytest.raises(AssertionError):
            encoder = TextEncoder(
                model_name="Qwen/Qwen2.5-0.5B",
                latent_dim=128,
                compression_factor=3,  # Not a power of 2
                freeze_backbone=True,
            )

    def test_device_property(self, encoder):
        """Test device property."""
        device = get_device()
        encoder = encoder.to(device)

        assert encoder.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test that encoder works on CUDA."""
        encoder = TextEncoder(
            model_name="Qwen/Qwen2.5-0.5B",
            latent_dim=128,
            compression_factor=4,
            freeze_backbone=True,
            num_latent_layers=1,
        ).cuda()

        input_ids = torch.randint(0, 1000, (2, 64), device="cuda")
        output = encoder(input_ids)

        assert output.device.type == "cuda"
        assert output.shape == (2, 16, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
