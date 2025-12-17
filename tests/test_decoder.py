"""Tests for TextDecoder module."""

import pytest
import torch

from src.model.decoder import TextDecoder


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTextDecoder:
    """Test suite for TextDecoder."""

    @pytest.fixture
    def decoder(self):
        """Create a minimal decoder for testing.

        Note: Uses a small model to reduce memory and runtime.
        """
        return TextDecoder(
            model_name="Qwen/Qwen2.5-0.5B",
            latent_dim=128,
            compression_factor=4,
            freeze_backbone=True,
            num_latent_layers=1,
        )

    def test_initialization(self, decoder):
        """Test decoder initialization."""
        assert decoder.latent_dim == 128
        assert decoder.compression_factor == 4
        assert decoder.num_latent_layers == 1

        # Check backbone is frozen
        for param in decoder.backbone.parameters():
            assert not param.requires_grad

    def test_forward_shape(self, decoder):
        """Test forward pass output shape."""
        device = get_device()
        decoder = decoder.to(device)

        batch, compressed_len, latent_dim = 2, 16, 128
        latent = torch.randn(batch, compressed_len, latent_dim, device=device)

        # Expand by factor of 4
        output = decoder(latent)

        expected_len = compressed_len * 4
        # vocab_size from model config
        assert output.shape[0] == batch
        assert output.shape[1] == expected_len
        assert output.shape[2] > 0  # vocab_size

    def test_with_target_len(self, decoder):
        """Test decoder with target length specification."""
        device = get_device()
        decoder = decoder.to(device)

        batch, compressed_len, latent_dim = 2, 16, 128
        latent = torch.randn(batch, compressed_len, latent_dim, device=device)

        target_len = 50
        output = decoder(latent, target_len=target_len)

        assert output.shape[0] == batch
        assert output.shape[1] == target_len

    def test_compression_factor(self):
        """Test different compression factors."""
        device = get_device()

        for compression_factor in [2, 4, 8]:
            decoder = TextDecoder(
                model_name="Qwen/Qwen2.5-0.5B",
                latent_dim=128,
                compression_factor=compression_factor,
                freeze_backbone=True,
                num_latent_layers=0,
            ).to(device)

            batch, compressed_len, latent_dim = 2, 16, 128
            latent = torch.randn(batch, compressed_len, latent_dim, device=device)

            output = decoder(latent)

            expected_len = compressed_len * compression_factor
            assert output.shape[1] == expected_len

    def test_gradient_flow_unfrozen(self):
        """Test gradient flow when backbone is unfrozen."""
        device = get_device()

        decoder = TextDecoder(
            model_name="Qwen/Qwen2.5-0.5B",
            latent_dim=128,
            compression_factor=4,
            freeze_backbone=False,  # Unfreeze for this test
            num_latent_layers=1,
        ).to(device)

        latent = torch.randn(2, 16, 128, device=device, requires_grad=True)

        output = decoder(latent)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert latent.grad is not None
        assert not torch.allclose(latent.grad, torch.zeros_like(latent.grad))

    def test_freeze_unfreeze_backbone(self, decoder):
        """Test freezing and unfreezing backbone."""
        # Initially frozen
        for param in decoder.backbone.parameters():
            assert not param.requires_grad

        # Unfreeze
        decoder.unfreeze_backbone()
        for param in decoder.backbone.parameters():
            assert param.requires_grad

        # Freeze again
        decoder.freeze_backbone()
        for param in decoder.backbone.parameters():
            assert not param.requires_grad

    def test_latent_refinement_layers(self):
        """Test decoder with different numbers of latent refinement layers."""
        device = get_device()

        for num_layers in [0, 1, 2]:
            decoder = TextDecoder(
                model_name="Qwen/Qwen2.5-0.5B",
                latent_dim=128,
                compression_factor=4,
                freeze_backbone=True,
                num_latent_layers=num_layers,
            ).to(device)

            assert len(decoder.latent_refinement) == num_layers

            latent = torch.randn(2, 16, 128, device=device)
            output = decoder(latent)

            assert output.shape[0] == 2
            assert output.shape[1] == 64  # 16 * 4

    def test_different_latent_dims(self):
        """Test decoder with different latent dimensions."""
        device = get_device()

        for latent_dim in [64, 128, 256]:
            decoder = TextDecoder(
                model_name="Qwen/Qwen2.5-0.5B",
                latent_dim=latent_dim,
                compression_factor=4,
                freeze_backbone=True,
                num_latent_layers=1,
            ).to(device)

            latent = torch.randn(2, 16, latent_dim, device=device)
            output = decoder(latent)

            assert output.shape[0] == 2
            assert output.shape[1] == 64

    def test_different_batch_sizes(self, decoder):
        """Test with different batch sizes."""
        device = get_device()
        decoder = decoder.to(device)

        for batch_size in [1, 2, 4]:
            latent = torch.randn(batch_size, 16, 128, device=device)
            output = decoder(latent)

            assert output.shape[0] == batch_size
            assert output.shape[1] == 64

    def test_different_sequence_lengths(self, decoder):
        """Test with different latent sequence lengths."""
        device = get_device()
        decoder = decoder.to(device)

        for compressed_len in [8, 16, 32]:
            latent = torch.randn(2, compressed_len, 128, device=device)
            output = decoder(latent)

            expected_len = compressed_len * 4
            assert output.shape[1] == expected_len

    def test_deterministic_in_eval_mode(self, decoder):
        """Test that decoder is deterministic in eval mode."""
        device = get_device()
        decoder = decoder.to(device)
        decoder.eval()

        torch.manual_seed(42)
        latent = torch.randn(2, 16, 128, device=device)

        with torch.no_grad():
            output1 = decoder(latent)
            output2 = decoder(latent)

        assert torch.allclose(output1, output2)

    def test_compression_factor_must_be_power_of_2(self):
        """Test that non-power-of-2 compression factors raise error."""
        with pytest.raises(AssertionError):
            decoder = TextDecoder(
                model_name="Qwen/Qwen2.5-0.5B",
                latent_dim=128,
                compression_factor=3,  # Not a power of 2
                freeze_backbone=True,
            )

    def test_output_logits_range(self, decoder):
        """Test that output logits are in reasonable range."""
        device = get_device()
        decoder = decoder.to(device)

        latent = torch.randn(2, 16, 128, device=device)
        output = decoder(latent)

        # Logits should be finite
        assert torch.isfinite(output).all()

        # Logits should have reasonable magnitude (not exploded)
        assert output.abs().max() < 1000

    def test_device_property(self, decoder):
        """Test device property."""
        device = get_device()
        decoder = decoder.to(device)

        assert decoder.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test that decoder works on CUDA."""
        decoder = TextDecoder(
            model_name="Qwen/Qwen2.5-0.5B",
            latent_dim=128,
            compression_factor=4,
            freeze_backbone=True,
            num_latent_layers=1,
        ).cuda()

        latent = torch.randn(2, 16, 128, device="cuda")
        output = decoder(latent)

        assert output.device.type == "cuda"
        assert output.shape[0] == 2
        assert output.shape[1] == 64

    def test_custom_vocab_size(self):
        """Test decoder with custom vocabulary size."""
        device = get_device()

        custom_vocab = 5000
        decoder = TextDecoder(
            model_name="Qwen/Qwen2.5-0.5B",
            latent_dim=128,
            compression_factor=4,
            freeze_backbone=True,
            vocab_size=custom_vocab,
        ).to(device)

        latent = torch.randn(2, 16, 128, device=device)
        output = decoder(latent)

        assert output.shape[2] == custom_vocab


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
