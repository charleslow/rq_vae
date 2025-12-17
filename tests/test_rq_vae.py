"""Tests for RQVAE model."""

import pytest
import torch
import tempfile
import os

from src.model.rq_vae import RQVAE


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestRQVAE:
    """Test suite for RQVAE model."""

    @pytest.fixture
    def model(self):
        """Create a minimal RQVAE model for testing."""
        return RQVAE(
            model_name="Qwen/Qwen2.5-0.5B",
            latent_dim=128,
            compression_factor=4,
            codebook_size=32,
            num_quantizers=4,
            commitment_weight=0.25,
            freeze_backbone=True,
        )

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.latent_dim == 128
        assert model.compression_factor == 4
        assert model.codebook_size == 32
        assert model.num_quantizers == 4

    def test_forward_shape(self, model):
        """Test forward pass output shapes."""
        device = get_device()
        model = model.to(device)

        batch, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)
        attention_mask = torch.ones(batch, seq_len, device=device)

        output = model(input_ids, attention_mask)

        assert "logits" in output
        assert "reconstruction_loss" in output
        assert "commitment_loss" in output
        assert "total_loss" in output
        assert "perplexity" in output
        assert "indices" in output
        assert "accuracy" in output

        # Check shapes
        assert output["logits"].shape[0] == batch
        assert output["logits"].shape[1] == seq_len
        assert output["logits"].shape[2] > 0  # vocab_size

        compressed_len = seq_len // 4
        assert output["indices"].shape == (batch, compressed_len, 4)

    def test_encode(self, model):
        """Test encode method."""
        device = get_device()
        model = model.to(device)

        batch, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)

        enc_output = model.encode(input_ids)

        assert "latent" in enc_output
        assert "quantized" in enc_output
        assert "indices" in enc_output
        assert "commitment_loss" in enc_output
        assert "perplexity" in enc_output

        compressed_len = seq_len // 4
        assert enc_output["latent"].shape == (batch, compressed_len, 128)
        assert enc_output["quantized"].shape == (batch, compressed_len, 128)
        assert enc_output["indices"].shape == (batch, compressed_len, 4)

    def test_decode(self, model):
        """Test decode method."""
        device = get_device()
        model = model.to(device)

        batch, compressed_len, latent_dim = 2, 16, 128
        quantized = torch.randn(batch, compressed_len, latent_dim, device=device)

        logits = model.decode(quantized, target_len=64)

        assert logits.shape[0] == batch
        assert logits.shape[1] == 64

    def test_decode_indices(self, model):
        """Test decode_indices method."""
        device = get_device()
        model = model.to(device)

        batch, compressed_len, num_quantizers = 2, 16, 4
        indices = torch.randint(0, 32, (batch, compressed_len, num_quantizers), device=device)

        decoded = model.decode_indices(indices)

        assert decoded.shape == (batch, compressed_len, 128)

    def test_encode_decode_cycle(self, model):
        """Test full encode-decode cycle."""
        device = get_device()
        model = model.to(device)

        batch, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)

        # Encode
        enc_output = model.encode(input_ids)

        # Decode
        logits = model.decode(enc_output["quantized"], target_len=seq_len)

        assert logits.shape == (batch, seq_len, model.decoder.vocab_size)

    def test_reconstruction_loss(self, model):
        """Test reconstruction loss computation."""
        device = get_device()
        model = model.to(device)

        batch, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)

        output = model(input_ids)

        # Loss should be positive
        assert output["reconstruction_loss"] > 0
        assert output["total_loss"] > 0

        # Total loss should include both reconstruction and commitment
        assert output["total_loss"] >= output["reconstruction_loss"]

    def test_gradient_flow(self, model):
        """Test gradient flow through the model."""
        device = get_device()
        model = model.to(device)

        input_ids = torch.randint(0, 1000, (2, 64), device=device)

        output = model(input_ids)
        loss = output["total_loss"]
        loss.backward()

        # Check gradients exist for trainable parameters
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break

        assert has_grad, "No gradients found in model"

    def test_freeze_unfreeze_backbone(self, model):
        """Test freezing and unfreezing backbone."""
        # Initially frozen
        encoder_frozen = not any(p.requires_grad for p in model.encoder.backbone.parameters())
        decoder_frozen = not any(p.requires_grad for p in model.decoder.backbone.parameters())
        assert encoder_frozen and decoder_frozen

        # Unfreeze
        model.unfreeze_backbone()
        encoder_frozen = not any(p.requires_grad for p in model.encoder.backbone.parameters())
        decoder_frozen = not any(p.requires_grad for p in model.decoder.backbone.parameters())
        assert not encoder_frozen and not decoder_frozen

        # Freeze again
        model.freeze_backbone()
        encoder_frozen = not any(p.requires_grad for p in model.encoder.backbone.parameters())
        decoder_frozen = not any(p.requires_grad for p in model.decoder.backbone.parameters())
        assert encoder_frozen and decoder_frozen

    def test_codebook_usage(self, model):
        """Test codebook usage statistics."""
        device = get_device()
        model = model.to(device)

        input_ids = torch.randint(0, 1000, (4, 64), device=device)
        model(input_ids)

        usage = model.get_codebook_usage()

        assert usage.shape == (4,)  # One per quantizer
        assert (usage >= 0).all() and (usage <= 1).all()

    def test_with_custom_labels(self, model):
        """Test forward pass with custom labels."""
        device = get_device()
        model = model.to(device)

        batch, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)
        labels = torch.randint(0, 1000, (batch, seq_len), device=device)

        output = model(input_ids, labels=labels)

        # Should compute loss against custom labels
        assert output["reconstruction_loss"] > 0

    def test_with_padding(self, model):
        """Test forward pass with padding (using -100 ignore index)."""
        device = get_device()
        model = model.to(device)

        batch, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch, seq_len), device=device)
        labels = input_ids.clone()

        # Add padding to labels
        labels[:, -10:] = -100  # Last 10 tokens are padding

        output = model(input_ids, labels=labels)

        # Should still compute loss, ignoring padding
        assert output["reconstruction_loss"] > 0
        assert torch.isfinite(output["reconstruction_loss"])

    def test_accuracy_computation(self, model):
        """Test accuracy computation."""
        device = get_device()
        model = model.to(device)

        input_ids = torch.randint(0, 1000, (2, 64), device=device)

        output = model(input_ids)

        # Accuracy should be between 0 and 1
        assert 0 <= output["accuracy"] <= 1

    def test_save_and_load(self, model):
        """Test saving and loading model."""
        device = get_device()
        model = model.to(device)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")

            # Save
            model.save_pretrained(save_path)
            assert os.path.exists(save_path)

            # Load
            loaded_model = RQVAE.from_pretrained(save_path)
            loaded_model = loaded_model.to(device)

            # Test that loaded model works
            input_ids = torch.randint(0, 1000, (2, 64), device=device)
            output1 = model(input_ids)
            output2 = loaded_model(input_ids)

            # Results should be close (allowing for small numerical differences)
            assert torch.allclose(
                output1["logits"],
                output2["logits"],
                atol=1e-4,
                rtol=1e-3
            )

    def test_different_compression_factors(self):
        """Test model with different compression factors."""
        device = get_device()

        for compression_factor in [2, 4, 8]:
            model = RQVAE(
                model_name="Qwen/Qwen2.5-0.5B",
                latent_dim=128,
                compression_factor=compression_factor,
                codebook_size=32,
                num_quantizers=4,
                freeze_backbone=True,
            ).to(device)

            input_ids = torch.randint(0, 1000, (2, 64), device=device)
            output = model(input_ids)

            compressed_len = 64 // compression_factor
            assert output["indices"].shape == (2, compressed_len, 4)

    def test_different_codebook_sizes(self):
        """Test model with different codebook sizes."""
        device = get_device()

        for codebook_size in [16, 32, 64]:
            model = RQVAE(
                model_name="Qwen/Qwen2.5-0.5B",
                latent_dim=128,
                compression_factor=4,
                codebook_size=codebook_size,
                num_quantizers=4,
                freeze_backbone=True,
            ).to(device)

            input_ids = torch.randint(0, 1000, (2, 64), device=device)
            output = model(input_ids)

            # Indices should be in valid range
            assert (output["indices"] >= 0).all()
            assert (output["indices"] < codebook_size).all()

    def test_different_num_quantizers(self):
        """Test model with different numbers of quantizers."""
        device = get_device()

        for num_quantizers in [2, 4, 8]:
            model = RQVAE(
                model_name="Qwen/Qwen2.5-0.5B",
                latent_dim=128,
                compression_factor=4,
                codebook_size=32,
                num_quantizers=num_quantizers,
                freeze_backbone=True,
            ).to(device)

            input_ids = torch.randint(0, 1000, (2, 64), device=device)
            output = model(input_ids)

            assert output["indices"].shape[2] == num_quantizers

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test that model works on CUDA."""
        model = RQVAE(
            model_name="Qwen/Qwen2.5-0.5B",
            latent_dim=128,
            compression_factor=4,
            codebook_size=32,
            num_quantizers=4,
            freeze_backbone=True,
        ).cuda()

        input_ids = torch.randint(0, 1000, (2, 64), device="cuda")
        output = model(input_ids)

        assert output["logits"].device.type == "cuda"
        assert output["indices"].device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
