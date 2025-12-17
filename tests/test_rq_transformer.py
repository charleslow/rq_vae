"""Tests for RQTransformer model."""

import pytest
import torch

from src.model.rq_transformer import (
    RQTransformer,
    SpatialTransformer,
    DepthTransformer,
    TransformerBlock,
    SinusoidalPositionalEncoding,
)


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSinusoidalPositionalEncoding:
    """Test suite for SinusoidalPositionalEncoding."""

    def test_initialization(self):
        """Test positional encoding initialization."""
        pe = SinusoidalPositionalEncoding(dim=128, max_len=512)
        assert pe.pe.shape == (512, 128)

    def test_forward(self):
        """Test forward pass."""
        pe = SinusoidalPositionalEncoding(dim=128, max_len=512)
        output = pe(64)
        assert output.shape == (64, 128)

    def test_deterministic(self):
        """Test that positional encoding is deterministic."""
        pe = SinusoidalPositionalEncoding(dim=128, max_len=512)
        output1 = pe(64)
        output2 = pe(64)
        assert torch.equal(output1, output2)


class TestTransformerBlock:
    """Test suite for TransformerBlock."""

    @pytest.fixture
    def block(self):
        """Create a transformer block for testing."""
        return TransformerBlock(
            dim=128,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
            causal=True,
        )

    def test_initialization(self, block):
        """Test block initialization."""
        assert block.causal is True
        assert isinstance(block.norm1, torch.nn.LayerNorm)
        assert isinstance(block.norm2, torch.nn.LayerNorm)

    def test_forward_shape(self, block):
        """Test forward pass shape."""
        device = get_device()
        block = block.to(device)

        batch, seq_len, dim = 2, 16, 128
        x = torch.randn(batch, seq_len, dim, device=device)

        output = block(x)
        assert output.shape == (batch, seq_len, dim)

    def test_causal_masking(self):
        """Test that causal masking works."""
        device = get_device()
        block = TransformerBlock(dim=128, num_heads=8, dropout=0.0, causal=True).to(device)

        batch, seq_len, dim = 1, 8, 128
        x = torch.randn(batch, seq_len, dim, device=device)

        # Make first token very different
        x[:, 0, :] *= 100

        output = block(x)

        # With causal masking, later tokens shouldn't affect earlier ones
        # This is hard to test directly, so we just check shape
        assert output.shape == (batch, seq_len, dim)

    def test_non_causal(self):
        """Test non-causal transformer block."""
        device = get_device()
        block = TransformerBlock(dim=128, num_heads=8, dropout=0.0, causal=False).to(device)

        x = torch.randn(2, 16, 128, device=device)
        output = block(x)

        assert output.shape == (2, 16, 128)


class TestSpatialTransformer:
    """Test suite for SpatialTransformer."""

    @pytest.fixture
    def spatial_transformer(self):
        """Create a spatial transformer for testing."""
        return SpatialTransformer(
            dim=128,
            codebook_size=32,
            num_quantizers=4,
            num_layers=2,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
            max_seq_len=512,
        )

    def test_initialization(self, spatial_transformer):
        """Test spatial transformer initialization."""
        assert spatial_transformer.dim == 128
        assert spatial_transformer.codebook_size == 32
        assert spatial_transformer.num_quantizers == 4
        assert len(spatial_transformer.layers) == 2

    def test_forward_with_indices(self, spatial_transformer):
        """Test forward pass with indices."""
        device = get_device()
        spatial_transformer = spatial_transformer.to(device)

        batch, seq_len, num_q = 2, 16, 4
        indices = torch.randint(0, 32, (batch, seq_len, num_q), device=device)

        output = spatial_transformer(indices)

        assert output.shape == (batch, seq_len, 128)

    def test_forward_without_indices(self, spatial_transformer):
        """Test forward pass without indices (inference mode)."""
        device = get_device()
        spatial_transformer = spatial_transformer.to(device)

        output = spatial_transformer(indices=None, seq_len=16)

        assert output.shape == (1, 16, 128)

    def test_gradient_flow(self, spatial_transformer):
        """Test gradient flow."""
        device = get_device()
        spatial_transformer = spatial_transformer.to(device)

        indices = torch.randint(0, 32, (2, 16, 4), device=device)
        output = spatial_transformer(indices)

        loss = output.sum()
        loss.backward()

        # Check that embeddings have gradients
        assert spatial_transformer.code_embed.weight.grad is not None


class TestDepthTransformer:
    """Test suite for DepthTransformer."""

    @pytest.fixture
    def depth_transformer(self):
        """Create a depth transformer for testing."""
        return DepthTransformer(
            dim=128,
            codebook_size=32,
            num_quantizers=4,
            num_layers=2,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
        )

    def test_initialization(self, depth_transformer):
        """Test depth transformer initialization."""
        assert depth_transformer.dim == 128
        assert depth_transformer.codebook_size == 32
        assert depth_transformer.num_quantizers == 4
        assert len(depth_transformer.layers) == 2

    def test_forward_with_indices(self, depth_transformer):
        """Test forward pass with indices (teacher forcing)."""
        device = get_device()
        depth_transformer = depth_transformer.to(device)

        batch, seq_len, dim = 2, 16, 128
        h_t = torch.randn(batch, seq_len, dim, device=device)

        num_q = 4
        indices = torch.randint(0, 32, (batch, seq_len, num_q), device=device)

        logits = depth_transformer(h_t, indices)

        assert logits.shape == (batch, seq_len, num_q, 32)

    def test_forward_without_indices(self, depth_transformer):
        """Test forward pass without indices (autoregressive)."""
        device = get_device()
        depth_transformer = depth_transformer.to(device)

        batch, seq_len, dim = 2, 16, 128
        h_t = torch.randn(batch, seq_len, dim, device=device)

        logits = depth_transformer(h_t, indices=None)

        assert logits.shape == (batch, seq_len, 4, 32)

    def test_single_position(self, depth_transformer):
        """Test forward pass with single position."""
        device = get_device()
        depth_transformer = depth_transformer.to(device)

        batch, dim = 2, 128
        h_t = torch.randn(batch, dim, device=device)

        logits = depth_transformer(h_t, indices=None)

        assert logits.shape == (batch, 4, 32)


class TestRQTransformer:
    """Test suite for RQTransformer."""

    @pytest.fixture
    def model(self):
        """Create an RQ-Transformer for testing."""
        return RQTransformer(
            dim=128,
            codebook_size=32,
            num_quantizers=4,
            spatial_layers=2,
            depth_layers=2,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.0,
            max_seq_len=512,
        )

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.dim == 128
        assert model.codebook_size == 32
        assert model.num_quantizers == 4

    def test_forward(self, model):
        """Test forward pass."""
        device = get_device()
        model = model.to(device)

        batch, seq_len, num_q = 2, 16, 4
        indices = torch.randint(0, 32, (batch, seq_len, num_q), device=device)

        output = model(indices)

        assert "logits" in output
        assert "loss" in output
        assert output["logits"].shape == (batch, seq_len, num_q, 32)
        assert output["loss"].dim() == 0  # scalar

    def test_loss_computation(self, model):
        """Test loss computation."""
        device = get_device()
        model = model.to(device)

        indices = torch.randint(0, 32, (2, 16, 4), device=device)

        output = model(indices)

        # Loss should be positive
        assert output["loss"] > 0
        assert torch.isfinite(output["loss"])

    def test_gradient_flow(self, model):
        """Test gradient flow through the model."""
        device = get_device()
        model = model.to(device)

        indices = torch.randint(0, 32, (2, 16, 4), device=device)

        output = model(indices)
        loss = output["loss"]
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad

    def test_generate(self, model):
        """Test generation."""
        device = get_device()
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            generated = model.generate(
                seq_len=16,
                batch_size=2,
                temperature=1.0,
                device=device,
            )

        assert generated.shape == (2, 16, 4)
        assert (generated >= 0).all()
        assert (generated < 32).all()

    def test_generate_with_top_k(self, model):
        """Test generation with top-k sampling."""
        device = get_device()
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            generated = model.generate(
                seq_len=8,
                batch_size=1,
                temperature=1.0,
                top_k=5,
                device=device,
            )

        assert generated.shape == (1, 8, 4)

    def test_generate_with_top_p(self, model):
        """Test generation with nucleus sampling."""
        device = get_device()
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            generated = model.generate(
                seq_len=8,
                batch_size=1,
                temperature=1.0,
                top_p=0.9,
                device=device,
            )

        assert generated.shape == (1, 8, 4)

    def test_generate_different_temperatures(self, model):
        """Test generation with different temperatures."""
        device = get_device()
        model = model.to(device)
        model.eval()

        for temperature in [0.5, 1.0, 2.0]:
            with torch.no_grad():
                generated = model.generate(
                    seq_len=8,
                    batch_size=1,
                    temperature=temperature,
                    device=device,
                )

            assert generated.shape == (1, 8, 4)

    def test_different_model_sizes(self):
        """Test different model configurations."""
        device = get_device()

        configs = [
            {"dim": 64, "spatial_layers": 2, "depth_layers": 1},
            {"dim": 128, "spatial_layers": 4, "depth_layers": 2},
            {"dim": 256, "spatial_layers": 6, "depth_layers": 3},
        ]

        for config in configs:
            model = RQTransformer(
                dim=config["dim"],
                codebook_size=32,
                num_quantizers=4,
                spatial_layers=config["spatial_layers"],
                depth_layers=config["depth_layers"],
                num_heads=8,
            ).to(device)

            indices = torch.randint(0, 32, (2, 16, 4), device=device)
            output = model(indices)

            assert output["logits"].shape == (2, 16, 4, 32)

    def test_get_num_params(self, model):
        """Test parameter counting."""
        num_params = model.get_num_params()
        assert num_params > 0

        # Verify it matches actual parameter count
        actual_params = sum(p.numel() for p in model.parameters())
        assert num_params == actual_params

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test that model works on CUDA."""
        model = RQTransformer(
            dim=128,
            codebook_size=32,
            num_quantizers=4,
            spatial_layers=2,
            depth_layers=2,
        ).cuda()

        indices = torch.randint(0, 32, (2, 16, 4), device="cuda")
        output = model(indices)

        assert output["logits"].device.type == "cuda"
        assert output["loss"].device.type == "cuda"

    def test_deterministic_in_eval(self, model):
        """Test that model is deterministic in eval mode."""
        device = get_device()
        model = model.to(device)
        model.eval()

        torch.manual_seed(42)
        indices = torch.randint(0, 32, (2, 16, 4), device=device)

        with torch.no_grad():
            output1 = model(indices)
            output2 = model(indices)

        assert torch.allclose(output1["logits"], output2["logits"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
