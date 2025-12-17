"""Tests for RQTransformer model."""

import pytest
import torch

from src.model.rq_transformer import RQTransformer


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestRQTransformer:
    """Essential tests for RQTransformer."""

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

    def test_forward(self, model):
        """Test forward pass with loss computation."""
        device = get_device()
        model = model.to(device)

        batch, seq_len, num_q = 2, 16, 4
        indices = torch.randint(0, 32, (batch, seq_len, num_q), device=device)

        output = model(indices)

        assert output["logits"].shape == (batch, seq_len, num_q, 32)
        assert output["loss"].dim() == 0
        assert output["loss"] > 0

    def test_gradient_flow(self, model):
        """Test gradient flow through the model."""
        device = get_device()
        model = model.to(device)

        indices = torch.randint(0, 32, (2, 16, 4), device=device)
        output = model(indices)
        output["loss"].backward()

        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_generate(self, model):
        """Test autoregressive generation."""
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

    def test_generate_with_sampling(self, model):
        """Test generation with top-k and top-p sampling."""
        device = get_device()
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            gen_topk = model.generate(seq_len=8, batch_size=1, top_k=5, device=device)
            gen_topp = model.generate(seq_len=8, batch_size=1, top_p=0.9, device=device)

        assert gen_topk.shape == (1, 8, 4)
        assert gen_topp.shape == (1, 8, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
