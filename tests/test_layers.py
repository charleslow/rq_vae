"""Tests for neural network layers."""

import pytest
import torch

from src.model.layers import SwiGLU, SwiGLUTransformerLayer


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSwiGLU:
    """Essential tests for SwiGLU activation."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        device = get_device()
        swiglu = SwiGLU(dim=128, hidden_dim=None, dropout=0.0).to(device)

        x = torch.randn(4, 32, 128, device=device)
        output = swiglu(x)

        assert output.shape == (4, 32, 128)

    def test_gradient_flow(self):
        """Test that gradients flow through SwiGLU."""
        device = get_device()
        swiglu = SwiGLU(dim=128).to(device)

        x = torch.randn(2, 16, 128, device=device, requires_grad=True)
        output = swiglu(x)
        output.sum().backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestSwiGLUTransformerLayer:
    """Essential tests for SwiGLUTransformerLayer."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        device = get_device()
        layer = SwiGLUTransformerLayer(d_model=128, nhead=8, dropout=0.0).to(device)

        x = torch.randn(4, 32, 128, device=device)
        output = layer(x)

        assert output.shape == (4, 32, 128)

    def test_gradient_flow(self):
        """Test gradient flow through the layer."""
        device = get_device()
        layer = SwiGLUTransformerLayer(d_model=128, nhead=8, dropout=0.0).to(device)

        x = torch.randn(2, 16, 128, device=device, requires_grad=True)
        output = layer(x)
        output.sum().backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_residual_connections(self):
        """Test that residual connections work (output differs from input)."""
        device = get_device()
        layer = SwiGLUTransformerLayer(d_model=128, nhead=8, dropout=0.0).to(device)

        x = torch.randn(2, 16, 128, device=device)
        output = layer(x)

        assert not torch.allclose(output, x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
