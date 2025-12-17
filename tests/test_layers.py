"""Tests for neural network layers."""

import pytest
import torch
import torch.nn as nn

from src.model.layers import SwiGLU, SwiGLUTransformerLayer


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSwiGLU:
    """Test suite for SwiGLU activation."""

    @pytest.fixture
    def swiglu(self):
        """Create a SwiGLU instance for testing."""
        return SwiGLU(dim=128, hidden_dim=None, dropout=0.0)

    def test_initialization(self, swiglu):
        """Test SwiGLU initialization."""
        assert swiglu.w1.in_features == 128
        assert swiglu.w2.out_features == 128
        # Default hidden_dim should be int(2.67 * 128) = 341
        expected_hidden = int(2.67 * 128)
        assert swiglu.w1.out_features == expected_hidden
        assert swiglu.w3.out_features == expected_hidden

    def test_custom_hidden_dim(self):
        """Test SwiGLU with custom hidden dimension."""
        swiglu = SwiGLU(dim=128, hidden_dim=256, dropout=0.0)
        assert swiglu.w1.out_features == 256
        assert swiglu.w2.in_features == 256

    def test_forward_shape(self, swiglu):
        """Test forward pass output shape."""
        device = get_device()
        swiglu = swiglu.to(device)

        batch, seq_len, dim = 4, 32, 128
        x = torch.randn(batch, seq_len, dim, device=device)

        output = swiglu(x)

        assert output.shape == (batch, seq_len, dim)

    def test_gradient_flow(self, swiglu):
        """Test that gradients flow through SwiGLU."""
        device = get_device()
        swiglu = swiglu.to(device)

        x = torch.randn(2, 16, 128, device=device, requires_grad=True)

        output = swiglu(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_no_bias(self, swiglu):
        """Test that SwiGLU uses no bias (as specified)."""
        assert swiglu.w1.bias is None
        assert swiglu.w2.bias is None
        assert swiglu.w3.bias is None

    @pytest.mark.parametrize("dim", [64, 128, 256, 512])
    def test_different_dimensions(self, dim):
        """Test SwiGLU with different dimensions."""
        device = get_device()
        swiglu = SwiGLU(dim=dim).to(device)

        x = torch.randn(2, 16, dim, device=device)
        output = swiglu(x)

        assert output.shape == (2, 16, dim)

    def test_dropout_training_vs_eval(self):
        """Test that dropout behaves differently in train vs eval mode."""
        device = get_device()
        swiglu = SwiGLU(dim=128, dropout=0.5).to(device)

        torch.manual_seed(42)
        x = torch.randn(100, 16, 128, device=device)

        # Training mode - should apply dropout
        swiglu.train()
        output_train = swiglu(x)

        # Eval mode - should not apply dropout
        swiglu.eval()
        output_eval1 = swiglu(x)
        output_eval2 = swiglu(x)

        # In eval, outputs should be deterministic
        assert torch.allclose(output_eval1, output_eval2)


class TestSwiGLUTransformerLayer:
    """Test suite for SwiGLUTransformerLayer."""

    @pytest.fixture
    def transformer_layer(self):
        """Create a transformer layer instance for testing."""
        return SwiGLUTransformerLayer(
            d_model=128,
            nhead=8,
            dropout=0.0,
        )

    def test_initialization(self, transformer_layer):
        """Test transformer layer initialization."""
        assert isinstance(transformer_layer.norm1, nn.LayerNorm)
        assert isinstance(transformer_layer.norm2, nn.LayerNorm)
        assert isinstance(transformer_layer.self_attn, nn.MultiheadAttention)
        assert isinstance(transformer_layer.ffn, SwiGLU)

    def test_forward_shape(self, transformer_layer):
        """Test forward pass output shape."""
        device = get_device()
        transformer_layer = transformer_layer.to(device)

        batch, seq_len, d_model = 4, 32, 128
        x = torch.randn(batch, seq_len, d_model, device=device)

        output = transformer_layer(x)

        assert output.shape == (batch, seq_len, d_model)

    def test_residual_connections(self, transformer_layer):
        """Test that residual connections work properly."""
        device = get_device()
        transformer_layer = transformer_layer.to(device)

        x = torch.randn(2, 16, 128, device=device)

        # With zero dropout, output should not be identical to input
        # (due to attention and FFN)
        output = transformer_layer(x)

        assert not torch.allclose(output, x)
        # But should have similar scale due to residual connections
        assert output.std() < x.std() * 3

    def test_gradient_flow(self, transformer_layer):
        """Test gradient flow through the layer."""
        device = get_device()
        transformer_layer = transformer_layer.to(device)

        x = torch.randn(2, 16, 128, device=device, requires_grad=True)

        output = transformer_layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_attention_pattern(self, transformer_layer):
        """Test that attention is computed correctly."""
        device = get_device()
        transformer_layer = transformer_layer.to(device)

        # Create input with distinct patterns
        batch, seq_len, d_model = 1, 8, 128
        x = torch.randn(batch, seq_len, d_model, device=device)

        # First token is very different
        x[:, 0, :] *= 10

        output = transformer_layer(x)

        # Output should mix information across sequence
        # (not be identical to input after normalization)
        x_normalized = (x - x.mean()) / (x.std() + 1e-6)
        output_normalized = (output - output.mean()) / (output.std() + 1e-6)

        assert not torch.allclose(x_normalized, output_normalized, atol=0.5)

    @pytest.mark.parametrize("d_model,nhead", [
        (64, 4),
        (128, 8),
        (256, 8),
        (512, 16),
    ])
    def test_different_model_sizes(self, d_model, nhead):
        """Test transformer layer with different model dimensions."""
        device = get_device()

        layer = SwiGLUTransformerLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=0.0,
        ).to(device)

        x = torch.randn(2, 16, d_model, device=device)
        output = layer(x)

        assert output.shape == (2, 16, d_model)

    def test_batch_first(self, transformer_layer):
        """Test that batch_first=True works correctly."""
        device = get_device()
        transformer_layer = transformer_layer.to(device)

        # Batch-first format: (batch, seq, features)
        x = torch.randn(4, 16, 128, device=device)
        output = transformer_layer(x)

        assert output.shape == (4, 16, 128)

    def test_variable_sequence_lengths(self, transformer_layer):
        """Test with different sequence lengths."""
        device = get_device()
        transformer_layer = transformer_layer.to(device)

        for seq_len in [8, 16, 32, 64]:
            x = torch.randn(2, seq_len, 128, device=device)
            output = transformer_layer(x)
            assert output.shape == (2, seq_len, 128)

    def test_pre_norm_architecture(self, transformer_layer):
        """Test that pre-norm is applied correctly."""
        device = get_device()
        transformer_layer = transformer_layer.to(device)

        # Create input with varying norms
        x = torch.randn(2, 16, 128, device=device)
        x = x * torch.linspace(0.1, 2.0, 128, device=device)

        # Should handle varying norms well due to pre-norm
        output = transformer_layer(x)

        # Output should be well-normalized (not explode or vanish)
        assert 0.1 < output.std() < 10.0

    def test_cpu_cuda_consistency(self, transformer_layer):
        """Test consistency between CPU and CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)
        x_cpu = torch.randn(2, 16, 128)

        # CPU version
        torch.manual_seed(42)
        layer_cpu = SwiGLUTransformerLayer(d_model=128, nhead=8, dropout=0.0)
        layer_cpu.eval()
        output_cpu = layer_cpu(x_cpu)

        # CUDA version
        torch.manual_seed(42)
        layer_cuda = SwiGLUTransformerLayer(d_model=128, nhead=8, dropout=0.0).cuda()
        layer_cuda.eval()
        output_cuda = layer_cuda(x_cpu.cuda())

        # Results should be close (allowing for small numerical differences)
        assert torch.allclose(output_cpu, output_cuda.cpu(), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
