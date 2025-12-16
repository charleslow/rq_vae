"""Common neural network layers and building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation function: Swish-Gated Linear Unit.

    Used in LLaMA, PaLM, Mistral, and other modern transformers.
    Performs better than standard GELU FFN.

    Reference:
        Shazeer, "GLU Variants Improve Transformer", 2020
        https://arxiv.org/abs/2002.05202
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        """
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (default: int(2.67 * dim) for SwiGLU standard)
            dropout: Dropout rate
        """
        super().__init__()
        if hidden_dim is None:
            # Standard SwiGLU uses ~2.67x instead of 4x
            # This keeps parameter count similar to 4x GELU FFN
            hidden_dim = int(2.67 * dim)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: SwiGLU(x) = (Swish(W1(x)) ⊙ W3(x)) W2

        Also known as: SwiGLU(x) = (SiLU(W1(x)) * W3(x)) W2
        """
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class SwiGLUTransformerLayer(nn.Module):
    """Transformer encoder layer with SwiGLU FFN instead of standard GELU FFN.

    Uses pre-norm architecture (LayerNorm before attention/FFN) which is more stable
    for training and is the standard in modern transformers.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
    ):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dropout: Dropout rate
            dim_feedforward: FFN hidden dimension (default: int(2.67 * d_model))
        """
        super().__init__()

        # Pre-norm architecture (norm before attention/FFN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )

        # SwiGLU FFN
        self.ffn = SwiGLU(d_model, dim_feedforward, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm and residual connections.

        Args:
            src: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Pre-norm multi-head attention with residual
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout(src2)

        # Pre-norm SwiGLU FFN with residual
        src2 = self.norm2(src)
        src2 = self.ffn(src2)
        src = src + src2

        return src
