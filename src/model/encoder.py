"""Text Encoder using Qwen3 as backbone with downsampling."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from einops import rearrange


class TextEncoder(nn.Module):
    """Text encoder using pretrained Qwen3 with learned downsampling.

    Takes token embeddings, processes through Qwen3, then downsamples
    to create compressed latent representations.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        latent_dim: int = 512,
        compression_factor: int = 4,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name for Qwen3
            latent_dim: Dimension of latent space (output of encoder)
            compression_factor: How much to compress sequence length (must be power of 2)
            freeze_backbone: Whether to freeze the pretrained model
        """
        super().__init__()
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.compression_factor = compression_factor

        # Load pretrained Qwen3
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.hidden_size = self.config.hidden_size

        # Freeze backbone if specified
        if freeze_backbone:
            self.freeze_backbone()

        # Downsampling layers using strided convolutions
        # Each conv with stride 2 halves the sequence length
        num_downsample = self._log2(compression_factor)
        downsample_layers = []
        in_dim = self.hidden_size

        for i in range(num_downsample):
            out_dim = latent_dim if i == num_downsample - 1 else in_dim
            downsample_layers.extend([
                nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.LayerNorm(out_dim),
            ])
            in_dim = out_dim

        self.downsample = nn.Sequential(*downsample_layers)

        # Final projection to latent dim if needed
        if in_dim != latent_dim:
            self.proj = nn.Linear(in_dim, latent_dim)
        else:
            self.proj = nn.Identity()

    def _log2(self, x: int) -> int:
        """Compute log base 2, ensuring x is a power of 2."""
        import math
        result = int(math.log2(x))
        assert 2 ** result == x, f"compression_factor must be power of 2, got {x}"
        return result

    def freeze_backbone(self):
        """Freeze the pretrained backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self):
        """Unfreeze the pretrained backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode input tokens to compressed latent representations.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Latent representations (batch, seq_len // compression_factor, latent_dim)
        """
        # Get hidden states from backbone
        # Set backbone to eval mode if frozen to disable dropout
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()

        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Use last hidden state
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Downsample: Conv1d expects (batch, channels, seq_len)
        hidden = rearrange(hidden, "b s d -> b d s")

        # Apply downsampling convolutions
        for i, layer in enumerate(self.downsample):
            if isinstance(layer, nn.LayerNorm):
                # LayerNorm expects (batch, seq_len, dim)
                hidden = rearrange(hidden, "b d s -> b s d")
                hidden = layer(hidden)
                hidden = rearrange(hidden, "b s d -> b d s")
            else:
                hidden = layer(hidden)

        # Back to (batch, seq_len, dim)
        hidden = rearrange(hidden, "b d s -> b s d")

        # Final projection
        latent = self.proj(hidden)

        return latent

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
