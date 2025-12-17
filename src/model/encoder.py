"""Text Encoder using Qwen3 as backbone with downsampling."""

import torch
import torch.nn as nn
from einops import rearrange

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError):
    UNSLOTH_AVAILABLE = False

from .layers import SwiGLUTransformerLayer


class TextEncoder(nn.Module):
    """Text encoder using pretrained Qwen3 with learned downsampling.

    Takes token embeddings, processes through Qwen3, then downsamples
    to create compressed latent representations.

    Steps:
    1. Input token IDs: (batch, seq_len)
    2. Process through Qwen3 backbone: (batch, seq_len, hidden_size)
       - Full transformer processing for contextualization
       - Extract last hidden state
    3. Downsample via strided convolutions: (batch, seq_len // compression_factor, hidden_size)
       - Each layer 2x reduces sequence length
       - compression_factor=4 means 2 downsample layers
    4. Project to latent dimension: (batch, compressed_len, latent_dim)
       - Maps from hidden_size to latent_dim
    5. Refine latents with self-attention: (batch, compressed_len, latent_dim)
       - num_latent_layers transformer layers operate on compressed sequence
       - Allows latents to communicate after compression
    6. Output compressed latents: (batch, compressed_len, latent_dim)
       - Ready for RQ-VAE quantization
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        latent_dim: int = 512,
        compression_factor: int = 4,
        freeze_backbone: bool = True,
        num_latent_layers: int = 2,
    ):
        """
        Args:
            model_name: HuggingFace model name for Qwen3
            latent_dim: Dimension of latent space (output of encoder)
            compression_factor: How much to compress sequence length (must be power of 2)
            freeze_backbone: Whether to freeze the pretrained model
            num_latent_layers: Number of self-attention layers to refine latents (0 to disable)
        """
        super().__init__()
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.compression_factor = compression_factor
        self.num_latent_layers = num_latent_layers

        # Load pretrained model (use Unsloth if available, otherwise standard transformers)
        if UNSLOTH_AVAILABLE and torch.cuda.is_available():
            try:
                self.backbone, _ = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=2048,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                )
                self.config = self.backbone.config
                self.hidden_size = self.config.hidden_size
            except Exception:
                # Fallback to standard transformers if Unsloth fails
                from transformers import AutoModel
                self.backbone = AutoModel.from_pretrained(model_name)
                self.config = self.backbone.config
                self.hidden_size = self.config.hidden_size
        else:
            # Use standard transformers on CPU or when Unsloth not available
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(model_name)
            self.config = self.backbone.config
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

        # Latent refinement layers (self-attention on compressed sequence)
        self.latent_refinement = self._create_latent_refinement(num_latent_layers)

    def _log2(self, x: int) -> int:
        """Compute log base 2, ensuring x is a power of 2."""
        import math
        result = int(math.log2(x))
        assert 2 ** result == x, f"compression_factor must be power of 2, got {x}"
        return result

    def _create_latent_refinement(self, num_layers: int) -> nn.ModuleList:
        """Create self-attention layers for latent refinement with SwiGLU FFN.

        Args:
            num_layers: Number of transformer layers (0 to disable)

        Returns:
            ModuleList of custom transformer layers with SwiGLU
        """
        if num_layers == 0:
            return nn.ModuleList()

        layers = nn.ModuleList([
            SwiGLUTransformerLayer(
                d_model=self.latent_dim,
                nhead=8,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])
        return layers

    def _refine_latents(self, latent: torch.Tensor) -> torch.Tensor:
        """Refine latents using self-attention.

        Args:
            latent: Latent representations (batch, seq_len, latent_dim)

        Returns:
            Refined latent representations (batch, seq_len, latent_dim)
        """
        for layer in self.latent_refinement:
            latent = layer(latent)
        return latent

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
        for layer in self.downsample:
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

        # Refine latents with self-attention
        if self.num_latent_layers > 0:
            latent = self._refine_latents(latent)

        return latent

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
