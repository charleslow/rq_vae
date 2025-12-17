"""Text Decoder using Qwen3 as backbone with upsampling."""

import torch
import torch.nn as nn
from einops import rearrange

try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError):
    UNSLOTH_AVAILABLE = False

from .layers import SwiGLUTransformerLayer


class TextDecoder(nn.Module):
    """Text decoder using pretrained Qwen3 with learned upsampling.

    Decodes from compressed latent sequence (that has been decoded by RQ-transformer)
    and outputs logits over vocabulary (decoded into tokens).

    Steps:
    1. Input is a sequence of compressed latents:
        (batch, compressed_len, latent_dim)
    2. Refine latents with self-attention:
        (batch, compressed_len, latent_dim)
    3. Linear projection to hidden size:
        (batch, compressed_len, hidden_size)
    4. Upsample via transposed convolutions:
        (batch, compressed_len * compression_factor, hidden_size)
       - Each layer 2x expands sequence length
    5. Process through Qwen3 backbone (one-shot, not autoregressive):
        (batch, seq_len, hidden_size)
    6. Linear projection to vocabulary logits:
        (batch, seq_len, vocab_size)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        latent_dim: int = 512,
        compression_factor: int = 4,
        freeze_backbone: bool = True,
        vocab_size: int | None = None,
        num_latent_layers: int = 2,
    ):
        """
        Args:
            model_name: HuggingFace model name for Qwen3
            latent_dim: Dimension of latent space (input to decoder)
            compression_factor: How much to expand sequence length (must be power of 2)
            freeze_backbone: Whether to freeze the pretrained model
            vocab_size: Vocabulary size for output logits (if None, use model's vocab)
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
                self.vocab_size = vocab_size or self.config.vocab_size
            except Exception:
                # Fallback to standard transformers if Unsloth fails
                from transformers import AutoModel
                self.backbone = AutoModel.from_pretrained(model_name)
                self.config = self.backbone.config
                self.hidden_size = self.config.hidden_size
                self.vocab_size = vocab_size or self.config.vocab_size
        else:
            # Use standard transformers on CPU or when Unsloth not available
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(model_name)
            self.config = self.backbone.config
            self.hidden_size = self.config.hidden_size
            self.vocab_size = vocab_size or self.config.vocab_size

        if freeze_backbone:
            self.freeze_backbone()

        # Latent refinement layers (self-attention on compressed sequence)
        self.latent_refinement = self._create_latent_refinement(num_latent_layers)
        self.input_proj = nn.Linear(latent_dim, self.hidden_size)
        self.upsample_layers = self._create_upsample_layers(compression_factor)
        self.output_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def _log2(self, x: int) -> int:
        """Compute log base 2, ensuring x is a power of 2."""
        import math

        result = int(math.log2(x))
        assert 2**result == x, f"compression_factor must be power of 2, got {x}"
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

        layers = nn.ModuleList(
            [
                SwiGLUTransformerLayer(
                    d_model=self.latent_dim,
                    nhead=8,
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
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

    def _create_upsample_layers(self, compression_factor: int) -> nn.Sequential:
        """Create upsampling layers using transposed convolutions.

        How it works:
            ConvTranspose1d is a de-convolution that increases sequence length.
            First we add a space between each input token
            Then we use a kernel splat to fill in the gaps
            Then we trim away the padding (first and last token)
            This will return a sequence twice as long as the input.

        Args:
            compression_factor: How much to expand sequence length (must be power of 2)

        Returns:
            Sequential module containing upsampling layers
        """
        num_upsample = self._log2(compression_factor)
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers.extend(
                [
                    nn.ConvTranspose1d(self.hidden_size, self.hidden_size, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                    nn.LayerNorm(self.hidden_size),
                ]
            )
        return nn.Sequential(*upsample_layers)

    def _upsample(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply upsampling to hidden states.

        Args:
            hidden: Hidden states (batch, seq_len, dim)

        Returns:
            Upsampled hidden states (batch, seq_len * compression_factor, dim)
        """
        # Upsample: ConvTranspose1d expects (batch, channels, seq_len)
        hidden = rearrange(hidden, "b s d -> b d s")

        for layer in self.upsample_layers:
            if isinstance(layer, nn.LayerNorm):
                hidden = rearrange(hidden, "b d s -> b s d")
                hidden = layer(hidden)
                hidden = rearrange(hidden, "b s d -> b d s")
            else:
                hidden = layer(hidden)

        # Back to (batch, seq_len, dim)
        hidden = rearrange(hidden, "b d s -> b s d")

        return hidden

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
        latent: torch.Tensor,
        target_len: int | None = None,
    ) -> torch.Tensor:
        """Decode compressed latent representations to token logits.

        Args:
            latent: Latent representations (batch, compressed_len, latent_dim)
            target_len: Target sequence length (for trimming if needed)

        Returns:
            Logits over vocabulary (batch, seq_len, vocab_size)
        """
        batch, compressed_len, _ = latent.shape

        # First use self-attention to refine latent representation
        if self.num_latent_layers > 0:
            latent = self._refine_latents(latent)
        hidden = self.input_proj(latent)  # (batch, compressed_len, hidden_size)

        # Then use deconvolution to upsample to full sequence length
        hidden = self._upsample(hidden)  # (batch, seq_len, hidden_size)
        if target_len is not None and hidden.shape[1] > target_len:
            hidden = hidden[:, :target_len, :]

        # Process through backbone (non-causal for reconstruction)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()

        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            # Create attention mask (all ones since we're not padding)
            attention_mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long)

            # Use inputs_embeds instead of input_ids
            outputs = self.backbone(
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        logits = self.output_head(hidden)  # (batch, seq_len, vocab_size)

        return logits

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
