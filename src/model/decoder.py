"""Text Decoder using Qwen3 as backbone with upsampling."""

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoConfig, AutoModel


class TextDecoder(nn.Module):
    """Text decoder using pretrained Qwen3 with learned upsampling.

    Takes compressed latent representations, upsamples them, processes
    through Qwen3, and outputs logits over vocabulary.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        latent_dim: int = 512,
        compression_factor: int = 4,
        freeze_backbone: bool = True,
        vocab_size: int | None = None,
    ):
        """
        Args:
            model_name: HuggingFace model name for Qwen3
            latent_dim: Dimension of latent space (input to decoder)
            compression_factor: How much to expand sequence length (must be power of 2)
            freeze_backbone: Whether to freeze the pretrained model
            vocab_size: Vocabulary size for output logits (if None, use model's vocab)
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
        self.vocab_size = vocab_size or self.config.vocab_size

        # Freeze backbone if specified
        if freeze_backbone:
            self.freeze_backbone()

        # Initial projection from latent dim to hidden size
        self.input_proj = nn.Linear(latent_dim, self.hidden_size)

        # Upsampling layers using transposed convolutions
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

        self.upsample = nn.Sequential(*upsample_layers)

        # Output head to vocabulary logits
        self.output_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def _log2(self, x: int) -> int:
        """Compute log base 2, ensuring x is a power of 2."""
        import math

        result = int(math.log2(x))
        assert 2**result == x, f"compression_factor must be power of 2, got {x}"
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

        # Project to backbone hidden size
        hidden = self.input_proj(latent)  # (batch, compressed_len, hidden_size)

        # Upsample: ConvTranspose1d expects (batch, channels, seq_len)
        hidden = rearrange(hidden, "b s d -> b d s")

        for layer in self.upsample:
            if isinstance(layer, nn.LayerNorm):
                hidden = rearrange(hidden, "b d s -> b s d")
                hidden = layer(hidden)
                hidden = rearrange(hidden, "b s d -> b d s")
            else:
                hidden = layer(hidden)

        # Back to (batch, seq_len, dim)
        hidden = rearrange(hidden, "b d s -> b s d")

        # Trim to target length if specified
        if target_len is not None and hidden.shape[1] > target_len:
            hidden = hidden[:, :target_len, :]

        # Process through backbone (non-causal for reconstruction)
        # We use the backbone without causal masking since we're reconstructing
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

        # Project to vocabulary logits
        logits = self.output_head(hidden)  # (batch, seq_len, vocab_size)

        return logits

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
