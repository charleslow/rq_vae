"""Full RQ-VAE model combining encoder, quantizer, and decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TextEncoder
from .decoder import TextDecoder
from .quantizer import ResidualQuantizer


class RQVAE(nn.Module):
    """Residual Quantized VAE for text with Qwen3 backbone.

    Encodes text into compressed discrete latents using residual quantization,
    then decodes back to token logits.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        latent_dim: int = 512,
        compression_factor: int = 4,
        codebook_size: int = 512,
        codebook_levels: int = 8,
        commitment_weight: float = 0.25,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name for Qwen3 backbone
            latent_dim: Dimension of latent space
            compression_factor: Sequence compression ratio (must be power of 2)
            codebook_size: Number of entries per codebook (K)
            codebook_levels: Number of residual quantization levels (D)
            commitment_weight: Weight for commitment loss
            freeze_backbone: Whether to freeze pretrained backbone initially
        """
        super().__init__()
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.compression_factor = compression_factor
        self.codebook_size = codebook_size
        self.codebook_levels = codebook_levels

        # Encoder
        self.encoder = TextEncoder(
            model_name=model_name,
            latent_dim=latent_dim,
            compression_factor=compression_factor,
            freeze_backbone=freeze_backbone,
        )

        # Residual Quantizer
        self.quantizer = ResidualQuantizer(
            dim=latent_dim,
            codebook_size=codebook_size,
            codebook_levels=codebook_levels,
            commitment_weight=commitment_weight,
        )

        # Decoder
        self.decoder = TextDecoder(
            model_name=model_name,
            latent_dim=latent_dim,
            compression_factor=compression_factor,
            freeze_backbone=freeze_backbone,
        )

    def freeze_backbone(self):
        """Freeze both encoder and decoder backbones."""
        self.encoder.freeze_backbone()
        self.decoder.freeze_backbone()

    def unfreeze_backbone(self):
        """Unfreeze both encoder and decoder backbones."""
        self.encoder.unfreeze_backbone()
        self.decoder.unfreeze_backbone()

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode input tokens to quantized latents.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Dictionary containing:
                - latent: Continuous latent before quantization
                - quantized: Quantized latent
                - indices: Codebook indices (batch, compressed_len, codebook_levels)
                - commitment_loss: Commitment loss
                - perplexities: Per-level codebook perplexity (codebook_levels,)
        """
        # Encode to continuous latent
        latent = self.encoder(input_ids, attention_mask)

        # Quantize
        quant_out = self.quantizer(latent)

        return {
            "latent": latent,
            "quantized": quant_out["quantized"],
            "indices": quant_out["indices"],
            "commitment_loss": quant_out["commitment_loss"],
            "perplexities": quant_out["perplexities"],
        }

    def decode(
        self,
        quantized: torch.Tensor,
        target_len: int | None = None,
    ) -> torch.Tensor:
        """Decode quantized latents to token logits.

        Args:
            quantized: Quantized latent representations
            target_len: Target sequence length

        Returns:
            Logits over vocabulary (batch, seq_len, vocab_size)
        """
        return self.decoder(quantized, target_len)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices to continuous representation.

        Args:
            indices: Codebook indices (batch, compressed_len, codebook_levels)

        Returns:
            Decoded continuous representation (batch, compressed_len, latent_dim)
        """
        return self.quantizer.decode_indices(indices)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode -> quantize -> decode.

        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Target token IDs for loss computation (usually same as input_ids)

        Returns:
            Dictionary containing:
                - logits: Output logits (batch, seq_len, vocab_size)
                - reconstruction_loss: Cross-entropy reconstruction loss
                - commitment_loss: Commitment loss from quantizer
                - total_loss: Combined loss
                - perplexities: Per-level codebook perplexity (codebook_levels,)
                - indices: Codebook indices
                - accuracy: Token-level reconstruction accuracy
        """
        seq_len = input_ids.shape[1]

        # Encode and quantize
        enc_out = self.encode(input_ids, attention_mask)

        # Decode
        logits = self.decode(enc_out["quantized"], target_len=seq_len)

        # Compute reconstruction loss
        if labels is None:
            labels = input_ids

        reconstruction_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,  # Ignore padding if present
        )

        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            accuracy = (preds[mask] == labels[mask]).float().mean()

        # Total loss
        total_loss = reconstruction_loss + enc_out["commitment_loss"]

        return {
            "logits": logits,
            "reconstruction_loss": reconstruction_loss,
            "commitment_loss": enc_out["commitment_loss"],
            "total_loss": total_loss,
            "perplexities": enc_out["perplexities"],
            "indices": enc_out["indices"],
            "accuracy": accuracy,
        }

    def get_codebook_usage(self) -> torch.Tensor:
        """Get codebook utilization statistics."""
        return self.quantizer.get_codebook_usage()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "RQVAE":
        """Load a pretrained RQ-VAE model.

        Args:
            path: Path to saved checkpoint
            **kwargs: Additional arguments to override config

        Returns:
            Loaded RQVAE model
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        config.update(kwargs)

        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    def save_pretrained(self, path: str):
        """Save the model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "config": {
                "model_name": self.model_name,
                "latent_dim": self.latent_dim,
                "compression_factor": self.compression_factor,
                "codebook_size": self.codebook_size,
                "codebook_levels": self.codebook_levels,
            },
            "model_state_dict": self.state_dict(),
        }
        torch.save(checkpoint, path)
