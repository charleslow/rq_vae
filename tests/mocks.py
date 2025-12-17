"""Mock models for fast testing without loading real pretrained weights."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any


@dataclass
class MockConfig:
    """Mock configuration for pretrained models."""
    hidden_size: int = 256
    vocab_size: int = 1000
    num_attention_heads: int = 4
    num_hidden_layers: int = 2
    intermediate_size: int = 512


class MockModelOutput:
    """Mock output from transformer models."""

    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class MockBackbone(nn.Module):
    """Lightweight mock backbone that mimics HuggingFace model interface.

    This is ~1000x faster than loading a real model because:
    1. No weights to download
    2. No tokenizer initialization
    3. Tiny parameter count
    """

    def __init__(self, config: MockConfig | None = None):
        super().__init__()
        self.config = config or MockConfig()

        # Minimal layers to produce output of correct shape
        self.embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> MockModelOutput:
        """Forward pass mimicking HuggingFace model."""
        if inputs_embeds is not None:
            hidden = inputs_embeds
        elif input_ids is not None:
            hidden = self.embed(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Simple projection to simulate transformer processing
        hidden = self.proj(hidden)

        return MockModelOutput(last_hidden_state=hidden)


def create_mock_encoder(
    latent_dim: int = 128,
    compression_factor: int = 4,
    hidden_size: int = 256,
    vocab_size: int = 1000,
    num_latent_layers: int = 1,
    freeze_backbone: bool = True,
):
    """Create a mock TextEncoder with lightweight backbone.

    Returns an encoder that behaves identically to the real one
    but loads instantly and uses minimal memory.
    """
    from src.model.encoder import TextEncoder
    from src.model.layers import SwiGLUTransformerLayer
    from unittest.mock import patch, MagicMock

    config = MockConfig(hidden_size=hidden_size, vocab_size=vocab_size)
    mock_backbone = MockBackbone(config)

    # Patch the model loading to return our mock
    with patch('src.model.encoder.UNSLOTH_AVAILABLE', False):
        with patch('transformers.AutoModel.from_pretrained', return_value=mock_backbone):
            encoder = TextEncoder(
                model_name="mock-model",
                latent_dim=latent_dim,
                compression_factor=compression_factor,
                freeze_backbone=freeze_backbone,
                num_latent_layers=num_latent_layers,
            )

    return encoder


def create_mock_decoder(
    latent_dim: int = 128,
    compression_factor: int = 4,
    hidden_size: int = 256,
    vocab_size: int = 1000,
    num_latent_layers: int = 1,
    freeze_backbone: bool = True,
):
    """Create a mock TextDecoder with lightweight backbone."""
    from src.model.decoder import TextDecoder
    from unittest.mock import patch

    config = MockConfig(hidden_size=hidden_size, vocab_size=vocab_size)
    mock_backbone = MockBackbone(config)

    with patch('src.model.decoder.UNSLOTH_AVAILABLE', False):
        with patch('transformers.AutoModel.from_pretrained', return_value=mock_backbone):
            decoder = TextDecoder(
                model_name="mock-model",
                latent_dim=latent_dim,
                compression_factor=compression_factor,
                freeze_backbone=freeze_backbone,
                vocab_size=vocab_size,
                num_latent_layers=num_latent_layers,
            )

    return decoder


def create_mock_rqvae(
    latent_dim: int = 128,
    compression_factor: int = 4,
    codebook_size: int = 32,
    num_quantizers: int = 4,
    hidden_size: int = 256,
    vocab_size: int = 1000,
    freeze_backbone: bool = True,
):
    """Create a mock RQVAE with lightweight backbones."""
    from src.model.rq_vae import RQVAE
    from unittest.mock import patch

    config = MockConfig(hidden_size=hidden_size, vocab_size=vocab_size)

    def mock_from_pretrained(*args, **kwargs):
        return MockBackbone(config)

    with patch('src.model.encoder.UNSLOTH_AVAILABLE', False):
        with patch('src.model.decoder.UNSLOTH_AVAILABLE', False):
            with patch('transformers.AutoModel.from_pretrained', mock_from_pretrained):
                model = RQVAE(
                    model_name="mock-model",
                    latent_dim=latent_dim,
                    compression_factor=compression_factor,
                    codebook_size=codebook_size,
                    num_quantizers=num_quantizers,
                    freeze_backbone=freeze_backbone,
                )

    return model
