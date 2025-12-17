"""Quick test script to verify encoder/decoder can be instantiated and run."""

import torch
from src.model import TextEncoder, TextDecoder


def test_encoder():
    """Test encoder initialization and forward pass."""
    print("Testing TextEncoder...")

    # Create encoder with small settings for quick test
    encoder = TextEncoder(
        model_name="Qwen/Qwen3-0.6B",
        latent_dim=512,
        compression_factor=4,
        freeze_backbone=True,
        num_latent_layers=2,
    )

    # Create dummy input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Forward pass
    with torch.no_grad():
        latents = encoder(input_ids, attention_mask)

    expected_compressed_len = seq_len // 4  # compression_factor=4
    assert latents.shape == (batch_size, expected_compressed_len, 512)

    print(f"✓ Encoder output shape: {latents.shape}")
    print(f"✓ Encoder test passed!")
    return encoder


def test_decoder():
    """Test decoder initialization and forward pass."""
    print("\nTesting TextDecoder...")

    # Create decoder with small settings for quick test
    decoder = TextDecoder(
        model_name="Qwen/Qwen3-0.6B",
        latent_dim=512,
        compression_factor=4,
        freeze_backbone=True,
        num_latent_layers=2,
    )

    # Create dummy latent input
    batch_size = 2
    compressed_len = 32
    latent_dim = 512
    latents = torch.randn(batch_size, compressed_len, latent_dim)

    # Forward pass
    with torch.no_grad():
        logits = decoder(latents)

    expected_seq_len = compressed_len * 4  # compression_factor=4
    expected_vocab_size = decoder.vocab_size
    assert logits.shape == (batch_size, expected_seq_len, expected_vocab_size)

    print(f"✓ Decoder output shape: {logits.shape}")
    print(f"✓ Decoder test passed!")
    return decoder


def test_encoder_decoder_pipeline():
    """Test full encoder -> decoder pipeline."""
    print("\nTesting Encoder -> Decoder pipeline...")

    # Use same model for both
    model_name = "Qwen/Qwen3-0.6B"

    encoder = TextEncoder(
        model_name=model_name,
        latent_dim=512,
        compression_factor=4,
        freeze_backbone=True,
        num_latent_layers=2,
    )

    decoder = TextDecoder(
        model_name=model_name,
        latent_dim=512,
        compression_factor=4,
        freeze_backbone=True,
        num_latent_layers=2,
    )

    # Create input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Full pipeline
    with torch.no_grad():
        # Encode
        latents = encoder(input_ids, attention_mask)
        print(f"  Encoded: {input_ids.shape} -> {latents.shape}")

        # Decode
        logits = decoder(latents, target_len=seq_len)
        print(f"  Decoded: {latents.shape} -> {logits.shape}")

        # Verify reconstruction has correct shape
        assert logits.shape == (batch_size, seq_len, decoder.vocab_size)

    print("✓ Full pipeline test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("RQ-VAE Model Tests")
    print("=" * 60)

    try:
        test_encoder()
        test_decoder()
        test_encoder_decoder_pipeline()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()