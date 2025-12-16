#!/bin/bash
# RunPod setup script for RQ-VAE experiments
# Run this after starting a new RunPod instance

set -e

echo "=== RQ-VAE RunPod Setup ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Navigate to project directory (adjust path as needed)
cd /workspace/rq_vae

# Create virtual environment and install dependencies
echo "Installing dependencies with uv..."
uv sync

# Install flash-attn separately (requires special build)
echo "Installing flash-attn..."
uv pip install flash-attn --no-build-isolation

# Login to wandb (you'll need to set WANDB_API_KEY environment variable)
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging into wandb..."
    uv run wandb login "$WANDB_API_KEY"
else
    echo "Warning: WANDB_API_KEY not set. Run 'wandb login' manually."
fi

# Login to HuggingFace (needed for Qwen3 access)
if [ -n "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    uv run huggingface-cli login --token "$HF_TOKEN"
else
    echo "Warning: HF_TOKEN not set. Run 'huggingface-cli login' manually."
fi

# Create checkpoint directories
mkdir -p checkpoints/phase0 checkpoints/phase1 checkpoints/phase2

# Verify GPU is available
echo "Checking GPU..."
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "=== Setup complete! ==="
echo ""
echo "To run Phase 0 (minimal validation):"
echo "  uv run python src/train_vae.py --max-length 128 --num-samples 50000 --num-epochs 5 --output-dir ./checkpoints/phase0"
echo ""
echo "To run Phase 1 (full VAE training):"
echo "  uv run python src/train_vae.py --max-length 256 --num-samples 500000 --compression-factor 8 --num-quantizers 8 --output-dir ./checkpoints/phase1"
echo ""
echo "To run Phase 2 (RQ-Transformer):"
echo "  uv run python src/train_transformer.py --vae-checkpoint ./checkpoints/phase1/rq_vae_final.pt --output-dir ./checkpoints/phase2"
