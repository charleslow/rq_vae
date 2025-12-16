#!/bin/bash
# Run Phase 1: Full RQ-VAE training
# Expected runtime: 12-24 hours on A100
# Expected cost: ~$50

set -e

cd "$(dirname "$0")/.."

echo "=== Phase 1: Full RQ-VAE Training ==="

uv run python src/train_vae.py \
    --model-name "Qwen/Qwen3-0.6B" \
    --latent-dim 512 \
    --compression-factor 8 \
    --codebook-size 512 \
    --num-quantizers 8 \
    --max-length 256 \
    --num-samples 500000 \
    --batch-size 32 \
    --lr 1e-4 \
    --num-epochs 10 \
    --warmup-epochs 2 \
    --gradient-accumulation-steps 2 \
    --wandb-run-name "phase1-full-vae" \
    --log-interval 100 \
    --eval-interval 1000 \
    --save-interval 5000 \
    --output-dir ./checkpoints/phase1 \
    --bf16

echo "=== Phase 1 Complete ==="
