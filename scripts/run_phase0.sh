#!/bin/bash
# Run Phase 0: Minimal validation experiment
# Expected runtime: 2-4 hours on A100
# Expected cost: ~$10

set -e

cd "$(dirname "$0")/.."

echo "=== Phase 0: Minimal RQ-VAE Validation ==="

uv run python src/train_vae.py \
    --model-name "Qwen/Qwen3-0.6B" \
    --latent-dim 512 \
    --compression-factor 4 \
    --codebook-size 512 \
    --num-quantizers 4 \
    --max-length 128 \
    --num-samples 50000 \
    --batch-size 32 \
    --lr 1e-4 \
    --num-epochs 5 \
    --warmup-epochs 1 \
    --wandb-run-name "phase0-minimal" \
    --log-interval 50 \
    --eval-interval 500 \
    --save-interval 1000 \
    --output-dir ./checkpoints/phase0 \
    --bf16

echo "=== Phase 0 Complete ==="
