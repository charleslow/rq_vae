#!/bin/bash
# Run Phase 2: RQ-Transformer training
# Expected runtime: 8-12 hours on A100
# Expected cost: ~$30
# Requires: Trained RQ-VAE from Phase 1

set -e

cd "$(dirname "$0")/.."

# Check if VAE checkpoint exists
VAE_CHECKPOINT="./checkpoints/phase1/rq_vae_final.pt"
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "Error: VAE checkpoint not found at $VAE_CHECKPOINT"
    echo "Please run Phase 1 first or specify a different checkpoint path."
    exit 1
fi

echo "=== Phase 2: RQ-Transformer Training ==="

uv run python src/train_transformer.py \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --dim 512 \
    --spatial-layers 12 \
    --depth-layers 4 \
    --num-heads 8 \
    --tokenizer-name "Qwen/Qwen3-0.6B" \
    --max-length 256 \
    --num-samples 500000 \
    --batch-size 64 \
    --lr 3e-4 \
    --num-epochs 10 \
    --wandb-run-name "phase2-transformer" \
    --log-interval 100 \
    --eval-interval 1000 \
    --save-interval 5000 \
    --output-dir ./checkpoints/phase2 \
    --bf16

echo "=== Phase 2 Complete ==="
