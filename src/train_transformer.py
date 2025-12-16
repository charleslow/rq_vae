"""Stage 2: Train RQ-Transformer for latent code prediction.

Trains the spatial-depth transformer to predict RQ-VAE latent codes.
Uses pre-trained RQ-VAE to extract codes from text.
"""

import os
import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from model import RQVAE, RQTransformer
from data import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-Transformer for latent prediction")

    # RQ-VAE checkpoint (required)
    parser.add_argument("--vae-checkpoint", type=str, required=True,
                        help="Path to trained RQ-VAE checkpoint")

    # RQ-Transformer arguments
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--spatial-layers", type=int, default=12)
    parser.add_argument("--depth-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Data arguments
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--tokenizer-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--text-column", type=str, default="text")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)

    # Logging arguments
    parser.add_argument("--wandb-project", type=str, default="rq-vae-text")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1000)

    # Checkpoint arguments
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--resume-from", type=str, default=None)

    # Device arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true", default=True)

    return parser.parse_args()


def extract_codes(vae, dataloader, device, dtype):
    """Extract latent codes from text using pre-trained VAE."""
    vae.eval()
    all_codes = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting codes"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=dtype):
                enc_out = vae.encode(input_ids, attention_mask)

            all_codes.append(enc_out["indices"].cpu())

    return torch.cat(all_codes, dim=0)


@torch.no_grad()
def evaluate(transformer, vae, dataloader, tokenizer, device, dtype, num_batches=50):
    """Evaluate code prediction and end-to-end reconstruction."""
    transformer.eval()
    vae.eval()

    total_loss = 0
    total_code_accuracy = 0
    total_text_accuracy = 0
    num_samples = 0

    example_texts = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=dtype):
            # Get ground truth codes
            enc_out = vae.encode(input_ids, attention_mask)
            gt_codes = enc_out["indices"]  # (batch, seq_len, num_quantizers)

            # Predict codes
            outputs = transformer(gt_codes)

            # Code-level accuracy
            pred_codes = outputs["logits"].argmax(dim=-1)
            code_accuracy = (pred_codes == gt_codes).float().mean()

            # Decode predicted codes and compute text accuracy
            quantized = vae.decode_indices(pred_codes)
            logits = vae.decode(quantized, target_len=input_ids.size(1))
            pred_tokens = logits.argmax(dim=-1)
            text_accuracy = (pred_tokens == input_ids).float().mean()

        total_loss += outputs["loss"].item() * input_ids.size(0)
        total_code_accuracy += code_accuracy.item() * input_ids.size(0)
        total_text_accuracy += text_accuracy.item() * input_ids.size(0)
        num_samples += input_ids.size(0)

        # Collect example reconstructions
        if len(example_texts) < 5:
            for j in range(min(2, input_ids.size(0))):
                original = tokenizer.decode(input_ids[j], skip_special_tokens=True)
                reconstructed = tokenizer.decode(pred_tokens[j], skip_special_tokens=True)
                example_texts.append({
                    "original": original[:200],
                    "reconstructed": reconstructed[:200],
                })

    transformer.train()

    return {
        "eval_loss": total_loss / num_samples,
        "eval_code_accuracy": total_code_accuracy / num_samples,
        "eval_text_accuracy": total_text_accuracy / num_samples,
        "examples": example_texts,
    }


def main():
    args = parse_args()

    # Setup device
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or "rq-transformer",
        config=vars(args),
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load pre-trained VAE
    print("Loading pre-trained RQ-VAE...")
    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    vae_config = vae_checkpoint["config"]

    vae = RQVAE(
        model_name=vae_config["model_name"],
        latent_dim=vae_config["latent_dim"],
        compression_factor=vae_config["compression_factor"],
        codebook_size=vae_config["codebook_size"],
        num_quantizers=vae_config["num_quantizers"],
        freeze_backbone=True,
    )
    vae.load_state_dict(vae_checkpoint["model_state_dict"])
    vae = vae.to(device)
    vae.eval()

    # Freeze VAE
    for param in vae.parameters():
        param.requires_grad = False

    # Compute compressed sequence length
    compressed_len = args.max_length // vae_config["compression_factor"]

    # Create RQ-Transformer
    print("Creating RQ-Transformer...")
    transformer = RQTransformer(
        dim=args.dim,
        codebook_size=vae_config["codebook_size"],
        num_quantizers=vae_config["num_quantizers"],
        spatial_layers=args.spatial_layers,
        depth_layers=args.depth_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        max_seq_len=compressed_len,
    )
    transformer = transformer.to(device)

    # Count parameters
    num_params = transformer.get_num_params()
    print(f"RQ-Transformer parameters: {num_params:,}")
    wandb.log({"transformer_params": num_params})

    # Create dataloader
    print("Loading data...")
    train_loader = create_dataloader(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split="train",
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        shuffle=True,
        text_column=args.text_column,
    )

    # Create optimizer
    optimizer = AdamW(
        transformer.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        transformer.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]

    # Training loop
    print("Starting training...")
    transformer.train()

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Extract codes from VAE (no grad needed)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    enc_out = vae.encode(input_ids, attention_mask)
                    codes = enc_out["indices"]  # (batch, compressed_len, num_quantizers)

            # Forward pass through transformer
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = transformer(codes)
                loss = outputs["loss"] / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Compute accuracy
            with torch.no_grad():
                pred_codes = outputs["logits"].argmax(dim=-1)
                accuracy = (pred_codes == codes).float().mean()

            # Gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Logging
            epoch_loss += outputs["loss"].item()
            epoch_accuracy += accuracy.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": outputs["loss"].item(),
                "acc": accuracy.item(),
            })

            # Log to wandb
            if global_step % args.log_interval == 0:
                wandb.log({
                    "train/loss": outputs["loss"].item(),
                    "train/code_accuracy": accuracy.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }, step=global_step)

            # Evaluation
            if global_step % args.eval_interval == 0:
                eval_results = evaluate(
                    transformer, vae, train_loader, tokenizer, device, dtype
                )
                wandb.log({
                    "eval/loss": eval_results["eval_loss"],
                    "eval/code_accuracy": eval_results["eval_code_accuracy"],
                    "eval/text_accuracy": eval_results["eval_text_accuracy"],
                }, step=global_step)

                # Log example reconstructions
                for i, ex in enumerate(eval_results["examples"][:3]):
                    wandb.log({
                        f"examples/original_{i}": ex["original"],
                        f"examples/reconstructed_{i}": ex["reconstructed"],
                    }, step=global_step)

                transformer.train()

            # Save checkpoint
            if global_step % args.save_interval == 0:
                checkpoint_path = os.path.join(
                    args.output_dir, f"transformer_step_{global_step}.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": transformer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": {
                        "dim": args.dim,
                        "codebook_size": vae_config["codebook_size"],
                        "num_quantizers": vae_config["num_quantizers"],
                        "spatial_layers": args.spatial_layers,
                        "depth_layers": args.depth_layers,
                        "num_heads": args.num_heads,
                        "mlp_ratio": args.mlp_ratio,
                        "dropout": args.dropout,
                        "max_seq_len": compressed_len,
                    },
                    "vae_checkpoint": args.vae_checkpoint,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        # End of epoch
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

        # Save end-of-epoch checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"transformer_epoch_{epoch + 1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": transformer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": {
                "dim": args.dim,
                "codebook_size": vae_config["codebook_size"],
                "num_quantizers": vae_config["num_quantizers"],
                "spatial_layers": args.spatial_layers,
                "depth_layers": args.depth_layers,
                "num_heads": args.num_heads,
                "mlp_ratio": args.mlp_ratio,
                "dropout": args.dropout,
                "max_seq_len": compressed_len,
            },
            "vae_checkpoint": args.vae_checkpoint,
        }, checkpoint_path)

    # Save final model
    final_path = os.path.join(args.output_dir, "rq_transformer_final.pt")
    torch.save({
        "model_state_dict": transformer.state_dict(),
        "config": {
            "dim": args.dim,
            "codebook_size": vae_config["codebook_size"],
            "num_quantizers": vae_config["num_quantizers"],
            "spatial_layers": args.spatial_layers,
            "depth_layers": args.depth_layers,
            "num_heads": args.num_heads,
            "mlp_ratio": args.mlp_ratio,
            "dropout": args.dropout,
            "max_seq_len": compressed_len,
        },
        "vae_checkpoint": args.vae_checkpoint,
    }, final_path)
    print(f"Saved final model to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
