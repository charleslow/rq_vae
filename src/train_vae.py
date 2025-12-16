"""Stage 1: Train RQ-VAE for text reconstruction.

Trains the encoder-quantizer-decoder pipeline to reconstruct text.
Uses Qwen3 backbone with frozen weights initially, then optionally unfreezes.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from model import RQVAE
from data import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-VAE for text reconstruction")

    # Model arguments
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--compression-factor", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--num-quantizers", type=int, default=8)
    parser.add_argument("--commitment-weight", type=float, default=0.25)

    # Data arguments
    parser.add_argument("--dataset", type=str, default="openwebtext")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--text-column", type=str, default="text")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="Number of epochs to train with frozen backbone")
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


def evaluate(model, dataloader, tokenizer, device, num_batches=50):
    """Evaluate reconstruction quality."""
    model.eval()

    total_loss = 0
    total_accuracy = 0
    total_perplexity = 0
    num_samples = 0

    example_texts = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask, labels=input_ids)

            total_loss += outputs["reconstruction_loss"].item() * input_ids.size(0)
            total_accuracy += outputs["accuracy"].item() * input_ids.size(0)
            total_perplexity += outputs["perplexity"].item() * input_ids.size(0)
            num_samples += input_ids.size(0)

            # Collect example reconstructions
            if len(example_texts) < 5:
                preds = outputs["logits"].argmax(dim=-1)
                for j in range(min(2, input_ids.size(0))):
                    original = tokenizer.decode(input_ids[j], skip_special_tokens=True)
                    reconstructed = tokenizer.decode(preds[j], skip_special_tokens=True)
                    example_texts.append({
                        "original": original[:200],
                        "reconstructed": reconstructed[:200],
                    })

    model.train()

    return {
        "eval_loss": total_loss / num_samples,
        "eval_accuracy": total_accuracy / num_samples,
        "eval_perplexity": total_perplexity / num_samples,
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
        name=args.wandb_run_name,
        config=vars(args),
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    print("Creating model...")
    model = RQVAE(
        model_name=args.model_name,
        latent_dim=args.latent_dim,
        compression_factor=args.compression_factor,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        commitment_weight=args.commitment_weight,
        freeze_backbone=True,  # Start with frozen backbone
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    wandb.log({"total_params": total_params, "trainable_params": trainable_params})

    # Create dataloaders
    print("Loading data...")
    train_loader = create_dataloader(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split="train",
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        shuffle=True,
        text_column=args.text_column,
    )

    # Create optimizer (only for trainable params initially)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]

    # Training loop
    print("Starting training...")
    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        # Unfreeze backbone after warmup
        if epoch == args.warmup_epochs:
            print("Unfreezing backbone...")
            model.unfreeze_backbone()

            # Recreate optimizer with all parameters
            optimizer = AdamW(
                model.parameters(),
                lr=args.lr * 0.1,  # Lower LR for fine-tuning
                weight_decay=args.weight_decay,
            )
            remaining_steps = (args.num_epochs - epoch) * len(train_loader) // args.gradient_accumulation_steps
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining_steps, eta_min=args.lr * 0.01)

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters after unfreeze: {trainable_params:,}")
            wandb.log({"trainable_params_after_unfreeze": trainable_params})

        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.bf16):
                outputs = model(input_ids, attention_mask, labels=input_ids)
                loss = outputs["total_loss"] / args.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Logging
            epoch_loss += outputs["total_loss"].item()
            epoch_accuracy += outputs["accuracy"].item()
            num_batches += 1

            pbar.set_postfix({
                "loss": outputs["total_loss"].item(),
                "acc": outputs["accuracy"].item(),
                "ppl": outputs["perplexity"].item(),
            })

            # Log to wandb
            if global_step % args.log_interval == 0:
                wandb.log({
                    "train/loss": outputs["total_loss"].item(),
                    "train/reconstruction_loss": outputs["reconstruction_loss"].item(),
                    "train/commitment_loss": outputs["commitment_loss"].item(),
                    "train/accuracy": outputs["accuracy"].item(),
                    "train/perplexity": outputs["perplexity"].item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                }, step=global_step)

                # Log codebook usage
                usage = model.get_codebook_usage()
                for i, u in enumerate(usage):
                    wandb.log({f"codebook/usage_level_{i}": u.item()}, step=global_step)

            # Evaluation
            if global_step % args.eval_interval == 0:
                eval_results = evaluate(model, train_loader, tokenizer, device)
                wandb.log({
                    "eval/loss": eval_results["eval_loss"],
                    "eval/accuracy": eval_results["eval_accuracy"],
                    "eval/perplexity": eval_results["eval_perplexity"],
                }, step=global_step)

                # Log example reconstructions
                for i, ex in enumerate(eval_results["examples"][:3]):
                    wandb.log({
                        f"examples/original_{i}": ex["original"],
                        f"examples/reconstructed_{i}": ex["reconstructed"],
                    }, step=global_step)

                model.train()

            # Save checkpoint
            if global_step % args.save_interval == 0:
                checkpoint_path = os.path.join(
                    args.output_dir, f"checkpoint_step_{global_step}.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": vars(args),
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        # End of epoch
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

        # Save end-of-epoch checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": vars(args),
        }, checkpoint_path)

    # Save final model
    final_path = os.path.join(args.output_dir, "rq_vae_final.pt")
    model.save_pretrained(final_path)
    print(f"Saved final model to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
