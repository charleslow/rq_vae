"""Stage 1: Train RQ-VAE for text reconstruction using PyTorch Lightning.

Trains the encoder-quantizer-decoder pipeline to reconstruct text.
Uses Qwen3 backbone with frozen weights initially, then optionally unfreezes.
"""

import argparse
import os
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import create_dataloader
from model import RQVAE


class RQVAELightningModule(L.LightningModule):
    """PyTorch Lightning module for RQ-VAE training."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        latent_dim: int = 512,
        compression_factor: int = 4,
        codebook_size: int = 512,
        codebook_levels: int = 8,
        commitment_weight: float = 0.25,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 2,
        num_epochs: int = 10,
        total_steps: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model = RQVAE(
            model_name=model_name,
            latent_dim=latent_dim,
            compression_factor=compression_factor,
            codebook_size=codebook_size,
            codebook_levels=codebook_levels,
            commitment_weight=commitment_weight,
            freeze_backbone=True,
        )

        # Store config
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs
        self.total_steps = total_steps
        self.backbone_unfrozen = False

        # For logging examples
        self.tokenizer = None

    def setup(self, stage: str) -> None:
        """Setup tokenizer for decoding examples."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask, labels=labels)

    def on_train_epoch_start(self) -> None:
        """Unfreeze backbone after warmup epochs."""
        if self.current_epoch == self.warmup_epochs and not self.backbone_unfrozen:
            self.print("Unfreezing backbone...")
            self.model.unfreeze_backbone()
            self.backbone_unfrozen = True

            # Log trainable params
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            self.log("trainable_params_after_unfreeze", float(trainable_params))

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self.model(input_ids, attention_mask, labels=input_ids)

        # Log metrics
        self.log("train/loss", outputs["total_loss"], prog_bar=True)
        self.log("train/reconstruction_loss", outputs["reconstruction_loss"])
        self.log("train/commitment_loss", outputs["commitment_loss"])
        self.log("train/accuracy", outputs["accuracy"], prog_bar=True)

        # Log per-level perplexities
        perplexities = outputs["perplexities"]
        for i, ppl in enumerate(perplexities):
            self.log(f"train/perplexity_level_{i}", ppl.item())
        self.log("train/perplexity_mean", perplexities.mean().item())

        # Log codebook usage periodically
        if batch_idx % 100 == 0:
            usage = self.model.get_codebook_usage()
            for i, u in enumerate(usage):
                self.log(f"codebook/usage_level_{i}", u.item())

        return outputs["total_loss"]

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self.model(input_ids, attention_mask, labels=input_ids)

        self.log("val/loss", outputs["total_loss"], prog_bar=True, sync_dist=True)
        self.log("val/reconstruction_loss", outputs["reconstruction_loss"], sync_dist=True)
        self.log("val/accuracy", outputs["accuracy"], prog_bar=True, sync_dist=True)

        # Log per-level perplexities
        perplexities = outputs["perplexities"]
        for i, ppl in enumerate(perplexities):
            self.log(f"val/perplexity_level_{i}", ppl.item(), sync_dist=True)
        self.log("val/perplexity_mean", perplexities.mean().item(), sync_dist=True)

        # Log example reconstructions for first batch
        if batch_idx == 0 and self.tokenizer is not None:
            preds = outputs["logits"].argmax(dim=-1)
            for j in range(min(3, input_ids.size(0))):
                original = self.tokenizer.decode(input_ids[j], skip_special_tokens=True)
                reconstructed = self.tokenizer.decode(preds[j], skip_special_tokens=True)
                if self.logger:
                    self.logger.experiment.log({
                        f"examples/original_{j}": original[:200],
                        f"examples/reconstructed_{j}": reconstructed[:200],
                    })

        return outputs["total_loss"]

    def configure_optimizers(self):
        # Use different learning rates based on whether backbone is frozen
        if self.backbone_unfrozen:
            lr = self.lr * 0.1
        else:
            lr = self.lr

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=self.weight_decay,
        )

        # Calculate total steps if not provided
        if self.total_steps is None:
            # Estimate from trainer
            if self.trainer and self.trainer.estimated_stepping_batches:
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = 10000  # Default fallback
        else:
            total_steps = self.total_steps

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=lr * 0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class RQVAEDataModule(L.LightningDataModule):
    """DataModule for RQ-VAE training."""

    def __init__(
        self,
        dataset_name: str = "openwebtext",
        dataset_config: str | None = None,
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
        max_length: int = 128,
        batch_size: int = 32,
        num_samples: int | None = None,
        num_workers: int = 4,
        text_column: str = "text",
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        pass  # DataLoaders are created dynamically

    def train_dataloader(self) -> DataLoader:
        return create_dataloader(
            dataset_name=self.hparams.dataset_name,
            dataset_config=self.hparams.dataset_config,
            split="train",
            tokenizer_name=self.hparams.tokenizer_name,
            max_length=self.hparams.max_length,
            batch_size=self.hparams.batch_size,
            num_samples=self.hparams.num_samples,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            text_column=self.hparams.text_column,
        )

    def val_dataloader(self) -> DataLoader:
        # Use a subset of train for validation (or specify a val split)
        return create_dataloader(
            dataset_name=self.hparams.dataset_name,
            dataset_config=self.hparams.dataset_config,
            split="train",
            tokenizer_name=self.hparams.tokenizer_name,
            max_length=self.hparams.max_length,
            batch_size=self.hparams.batch_size,
            num_samples=min(self.hparams.num_samples or 1000, 1000),  # Limit val size
            num_workers=self.hparams.num_workers,
            shuffle=False,
            text_column=self.hparams.text_column,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-VAE for text reconstruction")

    # Model arguments
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--compression-factor", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--codebook-levels", type=int, default=8)
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
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)

    # Logging arguments
    parser.add_argument("--wandb-project", type=str, default="rq-vae-text")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--val-check-interval", type=float, default=0.25)

    # Checkpoint arguments
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume-from", type=str, default=None)

    # Device arguments
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create data module
    data_module = RQVAEDataModule(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        text_column=args.text_column,
    )

    # Create model
    model = RQVAELightningModule(
        model_name=args.model_name,
        latent_dim=args.latent_dim,
        compression_factor=args.compression_factor,
        codebook_size=args.codebook_size,
        codebook_levels=args.codebook_levels,
        commitment_weight=args.commitment_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        num_epochs=args.num_epochs,
    )

    # Setup logger
    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        save_dir=args.output_dir,
        config=vars(args),
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="rqvae-{epoch:02d}-{val/loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm,
        val_check_interval=args.val_check_interval,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
    )

    # Train
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_from,
    )

    # Save final model
    final_path = os.path.join(args.output_dir, "rq_vae_final.pt")
    torch.save({
        "model_state_dict": model.model.state_dict(),
        "config": {
            "model_name": args.model_name,
            "latent_dim": args.latent_dim,
            "compression_factor": args.compression_factor,
            "codebook_size": args.codebook_size,
            "codebook_levels": args.codebook_levels,
        },
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
