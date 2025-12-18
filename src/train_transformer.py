"""Stage 2: Train RQ-Transformer for latent code prediction using PyTorch Lightning.

Trains the spatial-depth transformer to predict RQ-VAE latent codes.
Uses pre-trained RQ-VAE to extract codes from text.
"""

import argparse
import os

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
from model import RQVAE, RQTransformer


class RQTransformerLightningModule(L.LightningModule):
    """PyTorch Lightning module for RQ-Transformer training."""

    def __init__(
        self,
        # VAE config (loaded from checkpoint)
        vae_checkpoint: str,
        # Transformer config
        dim: int = 512,
        spatial_layers: int = 12,
        depth_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rope: bool = False,
        # Training config
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        total_steps: int | None = None,
        # Data config
        max_length: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load VAE checkpoint and extract config
        vae_ckpt = torch.load(vae_checkpoint, map_location="cpu", weights_only=False)
        self.vae_config = vae_ckpt["config"]

        # Compute compressed sequence length
        self.compressed_len = max_length // self.vae_config["compression_factor"]

        # Support both old (num_quantizers) and new (codebook_levels) config keys
        codebook_levels = self.vae_config.get("codebook_levels") or self.vae_config.get("num_quantizers")

        # Create VAE (will be frozen)
        self.vae = RQVAE(
            model_name=self.vae_config["model_name"],
            latent_dim=self.vae_config["latent_dim"],
            compression_factor=self.vae_config["compression_factor"],
            codebook_size=self.vae_config["codebook_size"],
            codebook_levels=codebook_levels,
            freeze_backbone=True,
        )
        self.vae.load_state_dict(vae_ckpt["model_state_dict"])

        # Freeze VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # Create transformer
        self.transformer = RQTransformer(
            dim=dim,
            codebook_size=self.vae_config["codebook_size"],
            codebook_levels=codebook_levels,
            spatial_layers=spatial_layers,
            depth_layers=depth_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_seq_len=self.compressed_len,
            use_rope=use_rope,
        )

        # Store codebook_levels for later use
        self.codebook_levels = codebook_levels

        # Store config
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_steps = total_steps

        # For logging examples
        self.tokenizer = None

    def setup(self, stage: str) -> None:
        """Setup tokenizer for decoding examples."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.vae_config["model_name"])
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, codes):
        return self.transformer(codes)

    def _extract_codes(self, input_ids, attention_mask):
        """Extract latent codes from text using frozen VAE."""
        with torch.no_grad():
            enc_out = self.vae.encode(input_ids, attention_mask)
        return enc_out["indices"]

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Extract codes from VAE
        codes = self._extract_codes(input_ids, attention_mask)

        # Forward pass through transformer
        outputs = self.transformer(codes)

        # Compute accuracy
        pred_codes = outputs["logits"].argmax(dim=-1)
        accuracy = (pred_codes == codes).float().mean()

        # Log metrics
        self.log("train/loss", outputs["loss"], prog_bar=True)
        self.log("train/code_accuracy", accuracy, prog_bar=True)

        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Extract codes from VAE
        codes = self._extract_codes(input_ids, attention_mask)

        # Forward pass through transformer
        outputs = self.transformer(codes)

        # Code-level accuracy
        pred_codes = outputs["logits"].argmax(dim=-1)
        code_accuracy = (pred_codes == codes).float().mean()

        # Decode predicted codes and compute text accuracy
        quantized = self.vae.decode_indices(pred_codes)
        logits = self.vae.decode(quantized, target_len=input_ids.size(1))
        pred_tokens = logits.argmax(dim=-1)
        text_accuracy = (pred_tokens == input_ids).float().mean()

        self.log("val/loss", outputs["loss"], prog_bar=True, sync_dist=True)
        self.log("val/code_accuracy", code_accuracy, prog_bar=True, sync_dist=True)
        self.log("val/text_accuracy", text_accuracy, sync_dist=True)

        # Log example reconstructions for first batch
        if batch_idx == 0 and self.tokenizer is not None:
            for j in range(min(3, input_ids.size(0))):
                original = self.tokenizer.decode(input_ids[j], skip_special_tokens=True)
                reconstructed = self.tokenizer.decode(pred_tokens[j], skip_special_tokens=True)
                if self.logger:
                    self.logger.experiment.log({
                        f"examples/original_{j}": original[:200],
                        f"examples/reconstructed_{j}": reconstructed[:200],
                    })

        return outputs["loss"]

    def configure_optimizers(self):
        optimizer = AdamW(
            self.transformer.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Calculate total steps if not provided
        if self.total_steps is None:
            if self.trainer and self.trainer.estimated_stepping_batches:
                total_steps = self.trainer.estimated_stepping_batches
            else:
                total_steps = 10000
        else:
            total_steps = self.total_steps

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=self.lr * 0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_save_checkpoint(self, checkpoint):
        """Save transformer config in checkpoint."""
        checkpoint["transformer_config"] = {
            "dim": self.hparams.dim,
            "codebook_size": self.vae_config["codebook_size"],
            "codebook_levels": self.codebook_levels,
            "spatial_layers": self.hparams.spatial_layers,
            "depth_layers": self.hparams.depth_layers,
            "num_heads": self.hparams.num_heads,
            "mlp_ratio": self.hparams.mlp_ratio,
            "dropout": self.hparams.dropout,
            "max_seq_len": self.compressed_len,
            "use_rope": self.hparams.use_rope,
        }
        checkpoint["vae_checkpoint"] = self.hparams.vae_checkpoint


class RQTransformerDataModule(L.LightningDataModule):
    """DataModule for RQ-Transformer training."""

    def __init__(
        self,
        dataset_name: str = "openwebtext",
        dataset_config: str | None = None,
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
        max_length: int = 128,
        batch_size: int = 64,
        num_samples: int | None = None,
        num_workers: int = 4,
        text_column: str = "text",
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        pass

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
        return create_dataloader(
            dataset_name=self.hparams.dataset_name,
            dataset_config=self.hparams.dataset_config,
            split="train",
            tokenizer_name=self.hparams.tokenizer_name,
            max_length=self.hparams.max_length,
            batch_size=self.hparams.batch_size,
            num_samples=min(self.hparams.num_samples or 1000, 1000),
            num_workers=self.hparams.num_workers,
            shuffle=False,
            text_column=self.hparams.text_column,
        )


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
    parser.add_argument("--use-rope", action="store_true", default=False,
                        help="Use Rotary Position Embeddings")

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
    data_module = RQTransformerDataModule(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        text_column=args.text_column,
    )

    # Create model
    model = RQTransformerLightningModule(
        vae_checkpoint=args.vae_checkpoint,
        dim=args.dim,
        spatial_layers=args.spatial_layers,
        depth_layers=args.depth_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_rope=args.use_rope,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
    )

    # Log parameter count
    num_params = model.transformer.get_num_params()
    print(f"RQ-Transformer parameters: {num_params:,}")

    # Setup logger
    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name or "rq-transformer",
        save_dir=args.output_dir,
        config=vars(args),
    )
    logger.experiment.config["transformer_params"] = num_params

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="rqtransformer-{epoch:02d}-{val/loss:.4f}",
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
    final_path = os.path.join(args.output_dir, "rq_transformer_final.pt")
    torch.save({
        "model_state_dict": model.transformer.state_dict(),
        "config": {
            "dim": args.dim,
            "codebook_size": model.vae_config["codebook_size"],
            "codebook_levels": model.codebook_levels,
            "spatial_layers": args.spatial_layers,
            "depth_layers": args.depth_layers,
            "num_heads": args.num_heads,
            "mlp_ratio": args.mlp_ratio,
            "dropout": args.dropout,
            "max_seq_len": model.compressed_len,
            "use_rope": args.use_rope,
        },
        "vae_checkpoint": args.vae_checkpoint,
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
