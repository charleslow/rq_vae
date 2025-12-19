"""Data loading utilities for RQ-VAE training."""

from collections.abc import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# TODO: Change to use fineweb instead of openwebtext


class TextDataset(Dataset):
    """Dataset for text autoencoding.

    Loads text from HuggingFace datasets and tokenizes for RQ-VAE training.
    """

    def __init__(
        self,
        dataset_name: str = "openwebtext",
        dataset_config: str | None = None,
        split: str = "train",
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
        max_length: int = 256,
        num_samples: int | None = None,
        text_column: str = "text",
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration (e.g., "en" for some datasets)
            split: Dataset split to use
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            num_samples: Number of samples to use (None for all)
            text_column: Name of text column in dataset
        """
        self.max_length = max_length
        self.text_column = text_column

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)

        # Limit samples if specified
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.dataset[idx][self.text_column]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels with -100 for padding positions (ignored in loss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class StreamingTextDataset:
    """Streaming dataset for large-scale training.

    Uses HuggingFace streaming to avoid loading entire dataset into memory.
    """

    def __init__(
        self,
        dataset_name: str = "openwebtext",
        dataset_config: str | None = None,
        split: str = "train",
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
        max_length: int = 256,
        text_column: str = "text",
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            split: Dataset split to use
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            text_column: Name of text column in dataset
        """
        self.max_length = max_length
        self.text_column = text_column

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load streaming dataset
        if dataset_config:
            self.dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        else:
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for example in self.dataset:
            text = example[self.text_column]

            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

            # Create labels with -100 for padding positions (ignored in loss)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }


def create_dataloader(
    dataset_name: str = "openwebtext",
    dataset_config: str | None = None,
    split: str = "train",
    tokenizer_name: str = "Qwen/Qwen3-0.6B",
    max_length: int = 256,
    batch_size: int = 32,
    num_samples: int | None = None,
    num_workers: int = 4,
    shuffle: bool = True,
    text_column: str = "text",
) -> DataLoader:
    """Create a DataLoader for text autoencoding.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        split: Dataset split to use
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length
        batch_size: Batch size
        num_samples: Number of samples to use
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        text_column: Name of text column

    Returns:
        DataLoader instance
    """
    dataset = TextDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        num_samples=num_samples,
        text_column=text_column,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def create_streaming_dataloader(
    dataset_name: str = "openwebtext",
    dataset_config: str | None = None,
    split: str = "train",
    tokenizer_name: str = "Qwen/Qwen3-0.6B",
    max_length: int = 256,
    batch_size: int = 32,
    text_column: str = "text",
) -> Iterator[dict[str, torch.Tensor]]:
    """Create a streaming iterator for large-scale training.

    Yields batches of tokenized text without loading full dataset.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        split: Dataset split to use
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length
        batch_size: Batch size
        text_column: Name of text column

    Yields:
        Batches of tokenized text
    """
    dataset = StreamingTextDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        text_column=text_column,
    )

    batch = {"input_ids": [], "attention_mask": [], "labels": []}

    for example in dataset:
        batch["input_ids"].append(example["input_ids"])
        batch["attention_mask"].append(example["attention_mask"])
        batch["labels"].append(example["labels"])

        if len(batch["input_ids"]) == batch_size:
            yield {
                "input_ids": torch.stack(batch["input_ids"]),
                "attention_mask": torch.stack(batch["attention_mask"]),
                "labels": torch.stack(batch["labels"]),
            }
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
