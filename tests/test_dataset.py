"""Tests for dataset utilities."""

import pytest
import torch
from unittest.mock import MagicMock, patch


class MockHFDataset:
    """Mock HuggingFace dataset that supports subscripting."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def select(self, indices):
        return MockHFDataset([self.data[i] for i in indices])

    def __iter__(self):
        return iter(self.data)


@pytest.fixture
def mock_dataset():
    """Create a mock HuggingFace dataset."""
    return MockHFDataset([{"text": f"Test sentence {i}."} for i in range(10)])


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "<eos>"

    def tokenize_fn(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt"):
        input_ids = torch.randint(0, 1000, (1, max_length))
        attention_mask = torch.ones(1, max_length)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    mock_tok.side_effect = tokenize_fn
    return mock_tok


class TestTextDataset:
    """Essential tests for TextDataset."""

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_getitem_returns_correct_shape(self, mock_tokenizer_cls, mock_load_dataset, mock_dataset, mock_tokenizer):
        """Test that __getitem__ returns properly shaped tensors."""
        from src.data.dataset import TextDataset

        mock_load_dataset.return_value = mock_dataset
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        dataset = TextDataset(
            dataset_name="test_dataset",
            split="train",
            tokenizer_name="test_tokenizer",
            max_length=128,
        )

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert item["input_ids"].shape == (128,)
        assert item["attention_mask"].shape == (128,)


class TestCreateDataloader:
    """Essential tests for create_dataloader."""

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_dataloader_yields_batches(self, mock_tokenizer_cls, mock_load_dataset, mock_dataset, mock_tokenizer):
        """Test dataloader produces correct batch shapes."""
        from src.data.dataset import create_dataloader

        mock_load_dataset.return_value = mock_dataset
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        dataloader = create_dataloader(
            dataset_name="test_dataset",
            split="train",
            tokenizer_name="test_tokenizer",
            max_length=128,
            batch_size=2,
            num_samples=4,
            num_workers=0,
            shuffle=False,
        )

        batch = next(iter(dataloader))
        assert batch["input_ids"].shape == (2, 128)
        assert batch["attention_mask"].shape == (2, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
