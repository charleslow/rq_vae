"""Tests for dataset utilities."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from src.data.dataset import (
    TextDataset,
    StreamingTextDataset,
    create_dataloader,
    create_streaming_dataloader,
)


class TestTextDataset:
    """Test suite for TextDataset."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock HuggingFace dataset."""
        mock_data = [
            {"text": "This is a test sentence."},
            {"text": "Another test sentence here."},
            {"text": "A third example sentence."},
            {"text": "One more sentence for testing."},
        ]

        mock_ds = Mock()
        mock_ds.__len__ = Mock(return_value=len(mock_data))
        mock_ds.__getitem__ = lambda self, idx: mock_data[idx]
        mock_ds.select = Mock(return_value=mock_ds)

        return mock_ds

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        mock_tok = Mock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"

        def tokenize_fn(text, max_length, padding, truncation, return_tensors):
            # Simple mock tokenization
            tokens = text.split()[:max_length]
            input_ids = torch.randint(0, 1000, (1, max_length))
            attention_mask = torch.ones(1, max_length)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        mock_tok.__call__ = tokenize_fn
        return mock_tok

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_initialization(self, mock_tokenizer_cls, mock_load_dataset, mock_dataset, mock_tokenizer):
        """Test dataset initialization."""
        mock_load_dataset.return_value = mock_dataset
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        dataset = TextDataset(
            dataset_name="test_dataset",
            split="train",
            tokenizer_name="test_tokenizer",
            max_length=128,
            num_samples=None,
        )

        assert dataset.max_length == 128
        mock_load_dataset.assert_called_once()

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_len(self, mock_tokenizer_cls, mock_load_dataset, mock_dataset, mock_tokenizer):
        """Test dataset length."""
        mock_load_dataset.return_value = mock_dataset
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        dataset = TextDataset(
            dataset_name="test_dataset",
            split="train",
            tokenizer_name="test_tokenizer",
            max_length=128,
        )

        assert len(dataset) == 4

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_getitem(self, mock_tokenizer_cls, mock_load_dataset, mock_dataset, mock_tokenizer):
        """Test getting items from dataset."""
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

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_num_samples_limit(self, mock_tokenizer_cls, mock_load_dataset, mock_dataset, mock_tokenizer):
        """Test limiting number of samples."""
        mock_load_dataset.return_value = mock_dataset
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        dataset = TextDataset(
            dataset_name="test_dataset",
            split="train",
            tokenizer_name="test_tokenizer",
            max_length=128,
            num_samples=2,
        )

        # Should call select to limit samples
        mock_dataset.select.assert_called()


class TestStreamingTextDataset:
    """Test suite for StreamingTextDataset."""

    @pytest.fixture
    def mock_streaming_dataset(self):
        """Create a mock streaming dataset."""
        mock_data = [
            {"text": "This is a test sentence."},
            {"text": "Another test sentence here."},
            {"text": "A third example sentence."},
        ]

        mock_ds = Mock()
        mock_ds.__iter__ = Mock(return_value=iter(mock_data))

        return mock_ds

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        mock_tok = Mock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"

        def tokenize_fn(text, max_length, padding, truncation, return_tensors):
            input_ids = torch.randint(0, 1000, (1, max_length))
            attention_mask = torch.ones(1, max_length)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        mock_tok.__call__ = tokenize_fn
        return mock_tok

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_initialization(self, mock_tokenizer_cls, mock_load_dataset, mock_streaming_dataset, mock_tokenizer):
        """Test streaming dataset initialization."""
        mock_load_dataset.return_value = mock_streaming_dataset
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        dataset = StreamingTextDataset(
            dataset_name="test_dataset",
            split="train",
            tokenizer_name="test_tokenizer",
            max_length=128,
        )

        assert dataset.max_length == 128
        mock_load_dataset.assert_called_once_with("test_dataset", split="train", streaming=True)

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_iter(self, mock_tokenizer_cls, mock_load_dataset, mock_streaming_dataset, mock_tokenizer):
        """Test iterating over streaming dataset."""
        mock_load_dataset.return_value = mock_streaming_dataset
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        dataset = StreamingTextDataset(
            dataset_name="test_dataset",
            split="train",
            tokenizer_name="test_tokenizer",
            max_length=128,
        )

        items = list(dataset)

        assert len(items) == 3
        for item in items:
            assert "input_ids" in item
            assert "attention_mask" in item


class TestCreateDataloader:
    """Test suite for create_dataloader function."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        mock_data = [
            {"text": f"Test sentence {i}."} for i in range(10)
        ]

        mock_ds = Mock()
        mock_ds.__len__ = Mock(return_value=len(mock_data))
        mock_ds.__getitem__ = lambda self, idx: mock_data[idx]
        mock_ds.select = Mock(return_value=mock_ds)

        return mock_ds

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        mock_tok = Mock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"

        def tokenize_fn(text, max_length, padding, truncation, return_tensors):
            input_ids = torch.randint(0, 1000, (1, max_length))
            attention_mask = torch.ones(1, max_length)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        mock_tok.__call__ = tokenize_fn
        return mock_tok

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_create_dataloader(self, mock_tokenizer_cls, mock_load_dataset, mock_dataset, mock_tokenizer):
        """Test creating a dataloader."""
        mock_load_dataset.return_value = mock_dataset
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        dataloader = create_dataloader(
            dataset_name="test_dataset",
            split="train",
            tokenizer_name="test_tokenizer",
            max_length=128,
            batch_size=4,
            num_samples=None,
            num_workers=0,  # Use 0 for testing
            shuffle=True,
        )

        # Check dataloader properties
        assert dataloader.batch_size == 4
        assert dataloader.drop_last is True
        assert dataloader.pin_memory is True

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_dataloader_iteration(self, mock_tokenizer_cls, mock_load_dataset, mock_dataset, mock_tokenizer):
        """Test iterating over dataloader."""
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

        # Iterate through one batch
        for batch in dataloader:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert batch["input_ids"].shape[0] == 2  # batch size
            assert batch["input_ids"].shape[1] == 128  # max length
            break


class TestCreateStreamingDataloader:
    """Test suite for create_streaming_dataloader function."""

    @pytest.fixture
    def mock_streaming_dataset(self):
        """Create a mock streaming dataset."""
        mock_data = [
            {"text": f"Test sentence {i}."} for i in range(10)
        ]

        mock_ds = Mock()
        mock_ds.__iter__ = Mock(return_value=iter(mock_data))

        return mock_ds

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        mock_tok = Mock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"

        def tokenize_fn(text, max_length, padding, truncation, return_tensors):
            input_ids = torch.randint(0, 1000, (1, max_length))
            attention_mask = torch.ones(1, max_length)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        mock_tok.__call__ = tokenize_fn
        return mock_tok

    @patch("src.data.dataset.load_dataset")
    @patch("src.data.dataset.AutoTokenizer")
    def test_create_streaming_dataloader(self, mock_tokenizer_cls, mock_load_dataset, mock_streaming_dataset, mock_tokenizer):
        """Test creating a streaming dataloader."""
        mock_load_dataset.return_value = mock_streaming_dataset
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        dataloader = create_streaming_dataloader(
            dataset_name="test_dataset",
            split="train",
            tokenizer_name="test_tokenizer",
            max_length=128,
            batch_size=4,
        )

        # Dataloader should be an iterator
        batch = next(dataloader)

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 4  # batch size
        assert batch["input_ids"].shape[1] == 128  # max length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
