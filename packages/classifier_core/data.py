"""Data loading and preprocessing utilities."""

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from .config import get_settings, LABEL2ID


def load_splits(processed_dir: Path | None = None) -> DatasetDict:
    """Load train, validation, and test splits as HuggingFace DatasetDict."""
    settings = get_settings()
    processed_dir = processed_dir or settings.data_processed_dir

    train_df = pd.read_csv(processed_dir / "train.csv")

    extra_train_path = processed_dir / "extra_train.csv"
    if extra_train_path.exists():
        extra_df = pd.read_csv(extra_train_path)
        train_df = pd.concat([train_df, extra_df], ignore_index=True)

    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    return DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })


def get_tokenizer(model_name: str | None = None) -> AutoTokenizer:
    """Load tokenizer for the specified model."""
    settings = get_settings()
    model_name = model_name or settings.model_name
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer | None = None,
    max_length: int | None = None,
) -> DatasetDict:
    """Tokenize the dataset for model training.

    Args:
        dataset: HuggingFace DatasetDict with train/validation/test splits
        tokenizer: Tokenizer to use (loads default if None)
        max_length: Maximum sequence length (uses settings default if None)

    Returns:
        Tokenized DatasetDict ready for training
    """
    settings = get_settings()
    tokenizer = tokenizer or get_tokenizer()
    max_length = max_length or settings.max_length

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # Tokenize all splits
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "label"],
    )

    # Rename label_id to labels (expected by HuggingFace Trainer)
    tokenized = tokenized.rename_column("label_id", "labels")

    # Set format for PyTorch
    tokenized.set_format("torch")

    return tokenized


def prepare_data_for_training(
    processed_dir: Path | None = None,
    model_name: str | None = None,
) -> tuple[DatasetDict, AutoTokenizer]:
    """Complete data preparation pipeline.

    Args:
        processed_dir: Directory with processed CSV splits
        model_name: Model name for tokenizer

    Returns:
        Tuple of (tokenized_dataset, tokenizer)
    """
    # Load splits
    dataset = load_splits(processed_dir)

    # Get tokenizer
    tokenizer = get_tokenizer(model_name)

    # Tokenize
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    return tokenized_dataset, tokenizer
