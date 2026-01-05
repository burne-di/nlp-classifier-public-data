"""Script to prepare and split the dataset for training."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from classifier_core.config import LABEL2ID, get_settings


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load raw CSV data."""
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records")
    return df


def clean_data(df: pd.DataFrame, min_length: int = 50) -> pd.DataFrame:
    """Clean the dataset.

    - Rename columns to standard names (text, label)
    - Remove rows with null values
    - Remove rows with text shorter than min_length
    - Strip whitespace from text
    """
    print("Cleaning data...")

    # Rename columns
    df = df.rename(columns={"news": "text", "Type": "label"})

    # Drop url column if exists
    if "url" in df.columns:
        df = df.drop(columns=["url"])

    # Remove nulls
    initial_count = len(df)
    df = df.dropna(subset=["text", "label"])
    print(f"Removed {initial_count - len(df)} rows with null values")

    # Strip whitespace
    df["text"] = df["text"].str.strip()

    # Filter short texts
    initial_count = len(df)
    df = df[df["text"].str.len() >= min_length]
    print(f"Removed {initial_count - len(df)} rows with text < {min_length} chars")

    # Add label_id column
    df["label_id"] = df["label"].map(LABEL2ID)

    print(f"Final dataset: {len(df)} records")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets with stratification."""
    print(f"Splitting data (test={test_size}, val={val_size})...")

    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_seed,
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val["label"],
        random_state=random_seed,
    )

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


def print_distribution(df: pd.DataFrame, name: str) -> None:
    """Print label distribution for a dataset."""
    print(f"\n{name} distribution:")
    dist = df["label"].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument(
        "--input",
        type=str,
        default="df_total.csv",
        help="Input CSV filename in data/raw/",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum text length in characters",
    )
    args = parser.parse_args()

    settings = get_settings()

    # Create output directory
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)

    # Load and clean data
    input_path = settings.data_raw_dir / args.input
    df = load_raw_data(input_path)
    df = clean_data(df, min_length=args.min_length)

    # Split data
    train, val, test = split_data(
        df,
        test_size=settings.test_size,
        val_size=settings.val_size,
        random_seed=settings.random_seed,
    )

    # Print distributions
    print_distribution(train, "Train")
    print_distribution(val, "Validation")
    print_distribution(test, "Test")

    # Save splits
    train_path = settings.data_processed_dir / "train.csv"
    val_path = settings.data_processed_dir / "val.csv"
    test_path = settings.data_processed_dir / "test.csv"

    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"\nSaved splits to {settings.data_processed_dir}")
    print(f"  - train.csv: {len(train)} records")
    print(f"  - val.csv: {len(val)} records")
    print(f"  - test.csv: {len(test)} records")


if __name__ == "__main__":
    main()
