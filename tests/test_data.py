"""Tests for data loading and preprocessing."""

import sys
from pathlib import Path

import pytest
import pandas as pd

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from classifier_core import get_settings, LABEL2ID, ID2LABEL


class TestConfig:
    """Tests for configuration."""

    def test_settings_loads(self):
        """Test that settings load correctly."""
        settings = get_settings()
        assert settings.model_name == "distilbert-base-multilingual-cased"
        assert settings.num_labels == 7
        assert settings.max_length == 512

    def test_label_mappings(self):
        """Test that label mappings are consistent."""
        assert len(LABEL2ID) == 7
        assert len(ID2LABEL) == 7
        for label, id_ in LABEL2ID.items():
            assert ID2LABEL[id_] == label


class TestDataProcessing:
    """Tests for data processing."""

    def test_processed_data_exists(self):
        """Test that processed data files exist."""
        settings = get_settings()
        assert (settings.data_processed_dir / "train.csv").exists()
        assert (settings.data_processed_dir / "val.csv").exists()
        assert (settings.data_processed_dir / "test.csv").exists()

    def test_processed_data_columns(self):
        """Test that processed data has correct columns."""
        settings = get_settings()
        train_df = pd.read_csv(settings.data_processed_dir / "train.csv")

        assert "text" in train_df.columns
        assert "label" in train_df.columns
        assert "label_id" in train_df.columns

    def test_labels_are_valid(self):
        """Test that all labels in data are valid."""
        settings = get_settings()
        train_df = pd.read_csv(settings.data_processed_dir / "train.csv")

        valid_labels = set(LABEL2ID.keys())
        data_labels = set(train_df["label"].unique())

        assert data_labels.issubset(valid_labels)

    def test_label_ids_are_valid(self):
        """Test that label IDs are in valid range."""
        settings = get_settings()
        train_df = pd.read_csv(settings.data_processed_dir / "train.csv")

        assert train_df["label_id"].min() >= 0
        assert train_df["label_id"].max() < len(LABEL2ID)
