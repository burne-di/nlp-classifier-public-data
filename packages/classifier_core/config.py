"""Configuration settings for the NLP classifier."""

from pathlib import Path
from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model settings
    model_name: str = Field(default="distilbert-base-multilingual-cased")
    max_length: int = Field(default=512)
    num_labels: int = Field(default=7)

    # Training settings
    batch_size: int = Field(default=16)
    learning_rate: float = Field(default=2e-5)
    num_epochs: int = Field(default=3)
    warmup_ratio: float = Field(default=0.1)
    weight_decay: float = Field(default=0.01)

    # Data settings
    test_size: float = Field(default=0.1)
    val_size: float = Field(default=0.1)
    min_text_length: int = Field(default=50)
    random_seed: int = Field(default=42)

    # MLflow settings
    mlflow_tracking_uri: str = Field(default="mlruns")
    mlflow_experiment_name: str = Field(default="nlp-classifier")

    @computed_field
    @property
    def project_root(self) -> Path:
        """Project root directory."""
        return PROJECT_ROOT

    @computed_field
    @property
    def data_raw_dir(self) -> Path:
        """Raw data directory."""
        return PROJECT_ROOT / "data" / "raw"

    @computed_field
    @property
    def data_processed_dir(self) -> Path:
        """Processed data directory."""
        return PROJECT_ROOT / "data" / "processed"

    @computed_field
    @property
    def models_dir(self) -> Path:
        """Models directory."""
        return PROJECT_ROOT / "models"



# Label mappings
LABEL2ID = {
    "Macroeconomia": 0,
    "Alianzas": 1,
    "Innovacion": 2,
    "Regulaciones": 3,
    "Sostenibilidad": 4,
    "Otra": 5,
    "Reputacion": 6,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@lru_cache
def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()
