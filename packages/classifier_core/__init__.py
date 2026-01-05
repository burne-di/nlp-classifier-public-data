"""NLP Classifier Core Package."""

from .config import get_settings, LABEL2ID, ID2LABEL, Settings
from .data import load_splits, get_tokenizer, tokenize_dataset, prepare_data_for_training
from .model import load_pretrained_model, load_trained_model, TextClassifier
from .metrics import compute_metrics, get_classification_report, get_confusion_matrix
from .train import create_trainer, train_model
from .llm_client import LLMClient, ClaudeClient, OpenAIClient, GroqClient, get_llm_client
from .hybrid_classifier import HybridClassifier, ClassificationResult, ClassificationStats

__all__ = [
    # Config
    "get_settings",
    "Settings",
    "LABEL2ID",
    "ID2LABEL",
    # Data
    "load_splits",
    "get_tokenizer",
    "tokenize_dataset",
    "prepare_data_for_training",
    # Model
    "load_pretrained_model",
    "load_trained_model",
    "TextClassifier",
    # Metrics
    "compute_metrics",
    "get_classification_report",
    "get_confusion_matrix",
    # Training
    "create_trainer",
    "train_model",
    # LLM Client
    "LLMClient",
    "ClaudeClient",
    "OpenAIClient",
    "GroqClient",
    "get_llm_client",
    # Hybrid Classifier
    "HybridClassifier",
    "ClassificationResult",
    "ClassificationStats",
]
