"""Model loading and inference utilities."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import get_settings, ID2LABEL, LABEL2ID


def load_pretrained_model(
    model_name: str | None = None,
    num_labels: int | None = None,
) -> AutoModelForSequenceClassification:
    """Load a pretrained model for sequence classification.

    Args:
        model_name: HuggingFace model name
        num_labels: Number of classification labels

    Returns:
        Model ready for fine-tuning
    """
    settings = get_settings()
    model_name = model_name or settings.model_name
    num_labels = num_labels or settings.num_labels

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    return model


def load_trained_model(
    model_path: Path | str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load a trained model and tokenizer from disk.

    Args:
        model_path: Path to the saved model directory

    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = Path(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


class TextClassifier:
    """High-level classifier for inference."""

    def __init__(self, model_path: Path | str):
        """Initialize classifier with a trained model.

        Args:
            model_path: Path to the saved model directory
        """
        self.model, self.tokenizer = load_trained_model(model_path)
        self.model.eval()

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.settings = get_settings()

    def predict(self, text: str) -> dict:
        """Predict the class for a single text.

        Args:
            text: Input text to classify

        Returns:
            Dict with 'label' and 'confidence' keys
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.settings.max_length,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get prediction
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class].item()

        return {
            "label": ID2LABEL[predicted_class],
            "confidence": round(confidence, 4),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict classes for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of dicts with 'label' and 'confidence' keys
        """
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.settings.max_length,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get predictions
        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)

        results = []
        for i, pred_class in enumerate(predicted_classes):
            confidence = probabilities[i, pred_class].item()
            results.append({
                "label": ID2LABEL[pred_class.item()],
                "confidence": round(confidence, 4),
            })

        return results
