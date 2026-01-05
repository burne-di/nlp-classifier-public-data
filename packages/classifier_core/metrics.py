"""Evaluation metrics for the classifier."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import ID2LABEL


def compute_metrics(eval_pred) -> dict:
    """Compute metrics for HuggingFace Trainer.

    Args:
        eval_pred: EvalPrediction object with predictions and labels

    Returns:
        Dict with accuracy, f1_macro, precision_macro, recall_macro
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "precision_macro": precision_score(labels, predictions, average="macro"),
        "recall_macro": recall_score(labels, predictions, average="macro"),
    }


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dict: bool = False,
) -> str | dict:
    """Generate a classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dict: If True, return dict instead of string

    Returns:
        Classification report as string or dict
    """
    target_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]

    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=output_dict,
    )


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Generate confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)
