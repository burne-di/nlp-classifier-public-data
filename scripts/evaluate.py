"""Script to evaluate a trained model on the test set."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from classifier_core import (
    get_settings,
    TextClassifier,
    get_classification_report,
    get_confusion_matrix,
    ID2LABEL,
)


def evaluate_model(model_path: Path) -> dict:
    """Evaluate a trained model on the test set.

    Args:
        model_path: Path to the trained model directory

    Returns:
        Dict with evaluation metrics
    """
    settings = get_settings()

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    classifier = TextClassifier(model_path)

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(settings.data_processed_dir / "test.csv")
    print(f"Test samples: {len(test_df)}")

    # Get predictions
    print("\nRunning predictions...")
    texts = test_df["text"].tolist()
    predictions = classifier.predict_batch(texts)

    # Extract predicted labels
    y_pred = [pred["label"] for pred in predictions]
    y_true = test_df["label"].tolist()

    # Convert to numeric
    label2id = {v: k for k, v in ID2LABEL.items()}
    y_pred_ids = [label2id[label] for label in y_pred]
    y_true_ids = [label2id[label] for label in y_true]

    # Classification report
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    report = get_classification_report(
        np.array(y_true_ids),
        np.array(y_pred_ids),
    )
    print(report)

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = get_confusion_matrix(np.array(y_true_ids), np.array(y_pred_ids))
    labels = [ID2LABEL[i] for i in range(len(ID2LABEL))]

    # Print confusion matrix with labels
    print(f"{'':>15}", end="")
    for label in labels:
        print(f"{label[:10]:>12}", end="")
    print()

    for i, row in enumerate(cm):
        print(f"{labels[i][:15]:>15}", end="")
        for val in row:
            print(f"{val:>12}", end="")
        print()

    # Get metrics from report
    report_dict = get_classification_report(
        np.array(y_true_ids),
        np.array(y_pred_ids),
        output_dict=True,
    )

    results = {
        "accuracy": report_dict["accuracy"],
        "f1_macro": report_dict["macro avg"]["f1-score"],
        "precision_macro": report_dict["macro avg"]["precision"],
        "recall_macro": report_dict["macro avg"]["recall"],
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Accuracy:        {results['accuracy']:.4f}")
    print(f"F1 Macro:        {results['f1_macro']:.4f}")
    print(f"Precision Macro: {results['precision_macro']:.4f}")
    print(f"Recall Macro:    {results['recall_macro']:.4f}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model directory",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)

    evaluate_model(model_path)


if __name__ == "__main__":
    main()
