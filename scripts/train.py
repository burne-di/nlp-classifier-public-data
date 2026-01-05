"""Script to train the NLP classifier with MLflow tracking."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from classifier_core import (
    get_settings,
    prepare_data_for_training,
    load_pretrained_model,
    create_trainer,
    get_classification_report,
)


def train_with_mlflow(
    num_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> Path:
    """Train the model with MLflow tracking.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        Path to the saved model
    """
    settings = get_settings()

    # Override settings if provided
    num_epochs = num_epochs or settings.num_epochs
    batch_size = batch_size or settings.batch_size
    learning_rate = learning_rate or settings.learning_rate

    # Setup MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = settings.models_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"train_{timestamp}"):
        # Log parameters
        mlflow.log_params({
            "model_name": settings.model_name,
            "max_length": settings.max_length,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_ratio": settings.warmup_ratio,
            "weight_decay": settings.weight_decay,
        })

        print("=" * 60)
        print("NLP Classifier Training")
        print("=" * 60)
        print(f"Model: {settings.model_name}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Output: {output_dir}")
        print("=" * 60)

        # Prepare data
        print("\nLoading and tokenizing data...")
        tokenized_dataset, tokenizer = prepare_data_for_training()
        print(f"Train samples: {len(tokenized_dataset['train'])}")
        print(f"Validation samples: {len(tokenized_dataset['validation'])}")
        print(f"Test samples: {len(tokenized_dataset['test'])}")

        # Load model
        print(f"\nLoading model: {settings.model_name}")
        model = load_pretrained_model()

        # Create trainer
        trainer = create_trainer(
            output_dir=output_dir,
            tokenized_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            model=model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Train
        print("\nStarting training...")
        train_result = trainer.train()

        # Log training metrics
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        })

        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_results = trainer.evaluate(tokenized_dataset["validation"])
        mlflow.log_metrics({
            f"val_{k}": v for k, v in val_results.items()
            if isinstance(v, (int, float))
        })

        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate(tokenized_dataset["test"])
        mlflow.log_metrics({
            f"test_{k}": v for k, v in test_results.items()
            if isinstance(v, (int, float))
        })

        # Get predictions for classification report
        predictions = trainer.predict(tokenized_dataset["test"])
        y_pred = np.argmax(predictions.predictions, axis=-1)
        y_true = predictions.label_ids

        # Generate and log classification report
        report = get_classification_report(y_true, y_pred)
        print("\nClassification Report:")
        print(report)

        # Save report to file
        report_path = output_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(str(report_path))

        # Save final model
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))

        # Log model artifact
        mlflow.log_artifacts(str(final_model_path), artifact_path="model")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Test Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
        print(f"Test F1 Macro: {test_results.get('eval_f1_macro', 0):.4f}")
        print(f"Model saved to: {final_model_path}")
        print("=" * 60)

        return final_model_path


def main():
    parser = argparse.ArgumentParser(description="Train NLP classifier")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    args = parser.parse_args()

    train_with_mlflow(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
