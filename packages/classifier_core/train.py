"""Training utilities for the classifier."""

from pathlib import Path

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from .config import get_settings
from .data import prepare_data_for_training
from .model import load_pretrained_model
from .metrics import compute_metrics


def get_training_args(
    output_dir: Path,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    warmup_ratio: float | None = None,
    weight_decay: float | None = None,
) -> TrainingArguments:
    """Create training arguments for the Trainer.

    Args:
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size for training and evaluation
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio
        weight_decay: Weight decay

    Returns:
        TrainingArguments instance
    """
    settings = get_settings()

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs or settings.num_epochs,
        per_device_train_batch_size=batch_size or settings.batch_size,
        per_device_eval_batch_size=batch_size or settings.batch_size,
        learning_rate=learning_rate or settings.learning_rate,
        warmup_ratio=warmup_ratio or settings.warmup_ratio,
        weight_decay=weight_decay or settings.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        report_to="none",  # MLflow is handled separately
        seed=settings.random_seed,
    )


def create_trainer(
    output_dir: Path,
    tokenized_dataset=None,
    tokenizer=None,
    model=None,
    **training_kwargs,
) -> Trainer:
    """Create a Trainer instance.

    Args:
        output_dir: Directory to save checkpoints
        tokenized_dataset: Tokenized dataset (loads default if None)
        tokenizer: Tokenizer (loads default if None)
        model: Model (loads default if None)
        **training_kwargs: Additional arguments for TrainingArguments

    Returns:
        Configured Trainer instance
    """
    # Load data if not provided
    if tokenized_dataset is None or tokenizer is None:
        tokenized_dataset, tokenizer = prepare_data_for_training()

    # Load model if not provided
    if model is None:
        model = load_pretrained_model()

    # Create training arguments
    training_args = get_training_args(output_dir, **training_kwargs)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    return trainer


def train_model(
    output_dir: Path,
    **training_kwargs,
) -> tuple[Trainer, dict]:
    """Train the model and return results.

    Args:
        output_dir: Directory to save the model
        **training_kwargs: Additional training arguments

    Returns:
        Tuple of (trainer, eval_results)
    """
    # Create trainer
    trainer = create_trainer(output_dir, **training_kwargs)

    # Train
    trainer.train()

    # Evaluate on test set
    tokenized_dataset, _ = prepare_data_for_training()
    eval_results = trainer.evaluate(tokenized_dataset["test"])

    # Save the best model
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    trainer.tokenizer.save_pretrained(str(final_model_path))

    return trainer, eval_results
