"""Hybrid classifier combining Transformer model with LLM fallback."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from .model import TextClassifier
from .llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of a classification."""

    label: str
    confidence: float
    source: Literal["model", "llm"]
    cost_usd: float
    latency_ms: float
    model_confidence: float | None = None  # Original model confidence (for analytics)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "source": self.source,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "model_confidence": self.model_confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class ClassificationStats:
    """Statistics for classification decisions."""

    total_requests: int = 0
    model_decisions: int = 0
    llm_fallbacks: int = 0
    total_cost_usd: float = 0.0
    avg_model_latency_ms: float = 0.0
    avg_llm_latency_ms: float = 0.0

    @property
    def model_ratio(self) -> float:
        """Ratio of requests handled by model."""
        if self.total_requests == 0:
            return 0.0
        return self.model_decisions / self.total_requests

    @property
    def llm_ratio(self) -> float:
        """Ratio of requests requiring LLM fallback."""
        if self.total_requests == 0:
            return 0.0
        return self.llm_fallbacks / self.total_requests

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "model_decisions": self.model_decisions,
            "llm_fallbacks": self.llm_fallbacks,
            "model_ratio": round(self.model_ratio, 4),
            "llm_ratio": round(self.llm_ratio, 4),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_model_latency_ms": round(self.avg_model_latency_ms, 2),
            "avg_llm_latency_ms": round(self.avg_llm_latency_ms, 2),
        }


class HybridClassifier:
    """Hybrid classifier that uses a fast model with LLM fallback.

    The classifier uses a fine-tuned Transformer model for fast, cheap predictions.
    When the model's confidence is below the threshold, it falls back to an LLM
    for a more accurate (but slower and more expensive) prediction.

    Cost comparison (approximate):
    - Model inference: ~$0.0001 per request
    - LLM (Claude Haiku): ~$0.001-0.01 per request
    - LLM (GPT-4o-mini): ~$0.0005-0.005 per request

    Example usage:
        classifier = HybridClassifier(
            model_path="models/run_xxx/final_model",
            confidence_threshold=0.75,
            llm_provider="anthropic",
        )
        result = classifier.classify("El banco anunció nuevas regulaciones.")
        print(result.label, result.source, result.cost_usd)
    """

    # Estimated cost per model inference (CPU)
    MODEL_COST_PER_REQUEST = 0.0001

    def __init__(
        self,
        model_path: Path | str,
        confidence_threshold: float = 0.75,
        llm_provider: str = "anthropic",
        llm_api_key: str | None = None,
        enable_llm: bool = True,
    ):
        """Initialize hybrid classifier.

        Args:
            model_path: Path to the trained Transformer model
            confidence_threshold: Minimum confidence to use model prediction (0-1)
            llm_provider: LLM provider ("anthropic" or "openai")
            llm_api_key: API key for LLM (uses env var if not provided)
            enable_llm: Whether to enable LLM fallback (disable for testing)
        """
        self.model = TextClassifier(model_path)
        self.confidence_threshold = confidence_threshold
        self.enable_llm = enable_llm
        self.stats = ClassificationStats()

        # Initialize LLM client if enabled
        self.llm_client: LLMClient | None = None
        if enable_llm:
            try:
                self.llm_client = get_llm_client(llm_provider, api_key=llm_api_key)
                logger.info(f"LLM fallback enabled with provider: {llm_provider}")
            except ValueError as e:
                logger.warning(f"LLM fallback disabled: {e}")
                self.enable_llm = False

    def classify(self, text: str) -> ClassificationResult:
        """Classify text using hybrid approach.

        Args:
            text: Text to classify

        Returns:
            ClassificationResult with label, confidence, source, and cost
        """
        import time

        start_time = time.perf_counter()

        # Step 1: Get model prediction
        model_result = self.model.predict(text)
        model_latency = (time.perf_counter() - start_time) * 1000

        model_confidence = model_result["confidence"]
        model_label = model_result["label"]

        # Step 2: Decide whether to use model or LLM
        if model_confidence >= self.confidence_threshold:
            # Use model prediction
            self._update_stats_model(model_latency)

            return ClassificationResult(
                label=model_label,
                confidence=model_confidence,
                source="model",
                cost_usd=self.MODEL_COST_PER_REQUEST,
                latency_ms=model_latency,
                model_confidence=model_confidence,
            )

        # Step 3: Fall back to LLM
        if not self.enable_llm or self.llm_client is None:
            # LLM not available, use model anyway
            logger.warning(
                f"Low confidence ({model_confidence:.2f}) but LLM not available. "
                f"Using model prediction."
            )
            self._update_stats_model(model_latency)

            return ClassificationResult(
                label=model_label,
                confidence=model_confidence,
                source="model",
                cost_usd=self.MODEL_COST_PER_REQUEST,
                latency_ms=model_latency,
                model_confidence=model_confidence,
            )

        # Call LLM
        llm_start = time.perf_counter()
        llm_result = self.llm_client.classify(text)
        llm_latency = (time.perf_counter() - llm_start) * 1000
        total_latency = (time.perf_counter() - start_time) * 1000

        self._update_stats_llm(llm_latency, llm_result["cost_usd"])

        logger.info(
            f"LLM fallback used. Model: {model_label} ({model_confidence:.2f}) -> "
            f"LLM: {llm_result['label']}"
        )

        return ClassificationResult(
            label=llm_result["label"],
            confidence=llm_result["confidence"],
            source="llm",
            cost_usd=self.MODEL_COST_PER_REQUEST + llm_result["cost_usd"],
            latency_ms=total_latency,
            model_confidence=model_confidence,
        )

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResult objects
        """
        return [self.classify(text) for text in texts]

    def get_stats(self) -> ClassificationStats:
        """Get classification statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset classification statistics."""
        self.stats = ClassificationStats()

    def _update_stats_model(self, latency_ms: float) -> None:
        """Update stats for model decision."""
        self.stats.total_requests += 1
        self.stats.model_decisions += 1
        self.stats.total_cost_usd += self.MODEL_COST_PER_REQUEST

        # Update average latency (running average)
        n = self.stats.model_decisions
        prev_avg = self.stats.avg_model_latency_ms
        self.stats.avg_model_latency_ms = prev_avg + (latency_ms - prev_avg) / n

    def _update_stats_llm(self, latency_ms: float, cost_usd: float) -> None:
        """Update stats for LLM fallback."""
        self.stats.total_requests += 1
        self.stats.llm_fallbacks += 1
        self.stats.total_cost_usd += self.MODEL_COST_PER_REQUEST + cost_usd

        # Update average latency
        n = self.stats.llm_fallbacks
        prev_avg = self.stats.avg_llm_latency_ms
        self.stats.avg_llm_latency_ms = prev_avg + (latency_ms - prev_avg) / n
