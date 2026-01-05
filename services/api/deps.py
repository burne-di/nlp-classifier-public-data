"""Dependencies for the API."""

import os
import sys
from functools import lru_cache
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

from classifier_core import TextClassifier, HybridClassifier, get_settings


def _find_model_path() -> Path:
    """Find the model path from env var or models directory.

    Returns:
        Path to the model

    Raises:
        RuntimeError: If no model is found
    """
    settings = get_settings()

    # Check for MODEL_PATH env var
    model_path = os.environ.get("MODEL_PATH")

    if model_path:
        model_path = Path(model_path)
    else:
        # Find latest model in models directory
        models_dir = settings.models_dir
        if models_dir.exists():
            model_dirs = sorted(
                [d for d in models_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if model_dirs:
                # Look for final_model subdirectory
                for model_dir in model_dirs:
                    final_model = model_dir / "final_model"
                    if final_model.exists():
                        model_path = final_model
                        break

    if model_path is None or not model_path.exists():
        raise RuntimeError(
            "No model found. Train a model first or set MODEL_PATH environment variable."
        )

    return model_path


@lru_cache
def get_classifier() -> TextClassifier:
    """Get the text classifier instance.

    Returns:
        TextClassifier instance

    Raises:
        RuntimeError: If no model is found
    """
    model_path = _find_model_path()
    return TextClassifier(model_path)


@lru_cache
def get_hybrid_classifier() -> HybridClassifier:
    """Get the hybrid classifier instance.

    Configuration via environment variables:
    - MODEL_PATH: Path to the trained model
    - CONFIDENCE_THRESHOLD: Threshold for LLM fallback (default: 0.75)
    - LLM_PROVIDER: "anthropic", "openai", or "groq" (default: "groq")
    - ANTHROPIC_API_KEY: API key for Anthropic (for Claude)
    - OPENAI_API_KEY: API key for OpenAI (for GPT)
    - GROQ_API_KEY: API key for Groq
    - LLM_MODEL: Model name to use (optional)

    Returns:
        HybridClassifier instance

    Raises:
        RuntimeError: If no model is found
    """
    model_path = _find_model_path()

    confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
    llm_provider = os.environ.get("LLM_PROVIDER", "groq")

    # Check if LLM is available
    llm_api_key = None
    enable_llm = False

    if llm_provider == "anthropic":
        llm_api_key = os.environ.get("ANTHROPIC_API_KEY")
        enable_llm = bool(llm_api_key)
    elif llm_provider == "openai":
        llm_api_key = os.environ.get("OPENAI_API_KEY")
        enable_llm = bool(llm_api_key)
    elif llm_provider == "groq":
        llm_api_key = os.environ.get("GROQ_API_KEY")
        enable_llm = bool(llm_api_key)

    return HybridClassifier(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        enable_llm=enable_llm,
    )
