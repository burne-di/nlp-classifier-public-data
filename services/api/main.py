"""FastAPI application for NLP classification with hybrid (Model + LLM) support."""

import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header

from .schemas import (
    PredictRequest,
    PredictResponse,
    HybridPredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    BatchHybridPredictResponse,
    TrainRequest,
    TrainResponse,
    HealthResponse,
    StatsResponse,
)
from .deps import get_classifier, get_hybrid_classifier


# Global variables to track status
model_loaded = False
llm_enabled = False
classifier_instance = None
hybrid_classifier_instance = None
confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.75"))
training_enabled = os.environ.get("TRAINING_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
training_token = os.environ.get("TRAINING_TOKEN")
training_lock = threading.Lock()
project_root = Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load models on startup."""
    global model_loaded, llm_enabled, classifier_instance, hybrid_classifier_instance

    # Load base classifier
    try:
        classifier_instance = get_classifier()
        model_loaded = True
        print("Base model loaded successfully")
    except RuntimeError as e:
        print(f"Warning: {e}")
        model_loaded = False

    # Load hybrid classifier (if LLM API key is available)
    if model_loaded:
        try:
            hybrid_classifier_instance = get_hybrid_classifier()
            llm_enabled = hybrid_classifier_instance.enable_llm
            print(f"Hybrid classifier loaded (LLM enabled: {llm_enabled})")
        except Exception as e:
            print(f"Hybrid classifier not available: {e}")
            llm_enabled = False

    yield

    # Cleanup
    classifier_instance = None
    hybrid_classifier_instance = None


app = FastAPI(
    title="NLP Classifier API",
    description="""
    API for classifying Spanish text into categories.

    ## Modes
    - **Simple mode** (`/predict`): Uses only the Transformer model
    - **Hybrid mode** (`/predict/hybrid`): Uses model + LLM fallback for low confidence

    ## Categories
    - Macroeconomia
    - Alianzas
    - Innovacion
    - Regulaciones
    - Sostenibilidad
    - Otra
    - Reputacion
    """,
    version="0.2.0",
    lifespan=lifespan,
)


def get_model():
    """Dependency to get the base classifier."""
    if not model_loaded or classifier_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first.",
        )
    return classifier_instance


def get_hybrid():
    """Dependency to get the hybrid classifier."""
    if not model_loaded or hybrid_classifier_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Hybrid classifier not available.",
        )
    return hybrid_classifier_instance


def _tail_output(text: str, max_lines: int = 40) -> str | None:
    """Return the last N non-empty lines from a text block."""
    if not text:
        return None
    lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    return "\n".join(lines[-max_lines:])


def _extract_model_path(stdout_text: str) -> str | None:
    """Extract model path from training output if present."""
    for line in stdout_text.splitlines():
        if "Model saved to:" in line:
            return line.split("Model saved to:", 1)[1].strip()
    return None


# ============== Health & Stats ==============

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        llm_enabled=llm_enabled,
        confidence_threshold=confidence_threshold,
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def stats(hybrid=Depends(get_hybrid)):
    """Get classification statistics."""
    stats = hybrid.get_stats()
    return StatsResponse(**stats.to_dict())


@app.post("/stats/reset", tags=["System"])
async def reset_stats(hybrid=Depends(get_hybrid)):
    """Reset classification statistics."""
    hybrid.reset_stats()
    return {"message": "Statistics reset"}


# ============== Training (Blocking) ==============

@app.post("/train", response_model=TrainResponse, tags=["Training"])
def train_model(
    request: TrainRequest,
    x_train_token: str | None = Header(default=None, alias="X-Train-Token"),
):
    """Run training inside the container and return when it finishes."""
    if not training_enabled:
        raise HTTPException(
            status_code=403,
            detail="Training is disabled. Set TRAINING_ENABLED=true to enable.",
        )

    if training_token and x_train_token != training_token:
        raise HTTPException(status_code=401, detail="Invalid training token.")

    if not training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Training already in progress.")

    start_time = time.perf_counter()
    try:
        cmd = [sys.executable, "scripts/train.py"]
        if request.epochs is not None:
            cmd += ["--epochs", str(request.epochs)]
        if request.batch_size is not None:
            cmd += ["--batch-size", str(request.batch_size)]
        if request.lr is not None:
            cmd += ["--lr", str(request.lr)]

        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
        )

        duration = time.perf_counter() - start_time
        stdout_tail = _tail_output(result.stdout)
        stderr_tail = _tail_output(result.stderr)
        model_path = _extract_model_path(result.stdout)

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Training failed",
                    "exit_code": result.returncode,
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                },
            )

        return TrainResponse(
            status="completed",
            exit_code=result.returncode,
            duration_seconds=round(duration, 2),
            model_path=model_path,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
    finally:
        training_lock.release()


# ============== Simple Mode (Model Only) ==============

@app.post("/predict", response_model=PredictResponse, tags=["Simple Mode"])
async def predict(
    request: PredictRequest,
    classifier=Depends(get_model),
):
    """Predict using Transformer model only.

    Fast and cheap, but may be less accurate on edge cases.
    """
    result = classifier.predict(request.text)
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Simple Mode"])
async def predict_batch(
    request: BatchPredictRequest,
    classifier=Depends(get_model),
):
    """Batch predict using Transformer model only."""
    results = classifier.predict_batch(request.texts)
    predictions = [PredictResponse(**r) for r in results]
    return BatchPredictResponse(predictions=predictions)


# ============== Hybrid Mode (Model + LLM Fallback) ==============

@app.post("/predict/hybrid", response_model=HybridPredictResponse, tags=["Hybrid Mode"])
async def predict_hybrid(
    request: PredictRequest,
    hybrid=Depends(get_hybrid),
):
    """Predict using hybrid approach (Model + LLM fallback).

    Uses the fast Transformer model first. If confidence is below threshold,
    falls back to LLM for a more accurate prediction.

    Response includes:
    - `source`: "model" or "llm" (which system made the decision)
    - `cost_usd`: Estimated cost of the request
    - `model_confidence`: Original model confidence (for analytics)
    """
    result = hybrid.classify(request.text)
    return HybridPredictResponse(
        label=result.label,
        confidence=result.confidence,
        source=result.source,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        model_confidence=result.model_confidence,
    )


@app.post("/predict/hybrid/batch", response_model=BatchHybridPredictResponse, tags=["Hybrid Mode"])
async def predict_hybrid_batch(
    request: BatchPredictRequest,
    hybrid=Depends(get_hybrid),
):
    """Batch predict using hybrid approach."""
    results = hybrid.classify_batch(request.texts)
    predictions = [
        HybridPredictResponse(
            label=r.label,
            confidence=r.confidence,
            source=r.source,
            cost_usd=r.cost_usd,
            latency_ms=r.latency_ms,
            model_confidence=r.model_confidence,
        )
        for r in results
    ]
    return BatchHybridPredictResponse(predictions=predictions)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
