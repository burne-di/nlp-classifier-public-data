"""FastAPI application for NLP classification with hybrid (Model + LLM) support."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends

from .schemas import (
    PredictRequest,
    PredictResponse,
    HybridPredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    BatchHybridPredictResponse,
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
