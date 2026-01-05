"""Pydantic schemas for API requests and responses."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        description="Text to classify",
        json_schema_extra={"example": "El banco anunció una alianza estratégica con la fintech."},
    )


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""

    label: str = Field(..., description="Predicted category")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")


class HybridPredictResponse(BaseModel):
    """Response schema for hybrid prediction endpoint."""

    label: str = Field(..., description="Predicted category")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    source: Literal["model", "llm"] = Field(..., description="Source of prediction")
    cost_usd: float = Field(..., description="Estimated cost in USD")
    latency_ms: float = Field(..., description="Latency in milliseconds")
    model_confidence: float | None = Field(None, description="Original model confidence")


class BatchPredictRequest(BaseModel):
    """Request schema for batch prediction endpoint."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to classify (max 100)",
    )


class BatchPredictResponse(BaseModel):
    """Response schema for batch prediction endpoint."""

    predictions: list[PredictResponse] = Field(..., description="List of predictions")


class BatchHybridPredictResponse(BaseModel):
    """Response schema for batch hybrid prediction endpoint."""

    predictions: list[HybridPredictResponse] = Field(..., description="List of predictions")


class HealthResponse(BaseModel):
    """Response schema for health endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    llm_enabled: bool = Field(default=False, description="Whether LLM fallback is enabled")
    confidence_threshold: float = Field(default=0.75, description="Confidence threshold for LLM fallback")


class StatsResponse(BaseModel):
    """Response schema for statistics endpoint."""

    total_requests: int = Field(..., description="Total requests processed")
    model_decisions: int = Field(..., description="Requests handled by model")
    llm_fallbacks: int = Field(..., description="Requests requiring LLM fallback")
    model_ratio: float = Field(..., description="Ratio of model decisions")
    llm_ratio: float = Field(..., description="Ratio of LLM fallbacks")
    total_cost_usd: float = Field(..., description="Total cost in USD")
    avg_model_latency_ms: float = Field(..., description="Average model latency")
    avg_llm_latency_ms: float = Field(..., description="Average LLM latency")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    detail: str = Field(..., description="Error message")
