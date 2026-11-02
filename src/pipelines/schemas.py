"""Schemas for the training and promotion pipeline."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class EvaluationResult(BaseModel):
    """Metrics from a completed training run."""

    run_id: str
    model_name: str
    model_version: str
    accuracy: float
    f1_macro: float
    f1_weighted: float


class PromotionDecision(BaseModel):
    """Outcome of the promotion gate evaluation."""

    should_promote: bool
    reason: str
    candidate: EvaluationResult
    champion: EvaluationResult | None = None


class PromotionEvent(BaseModel):
    """Event emitted after a promotion decision — consumed by notification clients."""

    model_name: str
    model_version: str
    run_id: str
    outcome: Literal["promoted", "rejected"]
    reason: str
    metrics: dict[str, float]
    champion_metrics: dict[str, float] | None
    triggered_by: str
    timestamp: datetime
