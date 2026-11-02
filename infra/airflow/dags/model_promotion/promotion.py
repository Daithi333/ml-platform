"""Promotion and rejection tasks for the train_and_promote DAG."""

import asyncio

import structlog
from datetime import datetime, timezone

from src.logs import setup_logging
from src.pipelines.schemas import PromotionDecision, PromotionEvent

from .common import get_cached_services

setup_logging()
logger = structlog.getLogger(__name__)


def promote(decision: dict) -> None:
    """Set the champion alias on the winning model version."""
    parsed = PromotionDecision.model_validate(decision)
    svc = get_cached_services()

    svc.mlflow_client.set_registered_model_alias(
        name=parsed.candidate.model_name,
        alias="champion",
        version=parsed.candidate.model_version,
    )

    logger.info(
        "Model promoted to champion",
        model_name=parsed.candidate.model_name,
        version=parsed.candidate.model_version,
        reason=parsed.reason,
    )


def reject(decision: dict) -> None:
    """Log the rejection. No state changes — run remains in registry for inspection."""
    parsed = PromotionDecision.model_validate(decision)

    logger.warning(
        "Model promotion rejected",
        model_name=parsed.candidate.model_name,
        version=parsed.candidate.model_version,
        reason=parsed.reason,
    )


def notify(decision: dict, triggered_by: str = "manual") -> None:
    """Emit a promotion event to all configured notification clients."""
    parsed = PromotionDecision.model_validate(decision)

    outcome = "promoted" if parsed.should_promote else "rejected"

    event = PromotionEvent(
        model_name=parsed.candidate.model_name,
        model_version=parsed.candidate.model_version,
        run_id=parsed.candidate.run_id,
        outcome=outcome,
        reason=parsed.reason,
        metrics={
            "accuracy": parsed.candidate.accuracy,
            "f1_macro": parsed.candidate.f1_macro,
            "f1_weighted": parsed.candidate.f1_weighted,
        },
        champion_metrics={
            "accuracy": parsed.champion.accuracy,
            "f1_macro": parsed.champion.f1_macro,
            "f1_weighted": parsed.champion.f1_weighted,
        }
        if parsed.champion
        else None,
        triggered_by=triggered_by,
        timestamp=datetime.now(tz=timezone.utc),
    )

    svc = get_cached_services()
    asyncio.run(svc.notification.notify(event))
