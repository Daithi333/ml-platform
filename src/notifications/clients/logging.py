"""Logging notification client.

Always active — requires no external configuration.
Logs the promotion event to structlog at info level.
"""

import structlog

from src.pipelines.schemas import PromotionEvent

logger = structlog.getLogger(__name__)


class LoggingNotificationClient:
    """Logs promotion events via structlog. Zero-dependency fallback."""

    async def notify(self, event: PromotionEvent) -> None:
        log_fn = logger.info if event.outcome == "promoted" else logger.warning
        log_fn(
            "Promotion event",
            model_name=event.model_name,
            version=event.model_version,
            outcome=event.outcome,
            reason=event.reason,
            metrics=event.metrics,
            champion_metrics=event.champion_metrics,
            triggered_by=event.triggered_by,
            timestamp=event.timestamp.isoformat(),
        )
