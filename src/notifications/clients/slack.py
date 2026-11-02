"""Slack notification client.

Sends a formatted message to a Slack webhook URL.
Only active when NOTIFICATIONS__SLACK_WEBHOOK_URL is configured.
"""

import structlog
import httpx

from src.pipelines.schemas import PromotionEvent

logger = structlog.getLogger(__name__)


class SlackNotificationClient:
    """Posts promotion events to a Slack incoming webhook."""

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    async def notify(self, event: PromotionEvent) -> None:
        icon = ":white_check_mark:" if event.outcome == "promoted" else ":x:"
        title = (
            f"{icon} Model *{event.model_name}* v{event.model_version} *{event.outcome.upper()}*"
        )

        metrics_text = " | ".join(f"{k}: {v:.4f}" for k, v in event.metrics.items())
        champion_text = ""
        if event.champion_metrics:
            champion_text = "\n*Champion metrics:* " + " | ".join(
                f"{k}: {v:.4f}" for k, v in event.champion_metrics.items()
            )

        message = {
            "text": title,
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"{title}\n"
                            f"*Reason:* {event.reason}\n"
                            f"*Candidate metrics:* {metrics_text}"
                            f"{champion_text}\n"
                            f"*Triggered by:* {event.triggered_by}"
                        ),
                    },
                }
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self._webhook_url, json=message)
                response.raise_for_status()
        except Exception as e:
            # Notification failures should not halt the pipeline
            logger.error("Slack notification failed", error=str(e), model_name=event.model_name)
