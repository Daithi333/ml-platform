"""Composite notification client.

Fans out a promotion event to all configured clients.
Adding a new channel = instantiate its client and pass it to this class.
"""

import structlog

from src.notifications.base import NotificationClient
from src.pipelines.schemas import PromotionEvent

logger = structlog.getLogger(__name__)


class CompositeNotificationClient:
    """Broadcasts promotion events to a list of notification clients."""

    def __init__(self, clients: list[NotificationClient]) -> None:
        self._clients = clients

    async def notify(self, event: PromotionEvent) -> None:
        for client in self._clients:
            try:
                await client.notify(event)
            except Exception as e:
                # One failing client should not block the others
                logger.error(
                    "Notification client failed",
                    client=type(client).__name__,
                    error=str(e),
                    model_name=event.model_name,
                )
