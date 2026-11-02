"""Notification client protocol.

Any class implementing notify() can serve as a notification client.
New channels (email, PagerDuty, etc.) are added by creating a new client
class and registering it in factory.py — nothing else changes.
"""

from typing import Protocol, runtime_checkable

from src.pipelines.schemas import PromotionEvent


@runtime_checkable
class NotificationClient(Protocol):
    """Interface for promotion event notification clients."""

    async def notify(self, event: PromotionEvent) -> None:
        """Send a promotion event notification."""
        ...
