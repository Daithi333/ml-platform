"""Notification client factory.

Builds a CompositeNotificationClient from settings.
LoggingNotificationClient is always included.
Additional clients are added when their config is present.
"""

from src.config import NotificationSettings
from src.notifications.base import NotificationClient
from src.notifications.clients.composite import CompositeNotificationClient
from src.notifications.clients.logging import LoggingNotificationClient
from src.notifications.clients.slack import SlackNotificationClient


def build_notification_client(settings: NotificationSettings) -> CompositeNotificationClient:
    """Build a composite client from the provided settings.

    Always includes logging. Adds Slack when webhook is configured.
    New channels: add a client class and one conditional block here.
    """
    clients: list[NotificationClient] = [LoggingNotificationClient()]

    if settings.slack_webhook_url:
        clients.append(SlackNotificationClient(settings.slack_webhook_url))

    return CompositeNotificationClient(clients)
