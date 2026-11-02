"""Shared services for the train_and_promote DAG.

Follows the same cached services pattern as the rag-platform.
Services are instantiated once per Airflow worker process.
"""

from dataclasses import dataclass
from functools import lru_cache

import mlflow
import structlog
from mlflow.tracking import MlflowClient

from src.config import get_settings
from src.notifications.base import NotificationClient
from src.notifications.factory import build_notification_client

logger = structlog.getLogger(__name__)


@dataclass(frozen=True)
class Services:
    mlflow_client: MlflowClient
    notification: NotificationClient


@lru_cache(maxsize=1)
def get_cached_services() -> Services:
    """Build and cache services for the duration of the worker process."""
    settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)

    services = Services(
        mlflow_client=MlflowClient(tracking_uri=settings.mlflow.tracking_uri),
        notification=build_notification_client(settings.notifications),
    )

    logger.info("DAG services initialised", tracking_uri=settings.mlflow.tracking_uri)
    return services
