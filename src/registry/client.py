"""Model registry client.

Read-only view of the MLflow model registry for the platform API (listing and
inspecting models and versions). Model loading lives in the model server, which
loads the @champion version at startup.
"""

import structlog
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import MLflowSettings
from src.serving.schemas.registry import (
    ModelVersionDetail,
    ModelVersionSummary,
    RegisteredModelDetail,
    RegisteredModelSummary,
)

logger = structlog.getLogger(__name__)


class RegistryClient:
    """Client for browsing the MLflow model registry."""

    def __init__(self, settings: MLflowSettings):
        self._settings = settings
        self._client = MlflowClient(tracking_uri=settings.tracking_uri)
        logger.info("Registry client initialised", tracking_uri=settings.tracking_uri)

    def list_models(self) -> list[RegisteredModelSummary]:
        """List all registered models in the registry."""
        registered_models = self._client.search_registered_models()
        return [
            RegisteredModelSummary(
                name=rm.name,
                description=rm.description or "",
                latest_versions=[
                    ModelVersionSummary(
                        version=v.version,
                        status=v.status,
                        run_id=v.run_id or "",
                    )
                    for v in (rm.latest_versions or [])
                ],
            )
            for rm in registered_models
        ]

    def get_model_info(self, model_name: str) -> RegisteredModelDetail | None:
        """Get detailed info about a specific registered model."""
        try:
            model = self._client.get_registered_model(model_name)
            versions = self._client.search_model_versions(f"name='{model_name}'")
            return RegisteredModelDetail(
                name=model.name,
                description=model.description or "",
                tags=dict(model.tags) if model.tags else {},
                versions=[
                    ModelVersionDetail(
                        version=v.version,
                        status=v.status,
                        run_id=v.run_id or "",
                        creation_timestamp=v.creation_timestamp,
                        aliases=v.aliases if hasattr(v, "aliases") else [],
                    )
                    for v in versions
                ],
            )
        except MlflowException:
            return None
