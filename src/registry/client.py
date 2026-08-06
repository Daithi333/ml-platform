"""Model registry client.

Platform-level abstraction over MLflow's model registry. Provides model loading
and listing capabilities that are model-agnostic.

Production pattern: models are loaded at startup (or on first request) and held
in process memory for the lifetime of the container. Reloading is triggered by
container rollout (blue-green/canary), not per-request loading.
"""

import mlflow.sklearn
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
    """Client for interacting with the MLflow model registry.

    Models are loaded into memory and held for the process lifetime.
    This mirrors production behaviour (SageMaker endpoints, TF Serving, etc.)
    where the model lives in container memory and scaling is horizontal.
    """

    def __init__(self, settings: MLflowSettings):
        self._settings = settings
        self._client = MlflowClient(tracking_uri=settings.tracking_uri)
        self._loaded_models: dict[str, object] = {}
        logger.info("Registry client initialised", tracking_uri=settings.tracking_uri)

    def load_model(self, model_name: str, version: str | None = None):
        """Load a model from the registry, or return the already-loaded instance.

        Models are held in process memory after first load. This is the standard
        production pattern — loading is expensive (disk/network I/O + deserialisation),
        inference from RAM is fast.

        Loading priority:
        1. Specific version (if provided)
        2. 'champion' alias (production deployment pattern)
        3. Latest version

        Returns None if no model is registered.
        """
        cache_key = f"{model_name}:{version or 'default'}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        model = self._fetch_model(model_name, version)
        if model is not None:
            self._loaded_models[cache_key] = model

        return model

    def _fetch_model(self, model_name: str, version: str | None = None):
        """Fetch model from MLflow (network/disk I/O). Called once per model."""
        if version:
            try:
                model_uri = f"models:/{model_name}/{version}"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info("Model loaded", model_name=model_name, version=version)
                return model
            except MlflowException as e:
                logger.error(
                    "Failed to load model version",
                    model_name=model_name,
                    version=version,
                    error=str(e),
                )
                raise

        try:
            model_uri = f"models:/{model_name}@champion"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded", model_name=model_name, alias="champion")
            return model
        except MlflowException:
            pass

        try:
            model_uri = f"models:/{model_name}/latest"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded", model_name=model_name, version="latest")
            return model
        except MlflowException as e:
            logger.warning("No model found in registry", model_name=model_name, error=str(e))
            return None

    def reload_model(self, model_name: str) -> None:
        """Force reload a model from the registry.

        In production, this is triggered by deployment automation (not user requests).
        In local dev, it's useful after retraining to pick up the new version
        without restarting the container.
        """
        keys_to_remove = [k for k in self._loaded_models if k.startswith(f"{model_name}:")]
        for key in keys_to_remove:
            del self._loaded_models[key]

        # Pre-load the default version
        self.load_model(model_name)
        logger.info("Model reloaded", model_name=model_name)

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
