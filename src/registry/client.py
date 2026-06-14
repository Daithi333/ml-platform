"""Model registry client.

Platform-level abstraction over MLflow's model registry. Provides model loading,
caching, and listing capabilities that are model-agnostic.
"""

import mlflow
import structlog
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import MLflowSettings

logger = structlog.getLogger(__name__)


class RegistryClient:
    """Client for interacting with the MLflow model registry."""

    def __init__(self, settings: MLflowSettings):
        self._settings = settings
        self._model_cache: dict[str, object] = {}
        self._client = MlflowClient(tracking_uri=settings.tracking_uri)
        mlflow.set_tracking_uri(settings.tracking_uri)
        logger.info("Registry client initialised", tracking_uri=settings.tracking_uri)

    def load_model(self, model_name: str, version: str | None = None):
        """Load a model from the registry by name.

        Loading priority:
        1. Specific version (if provided)
        2. 'champion' alias (production deployment pattern)
        3. Latest version

        Models are cached in memory. Call clear_cache() after retraining.
        """
        cache_key = f"{model_name}:{version or 'default'}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model = None

        if version:
            try:
                model_uri = f"models:/{model_name}/{version}"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info("Model loaded", model_name=model_name, version=version)
            except MlflowException as e:
                logger.error(
                    "Failed to load model version",
                    model_name=model_name,
                    version=version,
                    error=str(e),
                )
                raise

        if model is None:
            try:
                model_uri = f"models:/{model_name}@champion"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info("Model loaded", model_name=model_name, alias="champion")
            except MlflowException:
                pass

        if model is None:
            try:
                model_uri = f"models:/{model_name}/latest"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info("Model loaded", model_name=model_name, version="latest")
            except MlflowException as e:
                logger.warning("No model found in registry", model_name=model_name, error=str(e))
                return None

        self._model_cache[cache_key] = model
        return model

    def list_models(self) -> list[dict]:
        """List all registered models in the registry."""
        registered_models = self._client.search_registered_models()
        return [
            {
                "name": rm.name,
                "description": rm.description or "",
                "latest_versions": [
                    {
                        "version": v.version,
                        "status": v.status,
                        "run_id": v.run_id,
                    }
                    for v in (rm.latest_versions or [])
                ],
            }
            for rm in registered_models
        ]

    def get_model_info(self, model_name: str) -> dict | None:
        """Get detailed info about a specific registered model."""
        try:
            model = self._client.get_registered_model(model_name)
            versions = self._client.search_model_versions(f"name='{model_name}'")
            return {
                "name": model.name,
                "description": model.description or "",
                "tags": dict(model.tags) if model.tags else {},
                "versions": [
                    {
                        "version": v.version,
                        "status": v.status,
                        "run_id": v.run_id,
                        "creation_timestamp": v.creation_timestamp,
                        "aliases": v.aliases if hasattr(v, "aliases") else [],
                    }
                    for v in versions
                ],
            }
        except MlflowException:
            return None

    def clear_cache(self, model_name: str | None = None) -> None:
        """Clear cached models. If model_name is provided, only clear that model."""
        if model_name:
            keys_to_remove = [k for k in self._model_cache if k.startswith(f"{model_name}:")]
            for key in keys_to_remove:
                del self._model_cache[key]
            logger.info("Model cache cleared", model_name=model_name)
        else:
            self._model_cache.clear()
            logger.info("All model caches cleared")
