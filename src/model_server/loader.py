"""Model loading logic for the model server.

Loads a single model from MLflow at startup and holds it in memory
for the lifetime of the process. This is the standard production pattern.
"""

import mlflow.sklearn
import structlog
from mlflow.exceptions import MlflowException

logger = structlog.getLogger(__name__)


def load_model_from_registry(model_name: str, tracking_uri: str):
    """Load a model from MLflow registry.

    Tries champion alias first, then falls back to latest version.
    Raises if no model can be loaded (fail-fast on startup).
    """
    mlflow.set_tracking_uri(tracking_uri)

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
        logger.error("Failed to load model", model_name=model_name, error=str(e))
        raise RuntimeError(
            f"Cannot start model server: model '{model_name}' not found in registry"
        ) from e
