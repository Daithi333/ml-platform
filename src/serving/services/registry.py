import structlog

from src.exceptions import NotFoundError
from src.registry.client import RegistryClient
from src.serving.schemas.registry import (
    ModelReloadResponse,
    RegisteredModelDetail,
    RegisteredModelSummary,
)

logger = structlog.getLogger(__name__)


async def get_registered_models(registry: RegistryClient) -> list[RegisteredModelSummary]:
    """List all models in the registry."""
    return registry.list_models()


async def get_model_details(registry: RegistryClient, model_name: str) -> RegisteredModelDetail:
    """Get detailed information about a registered model."""
    info = registry.get_model_info(model_name)
    if info is None:
        raise NotFoundError(resource="Model", identifier=model_name)
    return info


async def reload_model(registry: RegistryClient, model_name: str) -> ModelReloadResponse:
    """Reload a model from the registry into memory.

    In production, model updates are handled by container rollout (blue-green/canary).
    This endpoint exists for local dev convenience after retraining.
    """
    registry.reload_model(model_name)
    logger.info("Model reloaded from registry", model_name=model_name)
    return ModelReloadResponse(
        status="ok",
        message=f"Model '{model_name}' reloaded from registry.",
    )
