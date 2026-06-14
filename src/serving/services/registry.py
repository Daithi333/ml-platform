import structlog

from src.exceptions import NotFoundError
from src.registry.client import RegistryClient

logger = structlog.getLogger(__name__)


async def get_registered_models(registry: RegistryClient) -> list[dict]:
    """List all models in the registry."""
    return registry.list_models()


async def get_model_details(registry: RegistryClient, model_name: str) -> dict:
    """Get detailed information about a registered model."""
    info = registry.get_model_info(model_name)
    if info is None:
        raise NotFoundError(resource="Model", identifier=model_name)
    return info


async def reload_model(registry: RegistryClient, model_name: str) -> dict:
    """Clear cached model, forcing reload from registry on next prediction."""
    registry.clear_cache(model_name)
    logger.info("Model cache cleared", model_name=model_name)
    return {
        "status": "ok",
        "message": f"Cache cleared for '{model_name}'. Next prediction will load fresh.",
    }
