from src.exceptions import NotFoundError
from src.registry.client import RegistryClient
from src.serving.schemas.registry import RegisteredModelDetail, RegisteredModelSummary


async def get_registered_models(registry: RegistryClient) -> list[RegisteredModelSummary]:
    """List all models in the registry."""
    return registry.list_models()


async def get_model_details(registry: RegistryClient, model_name: str) -> RegisteredModelDetail:
    """Get detailed information about a registered model."""
    info = registry.get_model_info(model_name)
    if info is None:
        raise NotFoundError(resource="Model", identifier=model_name)
    return info
