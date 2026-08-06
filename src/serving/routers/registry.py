from fastapi import APIRouter

from src.serving.dependencies import RegistryDep
from src.serving.schemas.registry import (
    ModelReloadResponse,
    RegisteredModelDetail,
    RegisteredModelSummary,
)
from src.serving.services.registry import (
    get_model_details,
    get_registered_models,
    reload_model,
)

router = APIRouter(prefix="/registry", tags=["Registry"])


@router.get("/models", response_model=list[RegisteredModelSummary])
async def list_models(registry: RegistryDep) -> list[RegisteredModelSummary]:
    """List all models in the registry."""
    return await get_registered_models(registry)


@router.get("/models/{model_name}", response_model=RegisteredModelDetail)
async def model_details(model_name: str, registry: RegistryDep) -> RegisteredModelDetail:
    """Get detailed information about a registered model."""
    return await get_model_details(registry, model_name)


@router.post("/models/{model_name}/reload", response_model=ModelReloadResponse)
async def model_reload(model_name: str, registry: RegistryDep) -> ModelReloadResponse:
    """Clear the cached model, forcing a reload on next prediction."""
    return await reload_model(registry, model_name)
