from fastapi import APIRouter

from src.serving.dependencies import RegistryDep
from src.serving.schemas.registry import RegisteredModelDetail, RegisteredModelSummary
from src.serving.services.registry import get_model_details, get_registered_models

router = APIRouter(prefix="/registry", tags=["Registry"])


@router.get("/models", response_model=list[RegisteredModelSummary])
async def list_models(registry: RegistryDep) -> list[RegisteredModelSummary]:
    """List all models in the registry."""
    return await get_registered_models(registry)


@router.get("/models/{model_name}", response_model=RegisteredModelDetail)
async def model_details(model_name: str, registry: RegistryDep) -> RegisteredModelDetail:
    """Get detailed information about a registered model."""
    return await get_model_details(registry, model_name)
