from fastapi import APIRouter

from src.serving.dependencies import SettingsDep
from src.serving.schemas.health import HealthResponse
from src.serving.services.health import get_health

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(
    settings: SettingsDep,
) -> HealthResponse:
    """Health check endpoint with liveness status of backing services."""
    return await get_health(
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
    )
