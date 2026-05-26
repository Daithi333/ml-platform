import structlog
from typing import Literal
from src.serving.schemas.health import HealthResponse

logger = structlog.getLogger(__name__)


async def get_health(
    version: str,
    environment: str,
    service_name: str,
) -> HealthResponse:
    """Run all health checks and return the aggregated response."""
    services = {}

    # check services health

    any_unhealthy = any(s.status == "unhealthy" for s in services.values())
    any_degraded = any(s.status == "degraded" for s in services.values())

    status: Literal["ok", "degraded", "error"] = "ok"
    if any_unhealthy:
        status = "error"
    elif any_degraded:
        status = "degraded"

    return HealthResponse(
        status=status,
        version=version,
        environment=environment,
        service_name=service_name,
        services=services,
    )
