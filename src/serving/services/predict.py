"""Prediction service — proxies inference requests to model server containers.

The platform API does not load models itself. It routes to the correct
model server based on model name. Each model server is a separate container
that holds one model in memory.
"""

import httpx
import structlog

from src.exceptions import ExternalServiceError, NotFoundError
from src.serving.schemas.predict import PredictionResponse, PredictionResult

logger = structlog.getLogger(__name__)

# Model name -> model server base URL
# In production this comes from service discovery (ECS, K8s, Consul)
# Locally it maps to compose service names
MODEL_SERVER_REGISTRY: dict[str, str] = {
    "newsgroups-classifier": "http://model-newsgroups:8001",
}


async def run_prediction(
    model_name: str,
    texts: list[str],
    version: str | None = None,
) -> PredictionResponse:
    """Proxy a prediction request to the appropriate model server."""
    base_url = MODEL_SERVER_REGISTRY.get(model_name)
    if base_url is None:
        raise NotFoundError(resource="Model server", identifier=model_name)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/predict",
                json={"texts": texts},
            )
    except httpx.ConnectError as e:
        raise ExternalServiceError(
            service=f"model-server/{model_name}",
            message=f"Connection failed: {e}",
        ) from e

    if response.status_code != 200:
        raise ExternalServiceError(
            service=f"model-server/{model_name}",
            message=f"HTTP {response.status_code}: {response.text}",
        )

    data = response.json()
    logger.info(
        "Prediction proxied",
        model_name=model_name,
        num_texts=len(texts),
    )

    return PredictionResponse(
        predictions=[
            PredictionResult(category=p["label"], confidence=p["confidence"])
            for p in data["predictions"]
        ],
        model_name=data["model_name"],
        model_version=version,
    )
