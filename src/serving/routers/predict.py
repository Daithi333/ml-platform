from fastapi import APIRouter, Query

from src.serving.schemas.predict import PredictionRequest, PredictionResponse
from src.serving.services.predict import run_prediction

router = APIRouter(prefix="/models", tags=["Inference"])


@router.post("/{model_name}/predict", response_model=PredictionResponse)
async def predict(
    model_name: str,
    request: PredictionRequest,
    version: str | None = Query(None, description="Specific model version to use"),
) -> PredictionResponse:
    """Proxy inference request to the model's dedicated server container."""
    return await run_prediction(
        model_name=model_name,
        texts=request.texts,
        version=version,
    )
