from fastapi import APIRouter, Query

from src.serving.dependencies import RegistryDep
from src.serving.schemas.predict import PredictionRequest, PredictionResponse
from src.serving.services.predict import run_prediction

router = APIRouter(prefix="/models", tags=["Inference"])


@router.post("/{model_name}/predict", response_model=PredictionResponse)
async def predict(
    model_name: str,
    request: PredictionRequest,
    registry: RegistryDep,
    version: str | None = Query(None, description="Specific model version to use"),
) -> PredictionResponse:
    """Run inference against a registered model."""
    return await run_prediction(
        registry=registry,
        model_name=model_name,
        texts=request.texts,
        version=version,
    )
