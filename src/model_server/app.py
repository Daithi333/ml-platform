"""Model Server — generic inference container.

Loads exactly one model at startup (determined by MODEL_NAME env var) and
serves predictions. Deployed as N containers — one per model.

This mirrors the production pattern where each model runs in its own
container with horizontal scaling per-model based on traffic/SLA needs.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.model_server.config import get_model_server_settings
from src.model_server.loader import load_model_from_registry
from src.model_server.schemas import (
    HealthResponse,
    PredictRequest,
    PredictionResult,
    PredictResponse,
)
from src.models.schema import discover_model_configs, load_model_config

logger = structlog.getLogger(__name__)


def _get_labels(model_name: str) -> list[str]:
    """Load output labels from the model's YAML config."""
    configs = discover_model_configs()
    config_path = configs.get(model_name)
    if config_path is None:
        raise RuntimeError(f"No config YAML found for model '{model_name}'")
    config = load_model_config(config_path)
    return config.labels


def create_app(
    model=None, model_name: str | None = None, labels: list[str] | None = None
) -> FastAPI:
    """App factory. Accepts optional pre-loaded model for testing.

    In production: called with no args, lifespan loads from MLflow.
    In tests: called with model/model_name/labels to skip MLflow entirely.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        if model is not None:
            # Test mode: model already provided
            app.state.model = model
            app.state.model_name = model_name
            app.state.labels = labels
        else:
            # Production mode: load from registry
            settings = get_model_server_settings()
            app.state.model = load_model_from_registry(
                settings.model_name, settings.mlflow_tracking_uri
            )
            app.state.model_name = settings.model_name
            app.state.labels = _get_labels(settings.model_name)

        logger.info("Model server ready", model_name=app.state.model_name)
        yield
        logger.info("Model server shutting down", model_name=app.state.model_name)

    application = FastAPI(
        title="ML Model Server",
        description="Generic model inference server (one model per container)",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Health check — confirms model is loaded and ready."""
        return HealthResponse(
            status="ok",
            model_name=request.app.state.model_name,
            model_loaded=request.app.state.model is not None,
        )

    @application.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, body: PredictRequest) -> PredictResponse:
        """Run inference on the loaded model."""
        loaded_model = request.app.state.model
        name: str = request.app.state.model_name
        model_labels: list[str] = request.app.state.labels

        predictions = loaded_model.predict(body.texts)
        probabilities = loaded_model.predict_proba(body.texts)

        results = []
        for pred_idx, probs in zip(predictions, probabilities):
            label = model_labels[pred_idx]
            confidence = float(probs[pred_idx])
            results.append(PredictionResult(label=label, confidence=confidence))

        return PredictResponse(model_name=name, predictions=results)

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Catch-all error handler."""
        logger.error("Unhandled exception in model server", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "INTERNAL_ERROR", "message": "Model server error"},
        )

    return application


# Default app instance for production (uvicorn src.model_server.app:app)
app = create_app()


if __name__ == "__main__":
    settings = get_model_server_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
