import structlog

from src.exceptions import NotFoundError, ValidationError
from src.models.schema import discover_model_configs, load_model_config
from src.registry.client import RegistryClient
from src.serving.schemas.predict import PredictionResponse, PredictionResult

logger = structlog.getLogger(__name__)


def get_labels_for_model(model_name: str) -> list[str] | None:
    """Look up output labels from the model's config YAML."""
    configs = discover_model_configs()
    config_path = configs.get(model_name)
    if config_path is None:
        return None
    config = load_model_config(config_path)
    return config.labels


async def run_prediction(
    registry: RegistryClient,
    model_name: str,
    texts: list[str],
    version: str | None = None,
) -> PredictionResponse:
    """Load model from registry and run inference."""
    model = registry.load_model(model_name, version=version)
    if model is None:
        raise NotFoundError(resource="Model", identifier=model_name)

    labels = get_labels_for_model(model_name)
    if labels is None:
        raise ValidationError(
            message=f"No config found for model '{model_name}'",
            details={"model_name": model_name},
        )

    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)

    results = []
    for pred_idx, probs in zip(predictions, probabilities):
        category = labels[pred_idx]
        confidence = float(probs[pred_idx])
        results.append(PredictionResult(category=category, confidence=confidence))

    logger.info(
        "Prediction complete",
        model_name=model_name,
        version=version,
        num_texts=len(texts),
    )

    return PredictionResponse(
        predictions=results,
        model_name=model_name,
        model_version=version,
    )
