"""Training task for the train_and_promote DAG."""

import structlog

from src.exceptions import TrainingError
from src.logs import setup_logging
from src.models.schema import discover_model_configs, load_model_config
from src.models.train import train_model

setup_logging()
logger = structlog.getLogger(__name__)

CONFIGS_DIR = "src/models/configs"


def run_training(model_config: str) -> str:
    """Train the model and return the MLflow run_id (pushed to XCom by the @task wrapper)."""
    logger.info("Starting training task", model_config=model_config)

    config_path = discover_model_configs(CONFIGS_DIR).get(model_config)
    if config_path is None:
        raise TrainingError(message=f"Model config '{model_config}' not found in {CONFIGS_DIR}")

    config = load_model_config(config_path)

    try:
        run_id = train_model(config)
    except Exception as e:
        raise TrainingError(message=f"Training failed for '{model_config}': {e}") from e

    logger.info("Training complete", model_config=model_config, run_id=run_id)
    return run_id
