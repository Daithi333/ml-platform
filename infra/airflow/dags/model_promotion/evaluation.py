"""Evaluation task for the train_and_promote DAG."""

import structlog

from src.exceptions import PromotionConfigMissingError
from src.logs import setup_logging
from src.models.schema import discover_model_configs, load_model_config
from src.pipelines.promotion import evaluate_promotion, fetch_champion_metrics, fetch_run_metrics

from .common import get_cached_services

setup_logging()
logger = structlog.getLogger(__name__)

CONFIGS_DIR = "src/models/configs"


def evaluate_run(run_id: str, model_config: str) -> dict:
    """Evaluate the trained run against promotion criteria.

    Returns PromotionDecision as a dict (XCom-serialisable).
    """
    logger.info("Evaluating run", run_id=run_id, model_config=model_config)

    configs = discover_model_configs(CONFIGS_DIR)
    config_path = configs.get(model_config)
    config = load_model_config(config_path)

    if config.promotion is None:
        raise PromotionConfigMissingError(model_name=config.name)

    svc = get_cached_services()

    candidate = fetch_run_metrics(svc.mlflow_client, run_id, config.name)
    champion = fetch_champion_metrics(svc.mlflow_client, config.name)
    decision = evaluate_promotion(candidate, champion, config.promotion)

    logger.info(
        "Evaluation complete",
        model=config.name,
        should_promote=decision.should_promote,
        reason=decision.reason,
    )

    return decision.model_dump()
