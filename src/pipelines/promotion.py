"""Promotion evaluation logic.

Pure business logic — no Airflow dependencies. Callable from the DAG,
from scripts, or from tests without needing a running Airflow instance.
"""

import structlog
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.exceptions import MetricsMissingError, ModelNotRegisteredError
from src.models.schema import PromotionConfig
from src.pipelines.schemas import EvaluationResult, PromotionDecision

logger = structlog.getLogger(__name__)

_REQUIRED_METRICS = {"accuracy", "f1_macro", "f1_weighted"}


def fetch_run_metrics(client: MlflowClient, run_id: str, model_name: str) -> EvaluationResult:
    """Fetch metrics and registered version for a completed MLflow run."""
    run = client.get_run(run_id)
    metrics = run.data.metrics

    missing = list(_REQUIRED_METRICS - metrics.keys())
    if missing:
        raise MetricsMissingError(run_id=run_id, missing_metrics=missing)

    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise ModelNotRegisteredError(run_id=run_id, model_name=model_name)

    return EvaluationResult(
        run_id=run_id,
        model_name=model_name,
        model_version=versions[0].version,
        accuracy=metrics["accuracy"],
        f1_macro=metrics["f1_macro"],
        f1_weighted=metrics["f1_weighted"],
    )


def fetch_champion_metrics(client: MlflowClient, model_name: str) -> EvaluationResult | None:
    """Fetch metrics for the current champion version, if one exists."""
    try:
        model_version = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        logger.info("No champion found — first promotion candidate", model_name=model_name)
        return None

    run = client.get_run(model_version.run_id)
    metrics = run.data.metrics

    missing = list(_REQUIRED_METRICS - metrics.keys())
    if missing:
        # Champion exists but lacks metrics — log and treat as no champion
        # rather than failing the pipeline (graceful degradation)
        logger.warning(
            "Champion run missing metrics — treating as no champion",
            model_name=model_name,
            version=model_version.version,
            missing=missing,
        )
        return None

    return EvaluationResult(
        run_id=model_version.run_id,
        model_name=model_name,
        model_version=model_version.version,
        accuracy=metrics["accuracy"],
        f1_macro=metrics["f1_macro"],
        f1_weighted=metrics["f1_weighted"],
    )


def evaluate_promotion(
    candidate: EvaluationResult,
    champion: EvaluationResult | None,
    config: PromotionConfig,
) -> PromotionDecision:
    """Evaluate whether the candidate model should be promoted.

    Checks (in order):
    1. Absolute accuracy threshold
    2. Absolute F1 macro threshold
    3. Must improve on champion's primary_metric if require_improvement=True
       and a champion exists
    """
    if candidate.accuracy < config.min_accuracy:
        return PromotionDecision(
            should_promote=False,
            reason=(
                f"Accuracy {candidate.accuracy:.4f} below minimum "
                f"threshold {config.min_accuracy:.4f}"
            ),
            candidate=candidate,
            champion=champion,
        )

    if candidate.f1_macro < config.min_f1_macro:
        return PromotionDecision(
            should_promote=False,
            reason=(
                f"F1 macro {candidate.f1_macro:.4f} below minimum "
                f"threshold {config.min_f1_macro:.4f}"
            ),
            candidate=candidate,
            champion=champion,
        )

    if config.require_improvement and champion is not None:
        candidate_score = getattr(candidate, config.primary_metric)
        champion_score = getattr(champion, config.primary_metric)

        if candidate_score <= champion_score:
            return PromotionDecision(
                should_promote=False,
                reason=(
                    f"Candidate {config.primary_metric} {candidate_score:.4f} does not "
                    f"improve on current champion {champion_score:.4f}"
                ),
                candidate=candidate,
                champion=champion,
            )

    candidate_score = getattr(candidate, config.primary_metric)
    reason = (
        f"Passed thresholds (accuracy={candidate.accuracy:.4f}, f1_macro={candidate.f1_macro:.4f})"
    )
    if champion is not None:
        champion_score = getattr(champion, config.primary_metric)
        reason += (
            f" and improved {config.primary_metric} "
            f"from {champion_score:.4f} to {candidate_score:.4f}"
        )

    logger.info(
        "Promotion approved",
        model_name=candidate.model_name,
        version=candidate.model_version,
        reason=reason,
    )

    return PromotionDecision(
        should_promote=True,
        reason=reason,
        candidate=candidate,
        champion=champion,
    )
