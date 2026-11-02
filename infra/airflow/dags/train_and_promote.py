"""Train and Promote DAG.

Trains a model, evaluates it against promotion criteria, and if it passes,
sets the @champion alias in the MLflow registry. Serving picks up the new
champion when the model server restarts (see README: Train and Promote).

Trigger manually with a config parameter:
    airflow dags trigger train_and_promote --conf '{"model_config": "newsgroups-classifier"}'

Or via the Airflow UI under DAGs > train_and_promote > Trigger DAG w/ config.
"""

from datetime import datetime, timedelta

from airflow.sdk import dag, task, Param

from model_promotion.evaluation import evaluate_run as _evaluate_run
from model_promotion.promotion import notify as _notify
from model_promotion.promotion import promote as _promote
from model_promotion.promotion import reject as _reject
from model_promotion.training import run_training as _run_training

default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="train_and_promote",
    default_args=default_args,
    description=(
        "Train a model, evaluate against promotion criteria, and promote to champion if it passes."
    ),
    schedule=None,
    max_active_runs=1,
    params={
        "model_config": Param(
            "newsgroups-classifier",
            type="string",
            description="Name of the model config YAML (without .yaml extension)",
        ),
        "triggered_by": Param(
            "manual",
            type="string",
            description="What triggered this run: manual, scheduled, drift",
        ),
    },
    tags=["ml-platform", "training", "promotion"],
)
def train_and_promote():

    @task
    def run_training(model_config: str) -> str:
        """Train the model. Returns run_id."""
        return _run_training(model_config)

    @task
    def evaluate_run(mlflow_run_id: str, model_config: str) -> dict:
        """Evaluate run against promotion criteria. Returns PromotionDecision as dict."""
        return _evaluate_run(mlflow_run_id, model_config)

    @task.branch
    def gate(decision: dict) -> str:
        """Route to promote or reject based on the decision."""
        return "promote" if decision["should_promote"] else "reject"

    @task
    def promote(decision: dict) -> None:
        """Set @champion alias on the winning model version in MLflow."""
        _promote(decision)

    @task
    def reject(decision: dict) -> None:
        """Log the rejection. No state changes."""
        _reject(decision)

    @task(trigger_rule="none_failed_min_one_success")
    def notify(decision: dict, triggered_by: str) -> None:
        """Emit promotion event to all configured notification clients."""
        _notify(decision, triggered_by)

    # --- Wiring ---
    model_config = "{{ params.model_config }}"
    triggered_by = "{{ params.triggered_by }}"

    mlflow_run_id = run_training(model_config)
    decision = evaluate_run(mlflow_run_id, model_config)
    branch = gate(decision)

    promoted = promote(decision)
    rejected = reject(decision)
    notified = notify(decision, triggered_by)

    branch >> [promoted, rejected]
    [promoted, rejected] >> notified


dag_instance = train_and_promote()
