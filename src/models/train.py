"""Generic model training entry point.

Reads a model config YAML, loads the dataset, builds the architecture,
trains, evaluates, and registers in MLflow. Model-agnostic — the config
drives everything.

Usage:
    uv run python -m src.models.train --config newsgroups-classifier
    uv run python -m src.models.train --config-path src/models/configs/custom.yaml
"""

import argparse
import importlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import structlog
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import get_settings
from src.logs import setup_logging
from src.models.datasets.loaders import load_dataset
from src.models.schema import ModelConfig, discover_model_configs, load_model_config

setup_logging()
logger = structlog.getLogger(__name__)

CONFIGS_DIR = Path("src/models/configs")


def get_architecture(architecture_name: str):
    """Dynamically load an architecture module by name."""
    module_path = f"src.models.architectures.{architecture_name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Architecture '{architecture_name}' not found. Expected module at {module_path}"
        ) from e

    if not hasattr(module, "build_pipeline"):
        raise ValueError(
            f"Architecture '{architecture_name}' must define a build_pipeline() function"
        )
    return module


def train_model(config: ModelConfig) -> None:
    """Train a model end-to-end from a validated config."""
    settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(config.experiment)

    logger.info("Starting training", model=config.name, architecture=config.architecture)

    # Load data
    X_train, X_test, y_train, y_test, target_names = load_dataset(config.dataset)
    logger.info(
        "Data loaded",
        train_samples=len(X_train),
        test_samples=len(X_test),
        categories=target_names,
    )

    # Build pipeline
    architecture = get_architecture(config.architecture)
    pipeline = architecture.build_pipeline(config.params)

    # Train and evaluate within an MLflow run
    with mlflow.start_run() as run:
        mlflow.log_params(config.params)
        mlflow.set_tag("architecture", config.architecture)
        mlflow.set_tag("dataset_source", config.dataset.source)
        mlflow.set_tag("dataset_name", config.dataset.name)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
            }
        )

        report = classification_report(y_test, y_pred, target_names=target_names)
        logger.info("Training complete", accuracy=accuracy, f1_macro=f1_macro)
        logger.info("Classification report", report=report)

        # Register model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=config.name,
            input_example=X_test[:1],
        )

        logger.info(
            "Model registered",
            run_id=run.info.run_id,
            model_name=config.name,
        )


def main():
    parser = argparse.ArgumentParser(description="Train a model from config")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        help="Model config name (looks up in src/models/configs/<name>.yaml)",
    )
    group.add_argument(
        "--config-path",
        help="Explicit path to a model config YAML file",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available model configs and exit",
    )
    args = parser.parse_args()

    if args.list:
        configs = discover_model_configs(CONFIGS_DIR)
        if not configs:
            print("No model configs found in", CONFIGS_DIR)
        else:
            print("Available model configs:")
            for name, path in configs.items():
                print(f"  {name} -> {path}")
        return

    if args.config:
        config_path = CONFIGS_DIR / f"{args.config}.yaml"
    else:
        config_path = Path(args.config_path)

    model_config = load_model_config(config_path)
    train_model(model_config)


if __name__ == "__main__":
    main()
