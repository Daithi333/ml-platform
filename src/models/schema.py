"""Model configuration schema.

Defines the structure of model YAML config files. Validated with Pydantic
so misconfigurations are caught before training, not during.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """How to load and split the training data."""

    source: str = Field(description="Loader type: 'sklearn_builtin', 'csv', 'parquet'")
    name: str = Field(description="Dataset identifier (e.g. '20newsgroups', or a file path)")
    categories: list[str] | None = Field(
        default=None, description="Subset of classes to use (optional)"
    )
    target_column: str | None = Field(
        default=None, description="Column name for labels (csv/parquet sources)"
    )
    text_column: str | None = Field(
        default=None, description="Column name for input text (csv/parquet sources)"
    )
    test_size: float = 0.2
    random_state: int = 42


class ModelConfig(BaseModel):
    """Full model configuration — one YAML file = one trainable model."""

    name: str = Field(description="Registered model name in MLflow")
    experiment: str = Field(description="MLflow experiment name")
    architecture: str = Field(description="Architecture to use (maps to src/models/architectures/)")
    dataset: DatasetConfig
    params: dict[str, Any] = Field(
        default_factory=dict, description="Architecture-specific hyperparameters"
    )
    labels: list[str] = Field(description="Ordered output labels for predictions")


def load_model_config(config_path: str | Path) -> ModelConfig:
    """Load and validate a model config from YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return ModelConfig(**raw)


def discover_model_configs(configs_dir: str | Path = "src/models/configs") -> dict[str, Path]:
    """Find all model config YAML files and return a name -> path mapping."""
    configs_path = Path(configs_dir)
    if not configs_path.exists():
        return {}

    configs = {}
    for yaml_file in configs_path.glob("*.yaml"):
        with open(yaml_file) as f:
            raw = yaml.safe_load(f)
        if "name" in raw:
            configs[raw["name"]] = yaml_file

    return configs
