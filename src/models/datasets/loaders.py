"""Dataset loading abstraction.

Each source type has a loader function. The training pipeline dispatches
to the right loader based on the dataset config's 'source' field.

Data paths in model configs are relative (e.g. 'support-tickets.csv').
The loader resolves them against DATA_ROOT from settings, which can be:
  - A local path: 'data' (default)
  - An S3 URI: 's3://my-bucket/datasets'
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from src.config import get_settings
from src.models.schema import DatasetConfig


def _resolve_path(relative_name: str) -> str:
    """Resolve a dataset name to a full path using data_root from settings.

    If data_root is an S3 URI, returns an S3 path (pandas reads these natively).
    If data_root is local, returns a filesystem path.
    """
    settings = get_settings()
    root = settings.data.root

    if root.startswith("s3://"):
        return f"{root.rstrip('/')}/{relative_name}"

    from pathlib import Path

    return str(Path(root) / relative_name)


def load_sklearn_builtin(
    config: DatasetConfig,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, list[str]]:
    """Load a sklearn built-in dataset."""
    if config.name == "20newsgroups":
        dataset = fetch_20newsgroups(
            subset="all",
            categories=config.categories,
            remove=("headers", "footers", "quotes"),
        )
        target_names = list(dataset.target_names)
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=dataset.target,
        )
        return X_train, X_test, y_train, y_test, target_names

    raise ValueError(f"Unknown sklearn builtin dataset: {config.name}")


def load_csv(
    config: DatasetConfig,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, list[str]]:
    """Load data from a CSV file."""
    path = _resolve_path(config.name)
    df = pd.read_csv(path)

    text_col = config.text_column or "text"
    target_col = config.target_column or "label"

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in {config.name}")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {config.name}")

    texts = df[text_col].tolist()
    labels = pd.Categorical(df[target_col])
    target_names = list(labels.categories)
    y = labels.codes

    if config.categories:
        mask = df[target_col].isin(config.categories)
        texts = df.loc[mask, text_col].tolist()
        labels = pd.Categorical(df.loc[mask, target_col], categories=config.categories)
        target_names = list(labels.categories)
        y = labels.codes

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, target_names


def load_parquet(
    config: DatasetConfig,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, list[str]]:
    """Load data from a Parquet file."""
    path = _resolve_path(config.name)
    df = pd.read_parquet(path)

    text_col = config.text_column or "text"
    target_col = config.target_column or "label"

    texts = df[text_col].tolist()
    labels = pd.Categorical(df[target_col])
    target_names = list(labels.categories)
    y = labels.codes

    if config.categories:
        mask = df[target_col].isin(config.categories)
        texts = df.loc[mask, text_col].tolist()
        labels = pd.Categorical(df.loc[mask, target_col], categories=config.categories)
        target_names = list(labels.categories)
        y = labels.codes

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, target_names


LOADERS = {
    "sklearn_builtin": load_sklearn_builtin,
    "csv": load_csv,
    "parquet": load_parquet,
}


def load_dataset(config: DatasetConfig):
    """Dispatch to the correct loader based on source type."""
    loader = LOADERS.get(config.source)
    if loader is None:
        raise ValueError(
            f"Unknown dataset source '{config.source}'. Available: {list(LOADERS.keys())}"
        )
    return loader(config)
