"""Dataset loading abstraction.

Each source type has a loader function. The training pipeline dispatches
to the right loader based on the dataset config's 'source' field.

Data paths in model configs are relative (e.g. 'support-tickets.csv').
The loader resolves them against DATA_ROOT from settings, which can be:
  - A local path: 'data' (default)
  - An S3 URI: 's3://my-bucket/datasets'
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from src.config import get_settings
from src.exceptions import NotFoundError, ValidationError
from src.models.schema import DatasetConfig

# Type alias for the standard return shape of all loaders
DataSplit = tuple[list[str], list[str], NDArray[np.intp], NDArray[np.intp], list[str]]


def _resolve_path(relative_name: str) -> str:
    """Resolve a dataset name to a full path using data_root from settings.

    If data_root is an S3 URI, returns an S3 path (pandas reads these natively).
    If data_root is local, returns a filesystem path.
    """
    settings = get_settings()
    root: str = settings.data.root

    if root.startswith("s3://"):
        return f"{root.rstrip('/')}/{relative_name}"

    return str(Path(root) / relative_name)


def load_sklearn_builtin(config: DatasetConfig) -> DataSplit:
    """Load a sklearn built-in dataset."""
    if config.name != "20newsgroups":
        raise ValidationError(
            message=f"Unknown sklearn builtin dataset: {config.name}",
            details={"dataset": config.name, "source": config.source},
        )

    dataset: Bunch = fetch_20newsgroups(  # type: ignore[assignment]
        subset="all",
        categories=config.categories,
        remove=("headers", "footers", "quotes"),
    )
    target_names: list[str] = list(dataset.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=dataset.target,
    )
    return (
        list(X_train),
        list(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
        target_names,
    )


def load_csv(config: DatasetConfig) -> DataSplit:
    """Load data from a CSV file."""
    path: str = _resolve_path(config.name)

    if not path.startswith("s3://") and not Path(path).exists():
        raise NotFoundError(resource="Dataset file", identifier=path)

    df: pd.DataFrame = pd.read_csv(path)
    return _split_dataframe(df, config)


def load_parquet(config: DatasetConfig) -> DataSplit:
    """Load data from a Parquet file."""
    path: str = _resolve_path(config.name)

    if not path.startswith("s3://") and not Path(path).exists():
        raise NotFoundError(resource="Dataset file", identifier=path)

    df: pd.DataFrame = pd.read_parquet(path)
    return _split_dataframe(df, config)


def _split_dataframe(df: pd.DataFrame, config: DatasetConfig) -> DataSplit:
    """Common logic for splitting a dataframe into train/test sets."""
    text_col: str = config.text_column or "text"
    target_col: str = config.target_column or "label"

    if text_col not in df.columns:
        raise ValidationError(
            message=f"Text column '{text_col}' not found in dataset",
            details={"column": text_col, "available": list(df.columns)},
        )
    if target_col not in df.columns:
        raise ValidationError(
            message=f"Target column '{target_col}' not found in dataset",
            details={"column": target_col, "available": list(df.columns)},
        )

    if config.categories:
        mask = df[target_col].isin(config.categories)
        df = pd.DataFrame(df[mask])

    texts: list[str] = df[text_col].tolist()
    labels = pd.Categorical(df[target_col], categories=config.categories or None)
    target_names: list[str] = list(labels.categories)
    y: NDArray[np.intp] = np.asarray(labels.codes)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    return (
        list(X_train),
        list(X_test),
        np.asarray(y_train),
        np.asarray(y_test),
        target_names,
    )


LOADERS: dict[str, Callable[[DatasetConfig], DataSplit]] = {
    "sklearn_builtin": load_sklearn_builtin,
    "csv": load_csv,
    "parquet": load_parquet,
}


def load_dataset(config: DatasetConfig) -> DataSplit:
    """Dispatch to the correct loader based on source type."""
    loader = LOADERS.get(config.source)
    if loader is None:
        raise ValidationError(
            message=f"Unknown dataset source '{config.source}'",
            details={"source": config.source, "available": list(LOADERS.keys())},
        )
    return loader(config)
