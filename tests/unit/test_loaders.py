"""Unit tests for dataset loaders — pure logic, no external I/O."""

import pytest

from src.exceptions import NotFoundError, ValidationError
from src.models.schema import DatasetConfig


class TestResolveDatasetConfig:
    """Test config validation and loader dispatch."""

    def test_invalid_source_raises_validation_error(self):
        """Should raise ValidationError for unknown source type."""
        from src.models.datasets.loaders import load_dataset

        config = DatasetConfig(
            source="unknown_source",
            name="test",
            labels=["a", "b"],
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValidationError) as exc_info:
            load_dataset(config)

        assert "unknown_source" in exc_info.value.message

    def test_unknown_sklearn_dataset_raises_validation_error(self):
        """Should raise ValidationError for unknown sklearn builtin."""
        from src.models.datasets.loaders import load_sklearn_builtin

        config = DatasetConfig(
            source="sklearn_builtin",
            name="nonexistent_dataset",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(ValidationError) as exc_info:
            load_sklearn_builtin(config)

        assert "nonexistent_dataset" in exc_info.value.message

    def test_csv_file_not_found_raises_not_found_error(self):
        """Should raise NotFoundError when CSV file does not exist."""
        from src.models.datasets.loaders import load_csv

        config = DatasetConfig(
            source="csv",
            name="nonexistent_file.csv",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(NotFoundError):
            load_csv(config)

    def test_parquet_file_not_found_raises_not_found_error(self):
        """Should raise NotFoundError when Parquet file does not exist."""
        from src.models.datasets.loaders import load_parquet

        config = DatasetConfig(
            source="parquet",
            name="nonexistent_file.parquet",
            test_size=0.2,
            random_state=42,
        )

        with pytest.raises(NotFoundError):
            load_parquet(config)


class TestModelConfigSchema:
    """Test model config YAML schema validation."""

    def test_valid_config_parses(self):
        """Should parse a valid model config."""
        from src.models.schema import ModelConfig

        config = ModelConfig(
            name="test-model",
            experiment="test-experiment",
            architecture="text_classifier",
            dataset=DatasetConfig(
                source="sklearn_builtin",
                name="20newsgroups",
                categories=["sci.space"],
                test_size=0.2,
                random_state=42,
            ),
            params={"max_features": 5000},
            labels=["sci.space"],
        )

        assert config.name == "test-model"
        assert config.dataset.source == "sklearn_builtin"

    def test_missing_required_fields_raises(self):
        """Should reject config missing required fields."""
        from pydantic import ValidationError as PydanticValidationError
        from src.models.schema import ModelConfig

        with pytest.raises(PydanticValidationError):
            ModelConfig(
                name="test",
                # missing experiment, architecture, dataset, labels
            )
