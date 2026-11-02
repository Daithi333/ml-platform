from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    CONFLICT = "CONFLICT"
    NOT_FOUND = "NOT_FOUND"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """Base for all HTTP/API errors."""

    def __init__(self, message: str, code: ErrorCode, details: dict[str, Any] = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} with id '{identifier}' not found",
            code=ErrorCode.NOT_FOUND,
            details={"resource": resource, "identifier": identifier},
        )


class ConflictError(AppError):
    def __init__(self, message: str, details: dict[str, Any] = None):
        super().__init__(message=message, code=ErrorCode.CONFLICT, details=details)


class ValidationError(AppError):
    def __init__(self, message: str, details: dict[str, Any] = None):
        super().__init__(message=message, code=ErrorCode.VALIDATION_ERROR, details=details)


class ExternalServiceError(AppError):
    def __init__(self, service: str, message: str):
        super().__init__(
            message=f"{service}: {message}",
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            details={"service": service},
        )


# --- Pipeline errors ---


class PipelineError(Exception):
    """Base for pipeline execution errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class TrainingError(PipelineError):
    """Training run failed or produced an unusable result."""

    def __init__(self, message: str, run_id: str | None = None):
        super().__init__(
            message=message,
            details={"run_id": run_id} if run_id else {},
        )
        self.run_id = run_id


class MetricsMissingError(PipelineError):
    """Required metrics were not logged for a run."""

    def __init__(self, run_id: str, missing_metrics: list[str]):
        super().__init__(
            message=(
                f"Run '{run_id}' is missing required metrics: {missing_metrics}. "
                f"Ensure train.py logs all required metrics before registering."
            ),
            details={"run_id": run_id, "missing_metrics": missing_metrics},
        )
        self.run_id = run_id
        self.missing_metrics = missing_metrics


class ModelNotRegisteredError(PipelineError):
    """Training completed but no model version was registered in MLflow."""

    def __init__(self, run_id: str, model_name: str):
        super().__init__(
            message=(
                f"No registered model version found for run_id='{run_id}', "
                f"model='{model_name}'. Ensure train.py registers the model."
            ),
            details={"run_id": run_id, "model_name": model_name},
        )
        self.run_id = run_id
        self.model_name = model_name


class PromotionConfigMissingError(PipelineError):
    """Model config YAML has no promotion section."""

    def __init__(self, model_name: str):
        super().__init__(
            message=(
                f"Model '{model_name}' has no promotion config. "
                f"Add a 'promotion:' section to its YAML config before running this DAG."
            ),
            details={"model_name": model_name},
        )
        self.model_name = model_name
