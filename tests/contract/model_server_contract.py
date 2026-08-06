"""Model Server API Contract.

Defines the shared contract between the platform API (consumer) and model
server (provider). Both sides test against these schemas independently.

If a data science team deploys a new model server, their provider tests
verify it honours this contract before it reaches production.
"""

from pydantic import BaseModel, Field


class ContractPredictRequest(BaseModel):
    """What the platform API sends to the model server."""

    texts: list[str] = Field(min_length=1, max_length=100)


class ContractPredictionResult(BaseModel):
    """Single prediction in the model server response."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class ContractPredictResponse(BaseModel):
    """What the model server must return on POST /predict."""

    model_name: str
    predictions: list[ContractPredictionResult]


class ContractHealthResponse(BaseModel):
    """What the model server must return on GET /health."""

    status: str
    model_name: str
    model_loaded: bool
