from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Inference request for the model server."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of text inputs for inference",
    )


class PredictionResult(BaseModel):
    """Single prediction output."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Inference response from the model server."""

    model_name: str
    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    """Model server health status."""

    status: str
    model_name: str
    model_loaded: bool
