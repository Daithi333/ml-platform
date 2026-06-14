from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for model inference."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of text strings to classify (1-100 items)",
    )


class PredictionResult(BaseModel):
    """Single prediction result."""

    category: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Response body for model inference."""

    predictions: list[PredictionResult]
    model_name: str
    model_version: str | None = None
