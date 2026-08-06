from pydantic import BaseModel


class ModelVersionSummary(BaseModel):
    """Summary of a model version."""

    version: str
    status: str
    run_id: str


class RegisteredModelSummary(BaseModel):
    """Summary of a registered model."""

    name: str
    description: str
    latest_versions: list[ModelVersionSummary]


class ModelVersionDetail(BaseModel):
    """Detailed model version info."""

    version: str
    status: str
    run_id: str
    creation_timestamp: int
    aliases: list[str]


class RegisteredModelDetail(BaseModel):
    """Full details for a registered model."""

    name: str
    description: str
    tags: dict[str, str]
    versions: list[ModelVersionDetail]


class ModelReloadResponse(BaseModel):
    """Response after clearing model cache."""

    status: str
    message: str
