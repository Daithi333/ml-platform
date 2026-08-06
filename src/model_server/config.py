from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class ModelServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    model_name: str
    mlflow_tracking_uri: str = "http://localhost:5001"
    port: int = 8001
    environment: Literal["development", "staging", "production"] = "development"


def get_model_server_settings() -> ModelServerSettings:
    return ModelServerSettings()
