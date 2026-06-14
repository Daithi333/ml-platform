from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )


class MLflowSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="MLFLOW__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    tracking_uri: str = "http://localhost:5001"
    artifact_root: str = "/mlflow/artifacts"


class DataSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", str(ENV_FILE_PATH)],
        env_prefix="DATA__",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )

    root: str = "data"


class Settings(BaseConfigSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "ml-platform-api"

    mlflow: MLflowSettings = MLflowSettings()
    data: DataSettings = DataSettings()


def get_settings() -> Settings:
    return Settings()
