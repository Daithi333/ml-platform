from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from src.config import Settings
from src.registry.client import RegistryClient


@lru_cache
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_registry(request: Request) -> RegistryClient:
    """Get registry client from app state (initialised in lifespan)."""
    return request.app.state.registry


# Dependency annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]
RegistryDep = Annotated[RegistryClient, Depends(get_registry)]
