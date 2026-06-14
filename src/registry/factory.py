from src.config import get_settings
from src.registry.client import RegistryClient


def make_registry_client() -> RegistryClient:
    settings = get_settings()
    return RegistryClient(settings=settings.mlflow)
