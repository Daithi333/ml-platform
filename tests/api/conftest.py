import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.serving.app import app
from src.serving.dependencies import get_registry, get_settings


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure test environment."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def base_url():
    """Base URL for API endpoints."""
    return "/api/v1"


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return Settings(
        app_version="0.1.0",
        environment="development",
        service_name="ml-platform-api",
    )


@pytest.fixture
def mock_registry():
    """Mock registry client for testing."""
    return MagicMock()


@pytest.fixture
def client(mock_settings, mock_registry):
    app.dependency_overrides[get_settings] = lambda: mock_settings
    app.dependency_overrides[get_registry] = lambda: mock_registry
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
