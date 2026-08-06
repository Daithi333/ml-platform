"""Provider contract tests.

Verifies that the model server produces responses conforming to the
agreed contract. Uses the app factory with a mock model — no MLflow needed.

If a data science team changes the model server's response shape,
these tests catch the contract violation before deployment.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from src.model_server.app import create_app
from tests.contract.model_server_contract import (
    ContractHealthResponse,
    ContractPredictResponse,
)


@pytest.fixture
def mock_model():
    """A mock sklearn model that returns predictable results."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1])
    model.predict_proba.return_value = np.array(
        [
            [0.9, 0.05, 0.02, 0.02, 0.01],
            [0.1, 0.8, 0.05, 0.03, 0.02],
        ]
    )
    return model


@pytest.fixture
def model_server_client(mock_model):
    """Test client using the app factory with a pre-loaded mock model."""
    test_app = create_app(
        model=mock_model,
        model_name="test-classifier",
        labels=["cat-a", "cat-b", "cat-c", "cat-d", "cat-e"],
    )
    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client


class TestProviderHonoursContract:
    """Model server produces responses that match the contract."""

    def test_health_response_matches_contract(self, model_server_client):
        """GET /health returns a response conforming to the health contract."""
        response = model_server_client.get("/health")

        assert response.status_code == 200
        validated = ContractHealthResponse(**response.json())
        assert validated.status == "ok"
        assert validated.model_loaded is True
        assert validated.model_name == "test-classifier"

    def test_predict_response_matches_contract(self, model_server_client):
        """POST /predict returns a response conforming to the predict contract."""
        response = model_server_client.post(
            "/predict",
            json={"texts": ["input one", "input two"]},
        )

        assert response.status_code == 200
        validated = ContractPredictResponse(**response.json())
        assert validated.model_name == "test-classifier"
        assert len(validated.predictions) == 2
        assert validated.predictions[0].label == "cat-a"
        assert 0.0 <= validated.predictions[0].confidence <= 1.0

    def test_predict_single_input(self, model_server_client, mock_model):
        """Contract holds for single-item input."""
        mock_model.predict.return_value = np.array([2])
        mock_model.predict_proba.return_value = np.array(
            [
                [0.05, 0.05, 0.85, 0.03, 0.02],
            ]
        )

        response = model_server_client.post(
            "/predict",
            json={"texts": ["single input"]},
        )

        assert response.status_code == 200
        validated = ContractPredictResponse(**response.json())
        assert len(validated.predictions) == 1
        assert validated.predictions[0].label == "cat-c"

    def test_predict_empty_texts_rejected(self, model_server_client):
        """Provider rejects empty texts (validation error, not contract violation)."""
        response = model_server_client.post(
            "/predict",
            json={"texts": []},
        )

        assert response.status_code == 422

    def test_predict_missing_body_rejected(self, model_server_client):
        """Provider rejects missing request body."""
        response = model_server_client.post("/predict", json={})

        assert response.status_code == 422
