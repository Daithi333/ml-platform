from unittest.mock import patch

import pytest

from src.serving.schemas.predict import PredictionResponse, PredictionResult


class TestPredictEndpoint:
    """Tests for POST /api/v1/models/{model_name}/predict."""

    @pytest.fixture
    def successful_prediction(self):
        """Prediction response from service layer."""
        return PredictionResponse(
            model_name="newsgroups-classifier",
            predictions=[
                PredictionResult(category="sci.space", confidence=0.92),
                PredictionResult(category="comp.graphics", confidence=0.87),
            ],
            model_version=None,
        )

    @patch("src.serving.routers.predict.run_prediction")
    def test_predict_success(self, mock_run, client, base_url, successful_prediction):
        """Should return formatted predictions from model server."""
        mock_run.return_value = successful_prediction

        response = client.post(
            f"{base_url}/models/newsgroups-classifier/predict",
            json={"texts": ["NASA launched a satellite", "The game was exciting"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "newsgroups-classifier"
        assert len(data["predictions"]) == 2
        assert data["predictions"][0]["category"] == "sci.space"
        assert data["predictions"][0]["confidence"] == 0.92
        assert data["model_version"] is None

    @patch("src.serving.routers.predict.run_prediction")
    def test_predict_with_version(self, mock_run, client, base_url):
        """Should pass version through in response."""
        mock_run.return_value = PredictionResponse(
            model_name="newsgroups-classifier",
            predictions=[PredictionResult(category="sci.space", confidence=0.91)],
            model_version="3",
        )

        response = client.post(
            f"{base_url}/models/newsgroups-classifier/predict?version=3",
            json={"texts": ["test input"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "3"

    @patch("src.serving.routers.predict.run_prediction")
    def test_predict_model_not_found(self, mock_run, client, base_url):
        """Should return 404 when model server is not registered."""
        from src.exceptions import NotFoundError

        mock_run.side_effect = NotFoundError(resource="Model server", identifier="nonexistent")

        response = client.post(
            f"{base_url}/models/nonexistent/predict",
            json={"texts": ["test input"]},
        )

        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "NOT_FOUND"
        assert "nonexistent" in data["message"]

    def test_predict_empty_texts(self, client, base_url):
        """Should return 422 when texts list is empty."""
        response = client.post(
            f"{base_url}/models/newsgroups-classifier/predict",
            json={"texts": []},
        )

        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "VALIDATION_ERROR"

    def test_predict_missing_texts(self, client, base_url):
        """Should return 422 when texts field is missing."""
        response = client.post(
            f"{base_url}/models/newsgroups-classifier/predict",
            json={},
        )

        assert response.status_code == 422

    @patch("src.serving.routers.predict.run_prediction")
    def test_predict_model_server_connection_error(self, mock_run, client, base_url):
        """Should return 502 when model server is unreachable."""
        from src.exceptions import ExternalServiceError

        mock_run.side_effect = ExternalServiceError(
            service="model-server/newsgroups-classifier",
            message="Connection refused",
        )

        response = client.post(
            f"{base_url}/models/newsgroups-classifier/predict",
            json={"texts": ["test input"]},
        )

        assert response.status_code == 502
        data = response.json()
        assert data["error"] == "EXTERNAL_SERVICE_ERROR"

    @patch("src.serving.routers.predict.run_prediction")
    def test_predict_model_server_error_response(self, mock_run, client, base_url):
        """Should return 502 when model server returns non-200."""
        from src.exceptions import ExternalServiceError

        mock_run.side_effect = ExternalServiceError(
            service="model-server/newsgroups-classifier",
            message="HTTP 500: Internal Server Error",
        )

        response = client.post(
            f"{base_url}/models/newsgroups-classifier/predict",
            json={"texts": ["test input"]},
        )

        assert response.status_code == 502
        data = response.json()
        assert data["error"] == "EXTERNAL_SERVICE_ERROR"
