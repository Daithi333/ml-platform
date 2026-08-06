"""Consumer contract tests.

Verifies that the platform API's predict service correctly parses
responses that conform to the model server contract.

If these fail, the platform API has drifted from the agreed contract.
"""

import pytest
from pydantic import ValidationError

from tests.contract.model_server_contract import (
    ContractHealthResponse,
    ContractPredictRequest,
    ContractPredictResponse,
)


class TestConsumerUnderstandsContract:
    """Platform API can construct valid requests and parse valid responses."""

    def test_can_construct_valid_request(self):
        """Platform API sends a valid request shape."""
        request = ContractPredictRequest(texts=["NASA launched a satellite"])

        assert request.texts == ["NASA launched a satellite"]

    def test_can_parse_valid_predict_response(self):
        """Platform API can parse a well-formed model server response."""
        raw_response = {
            "model_name": "newsgroups-classifier",
            "predictions": [
                {"label": "sci.space", "confidence": 0.92},
                {"label": "comp.graphics", "confidence": 0.78},
            ],
        }

        response = ContractPredictResponse(**raw_response)

        assert response.model_name == "newsgroups-classifier"
        assert len(response.predictions) == 2
        assert response.predictions[0].label == "sci.space"
        assert response.predictions[0].confidence == 0.92

    def test_can_parse_valid_health_response(self):
        """Platform API can parse a well-formed health response."""
        raw_response = {
            "status": "ok",
            "model_name": "newsgroups-classifier",
            "model_loaded": True,
        }

        response = ContractHealthResponse(**raw_response)

        assert response.status == "ok"
        assert response.model_loaded is True

    def test_rejects_response_missing_model_name(self):
        """Platform API detects contract violation: missing model_name."""
        raw_response = {
            "predictions": [{"label": "sci.space", "confidence": 0.92}],
        }

        with pytest.raises(ValidationError):
            ContractPredictResponse(**raw_response)

    def test_rejects_response_missing_predictions(self):
        """Platform API detects contract violation: missing predictions."""
        raw_response = {
            "model_name": "newsgroups-classifier",
        }

        with pytest.raises(ValidationError):
            ContractPredictResponse(**raw_response)

    def test_rejects_prediction_with_invalid_confidence(self):
        """Platform API detects contract violation: confidence out of range."""
        raw_response = {
            "model_name": "newsgroups-classifier",
            "predictions": [{"label": "sci.space", "confidence": 1.5}],
        }

        with pytest.raises(ValidationError):
            ContractPredictResponse(**raw_response)

    def test_rejects_prediction_missing_label(self):
        """Platform API detects contract violation: prediction missing label."""
        raw_response = {
            "model_name": "newsgroups-classifier",
            "predictions": [{"confidence": 0.92}],
        }

        with pytest.raises(ValidationError):
            ContractPredictResponse(**raw_response)
