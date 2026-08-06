from src.serving.schemas.registry import (
    ModelVersionDetail,
    ModelVersionSummary,
    RegisteredModelDetail,
    RegisteredModelSummary,
)


class TestListModels:
    """Tests for GET /api/v1/registry/models."""

    def test_list_models_success(self, client, base_url, mock_registry):
        """Should return list of registered models."""
        mock_registry.list_models.return_value = [
            RegisteredModelSummary(
                name="newsgroups-classifier",
                description="Text classifier",
                latest_versions=[
                    ModelVersionSummary(version="1", status="READY", run_id="abc123"),
                ],
            ),
            RegisteredModelSummary(
                name="fraud-detector",
                description="",
                latest_versions=[],
            ),
        ]

        response = client.get(f"{base_url}/registry/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "newsgroups-classifier"
        assert data[0]["description"] == "Text classifier"
        assert data[0]["latest_versions"][0]["version"] == "1"
        assert data[1]["name"] == "fraud-detector"
        assert data[1]["latest_versions"] == []

    def test_list_models_empty(self, client, base_url, mock_registry):
        """Should return empty list when no models are registered."""
        mock_registry.list_models.return_value = []

        response = client.get(f"{base_url}/registry/models")

        assert response.status_code == 200
        assert response.json() == []


class TestModelDetails:
    """Tests for GET /api/v1/registry/models/{model_name}."""

    def test_model_details_success(self, client, base_url, mock_registry):
        """Should return detailed model info with versions."""
        mock_registry.get_model_info.return_value = RegisteredModelDetail(
            name="newsgroups-classifier",
            description="Text classifier for newsgroups",
            tags={"team": "ml-platform"},
            versions=[
                ModelVersionDetail(
                    version="2",
                    status="READY",
                    run_id="def456",
                    creation_timestamp=1700000000000,
                    aliases=["champion"],
                ),
                ModelVersionDetail(
                    version="1",
                    status="READY",
                    run_id="abc123",
                    creation_timestamp=1699000000000,
                    aliases=[],
                ),
            ],
        )

        response = client.get(f"{base_url}/registry/models/newsgroups-classifier")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "newsgroups-classifier"
        assert data["tags"] == {"team": "ml-platform"}
        assert len(data["versions"]) == 2
        assert data["versions"][0]["aliases"] == ["champion"]

    def test_model_details_not_found(self, client, base_url, mock_registry):
        """Should return 404 when model does not exist."""
        mock_registry.get_model_info.return_value = None

        response = client.get(f"{base_url}/registry/models/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "NOT_FOUND"
        assert "nonexistent" in data["message"]


class TestModelReload:
    """Tests for POST /api/v1/registry/models/{model_name}/reload."""

    def test_reload_success(self, client, base_url, mock_registry):
        """Should trigger model reload and return confirmation."""
        response = client.post(f"{base_url}/registry/models/newsgroups-classifier/reload")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "newsgroups-classifier" in data["message"]
        mock_registry.reload_model.assert_called_once_with("newsgroups-classifier")
