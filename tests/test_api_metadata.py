# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import create_app
from app.routes import api


METADATA = {
    "model": "test-model",
    "provider": "TestProvider",
    "context_window": 4096,
    "max_output_tokens": 1024,
    "safe_input_tokens": 3072,
}


def _client_with_mock_agent(monkeypatch, metadata=METADATA, healthy=True):
    provider = MagicMock()
    provider.get_model_metadata.return_value = metadata
    provider.get_model_name.return_value = "test-model"
    provider.health_check.return_value = healthy
    monkeypatch.setattr(api, "agent", SimpleNamespace(provider=provider))
    return TestClient(create_app())


def test_model_metadata_endpoint_returns_provider_limits(monkeypatch):
    client = _client_with_mock_agent(monkeypatch)

    response = client.get("/model-metadata")

    assert response.status_code == 200
    assert response.json() == METADATA


def test_model_metadata_endpoint_returns_503_when_agent_missing(monkeypatch):
    monkeypatch.setattr(api, "agent", None)
    client = TestClient(create_app())

    response = client.get("/model-metadata")

    assert response.status_code == 503


def test_health_healthy_response_includes_context_metadata(monkeypatch):
    client = _client_with_mock_agent(monkeypatch, healthy=True)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["context"] == METADATA


def test_health_unhealthy_response_includes_context_metadata(monkeypatch):
    client = _client_with_mock_agent(monkeypatch, healthy=False)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["context"] == METADATA
