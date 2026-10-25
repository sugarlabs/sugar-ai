# Copyright (C) 2026 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Tests for the /health endpoint."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes import api


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def set_provider(monkeypatch, healthy=True, error=None):
    provider = MagicMock()
    provider.get_model_name.return_value = "qwen3:1.7b"
    provider.health_check.return_value = healthy
    provider.health_check.side_effect = error
    monkeypatch.setattr(api, "agent", MagicMock(provider=provider))


def test_health_returns_200_when_provider_is_healthy(client, monkeypatch):
    set_provider(monkeypatch, healthy=True)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "qwen3:1.7b"


def test_health_returns_503_when_provider_is_unhealthy(client, monkeypatch):
    set_provider(monkeypatch, healthy=False)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"
    assert response.json()["model"] == "qwen3:1.7b"


def test_health_returns_503_when_health_check_raises(client, monkeypatch):
    set_provider(monkeypatch, error=RuntimeError("connection refused"))

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "error"


def test_health_returns_503_when_agent_is_not_initialized(client, monkeypatch):
    monkeypatch.setattr(api, "agent", None)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
