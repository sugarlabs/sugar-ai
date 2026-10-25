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

"""Tests for the /change-model endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routes import api


@pytest.fixture
def agent(monkeypatch):
    agent = MagicMock()
    monkeypatch.setattr(api, "agent", agent)
    monkeypatch.setattr(
        settings,
        "API_KEYS",
        {
            "admin-key": {"name": "Admin", "can_change_model": True},
            "user-key": {"name": "User", "can_change_model": False},
        },
    )
    monkeypatch.setattr(settings, "MODEL_CHANGE_PASSWORD", "secret")
    monkeypatch.setattr(settings, "AI_PROVIDER", "ollama")
    return agent


@pytest.fixture
def client(agent):
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def test_admin_can_change_model_with_configured_provider(client, agent):
    with patch("app.providers.create_provider") as create_provider:
        response = client.post(
            "/change-model",
            params={"model": "qwen3:1.7b", "api_key": "admin-key", "password": "secret"},
        )

    assert response.status_code == 200
    assert response.json() == {"message": "Model changed to qwen3:1.7b", "user": "Admin"}
    assert create_provider.call_args.kwargs["provider_name"] == "ollama"
    assert create_provider.call_args.kwargs["model_name"] == "qwen3:1.7b"
    agent.set_model.assert_called_once_with(create_provider.return_value)


@pytest.mark.parametrize(
    ("api_key", "password", "expected_status"),
    [
        ("unknown-key", "secret", 401),
        ("user-key", "secret", 403),
        ("admin-key", "wrong-password", 403),
    ],
)
def test_change_model_rejects_unauthorized_requests(client, agent, api_key, password, expected_status):
    with patch("app.providers.create_provider") as create_provider:
        response = client.post(
            "/change-model",
            params={"model": "qwen3:1.7b", "api_key": api_key, "password": password},
        )

    assert response.status_code == expected_status
    create_provider.assert_not_called()
    agent.set_model.assert_not_called()
