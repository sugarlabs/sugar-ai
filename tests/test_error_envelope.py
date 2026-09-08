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

"""Every API error must use the {"error": {"code", "message"}} envelope."""

import pytest
from fastapi.testclient import TestClient

from app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def assert_error_envelope(body: dict, expected_code: str):
    assert set(body.keys()) == {"error"}
    assert body["error"]["code"] == expected_code
    assert isinstance(body["error"]["message"], str)
    assert body["error"]["message"]


def test_missing_api_key_returns_unauthorized_envelope(client):
    response = client.post("/ask", params={"question": "hi"})
    assert response.status_code == 401
    assert_error_envelope(response.json(), "unauthorized")


def test_invalid_api_key_returns_unauthorized_envelope(client):
    response = client.post(
        "/ask",
        params={"question": "hi"},
        headers={"X-API-Key": "not-a-real-key"},
    )
    assert response.status_code == 401
    assert_error_envelope(response.json(), "unauthorized")


def test_invalid_payload_returns_validation_envelope(client, monkeypatch):
    # Auth runs before body validation, so a valid key is needed to reach it.
    from app.config import settings
    monkeypatch.setitem(settings.API_KEYS, "test-key", {"name": "tester"})

    # temperature above the allowed bound must fail model validation
    response = client.post(
        "/ask-llm-prompted",
        json={"chat": False, "question": "hi", "custom_prompt": "p", "temperature": 99},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 422
    body = response.json()
    assert_error_envelope(body, "validation_error")
    assert "temperature" in body["error"]["message"]


def test_unknown_route_returns_not_found_envelope(client):
    response = client.get("/no-such-route")
    assert response.status_code == 404
    assert_error_envelope(response.json(), "not_found")
