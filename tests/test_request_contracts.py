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

"""Requests are validated, and the legacy query-parameter style still works."""

import pytest
from fastapi.testclient import TestClient

from app import create_app
from app.config import settings
from app.routes import api
from app.schemas.requests import MAX_QUESTION_CHARS


class FakeProvider:
    def get_model_name(self):
        return "fake-model"

    def health_check(self):
        return True

    def generate(self, prompt, params=None):
        return "llm answer"


class FakeAgent:
    def __init__(self):
        self.provider = FakeProvider()

    def run(self, question):
        return "rag answer"

    def debug(self, code, context):
        return f"debug answer context={context}"

    def run_with_custom_prompt(self, question, custom_prompt, params=None):
        return "prompted answer"

    def run_chat_completion(self, messages, params=None):
        return "chat answer"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def fake_agent(monkeypatch):
    monkeypatch.setattr(api, "agent", FakeAgent())
    monkeypatch.setitem(settings.API_KEYS, "test-key", {"name": "tester"})


HEADERS = {"X-API-Key": "test-key"}


# --- JSON body (the contract) ----------------------------------------------


@pytest.mark.parametrize("path", ["/ask", "/ask-llm"])
def test_json_body_is_accepted(client, path):
    response = client.post(path, json={"question": "what is sugar?"}, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["answer"]


def test_debug_json_body_is_accepted(client):
    response = client.post(
        "/debug", json={"code": "print(1)", "context": True}, headers=HEADERS
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "debug answer context=True"


def test_debug_context_defaults_to_false(client):
    response = client.post("/debug", json={"code": "print(1)"}, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["answer"] == "debug answer context=False"


# --- Legacy query parameters (kept working) --------------------------------


@pytest.mark.parametrize("path", ["/ask", "/ask-llm"])
def test_legacy_query_parameter_still_works(client, path):
    response = client.post(path, params={"question": "what is sugar?"}, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["answer"]


def test_legacy_debug_query_parameters_still_work(client):
    response = client.post(
        "/debug", params={"code": "print(1)", "context": "true"}, headers=HEADERS
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "debug answer context=True"


# --- Validation ------------------------------------------------------------


def assert_validation_error(response):
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    return body["error"]["message"]


@pytest.mark.parametrize("path", ["/ask", "/ask-llm"])
def test_missing_question_is_rejected(client, path):
    message = assert_validation_error(client.post(path, headers=HEADERS))
    assert "question" in message


@pytest.mark.parametrize("path", ["/ask", "/ask-llm"])
def test_empty_question_is_rejected(client, path):
    assert_validation_error(client.post(path, json={"question": ""}, headers=HEADERS))
    assert_validation_error(client.post(path, params={"question": ""}, headers=HEADERS))


@pytest.mark.parametrize("path", ["/ask", "/ask-llm"])
def test_oversized_question_is_rejected(client, path):
    huge = "x" * (MAX_QUESTION_CHARS + 1)
    assert_validation_error(client.post(path, json={"question": huge}, headers=HEADERS))


def test_missing_code_is_rejected(client):
    message = assert_validation_error(client.post("/debug", headers=HEADERS))
    assert "code" in message


def test_wrong_type_is_rejected(client):
    assert_validation_error(
        client.post("/ask", json={"question": {"nested": "object"}}, headers=HEADERS)
    )


# --- ask-llm-prompted mode rules -------------------------------------------


def test_prompted_mode_needs_question_and_custom_prompt(client):
    response = client.post(
        "/ask-llm-prompted",
        json={"chat": False, "question": "hi", "custom_prompt": "be nice"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "prompted answer"


@pytest.mark.parametrize(
    "payload, missing",
    [
        ({"chat": False}, ["question", "custom_prompt"]),
        ({"chat": False, "question": "hi"}, ["custom_prompt"]),
        ({"chat": False, "custom_prompt": "be nice"}, ["question"]),
    ],
)
def test_prompted_mode_rejects_missing_fields(client, payload, missing):
    message = assert_validation_error(
        client.post("/ask-llm-prompted", json=payload, headers=HEADERS)
    )
    for name in missing:
        assert name in message


def test_chat_mode_needs_messages(client):
    response = client.post(
        "/ask-llm-prompted",
        json={"chat": True, "messages": [{"role": "user", "content": "hi"}]},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "chat answer"


@pytest.mark.parametrize("payload", [{"chat": True}, {"chat": True, "messages": []}])
def test_chat_mode_rejects_missing_messages(client, payload):
    message = assert_validation_error(
        client.post("/ask-llm-prompted", json=payload, headers=HEADERS)
    )
    assert "messages" in message


def test_chat_mode_ignores_prompted_fields(client):
    """chat=True does not need question/custom_prompt even if absent."""
    response = client.post(
        "/ask-llm-prompted",
        json={"chat": True, "messages": [{"role": "user", "content": "hi"}], "temperature": 0.1},
        headers=HEADERS,
    )
    assert response.status_code == 200
