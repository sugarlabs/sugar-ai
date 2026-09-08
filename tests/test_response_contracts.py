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

"""Every endpoint must return exactly its declared response shape."""

import pytest
from fastapi.testclient import TestClient

from app import create_app
from app.config import settings
from app.routes import api


class FakeProvider:
    def get_model_name(self):
        return "fake-model"

    def health_check(self):
        return True


class FakeAgent:
    """Stands in for RAGAgent; returns canned answers."""

    def __init__(self):
        self.provider = FakeProvider()

    def run(self, question):
        return "rag answer"

    def debug(self, code, context):
        return "debug answer"

    def run_with_custom_prompt(self, question, custom_prompt, params=None):
        return "prompted answer"

    def run_chat_completion(self, messages, params=None):
        return "chat answer"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def fake_agent(monkeypatch):
    agent = FakeAgent()
    monkeypatch.setattr(api, "agent", agent)
    monkeypatch.setattr(api.agent.provider, "generate", lambda q: "llm answer", raising=False)
    monkeypatch.setitem(settings.API_KEYS, "test-key", {"name": "tester"})
    return agent


HEADERS = {"X-API-Key": "test-key"}


def assert_quota(body):
    assert set(body["quota"].keys()) == {"remaining", "total"}


def test_ask_response_shape(client):
    response = client.post("/ask", params={"question": "hi"}, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"answer", "user", "quota"}
    assert body["answer"] == "rag answer"
    assert body["user"] == "tester"
    assert_quota(body)


def test_ask_llm_response_shape(client):
    response = client.post("/ask-llm", params={"question": "hi"}, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"answer", "user", "quota"}
    assert body["answer"] == "llm answer"


def test_debug_response_shape(client):
    response = client.post(
        "/debug", params={"code": "print(1)", "context": "false"}, headers=HEADERS
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"answer", "user", "quota"}
    assert body["answer"] == "debug answer"


def test_prompted_mode_response_shape(client):
    response = client.post(
        "/ask-llm-prompted",
        json={"chat": False, "question": "hi", "custom_prompt": "be nice"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"answer", "user", "quota", "generation_params"}
    assert body["answer"] == "prompted answer"
    assert set(body["generation_params"].keys()) == {
        "max_length", "truncation", "repetition_penalty",
        "temperature", "top_p", "top_k",
    }


def test_chat_mode_response_shape(client):
    response = client.post(
        "/ask-llm-prompted",
        json={"chat": True, "messages": [{"role": "user", "content": "hi"}]},
        headers=HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"choices", "user", "quota", "generation_params"}
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "chat answer"}
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"


def test_health_response_shape(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["provider"] == "FakeProvider"
    assert body["model"] == "fake-model"
    assert "detail" not in body  # omitted when None, as before


def test_openapi_documents_responses(client):
    spec = client.get("/openapi.json").json()
    ask = spec["paths"]["/ask"]["post"]["responses"]
    assert "200" in ask and "422" in ask
    schema_ref = ask["200"]["content"]["application/json"]["schema"]["$ref"]
    assert schema_ref.endswith("AskResponse")
