#!/usr/bin/env python3
"""
Smoke test all public Sugar-AI API endpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")
DEFAULT_TEST_API_KEY = os.getenv("TEST_API_KEY", "user_key_1")
DEFAULT_ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "sugarai2024")
DEFAULT_MODEL_CHANGE_PASSWORD = os.getenv("MODEL_CHANGE_PASSWORD", "sugarai2024")


@dataclass
class ResponseRecord:
    name: str
    status_code: int
    ok: bool
    detail: str


class LiveRunner:
    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0)

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        return self.client.request(method, path, **kwargs)

    def close(self) -> None:
        self.client.close()


class InternalRunner:
    def __init__(self):
        self._temp_dir = tempfile.TemporaryDirectory(prefix="sugar-ai-internal-db-")
        os.environ["DOC_PATHS"] = "[]"
        os.environ["LLM_PROVIDER_TYPE"] = "openai_compatible"
        os.environ["LLM_BASE_URL"] = "http://mock-llm.invalid/v1"
        os.environ["LLM_API_KEY"] = "not-needed"
        os.environ["LLM_MODEL_NAME"] = "mock-model"
        os.environ["LLM_DISPLAY_NAME"] = "Mock Model"
        os.environ["DATABASE_URL"] = f"sqlite:///{Path(self._temp_dir.name) / 'sugar_ai.db'}"
        os.environ["API_KEYS"] = json.dumps(
            {
                "sugarai2024": {"name": "Admin Key", "can_change_model": True},
                "user_key_1": {"name": "User 1", "can_change_model": False},
            }
        )
        os.environ["MODEL_CHANGE_PASSWORD"] = DEFAULT_MODEL_CHANGE_PASSWORD

        from fastapi.testclient import TestClient

        import main
        import app.runtime as runtime

        class FakeProvider:
            def __init__(self, model_name: str):
                self.model_name = model_name

            def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
                del kwargs
                last_message = messages[-1]["content"] if messages else ""
                return f"[{self.model_name}] {last_message[:120]}".strip()

        def fake_build(model_config):
            return FakeProvider(model_config.model_name)

        runtime.build_llm_provider = fake_build

        self._main = main
        self.client_ctx = TestClient(main.app)
        self.client = self.client_ctx.__enter__()

    def request(self, method: str, path: str, **kwargs: Any):
        return self.client.request(method, path, **kwargs)

    def close(self) -> None:
        self.client_ctx.__exit__(None, None, None)
        self._temp_dir.cleanup()


def expect(
    records: list[ResponseRecord],
    name: str,
    response,
    *,
    expected_status: int = 200,
    validator=None,
) -> Any:
    ok = response.status_code == expected_status
    detail = response.text[:300]

    payload = None
    if ok:
        try:
            payload = response.json()
        except Exception:
            payload = None

    if ok and validator is not None:
        try:
            validator(payload)
        except Exception as exc:
            ok = False
            detail = f"validator failed: {exc}"

    records.append(
        ResponseRecord(
            name=name,
            status_code=response.status_code,
            ok=ok,
            detail=detail,
        )
    )
    if not ok:
        raise RuntimeError(f"{name} failed with status {response.status_code}: {detail}")
    return payload


def require_key(payload: dict, key: str) -> None:
    if key not in payload:
        raise AssertionError(f"missing key '{key}'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Sugar-AI API endpoints.")
    parser.add_argument("--internal", action="store_true", help="Run against an in-process TestClient with a fake LLM provider.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL for the live API.")
    args = parser.parse_args()

    runner = InternalRunner() if args.internal else LiveRunner(args.base_url)
    records: list[ResponseRecord] = []

    user_headers = {"X-API-Key": DEFAULT_TEST_API_KEY}
    admin_headers = {"X-API-Key": DEFAULT_ADMIN_API_KEY}
    created_model_id: Optional[int] = None
    original_active_id: Optional[int] = None

    try:
        root = runner.request("GET", "/")
        expect(records, "GET /", root, validator=lambda _: None)

        ask = runner.request(
            "POST",
            "/ask",
            params={"question": "How do I make a Pygame window?"},
            headers=user_headers,
        )
        expect(records, "POST /ask", ask, validator=lambda payload: require_key(payload, "answer"))

        ask_llm = runner.request(
            "POST",
            "/ask-llm",
            params={"question": "Explain Python lists."},
            headers=user_headers,
        )
        expect(records, "POST /ask-llm", ask_llm, validator=lambda payload: require_key(payload, "answer"))

        prompted = runner.request(
            "POST",
            "/ask-llm-prompted",
            headers={**user_headers, "Content-Type": "application/json"},
            json={
                "question": "Write a hello world example.",
                "custom_prompt": "You are a concise Python tutor.",
                "max_length": 128,
                "temperature": 0.2,
                "top_p": 0.9,
            },
        )
        expect(
            records,
            "POST /ask-llm-prompted (prompted)",
            prompted,
            validator=lambda payload: require_key(payload, "answer"),
        )

        chat = runner.request(
            "POST",
            "/ask-llm-prompted",
            headers={**user_headers, "Content-Type": "application/json"},
            json={
                "chat": True,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello."},
                ],
                "max_length": 128,
                "temperature": 0.2,
                "top_p": 0.9,
            },
        )
        expect(
            records,
            "POST /ask-llm-prompted (chat)",
            chat,
            validator=lambda payload: require_key(payload, "choices"),
        )

        debug = runner.request(
            "POST",
            "/debug",
            params={"code": "print('hello')", "context": "false"},
            headers=user_headers,
        )
        expect(records, "POST /debug", debug, validator=lambda payload: require_key(payload, "answer"))

        listed = runner.request("GET", "/admin/models", headers=admin_headers)
        listed_payload = expect(
            records,
            "GET /admin/models",
            listed,
            validator=lambda payload: require_key(payload, "models"),
        )
        models = listed_payload["models"]
        active_models = [model for model in models if model["is_active"]]
        if not active_models:
            raise RuntimeError("No active model found in /admin/models response.")
        original_active_id = active_models[0]["id"]

        timestamp = int(time.time())
        create = runner.request(
            "POST",
            "/admin/models",
            headers={**admin_headers, "Content-Type": "application/json"},
            json={
                "name": f"Smoke Test Model {timestamp}",
                "provider_type": "openai_compatible",
                "base_url": "http://localhost:8002/v1",
                "api_key": "not-needed",
                "model_name": f"smoke-model-{timestamp}",
                "max_model_length": 2048,
                "is_active": False,
            },
        )
        created_payload = expect(
            records,
            "POST /admin/models",
            create,
            validator=lambda payload: require_key(payload, "model"),
        )
        created_model_id = created_payload["model"]["id"]

        update = runner.request(
            "PUT",
            f"/admin/models/{created_model_id}",
            headers={**admin_headers, "Content-Type": "application/json"},
            json={"name": f"Smoke Test Model {timestamp} Updated"},
        )
        expect(records, "PUT /admin/models/{id}", update, validator=lambda payload: require_key(payload, "model"))

        delete_active = runner.request(
            "DELETE",
            f"/admin/models/{original_active_id}",
            headers=admin_headers,
        )
        expect(records, "DELETE /admin/models/{active_id}", delete_active, expected_status=400)

        activate = runner.request(
            "POST",
            f"/admin/models/{created_model_id}/activate",
            headers=admin_headers,
        )
        expect(
            records,
            "POST /admin/models/{id}/activate",
            activate,
            validator=lambda payload: require_key(payload, "model"),
        )

        change_model = runner.request(
            "POST",
            "/change-model",
            params={
                "model_id": original_active_id,
                "api_key": DEFAULT_ADMIN_API_KEY,
                "password": DEFAULT_MODEL_CHANGE_PASSWORD,
            },
        )
        expect(records, "POST /change-model", change_model, validator=lambda payload: require_key(payload, "model"))

        delete_created = runner.request(
            "DELETE",
            f"/admin/models/{created_model_id}",
            headers=admin_headers,
        )
        expect(records, "DELETE /admin/models/{id}", delete_created, validator=lambda payload: require_key(payload, "model"))

    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        for record in records:
            status = "PASS" if record.ok else "FAIL"
            print(f"[{status}] {record.name}: {record.status_code} {record.detail}")
        runner.close()
        return 1

    for record in records:
        print(f"[PASS] {record.name}: {record.status_code}")

    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
