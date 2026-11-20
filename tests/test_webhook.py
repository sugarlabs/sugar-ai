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

"""Tests for the GitHub deployment webhook."""

import hashlib
import hmac
import json
from unittest.mock import patch
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes import webhook

SECRET = "test-secret"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(webhook, "_webhook_configured", True)
    monkeypatch.setattr(webhook, "WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr(webhook, "REPO_PATH_LOCALLY", "/srv/sugar-ai")
    monkeypatch.setattr(webhook, "GIT_PATH", "/usr/bin/git")

    app = FastAPI()
    app.include_router(webhook.router)
    return TestClient(app)


def deliver(client, event, payload, form_encoded=False, secret=SECRET):
    """Send a delivery the way GitHub does, in either content type."""
    raw = json.dumps(payload)
    if form_encoded:
        body = urlencode({"payload": raw}).encode()
        content_type = "application/x-www-form-urlencoded"
    else:
        body = raw.encode()
        content_type = "application/json"
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    with patch.object(webhook.os, "system", return_value=0) as system:
        response = client.post(
            "/webhook",
            content=body,
            headers={
                "X-GitHub-Event": event,
                "X-Hub-Signature-256": signature,
                "Content-Type": content_type,
            },
        )
    return response, system


@pytest.mark.parametrize("form_encoded", [False, True])
def test_push_to_main_deploys(client, form_encoded):
    response, system = deliver(client, "push", {"ref": "refs/heads/main"}, form_encoded)

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    commands = [call.args[0] for call in system.call_args_list]
    assert len(commands) == 3
    assert "fetch origin main" in commands[0]
    assert "reset --hard origin/main" in commands[1]
    assert "systemctl restart sugarai" in commands[2]


@pytest.mark.parametrize(
    ("event", "payload"),
    [
        ("ping", {"zen": "Keep it logically awesome."}),
        ("push", {"ref": "refs/heads/docker"}),
        ("push", {"ref": "refs/tags/v1.0"}),
        ("issue_comment", {"action": "created"}),
    ],
)
def test_other_deliveries_are_ignored(client, event, payload):
    response, system = deliver(client, event, payload)

    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    system.assert_not_called()


def test_invalid_signature_is_rejected(client):
    response, system = deliver(client, "push", {"ref": "refs/heads/main"}, secret="wrong-secret")

    assert response.status_code == 403
    system.assert_not_called()
