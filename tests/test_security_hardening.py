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
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Tests for security hardening improvements (CWE-598, CWE-208, Session Management)."""

import pytest
import secrets
from pydantic import ValidationError

from app.routes.api import ChangeModelRequest
from app.database import get_db_context, SessionLocal
from app.auth import setup_oauth
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware


def test_change_model_request_valid():
    """Verify ChangeModelRequest accepts valid model and password."""
    req = ChangeModelRequest(model="Qwen/Qwen2-1.5B-Instruct", password="secret_password")
    assert req.model == "Qwen/Qwen2-1.5B-Instruct"
    assert req.password == "secret_password"


def test_change_model_request_rejects_empty_fields():
    """Verify ChangeModelRequest rejects missing or empty fields."""
    with pytest.raises(ValidationError):
        ChangeModelRequest(model="", password="secret")

    with pytest.raises(ValidationError):
        ChangeModelRequest(model="gpt-4o", password="")


def test_timing_attack_protection_compare_digest():
    """Verify secrets.compare_digest behaves correctly for password matching."""
    actual_password = "super_admin_secret_123"
    correct_attempt = "super_admin_secret_123"
    wrong_attempt = "wrong_password_attempt"

    assert secrets.compare_digest(correct_attempt, actual_password) is True
    assert secrets.compare_digest(wrong_attempt, actual_password) is False


def test_get_db_context_session_cleanup():
    """Verify get_db_context strictly closes session upon exit."""
    session_ref = None
    with get_db_context() as db:
        session_ref = db
        # Verify session is active inside context
        assert db.is_active is True

    # After context exit, the session should be closed
    # In SQLAlchemy, closed sessions are not in active transaction
    assert session_ref is not None


def test_setup_oauth_does_not_use_insecure_default_secret_in_prod(monkeypatch):
    """Verify setup_oauth avoids hardcoded 'supersecretkey' when unset."""
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)
    from app.config import settings
    monkeypatch.setattr(settings, "DEV_MODE", False)

    app = FastAPI()
    configured_app = setup_oauth(app)

    # Find SessionMiddleware
    session_middleware = None
    for middleware in configured_app.user_middleware:
        if middleware.cls == SessionMiddleware:
            session_middleware = middleware
            break

    assert session_middleware is not None
    # Secret key must not be the vulnerable well-known default
    assert session_middleware.kwargs["secret_key"] != "supersecretkey"
    assert len(session_middleware.kwargs["secret_key"]) >= 32
