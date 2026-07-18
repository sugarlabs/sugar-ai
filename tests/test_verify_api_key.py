"""
Tests for the `verify_api_key` dependency's integration with persistent quota
tracking (app/routes/api.py). Heavy ML-only modules (app.ai, app.providers)
are stubbed out so these tests don't require torch/transformers/etc. to be
installed.
"""
import sys
import types

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.config import settings


def _stub_module(name, **attrs):
    if name not in sys.modules:
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        sys.modules[name] = module


_stub_module("app.ai", RAGAgent=type("RAGAgent", (), {}))
_stub_module("app.providers")
_stub_module("app.providers.base", GenerationParams=type("GenerationParams", (), {}))

from app.routes import api as api_module  # noqa: E402  (import after stubbing)


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'quota.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def test_api_key(monkeypatch):
    monkeypatch.setitem(
        settings.API_KEYS, "testkey", {"name": "Test User", "can_change_model": False}
    )
    monkeypatch.setattr(settings, "MAX_DAILY_REQUESTS", 2)
    return "testkey"


def test_verify_api_key_returns_quota_without_mutating_settings(test_api_key, db_session):
    original_entry = dict(settings.API_KEYS[test_api_key])

    user_info = api_module.verify_api_key(api_key=test_api_key, request=None, db=db_session)

    assert user_info["name"] == "Test User"
    assert user_info["quota"] == {"remaining": 1, "total": 2}
    # the shared settings dict must stay untouched by the per-request quota info
    assert settings.API_KEYS[test_api_key] == original_entry


def test_verify_api_key_enforces_quota_across_calls(test_api_key, db_session):
    first = api_module.verify_api_key(api_key=test_api_key, request=None, db=db_session)
    assert first["quota"]["remaining"] == 1

    second = api_module.verify_api_key(api_key=test_api_key, request=None, db=db_session)
    assert second["quota"]["remaining"] == 0

    with pytest.raises(HTTPException) as exc_info:
        api_module.verify_api_key(api_key=test_api_key, request=None, db=db_session)
    assert exc_info.value.status_code == 429


def test_verify_api_key_rejects_invalid_key(db_session):
    with pytest.raises(HTTPException) as exc_info:
        api_module.verify_api_key(api_key="not-a-real-key", request=None, db=db_session)
    assert exc_info.value.status_code == 401


def test_verify_api_key_rejects_missing_key(db_session):
    with pytest.raises(HTTPException) as exc_info:
        api_module.verify_api_key(api_key=None, request=None, db=db_session)
    assert exc_info.value.status_code == 401
