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

"""Tests for API key handling in the OAuth login callback."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.middleware.sessions import SessionMiddleware

from app.config import settings
from app.database import APIKey, Base, get_db
from app.routes import auth as auth_routes

EMAIL = "learner@example.org"


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def client(db_session, monkeypatch):
    monkeypatch.setattr(settings, "API_KEYS", {})

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-secret")
    app.include_router(auth_routes.router)
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app, follow_redirects=False)


def login_with_github(client):
    user_response = MagicMock()
    user_response.json.return_value = {"name": "Learner", "email": EMAIL}

    with patch.object(auth_routes.oauth, "github") as github:
        github.authorize_access_token = AsyncMock(return_value={"access_token": "token"})
        github.get = AsyncMock(return_value=user_response)
        return client.get("/auth/callback/github")


def test_oauth_login_creates_and_enables_key_for_new_user(client, db_session):
    response = login_with_github(client)

    key = db_session.query(APIKey).filter(APIKey.email == EMAIL).one()
    assert response.headers["location"] == "/dashboard"
    assert key.approved is True
    assert key.is_active is True
    assert key.key in settings.API_KEYS


def test_oauth_login_keeps_approved_active_key_enabled(client, db_session):
    db_session.add(APIKey(key="existing-key", name="Learner", email=EMAIL, approved=True, is_active=True))
    db_session.commit()

    login_with_github(client)

    assert "existing-key" in settings.API_KEYS


@pytest.mark.parametrize(
    ("approved", "is_active"),
    [
        (True, False),  # deactivated by an admin
        (False, True),  # request denied by an admin
        (False, False),  # request still pending
    ],
)
def test_oauth_login_does_not_enable_disabled_key(client, db_session, approved, is_active):
    db_session.add(APIKey(key="disabled-key", name="Learner", email=EMAIL, approved=approved, is_active=is_active))
    db_session.commit()

    response = login_with_github(client)

    assert response.headers["location"] == "/dashboard"
    assert "disabled-key" not in settings.API_KEYS
