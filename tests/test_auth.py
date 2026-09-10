import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import database
from app.auth import load_approved_keys_from_db
from app.config import settings


class StubAgent:
    def __init__(self):
        self.provider = self

    def generate(self, question):
        return f"answer: {question}"


# Importing the API router normally imports the heavyweight RAG implementation.
# The endpoint test only exercises authentication and therefore uses a stub.
stub_ai = types.ModuleType("app.ai")
stub_ai.RAGAgent = object
sys.modules.setdefault("app.ai", stub_ai)

from app.routes import api  # noqa: E402


def make_session():
    engine = create_engine("sqlite:///:memory:")
    database.Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_load_approved_active_keys_from_database():
    session = make_session()
    session.add_all([
        database.APIKey(
            key="oauth-key",
            name="OAuth User",
            email="oauth@example.com",
            approved=True,
            is_active=True,
            can_change_model=False,
        ),
        database.APIKey(
            key="pending-key",
            name="Pending User",
            email="pending@example.com",
            approved=False,
            is_active=False,
        ),
        database.APIKey(
            key="revoked-key",
            name="Revoked User",
            email="revoked@example.com",
            approved=True,
            is_active=False,
        ),
        database.APIKey(
            key="revoked-env-key",
            name="Revoked Environment User",
            email="revoked-env@example.com",
            approved=False,
            is_active=True,
        ),
    ])
    session.commit()

    settings.API_KEYS.clear()
    settings.API_KEYS["env-key"] = {
        "name": "Environment User",
        "can_change_model": True,
    }
    settings.API_KEYS["revoked-env-key"] = {
        "name": "Revoked Environment User",
        "can_change_model": False,
    }

    load_approved_keys_from_db(session)

    assert settings.API_KEYS == {
        "oauth-key": {
            "name": "OAuth User",
            "can_change_model": False,
        },
    }
    assert "revoked-env-key" not in settings.API_KEYS

    session.close()
    settings.API_KEYS.clear()


def test_restored_key_authenticates_against_api_endpoint():
    session = make_session()
    session.add(database.APIKey(
        key="oauth-key",
        name="OAuth User",
        email="oauth@example.com",
        approved=True,
        is_active=True,
    ))
    session.commit()

    # Simulate a fresh process: the database survives, but the runtime map is
    # empty until startup restores approved keys.
    settings.API_KEYS.clear()
    load_approved_keys_from_db(session)
    api.user_quotas.clear()
    api.agent = StubAgent()

    app = FastAPI()
    app.include_router(api.router)
    response = TestClient(app).post(
        "/ask-llm",
        params={"question": "What is Python?"},
        headers={"X-API-Key": "oauth-key"},
    )

    assert response.status_code == 200
    assert response.json()["user"] == "OAuth User"
    assert response.json()["answer"] == "answer: What is Python?"

    session.close()
    settings.API_KEYS.clear()
    api.user_quotas.clear()
    api.agent = None
