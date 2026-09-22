import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.routes.reflect import get_provider
from app.routes.reflect import router as reflect_router

TEST_API_KEY = "test-key"


class FakeProvider:
    """A scripted stand-in for the engine's provider seam -- no network,
    no model. Every call returns a well-formed model turn; pass dicts
    to override fields call by call (the last one repeats once they run
    out). Every (system, user, schema) call is recorded so a test can
    assert on the prompt that was built.
    """

    def __init__(self, *turns: dict) -> None:
        self.turns = list(turns)
        self.calls: list = []

    def complete(self, *, system: str, user: str, schema: dict,
                 images: tuple = ()) -> dict:
        self.calls.append((system, user, schema))
        self.images = images
        reply = {
            "text": "What made you choose that color?",
            "is_open": True,
            "asks_about_people": False,
            "engagement": "engaged",
            "child_wants_stop": False,
            "child_wants_more": False,
            "flag_own_turn": False,
        }
        if self.turns:
            index = min(len(self.calls) - 1, len(self.turns) - 1)
            reply.update(self.turns[index])
        return reply


@pytest.fixture(autouse=True)
def _test_api_key(monkeypatch):
    monkeypatch.setitem(
        settings.API_KEYS, TEST_API_KEY, {"name": "Test User", "can_change_model": False}
    )


@pytest.fixture
def api_key():
    return TEST_API_KEY


@pytest.fixture
def make_fake_provider():
    return FakeProvider


@pytest.fixture
def fake_provider():
    return FakeProvider()


@pytest.fixture
def app(fake_provider):
    test_app = FastAPI()
    test_app.include_router(reflect_router)
    test_app.dependency_overrides[get_provider] = lambda: fake_provider
    return test_app


@pytest.fixture
def client(app):
    return TestClient(app)
