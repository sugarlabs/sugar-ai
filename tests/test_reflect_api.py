from datetime import date

import pytest
from fastapi import HTTPException

from app.config import settings
from app.reflection.activities import category_for_activity
from app.reflection.bridge import ProviderBridge
from app.routes import api as api_routes
from app.routes.reflect import _quota_status, _to_engine_history, get_provider
from reflection_engine.trace import as_record
from reflection_engine.types import ChildTurn, EngineTurn

CHAT_PAYLOAD = {
    "title": "Rainbow Spirals",
    "description": "I made spirals with turtle",
    "activity_id": "org.laptop.TurtleArtActivity",
    "records": [],
}


def _engine_turn(text="What did you build?"):
    return as_record(EngineTurn(kind="question", text=text, open=True))


def _child_turn(text):
    return {"type": "child_turn", "by": "host", "text": text}


def _auth(api_key: str) -> dict:
    return {"X-API-Key": api_key}


# --- request/response mapping helpers ---

def test_category_for_activity_known_bundle_maps_to_category():
    assert category_for_activity("org.laptop.PippyActivity") == "programming"


def test_category_for_activity_unknown_bundle_defaults_to_creative():
    assert category_for_activity("org.example.SomeNewActivity") == "creative"


def test_to_engine_history_decodes_records_to_typed_objects():
    history = _to_engine_history([_engine_turn("q1"), _child_turn("a1")])
    assert isinstance(history[0], EngineTurn)
    assert isinstance(history[1], ChildTurn)
    assert history[1].text == "a1"


def test_to_engine_history_refuses_a_session_start_in_history():
    record = {
        "type": "session_start", "by": "host",
        "work": {"title": "t", "description": "", "category": "creative",
                 "previous_next_steps": None},
    }
    with pytest.raises(ValueError):
        _to_engine_history([record])


def test_quota_status_unknown_user_name_returns_full_remaining():
    # next() with no default would raise StopIteration for a name with
    # no matching API_KEYS entry; it must fall back to 0 used.
    assert _quota_status({"name": "nobody-registered"}) == {
        "remaining": settings.MAX_DAILY_REQUESTS,
        "total": settings.MAX_DAILY_REQUESTS,
    }


# --- /reflect/chat ---

def test_reflect_chat_happy_path(client, fake_provider, api_key):
    resp = client.post("/reflect/chat", json=CHAT_PAYLOAD, headers=_auth(api_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["record"]["type"] == "engine_turn"
    assert body["record"]["kind"] == "question"
    assert body["record"]["text"] == "What made you choose that color?"
    assert body["user"] == "Test User"
    assert body["quota"]["total"] == settings.MAX_DAILY_REQUESTS


def test_reflect_chat_first_turn_builds_context_from_the_work(client, fake_provider, api_key):
    resp = client.post("/reflect/chat", json=CHAT_PAYLOAD, headers=_auth(api_key))
    assert resp.status_code == 200
    assert len(fake_provider.calls) == 1
    system, user, schema = fake_provider.calls[0]
    assert "Rainbow Spirals" in user
    assert "I made spirals with turtle" in user
    assert isinstance(schema, dict)


def test_reflect_chat_requires_api_key(client):
    resp = client.post("/reflect/chat", json=CHAT_PAYLOAD)
    assert resp.status_code == 401


def test_reflect_chat_round_trips_history_records(client, fake_provider, api_key):
    payload = dict(CHAT_PAYLOAD, records=[
        _engine_turn("What did you build?"),
        _child_turn("a spiral machine"),
    ])
    resp = client.post("/reflect/chat", json=payload, headers=_auth(api_key))
    assert resp.status_code == 200
    assert resp.json()["record"]["type"] == "engine_turn"


def test_reflect_chat_rejects_an_out_of_contract_record(client, api_key):
    payload = dict(CHAT_PAYLOAD, records=[{"type": "mystery", "by": "host"}])
    resp = client.post("/reflect/chat", json=payload, headers=_auth(api_key))
    assert resp.status_code == 422
    assert "out of contract" in resp.json()["detail"]


def test_reflect_chat_rejects_an_unanswered_engine_turn(client, api_key):
    # The engine refuses to ask twice with no child turn between; that
    # refusal must surface as a client error, not a 500.
    payload = dict(CHAT_PAYLOAD, records=[_engine_turn()])
    resp = client.post("/reflect/chat", json=payload, headers=_auth(api_key))
    assert resp.status_code == 422


def test_reflect_chat_rejects_oversized_conversations(client, api_key):
    records = [_child_turn("hello")] * 65
    resp = client.post("/reflect/chat", json=dict(CHAT_PAYLOAD, records=records),
                       headers=_auth(api_key))
    assert resp.status_code == 422


def test_reflect_chat_rejects_an_oversized_record(client, api_key):
    resp = client.post(
        "/reflect/chat",
        json=dict(CHAT_PAYLOAD, records=[_child_turn("x" * 6001)]),
        headers=_auth(api_key),
    )
    assert resp.status_code == 422


@pytest.mark.parametrize("field, limit", [
    ("title", 512),
    ("description", 8192),
    ("activity_id", 128),
    ("previous_next_steps", 1000),
])
def test_reflect_chat_rejects_oversized_fields(client, api_key, field, limit):
    payload = dict(CHAT_PAYLOAD, **{field: "x" * (limit + 1)})
    resp = client.post("/reflect/chat", json=payload, headers=_auth(api_key))
    assert resp.status_code == 422


def test_reflect_chat_floors_the_turn_when_the_provider_raises(client, app, api_key):
    # A provider outage mid-call is the engine's floor_request, a 200
    # the client renders from its local floor bank -- not a 503. The
    # 503 is reserved for no model being loaded at all (get_provider).
    class _Boom:
        def complete(self, *, system, user, schema):
            raise RuntimeError("provider fell over")

    app.dependency_overrides[get_provider] = lambda: _Boom()
    resp = client.post("/reflect/chat", json=CHAT_PAYLOAD, headers=_auth(api_key))
    assert resp.status_code == 200
    body = resp.json()
    assert body["record"]["type"] == "engine_turn"
    assert body["record"]["kind"] == "floor_request"
    assert body["record"]["text"] is None


def test_reflect_chat_quota_remaining_counts_down_per_call(
        client, fake_provider, api_key, monkeypatch):
    monkeypatch.setitem(
        api_routes.user_quotas, api_key,
        {"count": 0, "date": date.today()})
    for n in range(1, 4):
        resp = client.post("/reflect/chat", json=CHAT_PAYLOAD,
                           headers=_auth(api_key))
        assert resp.status_code == 200
        assert resp.json()["quota"]["remaining"] == \
            settings.MAX_DAILY_REQUESTS - n


# --- the production provider seam (bypassed by the fixture override above) ---

def test_get_provider_returns_503_when_no_model_is_loaded(monkeypatch):
    monkeypatch.setattr(api_routes, "agent", None)
    with pytest.raises(HTTPException) as exc:
        get_provider()
    assert exc.value.status_code == 503


def test_get_provider_wraps_the_agents_provider_in_the_bridge(monkeypatch):
    class _StubProvider:
        def chat(self, messages, params=None):
            return '{"got": "json"}'

    class _StubAgent:
        provider = _StubProvider()

    monkeypatch.setattr(api_routes, "agent", _StubAgent())
    provider = get_provider()
    assert isinstance(provider, ProviderBridge)
    assert provider.complete(system="s", user="u", schema={}) == {"got": "json"}


# --- the real app, not the test harness's slim one ---

def test_real_app_registers_the_reflect_route():
    from fastapi.testclient import TestClient
    from app import create_app

    client = TestClient(create_app())
    # 401 (auth ran) proves the route is mounted; only 404 would fail this.
    assert client.post("/reflect/chat", json={}).status_code != 404


def test_reflect_chat_work_context_reaches_the_engine(client, fake_provider, api_key):
    payload = dict(CHAT_PAYLOAD, work_context={
        "images": [{"mime": "image/jpeg", "data": "anBnLWJ5dGVz",
                    "caption": "the jump finally worked"}],
        "spent_seconds": 300,
    })
    resp = client.post("/reflect/chat", json=payload, headers=_auth(api_key))
    assert resp.status_code == 200
    _, user, _ = fake_provider.calls[0]
    assert "the jump finally worked" in user
    assert "5 minutes" in user


def test_reflect_chat_rejects_context_the_engine_refuses(client, api_key):
    payload = dict(CHAT_PAYLOAD, work_context={
        "images": [{"mime": "image/png", "data": "not-base64!!"}],
    })
    resp = client.post("/reflect/chat", json=payload, headers=_auth(api_key))
    assert resp.status_code == 422


def test_reflect_chat_rejects_an_oversized_context(client, api_key):
    payload = dict(CHAT_PAYLOAD, work_context={"spent_seconds": 1,
                                               "pad": "x" * (6 * 1024 * 1024)})
    resp = client.post("/reflect/chat", json=payload, headers=_auth(api_key))
    assert resp.status_code == 422


def test_reflect_chat_context_cannot_override_server_fields(
        client, fake_provider, api_key):
    # The context envelope carries extras; the fields this server
    # derives itself must win over anything a client puts in it.
    payload = dict(CHAT_PAYLOAD, work_context={
        "title": "HIJACKED", "description": "INJECTED",
        "category": "conversation",
    })
    resp = client.post("/reflect/chat", json=payload, headers=_auth(api_key))
    assert resp.status_code == 200
    _, user, _ = fake_provider.calls[0]
    assert "HIJACKED" not in user
    assert "INJECTED" not in user
    assert "Rainbow Spirals" in user
