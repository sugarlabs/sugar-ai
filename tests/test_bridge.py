import json

import pytest

from app.providers.base import GenerationParams
from app.reflection.bridge import ProviderBridge

SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}}


class _StubProvider:
    def __init__(self, reply='{"text": "What changed?"}'):
        self.reply = reply
        self.received = None
        self.params = None

    def chat(self, messages, params=None):
        self.received = messages
        self.params = params
        return self.reply


def _complete(stub):
    return ProviderBridge(stub).complete(
        system="sys prompt", user="the chat so far", schema=SCHEMA
    )


def test_bridge_decodes_the_reply_against_the_asked_schema():
    stub = _StubProvider()
    assert _complete(stub) == {"text": "What changed?"}


def test_bridge_puts_the_schema_in_the_system_message():
    stub = _StubProvider()
    _complete(stub)
    system_msg, user_msg = stub.received
    assert system_msg["role"] == "system"
    assert system_msg["content"].startswith("sys prompt")
    assert json.dumps(SCHEMA) in system_msg["content"]
    assert user_msg == {"role": "user", "content": "the chat so far"}


def test_bridge_unwraps_a_fenced_reply():
    stub = _StubProvider('```json\n{"text": "What changed?"}\n```')
    assert _complete(stub) == {"text": "What changed?"}


def test_bridge_sizes_generation_for_reasoning_models():
    # 256 was the old free-text envelope; a schema reply plus any
    # reasoning preamble needs real room, or truncation silently eats
    # the JSON instead of surfacing as a guard event.
    stub = _StubProvider()
    _complete(stub)
    assert isinstance(stub.params, GenerationParams)
    assert stub.params.max_new_tokens >= 2048


def test_bridge_raises_on_empty_reply_instead_of_blank_jo():
    for reply in ("", "   \n"):
        with pytest.raises(RuntimeError):
            _complete(_StubProvider(reply))


def test_bridge_raises_on_a_reply_that_is_not_json():
    with pytest.raises(ValueError):
        _complete(_StubProvider("Sure! Here is my question: what changed?"))


def test_bridge_raises_on_json_that_is_not_an_object():
    with pytest.raises(ValueError):
        _complete(_StubProvider('["a", "list"]'))


def test_bridge_attaches_images_when_the_provider_can_carry_them():
    stub = _StubProvider()
    stub.supports_images = True
    ProviderBridge(stub).complete(
        system="s", user="the chat", schema=SCHEMA,
        images=(("the work itself", "image/png", b"png!"),),
    )
    _, user_msg = stub.received
    parts = user_msg["content"]
    assert parts[0] == {"type": "text", "text": "the chat"}
    assert parts[1]["text"] == "(the work itself:)"
    assert parts[2]["image_url"]["url"].startswith("data:image/png;base64,")


def test_bridge_drops_images_for_a_text_only_provider():
    stub = _StubProvider()
    result = ProviderBridge(stub).complete(
        system="s", user="the chat", schema=SCHEMA,
        images=(("the work itself", "image/png", b"png!"),),
    )
    assert result == {"text": "What changed?"}
    _, user_msg = stub.received
    assert user_msg["content"] == "the chat"
