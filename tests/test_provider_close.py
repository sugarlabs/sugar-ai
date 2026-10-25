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

"""Providers must release their HTTP client on close().

The provider modules are loaded by path so this test only needs httpx --
importing app.providers would pull in torch via the HuggingFace provider.
"""
import importlib.util
import pathlib
import sys

import pytest

_PROVIDERS = pathlib.Path(__file__).resolve().parent.parent / "app" / "providers"


def _load(name):
    """Import a provider module directly, bypassing app/__init__.py."""
    spec = importlib.util.spec_from_file_location(
        f"app.providers.{name}", _PROVIDERS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load("base")
gemini = _load("gemini")
ollama = _load("ollama")


@pytest.mark.parametrize(
    "make_provider",
    [
        lambda: base.BaseProvider(model_name="gpt-4o-mini", api_key="test-key"),
        lambda: gemini.GeminiProvider(model_name="gemini-2.0-flash", api_key="test-key"),
        lambda: ollama.OllamaProvider(model_name="llama3"),
    ],
    ids=["base", "gemini", "ollama"],
)
def test_close_releases_http_client(make_provider):
    """close() must shut the httpx client, not leak its connection pool."""
    provider = make_provider()
    assert provider._client.is_closed is False

    provider.close()

    assert provider._client.is_closed is True


def test_close_is_safe_without_http_client():
    """Providers with no _client (HuggingFaceProvider) must not raise.

    HuggingFaceProvider subclasses BaseProvider but loads a local model
    instead of calling super().__init__(), so it never has a _client.
    """
    provider = base.BaseProvider.__new__(base.BaseProvider)

    provider.close()
