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

"""Tests for serialized Hugging Face pipeline access."""

import asyncio
import threading

from app.providers.huggingface import HuggingFaceProvider


def make_provider():
    """Build a provider shell without loading a real model."""
    provider = object.__new__(HuggingFaceProvider)
    provider._inference_queue = asyncio.Queue()
    provider._inference_worker = None
    provider._closed = False
    return provider


async def test_pipeline_operations_are_serialized():
    provider = make_provider()
    first_started = threading.Event()
    release_first = threading.Event()
    active_operations = 0
    maximum_active_operations = 0
    completed = []

    def operation(name, wait_for_release=False):
        nonlocal active_operations, maximum_active_operations
        active_operations += 1
        maximum_active_operations = max(maximum_active_operations, active_operations)
        if wait_for_release:
            first_started.set()
            assert release_first.wait(timeout=2)
        completed.append(name)
        active_operations -= 1
        return name

    provider._generate_sync = lambda prompt, params: operation(prompt, prompt == "first")
    provider._chat_sync = lambda messages, params: operation("chat")
    provider._health_check_sync = lambda: operation("health")

    first = asyncio.create_task(provider.generate("first"))
    await asyncio.to_thread(first_started.wait)

    chat = asyncio.create_task(provider.chat([]))
    health = asyncio.create_task(provider.health_check())
    await asyncio.sleep(0)

    assert completed == []
    assert maximum_active_operations == 1

    release_first.set()

    assert await first == "first"
    assert await chat == "chat"
    assert await health == "health"
    assert completed == ["first", "chat", "health"]
    assert maximum_active_operations == 1

    await provider.close()


async def test_closed_provider_rejects_new_inference():
    provider = make_provider()

    await provider.close()

    try:
        await provider.health_check()
    except RuntimeError as error:
        assert str(error) == "HuggingFaceProvider is closed"
    else:
        raise AssertionError("closed provider accepted an inference request")
