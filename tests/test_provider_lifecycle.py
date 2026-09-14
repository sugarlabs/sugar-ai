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

"""Tests for replacing a provider while requests are running."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.ai import RAGAgent


class FakeProvider:
    """Provider double whose generation can be held open on demand."""

    def __init__(self, model_name, log=None):
        self.model_name = model_name
        self.log = log if log is not None else []
        self.close_calls = 0
        self.close_error = None
        self.prompts = []
        self.entered = asyncio.Event()
        self.release = None

    def get_model_name(self):
        return self.model_name

    async def generate(self, prompt, params=None):
        self.prompts.append(prompt)
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        return f"{self.model_name}:{prompt}"

    async def health_check(self):
        return True

    async def close(self):
        self.close_calls += 1
        self.log.append(f"close:{self.model_name}")
        if self.close_error is not None:
            raise self.close_error


async def settle(turns=20):
    """Let pending tasks reach their next await point."""
    for _ in range(turns):
        await asyncio.sleep(0)


def make_agent(provider):
    agent = RAGAgent(provider)
    agent.retriever = Mock()
    agent.retriever.invoke = Mock(return_value=[])
    return agent


def factory_for(provider, log=None, delay=0.0):
    """Build a synchronous factory, as /change-model passes in."""

    def factory():
        if log is not None:
            log.append(f"build:enter:{provider.model_name}")
        if delay:
            time.sleep(delay)
        if log is not None:
            log.append(f"build:exit:{provider.model_name}")
        return provider

    return factory


# --- Holding a provider ----------------------------------------------------


async def test_use_provider_holds_the_current_provider():
    provider = FakeProvider("old")
    agent = make_agent(provider)

    async with agent.use_provider() as held:
        assert held is provider
        assert agent._provider_users[id(provider)] == 1

    assert agent._provider_users == {}


async def test_provider_is_released_when_the_request_raises():
    agent = make_agent(FakeProvider("old"))

    with pytest.raises(ValueError):
        async with agent.use_provider():
            raise ValueError("boom")

    assert agent._provider_users == {}


async def test_provider_is_released_when_the_request_is_cancelled():
    old = FakeProvider("old")
    old.release = asyncio.Event()
    agent = make_agent(old)

    request = asyncio.create_task(agent.generate("q"))
    await old.entered.wait()

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    await settle()

    assert agent._provider_users == {}
    await asyncio.wait_for(
        agent.replace_provider(factory_for(FakeProvider("new"))), timeout=2
    )


async def test_held_provider_is_reused_without_being_counted_twice():
    provider = FakeProvider("old")
    agent = make_agent(provider)

    async with agent.use_provider() as held:
        answer = await agent.generate("hi", provider=held)
        assert agent._provider_users[id(provider)] == 1

    assert answer == "old:hi"
    assert agent._provider_users == {}


# --- Swapping while requests run -------------------------------------------


async def test_swap_does_not_wait_for_active_requests():
    old = FakeProvider("old")
    old.release = asyncio.Event()
    agent = make_agent(old)
    new = FakeProvider("new")

    request = asyncio.create_task(agent.generate("q"))
    await old.entered.wait()

    await asyncio.wait_for(agent.replace_provider(factory_for(new)), timeout=2)

    assert agent.provider is new
    assert agent.model_name == "new"
    assert old.close_calls == 0

    old.release.set()
    assert await request == "old:q"


async def test_retired_provider_is_closed_by_its_last_request():
    old = FakeProvider("old")
    old.release = asyncio.Event()
    agent = make_agent(old)

    requests = [asyncio.create_task(agent.generate(f"q{i}")) for i in range(3)]
    await old.entered.wait()
    await settle()

    await agent.replace_provider(factory_for(FakeProvider("new")))
    assert old.close_calls == 0

    old.release.set()
    await asyncio.gather(*requests)
    await settle()

    assert old.close_calls == 1


async def test_requests_keep_working_during_a_swap():
    old = FakeProvider("old")
    old.release = asyncio.Event()
    agent = make_agent(old)
    new = FakeProvider("new")

    in_flight = asyncio.create_task(agent.generate("during"))
    await old.entered.wait()

    await agent.replace_provider(factory_for(new))

    # A request arriving right after the swap is served, not rejected.
    assert await agent.generate("after") == "new:after"

    old.release.set()
    assert await in_flight == "old:during"


async def test_unused_provider_is_closed_immediately():
    old = FakeProvider("old")
    agent = make_agent(old)

    await agent.replace_provider(factory_for(FakeProvider("new")))

    assert old.close_calls == 1
    assert agent._provider_users == {}


async def test_provider_factory_runs_off_the_event_loop():
    agent = make_agent(FakeProvider("old"))
    new = FakeProvider("new")
    factory = factory_for(new)

    with patch(
        "app.ai.run_in_threadpool",
        new_callable=AsyncMock,
        return_value=new,
    ) as run_in_threadpool:
        await agent.replace_provider(factory)

    run_in_threadpool.assert_awaited_once_with(factory)


# --- Failure paths ---------------------------------------------------------


async def test_failed_build_leaves_the_current_provider_serving():
    old = FakeProvider("old")
    agent = make_agent(old)

    def failing_factory():
        raise RuntimeError("model not found")

    with pytest.raises(RuntimeError, match="model not found"):
        await agent.replace_provider(failing_factory)

    assert agent.provider is old
    assert agent.model_name == "old"
    assert old.close_calls == 0
    assert await agent.generate("q") == "old:q"


async def test_failed_build_does_not_disturb_an_active_request():
    old = FakeProvider("old")
    old.release = asyncio.Event()
    agent = make_agent(old)

    request = asyncio.create_task(agent.generate("q"))
    await old.entered.wait()

    def failing_factory():
        raise RuntimeError("out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        await agent.replace_provider(failing_factory)

    old.release.set()
    assert await request == "old:q"
    assert agent.provider is old


async def test_replacement_is_closed_when_its_model_name_fails():
    old = FakeProvider("old")
    agent = make_agent(old)
    new = FakeProvider("new")
    new.get_model_name = Mock(side_effect=RuntimeError("no name"))

    with pytest.raises(RuntimeError, match="no name"):
        await agent.replace_provider(factory_for(new))

    assert new.close_calls == 1
    assert agent.provider is old
    assert await agent.generate("q") == "old:q"


async def test_failure_to_close_the_old_provider_is_swallowed():
    old = FakeProvider("old")
    old.close_error = RuntimeError("close failed")
    agent = make_agent(old)
    new = FakeProvider("new")

    await agent.replace_provider(factory_for(new))

    assert agent.provider is new
    assert await agent.generate("q") == "new:q"


async def test_close_failure_does_not_break_the_last_request():
    old = FakeProvider("old")
    old.release = asyncio.Event()
    old.close_error = RuntimeError("close failed")
    agent = make_agent(old)

    request = asyncio.create_task(agent.generate("q"))
    await old.entered.wait()
    await agent.replace_provider(factory_for(FakeProvider("new")))

    old.release.set()
    assert await request == "old:q"


# --- Mutual exclusion ------------------------------------------------------


async def test_concurrent_model_changes_do_not_interleave():
    log = []
    agent = make_agent(FakeProvider("old", log=log))
    first = FakeProvider("first", log=log)
    second = FakeProvider("second", log=log)

    await asyncio.gather(
        agent.replace_provider(factory_for(first, log=log, delay=0.05)),
        agent.replace_provider(factory_for(second, log=log)),
    )

    builds = [entry for entry in log if entry.startswith("build:")]
    for index in range(0, len(builds), 2):
        assert builds[index].startswith("build:enter")
        assert builds[index + 1].startswith("build:exit")
        assert builds[index].split(":")[-1] == builds[index + 1].split(":")[-1]

    assert agent.provider in (first, second)


async def test_every_retired_provider_is_closed_after_repeated_swaps():
    old = FakeProvider("old")
    first = FakeProvider("first")
    second = FakeProvider("second")
    agent = make_agent(old)

    await agent.replace_provider(factory_for(first))
    await agent.replace_provider(factory_for(second))

    assert old.close_calls == 1
    assert first.close_calls == 1
    assert second.close_calls == 0
    assert agent.provider is second


# --- End to end ------------------------------------------------------------


async def test_provider_swap_under_concurrent_load():
    """Swap providers while five requests are in flight."""
    old = FakeProvider("old")
    old.release = asyncio.Event()
    agent = make_agent(old)
    new = FakeProvider("new")

    in_flight = [asyncio.create_task(agent.generate(f"q{i}")) for i in range(5)]
    await old.entered.wait()
    await settle()

    await asyncio.wait_for(agent.replace_provider(factory_for(new)), timeout=2)

    # Traffic continues on the new provider while the old ones are still busy.
    assert await agent.generate("during") == "new:during"
    assert old.close_calls == 0

    old.release.set()
    assert await asyncio.gather(*in_flight) == [f"old:q{i}" for i in range(5)]
    await settle()

    assert old.close_calls == 1
    assert agent._provider_users == {}
