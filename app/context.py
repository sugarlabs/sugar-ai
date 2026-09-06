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

"""Provider-neutral context-window budgeting helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


# A conservative approximation used when a provider does not expose a tokenizer.
def estimate_tokens(text: str) -> int:
    """Estimate token count from text without a model-specific tokenizer."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class ContextBudget:
    """The input/output allocation for one model invocation."""

    context_window: int
    output_tokens: int

    @property
    def input_tokens(self) -> int:
        return max(1, self.context_window - self.output_tokens)


def _message_tokens(message: dict, counter: Callable[[str], int]) -> int:
    return counter(str(message.get("content", ""))) + 4


def _compact(message: dict, max_tokens: int, counter: Callable[[str], int]) -> dict:
    """Keep a bounded excerpt of a message while preserving its role."""
    content = str(message.get("content", ""))
    suffix = "\n[…truncated…]"
    if counter(content) <= max_tokens:
        return {"role": message.get("role", "user"), "content": content}
    if counter(suffix) > max_tokens:
        return {"role": message.get("role", "user"), "content": ""}

    # Binary search over characters, but validate every candidate with the
    # supplied counter. This works with both the fallback estimator and an
    # exact provider tokenizer; no chars-per-token assumption is required.
    low, high = 0, len(content)
    best = suffix
    while low <= high:
        midpoint = (low + high) // 2
        candidate = content[:midpoint].rstrip() + suffix
        if counter(candidate) <= max_tokens:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return {
        "role": message.get("role", "user"),
        "content": best,
    }


def fit_messages(
    messages: Iterable[dict],
    budget: ContextBudget,
    counter: Callable[[str], int] = estimate_tokens,
) -> list[dict]:
    """Fit chat messages into a budget, compressing older turns first.

    System instructions and the most recent turns are retained. Older turns are
    represented by a short deterministic summary, making the behavior portable
    across providers and safe when no summarization model is available.
    """
    source = [
        {"role": item.get("role", "user"), "content": str(item.get("content", ""))}
        for item in messages
    ]
    if not source:
        return []

    system = [item for item in source if item["role"] == "system"]
    turns = [item for item in source if item["role"] != "system"]
    system_tokens = sum(_message_tokens(item, counter) for item in system)
    available = max(1, budget.input_tokens - system_tokens)

    kept: list[dict] = []
    used = 0
    for item in reversed(turns):
        cost = _message_tokens(item, counter)
        if used + cost > available:
            break
        kept.append(item)
        used += cost
    kept.reverse()

    dropped = turns[: len(turns) - len(kept)]
    if dropped:
        summary_lines = []
        remaining = max(1, available - used - 12)
        for item in dropped:
            prefix = f"{item['role']}: "
            excerpt_budget = max(1, remaining // max(1, len(dropped)))
            excerpt = _compact({"role": "user", "content": prefix + item["content"]}, excerpt_budget, counter)["content"]
            summary_lines.append(excerpt)
        summary = {
            "role": "system",
            "content": "Earlier conversation (compressed):\n" + "\n".join(summary_lines),
        }
        system = system + [summary]

    result = system + kept
    # A summary has framing overhead too; repeatedly compact system content
    # until the same accounting used above proves it fits.
    while sum(_message_tokens(item, counter) for item in result) > budget.input_tokens:
        candidates = [
            (index, counter(item["content"]))
            for index, item in enumerate(result)
            if item["role"] == "system" and item["content"]
        ]
        if not candidates:
            # The latest turn is more useful than an old one when the budget is
            # exceptionally small, so discard the oldest non-system turn.
            if len(result) > 1:
                result.pop(0)
            else:
                break
            continue
        index, current = max(candidates, key=lambda pair: pair[1])
        compacted = _compact(result[index], max(1, current - 4), counter)
        if compacted["content"] == result[index]["content"]:
            # The truncation marker is larger than the remaining budget.
            result[index] = {"role": "system", "content": ""}
        else:
            result[index] = compacted
    return result


def fit_text(text: str, budget: ContextBudget, counter: Callable[[str], int] = estimate_tokens) -> str:
    """Trim a plain prompt to the input side of a context budget."""
    if counter(text) <= budget.input_tokens:
        return text
    return _compact({"role": "user", "content": text}, budget.input_tokens, counter)["content"]
