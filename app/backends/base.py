"""app/backends/base.py - Abstract base class for all model backends.

Every backend in sugar-ai inherits from ModelBackend and implements
the same interface. The rest of the application never imports a
specific backend class directly — it goes through BackendRouter.

Design principles:
  - Backends are stateless across requests (no per-request session)
  - Conversation history is passed in, not stored in the backend
  - Token counting is best-effort, not exact (avoids heavy tokenizer deps)
  - All backends raise BackendError on failure, never raw exceptions
  - Backends declare their capabilities so the router can make decisions
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

logger = logging.getLogger("sugar_ai.backends")


# Data structures

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role.value, "content": self.content}


@dataclass
class GenerationConfig:
    """Unified generation parameters across all backends.

    Backends apply whichever of these they support and silently ignore
    the rest — no errors for unsupported params.
    """
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repeat_penalty: float = 1.1
    stop_sequences: list[str] = field(default_factory=lambda: ["User:", "\nUser:"])
    stream: bool = False


@dataclass
class BackendResponse:
    """Standardised response from any backend."""
    content: str
    model: str                        # actual model name used
    backend: str                      # backend family name
    input_tokens: int = 0             # best-effort estimate
    output_tokens: int = 0            # best-effort estimate
    latency_ms: float = 0.0
    finish_reason: str = "stop"       # stop | length | error
    raw: dict = field(default_factory=dict)  # raw provider response


@dataclass
class BackendCapabilities:
    """What a backend can do."""
    streaming: bool = False
    chat_history: bool = True         # supports multi-turn
    system_prompt: bool = True
    token_counting: bool = False      # can count tokens exactly
    model_switching: bool = False     # can switch models without restart
    local: bool = True                # runs locally (no external API)


# Exceptions

class BackendError(Exception):
    """Raised by any backend on unrecoverable failure."""

    def __init__(self, message: str, backend: str = "", retryable: bool = False):
        super().__init__(message)
        self.backend = backend
        self.retryable = retryable


class BackendUnavailableError(BackendError):
    """Backend is not installed or configured."""
    pass


class BackendTimeoutError(BackendError):
    """Backend took too long to respond."""
    def __init__(self, message: str, backend: str = ""):
        super().__init__(message, backend=backend, retryable=True)


class BackendQuotaError(BackendError):
    """API quota or rate limit exceeded."""
    def __init__(self, message: str, backend: str = ""):
        super().__init__(message, backend=backend, retryable=True)


# Abstract base

class ModelBackend(ABC):
    """Abstract base class for all sugar-ai model backends.

    Subclasses implement:
        - ask()          : single-turn or multi-turn inference
        - stream()       : token-by-token streaming (optional)
        - is_available() : check if backend can be used
        - capabilities   : property returning BackendCapabilities

    Subclasses must NOT:
        - Store conversation state between calls
        - Catch and swallow exceptions silently
        - Import heavy dependencies at module level (use lazy imports)
    """

    #: Human-readable name used in config and logs
    name: str = "base"

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Backend-specific configuration from sugar_ai.yaml or env vars.
            Each backend documents its own required and optional keys.
        """
        self._config = config
        self._system_prompt = config.get(
            "system_prompt",
            (
                "You are a helpful, friendly AI assistant for children "
                "using the Sugar learning platform. Keep answers short, "
                "simple, and encouraging. Never produce harmful content."
            ),
        )

    # Required interface

    @abstractmethod
    def ask(
        self,
        question: str,
        history: list[Message] | None = None,
        config: GenerationConfig | None = None,
    ) -> BackendResponse:
        """Generate a response to *question*.

        Parameters
        ----------
        question : str
            The user's input.
        history : list[Message] | None
            Prior conversation turns. May be None or empty for single-turn.
        config : GenerationConfig | None
            Generation parameters. If None, backend uses its defaults.

        Returns
        -------
        BackendResponse
            Standardised response. Never returns None.

        Raises
        ------
        BackendError
            On any failure. Subclasses use the appropriate subclass.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is installed and configured."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return what this backend supports."""
        ...

    # Optional interface (default implementations)

    def stream(
        self,
        question: str,
        history: list[Message] | None = None,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Stream tokens one by one.

        Default implementation calls ask() and yields the full response
        as a single chunk. Backends that support real streaming should
        override this.
        """
        response = self.ask(question, history=history, config=config)
        yield response.content

    def count_tokens(self, text: str) -> int:
        """Estimate token count. Override for exact counting."""
        # Rough approximation: 1 token ~= 4 chars for English
        return max(1, len(text) // 4)

    def count_history_tokens(self, history: list[Message]) -> int:
        """Estimate total tokens in a conversation history."""
        return sum(self.count_tokens(m.content) for m in history)

    def truncate_history(
        self,
        history: list[Message],
        max_tokens: int,
        reserve_tokens: int = 256,
    ) -> list[Message]:
        """Trim history so total tokens stay within budget.

        Removes oldest turns first. Always preserves the system message
        if present.

        Parameters
        ----------
        history : list[Message]
            Full conversation history.
        max_tokens : int
            Maximum allowed tokens for history.
        reserve_tokens : int
            Tokens to reserve for the response.
        """
        budget = max_tokens - reserve_tokens
        system_msgs = [m for m in history if m.role == Role.SYSTEM]
        other_msgs = [m for m in history if m.role != Role.SYSTEM]

        system_tokens = sum(self.count_tokens(m.content) for m in system_msgs)
        budget -= system_tokens

        # Keep newest turns, drop oldest
        kept = []
        running = 0
        for msg in reversed(other_msgs):
            t = self.count_tokens(msg.content)
            if running + t > budget:
                break
            kept.append(msg)
            running += t

        return system_msgs + list(reversed(kept))

    # Timing utility

    def _timed_call(self, fn, *args, **kwargs) -> tuple:
        """Call fn and return (result, latency_ms)."""
        t0 = time.monotonic()
        result = fn(*args, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000
        return result, latency_ms

    # Prompt building

    def _build_messages(
        self,
        question: str,
        history: list[Message] | None,
    ) -> list[dict]:
        """Build the messages list to send to the model.

        Prepends the system prompt and appends the current question.
        """
        messages = [{"role": Role.SYSTEM.value, "content": self._system_prompt}]
        if history:
            messages.extend(m.to_dict() for m in history)
        messages.append({"role": Role.USER.value, "content": question})
        return messages

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
