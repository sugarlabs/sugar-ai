"""app/backends/router.py - Backend router and registry.

The router is the single entry point for all inference requests.
It selects the appropriate backend based on configuration, handles
fallback chains, and provides health checking.

Architecture:
  BackendRegistry  - knows all registered backend classes
  BackendRouter    - instantiates backends from config, routes requests

Usage:
  router = BackendRouter.from_config(config)
  response = router.ask("What is photosynthesis?", history=[...])
"""

from __future__ import annotations

import logging
from typing import Iterator, Type

from app.backends.base import (
    BackendError,
    BackendResponse,
    BackendUnavailableError,
    GenerationConfig,
    Message,
    ModelBackend,
)

logger = logging.getLogger("sugar_ai.backends.router")



# Registry

class BackendRegistry:
    """Maps backend names to backend classes."""

    _registry: dict[str, Type[ModelBackend]] = {}

    @classmethod
    def register(cls, backend_cls: Type[ModelBackend]) -> Type[ModelBackend]:
        """Register a backend class. Used as a decorator."""
        cls._registry[backend_cls.name] = backend_cls
        logger.debug("Registered backend: %s", backend_cls.name)
        return backend_cls

    @classmethod
    def get(cls, name: str) -> Type[ModelBackend]:
        if name not in cls._registry:
            raise BackendUnavailableError(
                f"Unknown backend: {name!r}. "
                f"Available: {list(cls._registry.keys())}",
                backend=name,
            )
        return cls._registry[name]

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry.keys())


# Register all built-in backends
def _register_builtins():
    from app.backends.huggingface import HuggingFaceBackend
    from app.backends.llamacpp import LlamaCppBackend
    from app.backends.onnx import ONNXBackend
    from app.backends.openai_compat import OpenAICompatBackend

    BackendRegistry.register(HuggingFaceBackend)
    BackendRegistry.register(LlamaCppBackend)
    BackendRegistry.register(ONNXBackend)
    BackendRegistry.register(OpenAICompatBackend)


_register_builtins()


# Router

class BackendRouter:
    """Routes inference requests to the appropriate backend.

    Supports:
    - Primary backend selection by name
    - Fallback chain: if primary fails, try next
    - Token budget enforcement (truncate history if needed)
    - Health checking
    """

    def __init__(
        self,
        primary: ModelBackend,
        fallbacks: list[ModelBackend] | None = None,
        max_history_tokens: int = 1500,
    ):
        self._primary = primary
        self._fallbacks = fallbacks or []
        self._max_history_tokens = max_history_tokens

        logger.info(
            "BackendRouter initialized: primary=%s, fallbacks=%s",
            primary.name,
            [f.name for f in self._fallbacks],
        )

    @classmethod
    def from_config(cls, config: "SugarAIConfig") -> "BackendRouter":
        """Build a router from a SugarAIConfig object."""
        primary_cfg = config.primary_backend
        primary_cls = BackendRegistry.get(primary_cfg["type"])
        primary = primary_cls(primary_cfg)

        fallbacks = []
        for fb_cfg in config.fallback_backends:
            fb_cls = BackendRegistry.get(fb_cfg["type"])
            fb = fb_cls(fb_cfg)
            fallbacks.append(fb)

        return cls(
            primary=primary,
            fallbacks=fallbacks,
            max_history_tokens=config.max_history_tokens,
        )

    def ask(
        self,
        question: str,
        history: list[Message] | None = None,
        config: GenerationConfig | None = None,
    ) -> BackendResponse:
        """Route an inference request through the backend chain."""
        history = self._maybe_truncate(history)

        backends = [self._primary] + self._fallbacks
        last_error = None

        for backend in backends:
            if not backend.is_available():
                logger.warning("Backend %s not available, skipping", backend.name)
                continue
            try:
                logger.debug("Routing to backend: %s", backend.name)
                return backend.ask(question, history=history, config=config)
            except BackendError as e:
                logger.warning(
                    "Backend %s failed (%s), trying next...",
                    backend.name, e
                )
                last_error = e
                continue

        raise BackendError(
            f"All backends exhausted. Last error: {last_error}",
            backend="router",
        )

    def stream(
        self,
        question: str,
        history: list[Message] | None = None,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        """Stream tokens. Falls back to non-streaming if primary doesn't support it."""
        history = self._maybe_truncate(history)

        if self._primary.is_available() and self._primary.capabilities.streaming:
            yield from self._primary.stream(question, history=history, config=config)
        else:
            response = self.ask(question, history=history, config=config)
            yield response.content

    def health(self) -> dict:
        """Return health status of all backends."""
        return {
            "primary": {
                "name": self._primary.name,
                "available": self._primary.is_available(),
                "capabilities": {
                    k: v for k, v in
                    self._primary.capabilities.__dict__.items()
                },
            },
            "fallbacks": [
                {
                    "name": fb.name,
                    "available": fb.is_available(),
                }
                for fb in self._fallbacks
            ],
        }

    def _maybe_truncate(
        self,
        history: list[Message] | None,
    ) -> list[Message] | None:
        if not history:
            return history
        total = self._primary.count_history_tokens(history)
        if total > self._max_history_tokens:
            logger.debug(
                "Truncating history: %d tokens > %d limit",
                total, self._max_history_tokens,
            )
            return self._primary.truncate_history(
                history, self._max_history_tokens
            )
        return history

    @property
    def primary(self) -> ModelBackend:
        return self._primary

    @property
    def fallbacks(self) -> list[ModelBackend]:
        return self._fallbacks
