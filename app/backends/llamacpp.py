"""app/backends/llamacpp.py - llama-cpp-python GGUF backend.

Wraps the existing sugar-ai GGUF inference path in the standard
backend interface. Kept for backward compatibility with the existing
LlaMA-135-Claude-RUN2-q4.gguf model already in use.

Config keys:
  model_path    : path to .gguf file (required)
  n_ctx         : context size (default: 2048)
  n_threads     : CPU threads (default: 1)
  verbose       : bool (default: False)
"""

from __future__ import annotations

import logging
import os

from app.backends.base import (
    BackendCapabilities,
    BackendError,
    BackendResponse,
    BackendUnavailableError,
    GenerationConfig,
    Message,
    ModelBackend,
)

logger = logging.getLogger("sugar_ai.backends.llamacpp")

_llama = None


def _import_deps():
    global _llama
    if _llama is None:
        try:
            from llama_cpp import Llama
            _llama = Llama
        except ImportError as e:
            raise BackendUnavailableError(
                f"llama-cpp backend requires 'llama-cpp-python': {e}",
                backend="llamacpp",
            )
    return _llama


class LlamaCppBackend(ModelBackend):
    """llama-cpp-python GGUF inference backend."""

    name = "llamacpp"

    def __init__(self, config: dict):
        super().__init__(config)
        self._model_path = config.get("model_path", "")
        self._n_ctx = int(config.get("n_ctx", 2048))
        self._n_threads = int(config.get("n_threads", 1))
        self._verbose = bool(config.get("verbose", False))
        self._model = None

    def is_available(self) -> bool:
        try:
            _import_deps()
            return bool(self._model_path and os.path.exists(self._model_path))
        except BackendUnavailableError:
            return False

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=False,
            chat_history=True,
            system_prompt=True,
            token_counting=False,
            model_switching=False,
            local=True,
        )

    def ask(
        self,
        question: str,
        history: list[Message] | None = None,
        config: GenerationConfig | None = None,
    ) -> BackendResponse:
        config = config or GenerationConfig()
        Llama = _import_deps()
        self._ensure_loaded(Llama)

        messages = self._build_messages(question, history)

        try:
            def _infer():
                return self._model.create_chat_completion(
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repeat_penalty=config.repeat_penalty,
                    stop=config.stop_sequences,
                )

            response, latency_ms = self._timed_call(_infer)
            content = response["choices"][0]["message"]["content"].strip()
            usage = response.get("usage", {})

            return BackendResponse(
                content=content,
                model=os.path.basename(self._model_path),
                backend=self.name,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                latency_ms=latency_ms,
            )

        except Exception as e:
            raise BackendError(
                f"llama-cpp inference failed: {e}", backend=self.name
            ) from e

    def _ensure_loaded(self, Llama) -> None:
        if self._model is not None:
            return
        logger.info("Loading GGUF model: %s", self._model_path)
        self._model = Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            verbose=self._verbose,
            chat_format="chatml",
        )
        logger.info("GGUF model loaded")
