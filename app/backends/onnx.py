"""app/backends/onnx.py - ONNX Runtime inference backend.

Zero C++ compilation. Works on x86, ARM, and Raspberry Pi.
Ideal for XO hardware and Debian deployments where llama-cpp fails.

Recommended model:
  microsoft/Phi-3-mini-4k-instruct-onnx (INT4, ~2GB, CPU-optimised)

Config keys:
  model_path      : path to .onnx model file (required)
  tokenizer_path  : path to tokenizer dir or HF model ID
  max_new_tokens  : int (default: 256)
  temperature     : float (default: 0.7)
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

import numpy as np

from app.backends.base import (
    BackendCapabilities,
    BackendError,
    BackendResponse,
    BackendUnavailableError,
    GenerationConfig,
    Message,
    ModelBackend,
)

logger = logging.getLogger("sugar_ai.backends.onnx")

_ort = None
_transformers = None


def _import_deps():
    global _ort, _transformers
    if _ort is None:
        try:
            import onnxruntime as ort
            import transformers
            _ort = ort
            _transformers = transformers
        except ImportError as e:
            raise BackendUnavailableError(
                f"ONNX backend requires 'onnxruntime' and 'transformers': {e}",
                backend="onnx",
            )
    return _ort, _transformers


class ONNXBackend(ModelBackend):
    """ONNX Runtime inference backend.

    Session pool: model loaded once, reused across all requests.
    """

    name = "onnx"

    def __init__(self, config: dict):
        super().__init__(config)
        self._model_path = config.get("model_path", "")
        self._tokenizer_path = config.get(
            "tokenizer_path",
            os.path.dirname(self._model_path) if self._model_path else "",
        )
        self._session = None
        self._tokenizer = None

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
            token_counting=True,
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
        ort, transformers = _import_deps()

        self._ensure_session(ort, transformers)

        messages = self._build_messages(question, history)
        prompt = self._apply_chat_template(messages)

        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="np",
                truncation=True,
                max_length=2048,
            )
            input_ids = inputs["input_ids"].astype(np.int64)
            prompt_len = input_ids.shape[1]

            def _generate():
                generated = input_ids.copy()
                for _ in range(config.max_tokens):
                    feed = {"input_ids": generated}
                    outputs = self._session.run(None, feed)
                    logits = outputs[0][0, -1, :]
                    next_token = self._sample(logits, config)
                    generated = np.concatenate(
                        [generated, np.array([[next_token]], dtype=np.int64)],
                        axis=1,
                    )
                    if next_token == self._tokenizer.eos_token_id:
                        break
                return generated

            generated, latency_ms = self._timed_call(_generate)
            new_tokens = generated[0, prompt_len:]
            content = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            return BackendResponse(
                content=content,
                model=os.path.basename(self._model_path),
                backend=self.name,
                input_tokens=prompt_len,
                output_tokens=len(new_tokens),
                latency_ms=latency_ms,
            )

        except Exception as e:
            raise BackendError(f"ONNX inference failed: {e}", backend=self.name) from e

    def _sample(self, logits: np.ndarray, config: GenerationConfig) -> int:
        if config.temperature == 0:
            return int(np.argmax(logits))
        logits = logits / config.temperature
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        if config.top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, config.top_p) + 1
            mask = np.zeros_like(probs)
            mask[sorted_idx[:cutoff]] = 1.0
            probs = probs * mask
            probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def _apply_chat_template(self, messages: list[dict]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        parts = [f"<|{m['role']}|>\n{m['content']}<|end|>" for m in messages]
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def _ensure_session(self, ort, transformers) -> None:
        if self._session is not None:
            return
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        logger.info("Loading ONNX model: %s", self._model_path)
        self._session = ort.InferenceSession(
            self._model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._tokenizer_path, trust_remote_code=False
        )
        logger.info("ONNX session ready")

    def count_tokens(self, text: str) -> int:
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return super().count_tokens(text)
