"""app/backends/huggingface.py - HuggingFace Transformers backend.

Covers any model loadable via AutoModelForCausalLM + AutoTokenizer.
This is the existing sugar-ai inference path (Qwen/Qwen2-1.5B-Instruct)
wrapped in the standard backend interface.

Supported models (non-exhaustive):
  - Qwen/Qwen2-1.5B-Instruct   (current sugar-ai default)
  - Qwen/Qwen2.5-1.5B-Instruct (updated version)
  - HuggingFaceTB/SmolLM2-1.7B-Instruct (lightweight)
  - google/gemma-2-2b-it        (Google, Apache 2.0)
  - microsoft/phi-2              (small, strong reasoning)

Config keys:
  model_name  : HuggingFace model ID (required)
  device      : "cpu" | "cuda" | "auto"  (default: "cpu")
  torch_dtype : "float32" | "float16" | "bfloat16" (default: "float32")
  max_context : int  maximum context tokens (default: 2048)
"""

from __future__ import annotations

import logging
from typing import Iterator

from app.backends.base import (
    BackendCapabilities,
    BackendError,
    BackendResponse,
    BackendUnavailableError,
    GenerationConfig,
    Message,
    ModelBackend,
)

logger = logging.getLogger("sugar_ai.backends.huggingface")

# Lazy imports — only loaded when backend is used
_transformers = None
_torch = None


def _import_deps():
    global _transformers, _torch
    if _transformers is None:
        try:
            import transformers
            import torch
            _transformers = transformers
            _torch = torch
        except ImportError as e:
            raise BackendUnavailableError(
                f"HuggingFace backend requires 'transformers' and 'torch': {e}",
                backend="huggingface",
            )
    return _transformers, _torch


class HuggingFaceBackend(ModelBackend):
    """HuggingFace Transformers inference backend.

    Wraps AutoModelForCausalLM + AutoTokenizer with the standard interface.
    Supports model hot-swapping via load_model().
    """

    name = "huggingface"

    def __init__(self, config: dict):
        super().__init__(config)
        self._model_name = config.get("model_name", "Qwen/Qwen2-1.5B-Instruct")
        self._device = config.get("device", "cpu")
        self._torch_dtype_str = config.get("torch_dtype", "float32")
        self._max_context = int(config.get("max_context", 2048))

        self._model = None
        self._tokenizer = None
        self._loaded_model_name = None

    # Backend interface

    def is_available(self) -> bool:
        try:
            _import_deps()
            return True
        except BackendUnavailableError:
            return False

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=False,
            chat_history=True,
            system_prompt=True,
            token_counting=True,   
            model_switching=True,
            local=True,
        )

    def ask(
        self,
        question: str,
        history: list[Message] | None = None,
        config: GenerationConfig | None = None,
    ) -> BackendResponse:
        config = config or GenerationConfig()
        transformers, torch = _import_deps()

        self._ensure_loaded()

        messages = self._build_messages(question, history)

        try:
            # Apply chat template
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self._tokenizer(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=self._max_context,
            ).to(self._model.device)

            input_len = inputs["input_ids"].shape[1]

            def _generate():
                with torch.no_grad():
                    return self._model.generate(
                        **inputs,
                        max_new_tokens=config.max_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        repetition_penalty=config.repeat_penalty,
                        do_sample=config.temperature > 0,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

            generated_ids, latency_ms = self._timed_call(_generate)

            # Decode only new tokens
            new_ids = generated_ids[0][input_len:]
            content = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            output_len = len(new_ids)

            return BackendResponse(
                content=content,
                model=self._loaded_model_name,
                backend=self.name,
                input_tokens=input_len,
                output_tokens=output_len,
                latency_ms=latency_ms,
            )

        except Exception as e:
            raise BackendError(
                f"HuggingFace inference failed: {e}", backend=self.name
            ) from e

    def count_tokens(self, text: str) -> int:
        """Exact token count using the loaded tokenizer."""
        if self._tokenizer is None:
            return super().count_tokens(text)
        return len(self._tokenizer.encode(text))

    
    # Model management
    

    def load_model(self, model_name: str) -> None:
        """Load or hot-swap to a different model."""
        if model_name == self._loaded_model_name:
            return

        transformers, torch = _import_deps()

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self._torch_dtype_str, torch.float32)

        logger.info("Loading HuggingFace model: %s", model_name)
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=False
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=self._device,
                trust_remote_code=False,
            )
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._loaded_model_name = model_name
            self._model_name = model_name
            logger.info("Model loaded: %s", model_name)

        except Exception as e:
            raise BackendError(
                f"Failed to load model {model_name!r}: {e}", backend=self.name
            ) from e

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load_model(self._model_name)

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_model_name
