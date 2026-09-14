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


"""HuggingFace Transformers provider for Sugar-AI."""
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
from typing import Optional
from starlette.concurrency import run_in_threadpool

from app.providers.base import BaseProvider, GenerationParams

logger = logging.getLogger("sugar-ai")


class HuggingFaceProvider(BaseProvider):
    """Provider running HuggingFace models locally via transformers."""

    def __init__(self, model_name: str, quantize: bool = True, dev_mode: bool = False):
        """Load a HuggingFace model into memory."""
        self.model_name = model_name
        self._dev_mode = dev_mode

        use_quant = quantize and torch.cuda.is_available() and not dev_mode
        device = 0 if torch.cuda.is_available() and not dev_mode else -1
        dtype = torch.float16 if device == 0 else torch.float32

        if use_quant:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self._pipeline = pipeline(
                "text-generation",
                model=model_obj,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                truncation=True,
            )
        else:
            self._pipeline = pipeline(
                "text-generation",
                model=model_name,
                max_new_tokens=1024,
                truncation=True,
                torch_dtype=dtype,
                device=device,
            )

        logger.info("HuggingFaceProvider loaded model: %s (quantized=%s, device=%s)",
                    model_name, use_quant, device)

    async def generate(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
    ) -> str:
        """Generate text without blocking FastAPI's event loop."""
        params = params or GenerationParams()

        return await run_in_threadpool(self._generate_sync, prompt, params)

    def _generate_sync(self, prompt: str, params: GenerationParams) -> str:
        """Generate text from a plain string prompt."""

        response = self._pipeline(
            prompt,
            max_new_tokens=params.max_new_tokens,
            truncation=params.truncation,
            repetition_penalty=params.repetition_penalty,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            do_sample=params.do_sample,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
        )

        generated_text = response[0].get("generated_text", "")
        if isinstance(generated_text, str) and generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text

    async def chat(
        self,
        messages: list[dict],
        params: Optional[GenerationParams] = None,
    ) -> str:
        """Generate a chat response without blocking FastAPI's event loop."""
        params = params or GenerationParams()

        return await run_in_threadpool(self._chat_sync, messages, params)

    def _chat_sync(self, messages: list[dict], params: GenerationParams) -> str:
        """Generate response from chat messages."""
        normalized = self._normalize_chat_messages(messages)
        full_prompt = self._pipeline.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )

        response = self._pipeline(
            full_prompt,
            max_new_tokens=params.max_new_tokens,
            truncation=params.truncation,
            repetition_penalty=params.repetition_penalty,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            do_sample=params.do_sample,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
        )

        generated_text = response[0].get("generated_text", "")
        answer = self._extract_after_prompt(
            generated_text,
            full_prompt,
            getattr(self._pipeline.tokenizer, "eos_token", None),
        )

        return answer

    def get_eos_token(self) -> Optional[str]:
        return getattr(self._pipeline.tokenizer, "eos_token", None)

    async def close(self) -> None:
        """Release provider resources."""
        # The local pipeline does not own an HTTP client to close.
        return None

    async def health_check(self) -> bool:
        """Check the local pipeline without blocking FastAPI's event loop."""
        return await run_in_threadpool(self._health_check_sync)

    def _health_check_sync(self) -> bool:
        """Check if pipeline can generate text."""
        try:
            result = self._pipeline("test", max_new_tokens=1)
            return result is not None
        except Exception:
            return False

    def _normalize_chat_messages(self, messages: list[dict]) -> list[dict]:
        """Normalize messages for model-specific requirements."""
        system_content = ""
        for msg in messages:
            if msg.get("role") == "system" and msg.get("content"):
                system_content = msg["content"]
                break

        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
        if not non_system_messages:
            return []

        normalized = []
        first_role = non_system_messages[0].get("role")

        if first_role == "assistant" and system_content:
            normalized.append({"role": "user", "content": system_content})

        for i, msg in enumerate(non_system_messages):
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "assistant" and "gemma" in self.model_name.lower():
                role = "model"

            if role == "user" and i == 0 and first_role == "user" and system_content:
                content = f"{system_content}\n\n{content}"

            normalized.append({"role": role, "content": content})

        return normalized

    def _extract_after_prompt(self, full_text: str, prompt: str, eos_token: str = None) -> str:
        """Extract the model response, removing the echoed prompt."""
        if full_text.startswith(prompt):
            answer = full_text[len(prompt):].strip()
        else:
            answer = full_text.strip()

        if eos_token and eos_token in answer:
            answer = answer.split(eos_token)[0].strip()

        return answer
