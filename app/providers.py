from abc import ABC, abstractmethod

import torch
from openai import OpenAI
from transformers import pipeline


class BaseProvider(ABC):
    """Abstract base class for AI model providers."""

    @abstractmethod
    def run(self, question: str) -> str:
        """Generate a response to a question."""
        pass

    @abstractmethod
    def run_chat_completion(self, messages: list, **kwargs) -> str:
        """Generate a response from a conversation history."""
        pass


class HuggingFaceProvider(BaseProvider):
    """AI provider using local HuggingFace models."""

    def __init__(self, model: str):
        self.model_name = model
        self.model = pipeline(
            "text-generation",
            model=model,
            max_new_tokens=1024,
            truncation=True,
            torch_dtype=torch.float16,
            device=0 if torch.cuda.is_available() else -1,
        )

    def run(self, question: str) -> str:
        """Generate a response to a question."""
        result = self.model(question, return_full_text=False)
        return result[0]["generated_text"]

    def run_chat_completion(self, messages: list, **kwargs) -> str:
        """Generate a response from a conversation history."""
        prompt = messages[-1]["content"] if messages else ""
        result = self.model(prompt, return_full_text=False, **kwargs)
        return result[0]["generated_text"]
    

    class OpenAIProvider(BaseProvider):
        """AI provider using OpenAI API."""

    def __init__(self, api_key: str, model: str):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def run(self, question: str) -> str:
        """Generate a response to a question."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": question}],
        )
        return response.choices[0].message.content

    def run_chat_completion(self, messages: list, **kwargs) -> str:
        """Generate a response from a conversation history."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content