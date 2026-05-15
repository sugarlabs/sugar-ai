from abc import ABC, abstractmethod
from typing import AsyncIterator

class BaseProvider(ABC):
    """Abstract base class for all sugar-ai LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        ...

    @abstractmethod
    async def stream(
        self, messages: list[dict], **kwargs
    ) -> AsyncIterator[str]:
        ...

    @abstractmethod
    def health_check(self) -> bool:
        ...