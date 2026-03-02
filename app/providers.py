from abc import ABC, abstractmethod


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