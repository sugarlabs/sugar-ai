"""
Context-window management for Sugar-AI.

Backend-agnostic: only requires a count_tokens(str) -> int callable.
When a provider abstraction is added (issue #117), only that callable changes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List
import logging

logger = logging.getLogger("sugar-ai")


@dataclass
class ModelMetadata:
    model_name: str
    context_window: int
    max_output_tokens: int

    @property
    def safe_input_budget(self) -> int:
        """Tokens available for input = context_window - reserved output space."""
        return self.context_window - self.max_output_tokens


@dataclass
class ContextManager:
    metadata: ModelMetadata
    count_tokens: Callable[[str], int]

    def trim_chat_history(
        self,
        messages: List[dict],
        system_tokens: int = 0,
        new_user_tokens: int = 0,
    ) -> List[dict]:
        """
        Drop oldest non-system turns until history fits within safe_input_budget.

        Rules:
        - System messages are never dropped.
        - Walks newest-to-oldest, skipping any turn that would overflow
          (does not stop at first overflow — a large old turn should not
          evict smaller newer turns that fit).
        - Returns [] for history if budget is already exhausted by system + user.
        """
        budget = self.metadata.safe_input_budget - system_tokens - new_user_tokens

        if budget <= 0:
            logger.warning(
                "safe_input_budget exhausted by system prompt + user turn alone "
                "(system_tokens=%d, new_user_tokens=%d, safe_input_budget=%d).",
                system_tokens,
                new_user_tokens,
                self.metadata.safe_input_budget,
            )
            return []

        system_msgs = [m for m in messages if m.get("role") == "system"]
        history = [m for m in messages if m.get("role") != "system"]

        kept: List[dict] = []
        used = 0
        for msg in reversed(history):
            tokens = self.count_tokens(msg.get("content", ""))
            if used + tokens <= budget:
                kept.append(msg)
                used += tokens
            else:
                logger.info(
                    "History trim: dropping role=%s tokens=%d (used=%d budget=%d).",
                    msg.get("role"),
                    tokens,
                    used,
                    budget,
                )

        kept.reverse()
        dropped = len(history) - len(kept)
        if dropped:
            logger.info("Trimmed %d message(s) from chat history.", dropped)

        return system_msgs + kept

    def fits_in_budget(self, text: str) -> bool:
        """True if text fits within safe_input_budget."""
        return self.count_tokens(text) <= self.metadata.safe_input_budget
