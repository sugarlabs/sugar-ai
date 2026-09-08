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

"""Request contracts for the Sugar-AI API."""
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

# Generous bound; protects the backend from unbounded input, not a token limit.
MAX_QUESTION_CHARS = 32_000
MAX_CODE_CHARS = 64_000


class AskRequest(BaseModel):
    """JSON body for /ask and /ask-llm."""
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)


class DebugRequest(BaseModel):
    """JSON body for /debug."""
    code: str = Field(..., min_length=1, max_length=MAX_CODE_CHARS)
    context: bool = Field(
        False,
        description="True explains what the code does; False debugs it",
    )


class ChatMessage(BaseModel):
    """One message in a chat conversation."""
    role: str  # "system", "user", "assistant"
    content: str


class PromptedLLMRequest(BaseModel):
    """JSON body for /ask-llm-prompted.

    The endpoint serves two modes. chat=False answers a single question
    under a custom system prompt; chat=True continues a conversation.
    Each mode requires its own fields, enforced below.
    """
    chat: bool = Field(False, description="Enable chat mode (uses messages instead of question)")
    question: Optional[str] = Field(None, description="The question to ask (required if chat=False)")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt to replace system prompt (required if chat=False)")
    messages: Optional[List[ChatMessage]] = Field(None, description="List of chat messages (required if chat=True)")

    # Boundary validation added below:
    max_length: int = Field(1024, gt=0, le=8192, description="Maximum length of generated text")
    truncation: bool = Field(True, description="Whether to truncate input if too long")
    repetition_penalty: float = Field(1.1, gt=0.0, le=2.0, description="Repetition penalty")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for sampling")
    top_p: float = Field(0.9, gt=0.0, le=1.0, description="Top-p (nucleus) sampling parameter")
    top_k: int = Field(50, ge=0, description="Top-k sampling parameter")

    @model_validator(mode="after")
    def check_mode_fields(self):
        """Require the fields the selected mode actually uses."""
        if self.chat:
            if not self.messages:
                raise ValueError("messages is required when chat=True")
        else:
            missing = [
                name
                for name, value in (
                    ("question", self.question),
                    ("custom_prompt", self.custom_prompt),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f"{' and '.join(missing)} "
                    f"{'are' if len(missing) > 1 else 'is'} required when chat=False"
                )
        return self
