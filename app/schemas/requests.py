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
from typing import Annotated, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

from app.schemas.content import (
    AudioPart,
    ContentPart,
    ImagePart,
    TextPart,
    modalities_of,
)

# An attachment is media only; the question itself carries the text.
MediaPart = Annotated[Union[ImagePart, AudioPart], Field(discriminator="type")]

# Generous bound; protects the backend from unbounded input, not a token limit.
MAX_QUESTION_CHARS = 32_000
MAX_CODE_CHARS = 64_000


class AskRequest(BaseModel):
    """JSON body for /ask and /ask-llm.

    /ask-llm also takes attachments, so a question can refer to a picture
    or a recording. /ask ignores them: retrieval is over text.
    """
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    attachments: Optional[List[MediaPart]] = Field(
        None, description="Images or audio the question refers to"
    )

    def as_content(self) -> Union[str, list]:
        """Return the question alone, or the question beside its media."""
        if not self.attachments:
            return self.question
        return [TextPart(type="text", text=self.question), *self.attachments]

    def modalities(self) -> set:
        return modalities_of(self.as_content())


class DebugRequest(BaseModel):
    """JSON body for /debug."""
    code: str = Field(..., min_length=1, max_length=MAX_CODE_CHARS)
    context: bool = Field(
        False,
        description="True explains what the code does; False debugs it",
    )


class ChatMessage(BaseModel):
    """One message in a chat conversation.

    content is a plain string, as it always has been, or a list of typed
    parts when the message carries an image or a recording.
    """
    role: str  # "system", "user", "assistant"
    content: Union[str, List[ContentPart]]

    def modalities(self) -> set:
        """Return the modalities this message uses."""
        return modalities_of(self.content)

    def text(self) -> str:
        """Return the message's text, ignoring any non-text parts."""
        if isinstance(self.content, str):
            return self.content
        return " ".join(part.text for part in self.content if part.type == "text")


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
    attachments: Optional[List[MediaPart]] = Field(
        None, description="Images or audio the question refers to (prompted mode)"
    )

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
