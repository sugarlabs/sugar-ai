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

"""Response contracts for the Sugar-AI API.

These models describe exactly what each endpoint returns. Field names
and shapes match what the endpoints already emit, so declaring them is
a formalization, not a behavior change.
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class QuotaInfo(BaseModel):
    """Remaining daily request allowance for the calling API key."""
    remaining: int
    total: int


class AskResponse(BaseModel):
    """Response for /ask, /ask-llm and /debug."""
    answer: str
    user: str
    quota: QuotaInfo


class GenerationParamsInfo(BaseModel):
    """Echo of the generation parameters a request was served with."""
    max_length: int
    truncation: bool
    repetition_penalty: float
    temperature: float
    top_p: float
    top_k: int


class PromptedResponse(AskResponse):
    """Response for /ask-llm-prompted in prompted mode (chat=False)."""
    generation_params: GenerationParamsInfo


class ChatMessageOut(BaseModel):
    """A single assistant message in a chat completion."""
    role: str
    content: str


class ChatChoice(BaseModel):
    """One completion choice, mirroring the OpenAI chat format."""
    message: ChatMessageOut
    index: int = 0
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """Response for /ask-llm-prompted in chat mode (chat=True)."""
    choices: List[ChatChoice]
    user: str
    quota: QuotaInfo
    generation_params: GenerationParamsInfo


class HealthResponse(BaseModel):
    """Response for /health."""
    status: str
    provider: Optional[str] = None
    model: Optional[str] = None
    detail: Optional[str] = None


class ModelChangeResponse(BaseModel):
    """Response for /change-model."""
    message: str
    user: str


class ErrorDetail(BaseModel):
    """Machine-readable error code plus a human-readable message."""
    code: str = Field(..., description="Stable identifier, e.g. 'unauthorized'")
    message: str


class ErrorResponse(BaseModel):
    """Uniform envelope for every API error."""
    error: ErrorDetail
