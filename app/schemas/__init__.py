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

"""
Schema package for Sugar-AI.

Pydantic models that define the API's input and output contracts.
"""
from app.schemas.requests import AskRequest, DebugRequest
from app.schemas.responses import (
    AskResponse,
    ChatChoice,
    ChatCompletionResponse,
    ChatMessageOut,
    ErrorDetail,
    ErrorResponse,
    GenerationParamsInfo,
    HealthResponse,
    ModelChangeResponse,
    PromptedResponse,
    QuotaInfo,
)

__all__ = [
    "AskRequest",
    "DebugRequest",
    "AskResponse",
    "ChatChoice",
    "ChatCompletionResponse",
    "ChatMessageOut",
    "ErrorDetail",
    "ErrorResponse",
    "GenerationParamsInfo",
    "HealthResponse",
    "ModelChangeResponse",
    "PromptedResponse",
    "QuotaInfo",
]
