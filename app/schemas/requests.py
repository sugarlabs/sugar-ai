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
from pydantic import BaseModel, Field

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
