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
Reflection route for Sugar-AI.

The conversation crosses this server as the engine's own trace
records, both directions: the client sends the history as records,
this handler decodes them with the engine's from_record, calls
next_turn, and returns the result as one record via as_record. No
translation layer, no shapes of our own -- the engine spec's section
1 is the only wire definition.

Handlers are deliberately sync: the providers do blocking HTTP, and a
sync def hands them to Starlette's threadpool instead of parking the
event loop for the length of a model call.
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
import logging

from app.config import settings
from app.reflection.activities import category_for_activity
from app.reflection.bridge import ProviderBridge
from app.reflection.schemas import ReflectChatRequest, ReflectChatResponse
from app.routes import api as api_routes
from reflection_engine import next_turn
from reflection_engine.trace import as_record, from_record
from reflection_engine.types import ChildTurn, EngineTurn, Summary

router = APIRouter(tags=["reflection"])

# setup logging
logger = logging.getLogger("sugar-ai")

# What may appear in a request's history. A session_start is implied
# by the flat work fields, and a session_end means there is nothing
# left to ask; a client sending either has lost the plot, and 422
# says so rather than guessing.
_HISTORY_TYPES = (ChildTurn, EngineTurn, Summary)


def _to_engine_history(records: List[dict]) -> list:
    history = []
    for record in records:
        obj = from_record(record)  # ValueError when out of contract
        if not isinstance(obj, _HISTORY_TYPES):
            raise ValueError(
                f"a {record.get('type')} record does not belong in history"
            )
        history.append(obj)
    return history


def _log_safe(value: str) -> str:
    """One line, bounded -- request fields never format our log records."""
    return value.replace("\n", " ").replace("\r", " ")[:64]


def _quota_status(user_info: dict) -> dict:
    api_key = next(
        (key for key, value in settings.API_KEYS.items()
         if value["name"] == user_info["name"]),
        None,
    )
    remaining = (
        settings.MAX_DAILY_REQUESTS
        - api_routes.user_quotas.get(api_key, {}).get("count", 0)
    )
    return {"remaining": remaining, "total": settings.MAX_DAILY_REQUESTS}


def get_provider() -> ProviderBridge:
    """Adapt the app-wide RAGAgent's provider to the engine's seam."""
    if api_routes.agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ProviderBridge(api_routes.agent.provider)


@router.post("/reflect/chat", response_model=ReflectChatResponse)
def reflect_chat(
    request_data: ReflectChatRequest,
    user_info: dict = Depends(api_routes.verify_api_key),
    provider=Depends(get_provider),
    request: Request = None,
):
    """Advance the reflection session by one engine turn."""
    client_ip = request.client.host if request else "unknown"
    logger.info(
        f"REQUEST - /reflect/chat - User: {user_info['name']} - IP: {client_ip} "
        f"- Activity: {_log_safe(request_data.activity_id)}"
    )

    # Any rich context first, the flat fields after, so the
    # server-derived values always win; the merged work is decoded
    # by the engine's own from_record - this server never interprets
    # context, it only sets the ceiling on its size (schema).
    work_payload = {
        **(request_data.work_context or {}),
        "title": request_data.title,
        "description": request_data.description,
        "category": category_for_activity(request_data.activity_id),
        "previous_next_steps": request_data.previous_next_steps,
    }
    try:
        work = from_record(
            {"type": "session_start", "by": "host", "work": work_payload}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=422, detail=f"work context out of contract: {e}"
        )
    try:
        history = _to_engine_history(request_data.records)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"history out of contract: {e}")

    # A provider that raises mid-call floors the turn inside the
    # engine; this except only catches the engine refusing the call
    # shape itself (unanswered turn, summary in live history), which
    # is a client error, not an outage.
    try:
        result = next_turn(work, history, provider=provider)
    except ValueError as e:
        logger.error(
            f"ERROR - /reflect/chat - User: {user_info['name']} "
            f"- Error: {type(e).__name__}"
        )
        raise HTTPException(status_code=422, detail=f"history out of contract: {e}")

    return ReflectChatResponse(
        record=as_record(result),
        user=user_info["name"],
        quota=_quota_status(user_info),
    )
