"""
API routes for Sugar-AI.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.ai import RAGAgent
from app.config import settings
from app.database import get_db
from app.llm import (
    LLMConfigurationError,
    activate_llm_model,
)
from app.runtime import commit_model_and_sync_app


class ChatMessage(BaseModel):
    role: str
    content: str


class PromptedLLMRequest(BaseModel):
    """Request model for ask-llm-prompted endpoint."""

    chat: bool = Field(False, description="Enable chat mode (uses messages instead of question)")
    question: Optional[str] = Field(None, description="The question to ask (required if chat=False)")
    custom_prompt: Optional[str] = Field(None, description="Custom prompt to replace system prompt (required if chat=False)")
    messages: Optional[List[ChatMessage]] = Field(None, description="List of chat messages (required if chat=True)")
    max_length: int = Field(1024, description="Maximum length of generated text")
    truncation: bool = Field(True, description="Whether to truncate input if too long")
    temperature: float = Field(0.7, description="Temperature for sampling")
    top_p: float = Field(0.9, description="Top-p (nucleus) sampling parameter")


router = APIRouter(tags=["api"])
logger = logging.getLogger("sugar-ai")
user_quotas: Dict[str, Dict] = {}


def check_quota(api_key: str) -> bool:
    """Check if a user has exceeded their daily quota."""
    today = datetime.now().date()

    if api_key not in user_quotas:
        user_quotas[api_key] = {"count": 0, "date": today}
        return True

    if user_quotas[api_key]["date"] != today:
        user_quotas[api_key]["count"] = 0
        user_quotas[api_key]["date"] = today

    if user_quotas[api_key]["count"] >= settings.MAX_DAILY_REQUESTS:
        return False

    user_quotas[api_key]["count"] += 1
    return True


def verify_api_key(
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    request: Request = None,
):
    """Verify API key and check quota."""
    client_host = request.client.host if request else "unknown"
    if not api_key:
        logger.warning("API key missing: %s", client_host)
        raise HTTPException(status_code=401, detail="API key is missing")

    if api_key not in settings.API_KEYS:
        logger.warning("Invalid API key used: %s... from %s", api_key[:5], client_host)
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not check_quota(api_key):
        logger.warning("Quota exceeded for user: %s", settings.API_KEYS[api_key]["name"])
        raise HTTPException(status_code=429, detail="Daily request quota exceeded")

    return settings.API_KEYS[api_key]


def remaining_quota_for_user(user_info: dict) -> dict:
    api_key = next(
        key for key, value in settings.API_KEYS.items() if value["name"] == user_info["name"]
    )
    remaining = settings.MAX_DAILY_REQUESTS - user_quotas.get(api_key, {}).get("count", 0)
    return {"remaining": remaining, "total": settings.MAX_DAILY_REQUESTS}


def verify_model_change_access(api_key: str, password: str, request: Request = None) -> dict:
    client_ip = request.client.host if request else "unknown"
    logger.info(
        "REQUEST - /change-model - API Key: %s... - IP: %s",
        api_key[:5],
        client_ip,
    )

    if api_key not in settings.API_KEYS:
        logger.warning("Invalid API key used for model change: %s... from %s", api_key[:5], client_ip)
        raise HTTPException(status_code=401, detail="Invalid API key")

    user_info = settings.API_KEYS[api_key]
    if not user_info.get("can_change_model", False):
        logger.warning("Unauthorized model change attempt by: %s from %s", user_info["name"], client_ip)
        raise HTTPException(status_code=403, detail="User doesn't have permission to change model")

    if password != settings.MODEL_CHANGE_PASSWORD:
        logger.warning("Invalid password for model change by: %s from %s", user_info["name"], client_ip)
        raise HTTPException(status_code=403, detail="Invalid model change password")

    return user_info


def get_running_agent(request: Request) -> RAGAgent:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        detail = getattr(request.app.state, "startup_error", None) or "No active LLM model is configured."
        raise HTTPException(status_code=503, detail=detail)
    return agent


@router.post("/ask")
async def ask_question(
    question: str,
    request: Request,
    user_info: dict = Depends(verify_api_key),
):
    """Process a question using the RAG pipeline."""
    start_time = time.time()
    client_ip = request.client.host if request else "unknown"
    logger.info("REQUEST - /ask - User: %s - IP: %s - Question: %s...", user_info["name"], client_ip, question[:50])

    try:
        agent = get_running_agent(request)
        answer = agent.run(question)
        process_time = time.time() - start_time
        logger.info("RESPONSE - User: %s - Success - Time: %.2fs", user_info["name"], process_time)
        return {
            "answer": answer,
            "user": user_info["name"],
            "quota": remaining_quota_for_user(user_info),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ERROR - User: %s - Error: %s", user_info["name"], exc)
        raise HTTPException(status_code=500, detail=f"Error processing request: {exc}")


@router.post("/ask-llm")
async def ask_llm(
    question: str,
    request: Request,
    user_info: dict = Depends(verify_api_key),
):
    """Process a question with a direct LLM call (no retrieval)."""
    start_time = time.time()
    client_ip = request.client.host if request else "unknown"
    logger.info("REQUEST - /ask-llm - User: %s - IP: %s - Question: %s...", user_info["name"], client_ip, question[:50])

    try:
        agent = get_running_agent(request)
        answer = agent.run_direct(question)
        process_time = time.time() - start_time
        logger.info("RESPONSE - User: %s - Success - Time: %.2fs", user_info["name"], process_time)
        return {
            "answer": answer,
            "user": user_info["name"],
            "quota": remaining_quota_for_user(user_info),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ERROR - User: %s - Error: %s", user_info["name"], exc)
        raise HTTPException(status_code=500, detail=f"Error processing request: {exc}")


@router.post("/ask-llm-prompted")
async def ask_llm_prompted(
    request_data: PromptedLLMRequest,
    request: Request,
    user_info: dict = Depends(verify_api_key),
):
    """Custom prompt or chat-completions style endpoint."""
    start_time = time.time()
    client_ip = request.client.host if request else "unknown"

    try:
        agent = get_running_agent(request)
        if request_data.chat:
            if not request_data.messages:
                raise HTTPException(status_code=400, detail="messages field is required when chat=True")

            user_messages = [msg for msg in request_data.messages if msg.role == "user"]
            last_user_msg = user_messages[-1].content if user_messages else "No user message"
            logger.info(
                "REQUEST - /ask-llm-prompted (chat=True) - User: %s - IP: %s - Last message: %s...",
                user_info["name"],
                client_ip,
                last_user_msg[:200],
            )

            messages_dict = [{"role": msg.role, "content": msg.content} for msg in request_data.messages]
            answer = agent.run_chat_completion(
                messages=messages_dict,
                max_length=request_data.max_length,
                truncation=request_data.truncation,
                temperature=request_data.temperature,
                top_p=request_data.top_p,
            )

            process_time = time.time() - start_time
            logger.info(
                "RESPONSE - User: %s - Success - Time: %.2fs - Last message: %s...",
                user_info["name"],
                process_time,
                last_user_msg[:200],
            )
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": answer,
                    },
                    "index": 0,
                    "finish_reason": "stop",
                }],
                "user": user_info["name"],
                "quota": remaining_quota_for_user(user_info),
                "generation_params": {
                    "max_length": request_data.max_length,
                    "truncation": request_data.truncation,
                    "temperature": request_data.temperature,
                    "top_p": request_data.top_p,
                },
            }

        if not request_data.question or not request_data.custom_prompt:
            raise HTTPException(
                status_code=400,
                detail="question and custom_prompt fields are required when chat=False",
            )

        logger.info(
            "REQUEST - /ask-llm-prompted - User: %s - IP: %s - Question: %s...",
            user_info["name"],
            client_ip,
            request_data.question[:200],
        )
        logger.info(
            "CUSTOM PROMPT - User: %s - Prompt: %s...",
            user_info["name"],
            request_data.custom_prompt[:100],
        )

        answer = agent.run_with_custom_prompt(
            question=request_data.question,
            custom_prompt=request_data.custom_prompt,
            max_length=request_data.max_length,
            truncation=request_data.truncation,
            temperature=request_data.temperature,
            top_p=request_data.top_p,
        )
        process_time = time.time() - start_time
        logger.info("RESPONSE - User: %s - Success - Time: %.2fs", user_info["name"], process_time)
        return {
            "answer": answer,
            "user": user_info["name"],
            "quota": remaining_quota_for_user(user_info),
            "generation_params": {
                "max_length": request_data.max_length,
                "truncation": request_data.truncation,
                "temperature": request_data.temperature,
                "top_p": request_data.top_p,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ERROR - User: %s - Error: %s", user_info["name"], exc)
        raise HTTPException(status_code=500, detail=f"Error processing request: {exc}")


@router.post("/debug")
async def debug(
    code: str,
    context: bool,
    request: Request,
    user_info: dict = Depends(verify_api_key),
):
    """Process Python code for debugging."""
    start_time = time.time()
    client_ip = request.client.host if request else "unknown"
    logger.info("REQUEST - /debug - User: %s - IP: %s - code: %s...", user_info["name"], client_ip, code[:50])

    try:
        agent = get_running_agent(request)
        answer = agent.debug(code, context)
        process_time = time.time() - start_time
        logger.info("RESPONSE - User: %s - Success - Time: %.2fs", user_info["name"], process_time)
        return {
            "answer": answer,
            "user": user_info["name"],
            "quota": remaining_quota_for_user(user_info),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ERROR - User: %s - Error: %s", user_info["name"], exc)
        raise HTTPException(status_code=500, detail=f"Error processing request: {exc}")


@router.post("/change-model")
async def change_model(
    request: Request,
    model_id: int = Query(...),
    api_key: str = Query(...),
    password: str = Query(...),
    db: Session = Depends(get_db),
):
    """Change the active LLM model config (admin only)."""
    user_info = verify_model_change_access(api_key, password, request)

    try:
        active_model = activate_llm_model(db, model_id, auto_commit=False)
        active_model = commit_model_and_sync_app(request.app, db, active_model)
        logger.info("Active model changed to %s by %s", active_model.name, user_info["name"])
        return {
            "message": f"Active model changed to {active_model.name}",
            "user": user_info["name"],
            "model": active_model.to_dict(),
        }
    except LLMConfigurationError as exc:
        db.rollback()
        logger.error("Error changing model id %s by %s: %s", model_id, user_info["name"], exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        db.rollback()
        logger.error("Error changing model id %s by %s: %s", model_id, user_info["name"], exc)
        raise HTTPException(status_code=500, detail=f"Error changing model: {exc}")
