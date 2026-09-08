"""
API routes for Sugar-AI.
"""
from fastapi import APIRouter, Body, Depends, HTTPException, Header, Query, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, ValidationError
import time
import logging
import os
import json
from datetime import datetime
from typing import Dict, Optional, List

from typing import Union

from app.database import get_db, APIKey
from app.ai import RAGAgent
from app.providers.base import GenerationParams
from app.config import settings
from app.schemas.content import messages_to_provider
from app.schemas.requests import (
    AskRequest,
    ChatMessage,
    DebugRequest,
    PromptedLLMRequest,
)
from app.schemas import (
    AskResponse,
    ChatChoice,
    ChatCompletionResponse,
    ChatMessageOut,
    ErrorResponse,
    GenerationParamsInfo,
    HealthResponse,
    ModelChangeResponse,
    PromptedResponse,
    QuotaInfo,
)

# Documented error responses shared by all authenticated endpoints.
ERROR_RESPONSES = {
    401: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}

router = APIRouter(tags=["api"])

# setup logging
logger = logging.getLogger("sugar-ai")

# Initialize the agent
agent = None

# user quotas tracking
user_quotas: Dict[str, Dict] = {}

def check_quota(api_key: str) -> bool:
    """Check if a user has exceeded their daily quota"""
    today = datetime.now().date()
    
    if api_key not in user_quotas:
        user_quotas[api_key] = {"count": 0, "date": today}
        return True
        
    # reset quota daily
    if user_quotas[api_key]["date"] != today:
        user_quotas[api_key]["count"] = 0
        user_quotas[api_key]["date"] = today
        
    if user_quotas[api_key]["count"] >= settings.MAX_DAILY_REQUESTS:
        return False
        
    user_quotas[api_key]["count"] += 1
    return True

def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key"), request: Request = None):
    """Verify API key and check quota"""
    if not api_key:
        logger.warning(f"API key missing: {request.client.host if request else 'unknown'}")
        raise HTTPException(status_code=401, detail="API key is missing")
    
    if api_key not in settings.API_KEYS:
        logger.warning(f"Invalid API key used: {api_key[:5]}... from {request.client.host if request else 'unknown'}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    if not check_quota(api_key):
        logger.warning(f"Quota exceeded for user: {settings.API_KEYS[api_key]['name']}")
        raise HTTPException(status_code=429, detail="Daily request quota exceeded")
    
    return settings.API_KEYS[api_key]

def _resolve_ask(body: Optional[AskRequest], question: Optional[str]) -> AskRequest:
    """Take the JSON body, else fall back to the legacy ?question= parameter."""
    if body is not None:
        return body
    if question is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "validation_error",
                "message": "question is required, in the JSON body or as a query parameter",
            },
        )
    try:
        return AskRequest(question=question)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "validation_error", "message": _first_error(e)},
        )


def _first_error(exc: ValidationError) -> str:
    """Render a Pydantic error the same way the request-validation handler does."""
    errors = exc.errors()
    if not errors:
        return "Invalid request"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", []))
    message = first.get("msg", "Invalid request")
    return f"{location}: {message}" if location else message


@router.post("/ask", response_model=AskResponse, responses=ERROR_RESPONSES)
async def ask_question(
    body: Optional[AskRequest] = Body(None),
    question: Optional[str] = Query(
        None, description="Deprecated: send a JSON body instead"
    ),
    user_info: dict = Depends(verify_api_key),
    request: Request = None
):
    """Process a question using RAG pipeline"""
    start_time = time.time()
    question = _resolve_ask(body, question).question

    client_ip = request.client.host if request else "unknown"
    logger.info(f"REQUEST - /ask - User: {user_info['name']} - IP: {client_ip} - Question: {question[:50]}...")
    
    try:
        answer = agent.run(question)
        
        # log completion
        process_time = time.time() - start_time
        logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s")
        
        # check quota
        api_key = next(
            key for key, value in settings.API_KEYS.items()
            if value['name'] == user_info['name']
        )
        remaining = (
            settings.MAX_DAILY_REQUESTS
            - user_quotas.get(api_key, {}).get("count", 0)
        )
        
        return AskResponse(
            answer=answer,
            user=user_info["name"],
            quota=QuotaInfo(remaining=remaining, total=settings.MAX_DAILY_REQUESTS),
        )
    except Exception as e:
        logger.error(f"ERROR - User: {user_info['name']} - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/ask-llm", response_model=AskResponse, responses=ERROR_RESPONSES)
async def ask_llm(
    body: Optional[AskRequest] = Body(None),
    question: Optional[str] = Query(
        None, description="Deprecated: send a JSON body instead"
    ),
    user_info: dict = Depends(verify_api_key),
    request: Request = None
):
    """Process a question with direct LLM call (no retrieval)"""
    start_time = time.time()
    question = _resolve_ask(body, question).question

    client_ip = request.client.host if request else "unknown"
    logger.info(f"REQUEST - /ask-llm - User: {user_info['name']} - IP: {client_ip} - Question: {question[:50]}...")
    
    try:
        answer = agent.provider.generate(question)
        
        process_time = time.time() - start_time
        logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s")
        
        # check quota
        api_key = next(key for key, value in settings.API_KEYS.items() if value['name'] == user_info['name'])
        remaining = settings.MAX_DAILY_REQUESTS - user_quotas.get(api_key, {}).get("count", 0)
        
        return AskResponse(
            answer=answer,
            user=user_info["name"],
            quota=QuotaInfo(remaining=remaining, total=settings.MAX_DAILY_REQUESTS),
        )
    except Exception as e:
        logger.error(f"ERROR - User: {user_info['name']} - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

def _require_supported_modalities(messages) -> None:
    """Refuse a request whose media the active provider cannot accept."""
    requested = set()
    for message in messages:
        requested |= message.modalities()

    supported = getattr(agent.provider, "supported_modalities", {"text"})
    unsupported = sorted(requested - set(supported))
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "modality_not_supported",
                "message": (
                    f"{agent.provider.get_model_name()} does not accept "
                    f"{', '.join(unsupported)} input; it accepts "
                    f"{', '.join(sorted(supported))}"
                ),
            },
        )


def _generation_params_info(request_data: PromptedLLMRequest) -> GenerationParamsInfo:
    """Echo the generation parameters a request was served with."""
    return GenerationParamsInfo(
        max_length=request_data.max_length,
        truncation=request_data.truncation,
        repetition_penalty=request_data.repetition_penalty,
        temperature=request_data.temperature,
        top_p=request_data.top_p,
        top_k=request_data.top_k,
    )

@router.post(
    "/ask-llm-prompted",
    response_model=Union[ChatCompletionResponse, PromptedResponse],
    responses=ERROR_RESPONSES,
)
async def ask_llm_prompted(
    request_data: PromptedLLMRequest,
    user_info: dict = Depends(verify_api_key), 
    request: Request = None
):
    """This endpoint lets you ask a question to the model running on Sugar-AI using custom prompts and also provides options to change model parameters to tune the output.
    RAG is disabled for this endpoint. Set chat=True for chat completions mode.
    """
    start_time = time.time()
    client_ip = request.client.host if request else "unknown"
    
    # Check quota first
    api_key = next(key for key, value in settings.API_KEYS.items() if value['name'] == user_info['name'])
    remaining = settings.MAX_DAILY_REQUESTS - user_quotas.get(api_key, {}).get("count", 0)
    
    try:
        if request_data.chat:
            # Chat completions mode; the request model guarantees messages exist.
            _require_supported_modalities(request_data.messages)

            # Log the last user message for tracking
            user_messages = [msg for msg in request_data.messages if msg.role == "user"]
            last_user_msg = user_messages[-1].text() if user_messages else "No user message"
            logger.info(f"REQUEST - /ask-llm-prompted (chat=True) - User: {user_info['name']} - IP: {client_ip} - Last message: {last_user_msg[:200]}...")
            
            # Log system message if present
            system_messages = [msg for msg in request_data.messages if msg.role == "system"]
            if system_messages:
                logger.info(f"SYSTEM PROMPT - User: {user_info['name']} - Prompt: {system_messages[0].text()[:100]}...")

            # Convert Pydantic messages to dict format for the agent function
            messages_dict = messages_to_provider(request_data.messages)
            
            # Build generation params from request
            params = GenerationParams(
                max_new_tokens=request_data.max_length,
                temperature=request_data.temperature,
                top_p=request_data.top_p,
                top_k=request_data.top_k,
                repetition_penalty=request_data.repetition_penalty,
                truncation=request_data.truncation,
            )

            answer = agent.run_chat_completion(
                messages=messages_dict,
                params=params,
            )
            
            process_time = time.time() - start_time
            logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s - Last message: {last_user_msg[:200]}...")
            
            # Return chat format response
            return ChatCompletionResponse(
                choices=[
                    ChatChoice(
                        message=ChatMessageOut(role="assistant", content=answer),
                        index=0,
                        finish_reason="stop",
                    )
                ],
                user=user_info["name"],
                quota=QuotaInfo(remaining=remaining, total=settings.MAX_DAILY_REQUESTS),
                generation_params=_generation_params_info(request_data),
            )
        else:
            # Prompted mode; the request model guarantees question and custom_prompt.
            logger.info(f"REQUEST - /ask-llm-prompted - User: {user_info['name']} - IP: {client_ip} - Question: {request_data.question[:200]}...")
            logger.info(f"CUSTOM PROMPT - User: {user_info['name']} - Prompt: {request_data.custom_prompt[:100]}...")
            
            params = GenerationParams(
                max_new_tokens=request_data.max_length,
                temperature=request_data.temperature,
                top_p=request_data.top_p,
                top_k=request_data.top_k,
                repetition_penalty=request_data.repetition_penalty,
                truncation=request_data.truncation,
            )

            answer = agent.run_with_custom_prompt(
                question=request_data.question,
                custom_prompt=request_data.custom_prompt,
                params=params,
            )
            
            process_time = time.time() - start_time
            logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s")
            
            return PromptedResponse(
                answer=answer,
                user=user_info["name"],
                quota=QuotaInfo(remaining=remaining, total=settings.MAX_DAILY_REQUESTS),
                generation_params=_generation_params_info(request_data),
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERROR - User: {user_info['name']} - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
        
@router.post("/debug", response_model=AskResponse, responses=ERROR_RESPONSES)
async def debug(
    body: Optional[DebugRequest] = Body(None),
    code: Optional[str] = Query(None, description="Deprecated: send a JSON body instead"),
    context: Optional[bool] = Query(
        None, description="Deprecated: send a JSON body instead"
    ),
    user_info: dict = Depends(verify_api_key),
    request: Request = None
):
    """Process python code for debugging"""
    start_time = time.time()

    if body is None:
        if code is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "validation_error",
                    "message": "code is required, in the JSON body or as a query parameter",
                },
            )
        try:
            body = DebugRequest(code=code, context=bool(context))
        except ValidationError as e:
            raise HTTPException(
                status_code=422,
                detail={"code": "validation_error", "message": _first_error(e)},
            )
    code, context = body.code, body.context

    client_ip = request.client.host if request else "unknown"
    logger.info(f"REQUEST - /debug - User: {user_info['name']} - IP: {client_ip} - code: {code[:50]}...")
    
    try:
        response = agent.debug(code, context)
        answer = response
        
        process_time = time.time() - start_time
        logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s")
        
        # check quota
        api_key = next(key for key, value in settings.API_KEYS.items() if value['name'] == user_info['name'])
        remaining = settings.MAX_DAILY_REQUESTS - user_quotas.get(api_key, {}).get("count", 0)
        
        return AskResponse(
            answer=answer,
            user=user_info["name"],
            quota=QuotaInfo(remaining=remaining, total=settings.MAX_DAILY_REQUESTS),
        )

    except Exception as e:
        logger.error(f"ERROR - User: {user_info['name']} - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/change-model", response_model=ModelChangeResponse, responses=ERROR_RESPONSES)
async def change_model(
    model: str, 
    api_key: str = Query(...), 
    password: str = Query(...), 
    request: Request = None
):
    """Change the model used by the RAG agent (admin only)"""
    client_ip = request.client.host if request else "unknown"
    logger.info(f"REQUEST - /change-model - API Key: {api_key[:5]}... - IP: {client_ip} - Model: {model}")
    
    if api_key not in settings.API_KEYS:
        logger.warning(f"Invalid API key used for model change: {api_key[:5]}... from {client_ip}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    user_info = settings.API_KEYS[api_key]
    if not user_info.get("can_change_model", False):
        logger.warning(f"Unauthorized model change attempt by: {user_info['name']} from {client_ip}")
        raise HTTPException(status_code=403, detail="User doesn't have permission to change model")
    
    if password != settings.MODEL_CHANGE_PASSWORD:
        logger.warning(f"Invalid password for model change by: {user_info['name']} from {client_ip}")
        raise HTTPException(status_code=403, detail="Invalid model change password")
    
    try:
        from app.providers import create_provider
        from app.config import settings
        new_provider = create_provider(
            provider_name=settings.AI_PROVIDER,
            model_name=model,
            quantize=True,
            dev_mode=False,
            base_url=settings.OLLAMA_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            openai_base_url=settings.OPENAI_BASE_URL,
            gemini_api_key=settings.GEMINI_API_KEY,
            gemini_base_url=settings.GEMINI_BASE_URL,
            supported_modalities=settings.supported_modalities(),
        )
        agent.set_model(new_provider)
        logger.info(f"Model changed to {model} by {user_info['name']}")
        return ModelChangeResponse(
            message=f"Model changed to {model}",
            user=user_info["name"],
        )
    except Exception as e:
        logger.error(f"Error changing model to {model} by {user_info['name']}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error changing model: {str(e)}")


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
async def health_check():
    """Check if the AI backend is alive and responsive."""
    if agent is None:
        return HealthResponse(status="unavailable", detail="Agent not initialized")

    try:
        model_name = agent.provider.get_model_name()
        is_healthy = agent.provider.health_check()
        modalities = sorted(getattr(agent.provider, "supported_modalities", {"text"}))

        if is_healthy:
            return HealthResponse(
                status="healthy",
                provider=type(agent.provider).__name__,
                model=model_name,
                modalities=modalities,
            )
        else:
            return HealthResponse(
                status="unhealthy",
                provider=type(agent.provider).__name__,
                model=model_name,
                modalities=modalities,
                detail="Health check failed",
            )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(status="error", detail=str(e))
