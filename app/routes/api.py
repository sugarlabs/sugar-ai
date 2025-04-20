"""
API routes for Sugar-AI.
"""
from fastapi import APIRouter, Depends, HTTPException, Header, Query, Request
from sqlalchemy.orm import Session
import time
import logging
import os
import json
from datetime import datetime
from typing import Dict, Optional

from app.database import get_db, APIKey
from app.ai import RAGAgent
from app.config import settings

router = APIRouter(tags=["api"])

# setup logging
logger = logging.getLogger("sugar-ai")

# load ai agent and document paths
agent = RAGAgent(model=settings.DEFAULT_MODEL)
agent.retriever = agent.setup_vectorstore(settings.DOC_PATHS)

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

@router.post("/ask")
async def ask_question(
    question: str, 
    user_info: dict = Depends(verify_api_key), 
    request: Request = None
):
    """Process a question using RAG pipeline"""
    start_time = time.time()
    
    client_ip = request.client.host if request else "unknown"
    logger.info(f"REQUEST - /ask - User: {user_info['name']} - IP: {client_ip} - Question: {question[:50]}...")
    
    try:
        answer = agent.run(question)
        
        # log completion
        process_time = time.time() - start_time
        logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s")
        
        # check quota
        api_key = next(key for key, value in settings.API_KEYS.items() if value['name'] == user_info['name'])
        remaining = settings.MAX_DAILY_REQUESTS - user_quotas.get(api_key, {}).get("count", 0)
        
        return {
            "answer": answer, 
            "user": user_info["name"],
            "quota": {"remaining": remaining, "total": settings.MAX_DAILY_REQUESTS}
        }
    except Exception as e:
        logger.error(f"ERROR - User: {user_info['name']} - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/ask-llm")
async def ask_llm(
    question: str, 
    user_info: dict = Depends(verify_api_key), 
    request: Request = None
):
    """Process a question with direct LLM call (no retrieval)"""
    start_time = time.time()
    
    client_ip = request.client.host if request else "unknown"
    logger.info(f"REQUEST - /ask-llm - User: {user_info['name']} - IP: {client_ip} - Question: {question[:50]}...")
    
    try:
        response = agent.model(question)
        answer = response[0]['generated_text'].split("Answer:")[-1].strip()
        
        process_time = time.time() - start_time
        logger.info(f"RESPONSE - User: {user_info['name']} - Success - Time: {process_time:.2f}s")
        
        # check quota
        api_key = next(key for key, value in settings.API_KEYS.items() if value['name'] == user_info['name'])
        remaining = settings.MAX_DAILY_REQUESTS - user_quotas.get(api_key, {}).get("count", 0)
        
        return {
            "answer": answer, 
            "user": user_info["name"],
            "quota": {"remaining": remaining, "total": settings.MAX_DAILY_REQUESTS}
        }
    except Exception as e:
        logger.error(f"ERROR - User: {user_info['name']} - Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@router.post("/change-model")
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
        agent.set_model(model)
        logger.info(f"Model changed to {model} by {user_info['name']}")
        return {"message": f"Model changed to {model}", "user": user_info["name"]}
    except Exception as e:
        logger.error(f"Error changing model to {model} by {user_info['name']}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error changing model: {str(e)}")
