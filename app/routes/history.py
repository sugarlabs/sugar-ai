"""
History routes for Sugar-AI.
"""
from fastapi import APIRouter, Depends, Request, HTTPException, Query
from typing import Dict, Any
from app.routes.api import verify_api_key
from app.config import settings
from app.history import get_history, clear_history

router = APIRouter()

logger = __import__('logging').getLogger("sugar-ai")


def get_user_api_key(user_name: str) -> str:
    """Retrieve the API key for a given user name."""
    for key, value in settings.API_KEYS.items():
        if value.get("name") == user_name:
            return key
    raise HTTPException(status_code=401, detail="Invalid API key")


@router.get("/history")
async def get_question_history(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    user_info: Dict[str, Any] = Depends(verify_api_key),
):
    """Retrieve Q&A history for the authenticated user."""
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"REQUEST - /history - User: {user_info['name']} - IP: {client_ip}")

    api_key = get_user_api_key(user_info["name"])

    try:
        history = get_history(api_key, limit=limit)
        return {
            "user": user_info["name"],
            "returned_entries": len(history),
            "history": history
        }
    except Exception as e:
        logger.error(f"Error retrieving history for {user_info['name']}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


@router.delete("/history")
async def delete_question_history(
    request: Request,
    user_info: Dict[str, Any] = Depends(verify_api_key),
):
    """Clear all Q&A history for the authenticated user."""
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"REQUEST - DELETE /history - User: {user_info['name']} - IP: {client_ip}")

    api_key = get_user_api_key(user_info["name"])

    try:
        deleted_count = clear_history(api_key)
        return {
            "message": "History cleared successfully",
            "user": user_info["name"],
            "deleted_entries": deleted_count
        }
    except Exception as e:
        logger.error(f"Error deleting history for {user_info['name']}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear history")