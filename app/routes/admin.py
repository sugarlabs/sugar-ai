"""
Admin routes for Sugar-AI.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db, APIKey
from app.auth import get_current_user
from app.config import settings
from app.llm import (
    LLMConfigurationError,
    activate_llm_model,
    create_llm_model,
    get_llm_model_or_raise,
    list_llm_models,
    soft_delete_llm_model,
    update_llm_model,
)
from app.runtime import commit_model_and_sync_app

router = APIRouter(tags=["admin"])

# set up templates
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)


class LLMModelCreateRequest(BaseModel):
    name: str
    provider_type: str = Field(default="openai_compatible")
    base_url: str
    api_key: Optional[str] = None
    model_name: str
    max_model_length: Optional[int] = None
    is_active: bool = False


class LLMModelUpdateRequest(BaseModel):
    name: Optional[str] = None
    provider_type: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    max_model_length: Optional[int] = None
    is_active: Optional[bool] = None


def require_admin(user_data: tuple):
    user, authenticated = user_data
    if not authenticated or not user or not user.can_change_model:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return user


@router.get("/admin", response_class=HTMLResponse)
async def admin_panel(
    request: Request,
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Admin panel view"""
    require_admin(user_data)
    
    pending_keys = db.query(APIKey).filter(APIKey.approved == False, APIKey.is_active == False).all()
    approved_keys = db.query(APIKey).filter(APIKey.approved == True).all()
    denied_keys = db.query(APIKey).filter(APIKey.approved == False, APIKey.is_active == True).all()
    models = list_llm_models(db)
    
    return templates.TemplateResponse(
        request,
        "admin_panel.html",
        {
            "pending_keys": pending_keys,
            "approved_keys": approved_keys,
            "denied_keys": denied_keys,
            "models": models,
            "startup_error": getattr(request.app.state, "startup_error", None),
        },
    )

@router.post("/admin/approve/{key_id}")
async def approve_key(
    key_id: int,
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Approve an API key request"""
    require_admin(user_data)
    
    key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    key.approved = True
    key.is_active = True
    db.commit()
    
    # update in-memory API keys
    settings.API_KEYS[key.key] = {"name": key.name, "can_change_model": key.can_change_model}
    
    return {"status": "success", "message": "API key approved"}

@router.post("/admin/deny/{key_id}")
async def deny_key(
    key_id: int,
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deny an API key request"""
    require_admin(user_data)
    
    key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    key.approved = False
    key.is_active = True  # mark as processed but denied
    db.commit()
    
    return {"status": "success", "message": "API key request denied"}

@router.post("/admin/toggle-admin/{key_id}")
async def toggle_admin(
    key_id: int,
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggle admin status for an API key"""
    require_admin(user_data)
    
    key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    key.can_change_model = not key.can_change_model
    db.commit()
    
    # update in-memory API keys if needed
    if key.key in settings.API_KEYS:
        settings.API_KEYS[key.key]["can_change_model"] = key.can_change_model
    
    return {"status": "success", "message": "Admin status toggled"}

@router.post("/admin/toggle-status/{key_id}")
async def toggle_status(
    key_id: int,
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggle active status for an API key"""
    require_admin(user_data)
    
    key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    key.is_active = not key.is_active
    db.commit()
    
    # update in-memory API keys if needed
    if key.key in settings.API_KEYS and not key.is_active:
        del settings.API_KEYS[key.key]
    elif key.key not in settings.API_KEYS and key.is_active and key.approved:
        settings.API_KEYS[key.key] = {"name": key.name, "can_change_model": key.can_change_model}
    
    return {"status": "success", "message": "API key status toggled"}


@router.get("/admin/models")
async def get_models(
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List configured LLM models."""
    require_admin(user_data)
    return {"models": [model.to_dict() for model in list_llm_models(db)]}


@router.post("/admin/models")
async def create_model(
    request: Request,
    request_data: LLMModelCreateRequest,
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new LLM model config."""
    require_admin(user_data)
    try:
        model = create_llm_model(
            db,
            auto_commit=False,
            **request_data.model_dump(),
        )
        if model.is_active:
            model = commit_model_and_sync_app(request.app, db, model)
        else:
            db.commit()
            db.refresh(model)
        return {"model": model.to_dict()}
    except LLMConfigurationError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        raise


@router.put("/admin/models/{model_id}")
async def update_model(
    request: Request,
    model_id: int,
    request_data: LLMModelUpdateRequest,
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update an existing LLM model config."""
    require_admin(user_data)
    try:
        model = get_llm_model_or_raise(db, model_id)
        if model.is_active and request_data.is_active is False:
            raise LLMConfigurationError("Cannot deactivate the active model without activating another model.")
        changes = request_data.model_dump(exclude_unset=True)
        model = update_llm_model(
            db,
            model,
            auto_commit=False,
            **changes,
        )
        if model.is_active:
            model = commit_model_and_sync_app(request.app, db, model)
        else:
            db.commit()
            db.refresh(model)
        return {"model": model.to_dict()}
    except LLMConfigurationError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        raise


@router.delete("/admin/models/{model_id}")
async def delete_model(
    model_id: int,
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Soft-delete an LLM model config."""
    require_admin(user_data)
    try:
        model = soft_delete_llm_model(db, model_id)
        return {"model": model.to_dict(), "message": "Model deleted"}
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/admin/models/{model_id}/activate")
async def activate_model(
    request: Request,
    model_id: int,
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Activate an LLM model config."""
    require_admin(user_data)
    try:
        model = activate_llm_model(db, model_id, auto_commit=False)
        model = commit_model_and_sync_app(request.app, db, model)
        return {"model": model.to_dict(), "message": f"Active model changed to {model.name}"}
    except LLMConfigurationError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        raise
