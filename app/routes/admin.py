"""
Admin routes for Sugar-AI.
"""
from fastapi import APIRouter, Depends, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db, APIKey
from app.auth import get_current_user
from app.config import settings

router = APIRouter(tags=["admin"])

# set up templates
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)

@router.get("/admin", response_class=HTMLResponse)
async def admin_panel(
    request: Request,
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Admin panel view"""
    user, authenticated = user_data
    if not authenticated or not user or not user.can_change_model:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    pending_keys = db.query(APIKey).filter(APIKey.approved == False, APIKey.is_active == False).all()
    approved_keys = db.query(APIKey).filter(APIKey.approved == True).all()
    denied_keys = db.query(APIKey).filter(APIKey.approved == False, APIKey.is_active == True).all()
    
    return templates.TemplateResponse(
        "admin_panel.html", 
        {
            "request": request,
            "pending_keys": pending_keys,
            "approved_keys": approved_keys,
            "denied_keys": denied_keys
        }
    )

@router.post("/admin/approve/{key_id}")
async def approve_key(
    key_id: int,
    user_data: tuple = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Approve an API key request"""
    user, authenticated = user_data
    if not authenticated or not user or not user.can_change_model:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
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
    user, authenticated = user_data
    if not authenticated or not user or not user.can_change_model:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
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
    user, authenticated = user_data
    if not authenticated or not user or not user.can_change_model:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
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
    user, authenticated = user_data
    if not authenticated or not user or not user.can_change_model:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
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
