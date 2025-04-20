"""
Web routes handling HTML responses for Sugar-AI.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db, APIKey
from app.auth import get_oauth_user_info
from app.config import settings

router = APIRouter(tags=["web"])

# set up templates
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)

@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render welcome page"""
    return templates.TemplateResponse("welcome.html", {"request": request})

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Render dashboard page"""
    # check for OAuth user
    user = await get_oauth_user_info(request)
    api_key = None
    
    # if no OAuth user, check for API key in cookie
    if not user:
        api_key_cookie = request.cookies.get("admin_api_key")
        if api_key_cookie:
            # verify API key
            key_obj = db.query(APIKey).filter(
                APIKey.key == api_key_cookie,
                APIKey.approved == True,
                APIKey.is_active == True
            ).first()
            
            if key_obj:
                user = {
                    "name": key_obj.name, 
                    "email": key_obj.email,
                    "can_change_model": key_obj.can_change_model
                }
                api_key = api_key_cookie
        
        # no valid user found, redirect to login
        if not user:
            return RedirectResponse(url="/oauth-login")
    
    # if user from OAuth but no API key found yet
    if not api_key and user and "email" in user:
        user_key = db.query(APIKey).filter(APIKey.email == user["email"]).first()
        if user_key:
            api_key = user_key.key
            # ensure can_change_model is included
            if "can_change_model" not in user:
                user["can_change_model"] = user_key.can_change_model
    
    # add admin URL if user has admin privileges
    admin_url = None
    if user and user.get("can_change_model", False):
        admin_url = "/admin"
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "api_key": api_key,
        "admin_url": admin_url
    })
