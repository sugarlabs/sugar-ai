"""
Authentication routes for Sugar-AI.
"""
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import logging

from app.database import get_db, APIKey
from app.auth import oauth, generate_api_key
from app.config import settings

router = APIRouter(tags=["auth"])

# setup logging
logger = logging.getLogger("sugar-ai")

# set up templates
templates = Jinja2Templates(directory=settings.TEMPLATES_DIR)

@router.get("/logout")
async def logout(request: Request):
    """Log out a user by clearing session and cookies"""
    request.session.pop("user", None)
    response = RedirectResponse(url="/")
    response.delete_cookie("admin_api_key")
    return response

@router.get("/oauth-login", response_class=HTMLResponse)
async def login(request: Request):
    """Render OAuth login page"""
    return templates.TemplateResponse("oauth_login.html", {"request": request})

@router.get("/auth/github")
async def login_github(request: Request):
    """Redirect to GitHub OAuth login"""
    redirect_uri = request.url_for("auth_callback", provider="github")
    return await oauth.github.authorize_redirect(request, redirect_uri)

@router.get("/auth/google")
async def login_google(request: Request):
    """Redirect to Google OAuth login"""
    try:
        # create explicit redirect URI
        base_url = str(request.base_url).rstrip("/")
        redirect_uri = f"{base_url}/auth/callback/google"
        
        logger.info(f"Google OAuth redirect using URI: {redirect_uri}")
        return await oauth.google.authorize_redirect(request, redirect_uri=redirect_uri)
    except Exception as e:
        logger.error(f"Google OAuth error: {str(e)}")
        return HTMLResponse(f"OAuth Error: {str(e)}")

@router.get("/auth/callback/{provider}")
async def auth_callback(provider: str, request: Request, db: Session = Depends(get_db)):
    """Handle OAuth callback and create user if needed"""
    try:
        logger.info(f"OAuth callback for provider: {provider}")
        
        if provider == "github":
            token = await oauth.github.authorize_access_token(request)
            resp = await oauth.github.get(
                url="https://api.github.com/user", 
                token=token
            )
            user_info = resp.json()
            
            # user email isssue was happening so had to do a secondary request
            if not user_info.get('email'):
                emails_resp = await oauth.github.get(
                    url="https://api.github.com/user/emails",
                    token=token
                )
                emails_data = emails_resp.json()
                
                # primary email comes first
                primary_email = next((email['email'] for email in emails_data if email.get('primary')), None)
                if not primary_email and emails_data:
                    primary_email = emails_data[0]['email']
                
                if primary_email:
                    user_info['email'] = primary_email
                    logger.info(f"Retrieved email from secondary request: {primary_email}")
            
        elif provider == "google":
            token = await oauth.google.authorize_access_token(request)
            resp = await oauth.google.get(
                url="https://www.googleapis.com/oauth2/v1/userinfo",
                token=token
            )
            user_info = resp.json()
            logger.info(f"Google user info: {user_info}")
            
        else:
            logger.error(f"Unsupported OAuth provider: {provider}")
            return RedirectResponse(url="/oauth-login?error=Unsupported provider")
        
        # store user info in session
        request.session["user"] = user_info
        logger.info(f"OAuth login successful for: {user_info.get('email', 'unknown')}")
        
        # check if user exists in database or create API key
        email = user_info.get("email")
        if not email:
            logger.warning("No email found in OAuth user info")
            return RedirectResponse(url="/oauth-login?error=No email found")
            
        existing_key = db.query(APIKey).filter(APIKey.email == email).first()
        
        if not existing_key:
            # create new API key for OAuth user
            api_key = generate_api_key()
            new_key = APIKey(
                key=api_key,
                name=user_info.get("name", f"OAuth User {email}"),
                email=email,
                approved=True,  # auto-approve OAuth users
                is_active=True,
                can_change_model=False  # not allowed to change models tho
            )
            db.add(new_key)
            db.commit()
            logger.info(f"Created new API key for OAuth user: {email}")
            
            # update API_KEYS in memory
            settings.API_KEYS[api_key] = {"name": new_key.name, "can_change_model": new_key.can_change_model}
        else:
            # update in-memory API_KEYS for existing key
            settings.API_KEYS[existing_key.key] = {"name": existing_key.name, "can_change_model": existing_key.can_change_model}
        
        return RedirectResponse(url="/dashboard")
    except Exception as e:
        logger.error(f"OAuth error: {str(e)}")
        return RedirectResponse(url="/oauth-login?error=Authentication failed")

@router.get("/admin-login", response_class=HTMLResponse)
async def admin_login(request: Request):
    """Render admin login page"""
    return templates.TemplateResponse("admin_login.html", {"request": request})

@router.post("/admin-login")
async def admin_login_submit(
    request: Request,
    api_key: str = Form(...),
    db: Session = Depends(get_db)
):
    """Process admin login via API key"""
    # check if API key exists and is active
    key = db.query(APIKey).filter(
        APIKey.key == api_key,
        APIKey.approved == True,
        APIKey.is_active == True
    ).first()
    
    if not key:
        return templates.TemplateResponse(
            "admin_login.html", 
            {"request": request, "error": "Invalid API key"}
        )
    
    # redirect based on permissions
    response = RedirectResponse(url="/dashboard" if not key.can_change_model else "/admin", status_code=303)
    response.set_cookie(key="admin_api_key", value=api_key, httponly=True)
    return response

@router.get("/request-key", response_class=HTMLResponse)
async def request_key_form(request: Request):
    """Render API key request page"""
    return templates.TemplateResponse("request_key.html", {"request": request})

@router.post("/request-key", response_class=HTMLResponse)
async def submit_key_request(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    reason: str = Form(...),
    db: Session = Depends(get_db)
):
    """Process API key request"""
    # generate API key but keep it unapproved
    api_key = generate_api_key()
    new_key = APIKey(
        key=api_key,
        name=name,
        email=email,
        request_reason=reason,
        approved=False,
        is_active=False
    )
    db.add(new_key)
    db.commit()
    
    return templates.TemplateResponse(
        "request_key.html", 
        {
            "request": request, 
            "message": "Your API key request has been submitted. You will be notified once it's approved.",
            "success": True
        }
    )
