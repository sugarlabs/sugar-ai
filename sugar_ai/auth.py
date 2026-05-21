"""
Authentication handling for Sugar-AI including API keys and OAuth.
"""
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
import secrets
import os
import json
import logging
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from starlette.responses import RedirectResponse
from dotenv import load_dotenv

from sugar_ai.database import APIKey, get_db

# load environment variables
load_dotenv()

# setup logging
logger = logging.getLogger("sugar-ai")

# set up OAuth
oauth = OAuth()
oauth.register(
    name="github",
    client_id=os.getenv("GITHUB_CLIENT_ID", ""),
    client_secret=os.getenv("GITHUB_CLIENT_SECRET", ""),
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "read:user user:email"} 
)

oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    access_token_url="https://oauth2.googleapis.com/token",
    userinfo_endpoint="https://www.googleapis.com/oauth2/v1/userinfo",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def setup_oauth(app):
    """Configure the FastAPI app with OAuth and session middleware"""
    app.add_middleware(
        SessionMiddleware, 
        secret_key=os.getenv("SESSION_SECRET_KEY", "supersecretkey")
    )
    return app

async def get_oauth_user_info(request: Request) -> Optional[Dict[str, Any]]:
    """Get the current OAuth user info from session"""
    user_info = request.session.get("user")
    
    if user_info and "email" in user_info:
        try:
            db = next(get_db())
            api_key = db.query(APIKey).filter(APIKey.email == user_info["email"]).first()
            if api_key:
                user_info["can_change_model"] = api_key.can_change_model
        except Exception as e:
            logger.error(f"Error enriching OAuth user info: {str(e)}")
    
    return user_info

async def require_oauth_login(request: Request):
    """Middleware-style function to require OAuth login"""
    user = await get_oauth_user_info(request)
    if not user:
        return RedirectResponse(url="/oauth-login")
    return user

async def get_current_user(request: Request, db: Session = Depends(get_db)) -> Tuple[Optional[APIKey], bool]:
    """Get the current authenticated user from either API key or OAuth session"""
    # try header-based authentication
    api_key = request.headers.get("X-API-Key")
    
    # if not in header, try cookie
    if not api_key:
        api_key = request.cookies.get("admin_api_key")
    
    # if not in cookie or header, try OAuth session
    if not api_key:
        user_info = request.session.get("user")
        if user_info and "email" in user_info:
            # find API key by email
            user = db.query(APIKey).filter(
                APIKey.email == user_info["email"],
                APIKey.approved == True,
                APIKey.is_active == True
            ).first()
            if user:
                return user, True
    
    if not api_key:
        return None, False
    
    user = db.query(APIKey).filter(
        APIKey.key == api_key,
        APIKey.approved == True,
        APIKey.is_active == True
    ).first()
    
    if user:
        return user, True
    
    return None, False

def generate_api_key() -> str:
    """Generate a secure random API key"""
    return secrets.token_urlsafe(32)

def sync_and_load_keys(db: Session):
    """
    1. Sync new keys from .env file to database.
    2. Load all approved and active keys from database into memory (settings.API_KEYS).
    """
    from sugar_ai.config import settings
    
    # Phase 1: Sync from .env to DB
    try:
        api_keys_str = os.getenv("API_KEYS", "{}")
        api_keys = json.loads(api_keys_str)
        
        for key, data in api_keys.items():
            existing = db.query(APIKey).filter(APIKey.key == key).first()
            if not existing:
                db_key = APIKey(
                    key=key,
                    name=data.get("name", "Unknown"),
                    can_change_model=data.get("can_change_model", False),
                    is_active=True,
                    approved=True,
                    email="migrated_from_env@example.com"
                )
                db.add(db_key)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error syncing .env keys: {e}")

    # Phase 2: Load from DB to Memory (settings.API_KEYS)
    try:
        all_approved = db.query(APIKey).filter(
            APIKey.approved == True,
            APIKey.is_active == True
        ).all()
        
        # Hydrate settings.API_KEYS - this is the source of truth for the API validator
        for key_obj in all_approved:
            settings.API_KEYS[key_obj.key] = {
                "name": key_obj.name,
                "can_change_model": key_obj.can_change_model
            }
        
        logger.info(f"Successfully reloaded {len(all_approved)} API keys from database.")
    except Exception as e:
        logger.error(f"Error loading keys into memory: {e}")
