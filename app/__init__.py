"""
Sugar-AI application package.
"""
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import logging

from app.auth import setup_oauth
from app.database import create_tables

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sugar_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sugar-ai")

def create_app() -> FastAPI:
    app = FastAPI()
    
    # apply middlewares
    app = setup_oauth(app)
    
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["localhost", "127.0.0.1", "*"]  
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ensure DB tables exist
    create_tables()
    
    # mount static files
    static_dir = "static"
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    else:
        logger.warning(f"Static directory '{static_dir}' does not exist")
    
    # register routers
    from app.routes.api import router as api_router
    from app.routes.admin import router as admin_router
    from app.routes.auth import router as auth_router
    from app.routes.web import router as web_router
    from app.routes.webhook import router as webhook_router
    
    app.include_router(api_router)
    app.include_router(admin_router)
    app.include_router(auth_router)
    app.include_router(web_router)
    app.include_router(webhook_router)
    
    return app
