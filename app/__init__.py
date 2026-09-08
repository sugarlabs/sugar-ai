"""
Sugar-AI application package.
"""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
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

# Stable error codes for the uniform error envelope, keyed by HTTP status.
_ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
    429: "quota_exceeded",
    500: "internal_error",
}


def _register_error_handlers(app: FastAPI) -> None:
    """Make every error leave the API as {"error": {"code", "message"}}."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail and "message" in detail:
            error = {"code": detail["code"], "message": detail["message"]}
        else:
            error = {
                "code": _ERROR_CODES.get(exc.status_code, "error"),
                "message": str(detail),
            }
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": error},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        first = errors[0] if errors else {}
        location = ".".join(str(part) for part in first.get("loc", []))
        message = first.get("msg", "Invalid request")
        if location:
            message = f"{location}: {message}"
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": message}},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error"}},
        )


def create_app() -> FastAPI:
    app = FastAPI()

    _register_error_handlers(app)

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
