"""
Main routes for Sugar-AI.
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["main"])

# setup templates
templates = Jinja2Templates(directory="templates")

@router.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the root welcome page with links to documentation"""
    return templates.TemplateResponse("welcome.html", {"request": request})
