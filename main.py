# Copyright (C) 2024 Sugar Labs, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
Main entry point for Sugar-AI application.
"""

import uvicorn
import logging
from sqlalchemy.orm import Session
import os

from app import create_app
from app.ai import RAGAgent
from app.providers import create_provider
from app.database import get_db
from app.auth import sync_env_keys_to_db
from app.config import settings
from app.routes import api

# setup logging
logger = logging.getLogger("sugar-ai")

app = create_app()


def get_active_model() -> str:
    """Resolve the configured model and fail clearly when none is selected."""
    active_model = (
        settings.AI_MODEL
        or (settings.DEV_MODEL_NAME if settings.DEV_MODE else settings.PROD_MODEL_NAME)
    )
    if active_model:
        return active_model

    fallback_name = "DEV_MODEL_NAME" if settings.DEV_MODE else "PROD_MODEL_NAME"
    raise RuntimeError(
        "No model is configured. Set AI_MODEL, or set "
        f"{fallback_name} for the selected DEV_MODE value in .env."
    )


@app.on_event("startup")
async def startup_event():
    """Initialize data on app startup"""
    active_model = get_active_model()
    db = next(get_db())
    sync_env_keys_to_db(db)

    logger.info(
        "Starting with provider=%s, model=%s",
        settings.AI_PROVIDER,
        active_model,
    )

    provider = create_provider(
        provider_name=settings.AI_PROVIDER,
        model_name=active_model,
        quantize=True,
        dev_mode=settings.DEV_MODE,
        base_url=settings.OLLAMA_BASE_URL,
        api_key=settings.OPENAI_API_KEY,
        openai_base_url=settings.OPENAI_BASE_URL,
        gemini_api_key=settings.GEMINI_API_KEY,
        gemini_base_url=settings.GEMINI_BASE_URL,
    )

    initialized_agent = RAGAgent(provider=provider)
    initialized_agent.retriever = initialized_agent.setup_vectorstore(settings.DOC_PATHS)

    # Inject this instance into the API module
    api.agent = initialized_agent

    app.state.agent = initialized_agent


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Sugar-AI on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
