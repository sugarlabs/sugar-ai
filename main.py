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
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app import create_app
from app.database import get_db, create_tables
from app.auth import sync_env_keys_to_db
from app.config import settings
from app.routes import api
from app.ai import RAGAgent

# Setup logging
logger = logging.getLogger("sugar-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the startup and shutdown lifecycle of the application.
    Replaces deprecated @app.on_event.
    """
    try:
        db = next(get_db())
        sync_env_keys_to_db(db)
        create_tables()

        if getattr(settings, "DEV_MODE", False):
            active_model = getattr(settings, "DEV_MODEL_NAME", settings.DEFAULT_MODEL)
            logger.info(f"DEV_MODE active. Loading model: {active_model}")
        else:
            active_model = getattr(settings, "PROD_MODEL_NAME", settings.DEFAULT_MODEL)
            logger.info(f"PRODUCTION mode. Loading model: {active_model}")

        initialized_agent = RAGAgent(model=active_model)
        initialized_agent.retriever = initialized_agent.setup_vectorstore(settings.DOC_PATHS)

        # Inject into API
        api.agent = initialized_agent
        app.state.agent = initialized_agent

        logger.info(f"Starting Sugar-AI with model: {active_model}")

    except Exception as e:
        logger.error(f"Failed to initialize app during startup: {e}")
        raise e

    yield

    logger.info("Shutting down Sugar-AI...")


# Create FastAPI app with lifespan
app = create_app(lifespan=lifespan)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )