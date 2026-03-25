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

from sugar_ai import create_app
from sugar_ai.ai import RAGAgent
from sugar_ai.database import get_db
from sugar_ai.auth import sync_and_load_keys
from sugar_ai.config import settings
from sugar_ai.routes import api

# setup logging
logger = logging.getLogger("sugar-ai")

app = create_app()

@app.on_event("startup")
async def startup_event():
    """Initialize data and sync keys on app startup"""
    db = next(get_db())
    sync_and_load_keys(db)
    if settings.DEV_MODE:
        active_model = settings.DEV_MODEL_NAME
        logger.info(f"DEV_MODE active. Loading lightweight model: {active_model}")
    else:
        active_model = settings.PROD_MODEL_NAME
        logger.info(f"PRODUCTION mode. Loading full model: {active_model}")

    initialized_agent = RAGAgent(model=active_model)
    initialized_agent.retriever = initialized_agent.setup_vectorstore(settings.DOC_PATHS)

    # Inject this instance into the API module
    # This updates the 'agent = None' in api.py to be the real loaded model
    api.agent = initialized_agent
    
    app.state.agent = initialized_agent


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Sugar-AI on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

