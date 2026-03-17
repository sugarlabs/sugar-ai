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
from app.database import get_db
from app.auth import sync_env_keys_to_db
from app.config import settings
from app.routes import api

# setup logging
logger = logging.getLogger("sugar-ai")

app = create_app()

def initialize_agent(model_name: str):
            """Initialize the RAG agent with the given model."""
            return RAGAgent(model=model_name)


@app.on_event("startup")
async def startup_event():
        """Initialize data on app startup"""
    try:
        db = next(get_db())
        sync_env_keys_to_db(db)
        
        if settings.DEV_MODE:
            active_model = settings.DEV_MODEL_NAME
            logger.info(f"DEV_MODE active. Loading lightweight model: {active_model}")
        else:
            active_model = settings.PROD_MODEL_NAME
            logger.info(f"PRODUCTION mode. Loading full model: {active_model}")

        initialized_agent = initialize_agent(active_model)
        
        try:
            initialized_agent.retriever = initialized_agent.setup_vectorstore(settings.DOC_PATHS)
        except Exception:
            logger.error("Failed to initialize vectorstore", exc_info=True)


        # Inject this instance into the API module
        api.agent = initialized_agent
    
        app.state.agent = initialized_agent
        
        logger.info("Sugar-AI startup completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise  # Re-raise to prevent app from starting if initialization fails
    


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Sugar-AI on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

