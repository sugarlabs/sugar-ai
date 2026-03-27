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

from app import create_app
from app.database import get_db
from app.auth import sync_env_keys_to_db
from app.config import settings
from app.llm import ensure_active_llm_model, LLMConfigurationError
from app.runtime import sync_app_agent

# setup logging
logger = logging.getLogger("sugar-ai")

app = create_app()
app.state.settings = settings

@app.on_event("startup")
async def startup_event():
    """Initialize data on app startup"""
    db = next(get_db())
    try:
        sync_env_keys_to_db(db)
        active_model = ensure_active_llm_model(db)
        sync_app_agent(app, active_model)
        logger.info(
            "Using provider '%s' with model '%s' at '%s'.",
            active_model.provider_type,
            active_model.model_name,
            active_model.base_url,
        )
    except LLMConfigurationError as exc:
        app.state.agent = None
        app.state.startup_error = str(exc)
        logger.warning("LLM configuration unavailable at startup: %s", exc)
    finally:
        db.close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Sugar-AI on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
