"""
Runtime helpers for loading and swapping the active RAG agent.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from sqlalchemy.orm import Session

from app.ai import RAGAgent
from app.config import settings
from app.database import LLMModel
from app.llm import build_llm_provider

logger = logging.getLogger("sugar-ai")


def build_app_agent(app: FastAPI, model: LLMModel) -> RAGAgent:
    """Build a replacement agent without mutating app state."""
    runtime_settings = getattr(app.state, "settings", None) or settings
    provider = build_llm_provider(model)
    agent = RAGAgent(provider=provider)

    current_agent = getattr(app.state, "agent", None)
    if current_agent is not None and current_agent.retriever is not None:
        agent.retriever = current_agent.retriever
    else:
        agent.retriever = agent.setup_vectorstore(runtime_settings.DOC_PATHS)

    return agent


def install_app_agent(app: FastAPI, agent: RAGAgent, model: LLMModel) -> RAGAgent:
    """Swap in a fully prepared agent."""
    logger.info("Loaded in-memory agent for model '%s'.", model.name)
    app.state.agent = agent
    app.state.startup_error = None
    return agent


def commit_model_and_sync_app(app: FastAPI, db: Session, model: LLMModel) -> LLMModel:
    """Commit a pending active-model change and then swap the runtime agent."""
    agent = build_app_agent(app, model)
    db.commit()
    db.refresh(model)
    install_app_agent(app, agent, model)
    return model


def sync_app_agent(app: FastAPI, model: LLMModel) -> RAGAgent:
    """Create or update the in-memory agent for the active model."""
    agent = build_app_agent(app, model)
    return install_app_agent(app, agent, model)
