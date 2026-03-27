"""
LLM provider abstractions and model configuration helpers.
"""
from __future__ import annotations

import datetime
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from dotenv import dotenv_values
from openai import OpenAI
from sqlalchemy.orm import Session

from app.config import settings
from app.database import LLMModel

logger = logging.getLogger("sugar-ai")


class LLMConfigurationError(RuntimeError):
    """Raised when no valid LLM configuration is available."""


class LLMProvider(ABC):
    """Abstract interface for chat-based text generation."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Return the assistant text for the provided messages."""


class OpenAICompatibleLLMProvider(LLMProvider):
    """LLM provider backed by the OpenAI Python SDK."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str],
        model_name: str,
        max_model_length: Optional[int] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "not-needed"
        self.model_name = model_name
        self.max_model_length = max_model_length
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        response = self.client.chat.completions.create(
            **request_kwargs,
        )
        return _extract_message_text(response)


def _extract_message_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text"))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


def build_llm_provider(model_config: LLMModel) -> LLMProvider:
    if model_config.provider_type != "openai_compatible":
        raise LLMConfigurationError(
            f"Unsupported provider_type '{model_config.provider_type}'."
        )

    return OpenAICompatibleLLMProvider(
        base_url=model_config.base_url,
        api_key=model_config.api_key,
        model_name=model_config.model_name,
        max_model_length=model_config.max_model_length,
    )


def validate_llm_model_config(model_config: LLMModel) -> None:
    if not model_config.base_url:
        raise LLMConfigurationError("LLM base_url is required.")
    if not model_config.model_name:
        raise LLMConfigurationError("LLM model_name is required.")
    build_llm_provider(model_config)


def get_active_llm_model(db: Session) -> Optional[LLMModel]:
    return (
        db.query(LLMModel)
        .filter(
            LLMModel.is_active == True,
            LLMModel.deleted_at.is_(None),
        )
        .order_by(LLMModel.id.asc())
        .first()
    )


def list_llm_models(db: Session) -> list[LLMModel]:
    return (
        db.query(LLMModel)
        .filter(LLMModel.deleted_at.is_(None))
        .order_by(LLMModel.id.asc())
        .all()
    )


def create_llm_model(
    db: Session,
    *,
    auto_commit: bool = True,
    **data: Any,
) -> LLMModel:
    candidate = _build_candidate_model(data)
    if data.get("is_active"):
        validate_llm_model_config(candidate)

    if data.get("is_active"):
        _deactivate_all_models(db)

    model = candidate
    db.add(model)
    if auto_commit:
        db.commit()
        db.refresh(model)
    else:
        db.flush()
    return model


def update_llm_model(
    db: Session,
    model: LLMModel,
    *,
    auto_commit: bool = True,
    **changes: Any,
) -> LLMModel:
    candidate = _build_candidate_model(_merge_model_changes(model, changes))
    if model.is_active or changes.get("is_active"):
        validate_llm_model_config(candidate)

    if changes.get("is_active"):
        _deactivate_all_models(db, exclude_id=model.id)

    for field, value in changes.items():
        setattr(model, field, value)

    if auto_commit:
        db.commit()
        db.refresh(model)
    else:
        db.flush()
    return model


def activate_llm_model(
    db: Session,
    model_id: int,
    *,
    auto_commit: bool = True,
) -> LLMModel:
    model = get_llm_model_or_raise(db, model_id)
    validate_llm_model_config(model)

    _deactivate_all_models(db, exclude_id=model.id)
    model.is_active = True
    if auto_commit:
        db.commit()
        db.refresh(model)
    else:
        db.flush()
    return model


def soft_delete_llm_model(db: Session, model_id: int) -> LLMModel:
    model = get_llm_model_or_raise(db, model_id)
    if model.is_active:
        raise LLMConfigurationError("Cannot delete the active model. Activate another model first.")

    model.deleted_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(model)
    return model


def get_llm_model_or_raise(db: Session, model_id: int) -> LLMModel:
    model = (
        db.query(LLMModel)
        .filter(LLMModel.id == model_id, LLMModel.deleted_at.is_(None))
        .first()
    )
    if not model:
        raise LLMConfigurationError(f"Model id {model_id} not found.")
    return model


def ensure_active_llm_model(db: Session) -> LLMModel:
    active_model = get_active_llm_model(db)
    if active_model:
        validate_llm_model_config(active_model)
        return active_model

    if db.query(LLMModel.id).first() is not None:
        raise LLMConfigurationError(
            "No active LLM model configured in the database. Activate or create one from /admin/models."
        )

    seed = _get_env_seed()
    if not seed:
        legacy_values = _get_legacy_model_env_values()
        if legacy_values:
            legacy_keys = ", ".join(legacy_values.keys())
            raise LLMConfigurationError(
                "Detected legacy model configuration keys "
                f"({legacy_keys}), but this version no longer boots from DEV_MODEL_NAME, "
                "PROD_MODEL_NAME, or DEFAULT_MODEL. Configure LLM_BASE_URL and "
                "LLM_MODEL_NAME before first startup. If you are upgrading a Docker "
                "deployment, migrate the existing database as described in README.md and restart."
            )
        raise LLMConfigurationError(
            "No LLM model configured. Set LLM_BASE_URL and LLM_MODEL_NAME for the first startup, or create a model from /admin/models."
        )

    validate_llm_model_config(_build_candidate_model({**seed, "is_active": True}))
    model = create_llm_model(
        db,
        name=seed["name"],
        provider_type=seed["provider_type"],
        base_url=seed["base_url"],
        api_key=seed["api_key"],
        model_name=seed["model_name"],
        max_model_length=seed["max_model_length"],
        is_active=True,
    )
    logger.info("Bootstrapped initial active LLM model '%s' from environment.", model.name)
    return model


def _deactivate_all_models(db: Session, exclude_id: Optional[int] = None) -> None:
    query = db.query(LLMModel).filter(
        LLMModel.is_active == True,
        LLMModel.deleted_at.is_(None),
    )
    if exclude_id is not None:
        query = query.filter(LLMModel.id != exclude_id)
    query.update({"is_active": False}, synchronize_session=False)


def _get_env_seed() -> Optional[dict[str, Any]]:
    if not settings.LLM_BASE_URL or not settings.LLM_MODEL_NAME:
        return None

    provider_type = settings.LLM_PROVIDER_TYPE or "openai_compatible"
    display_name = settings.LLM_DISPLAY_NAME or settings.LLM_MODEL_NAME
    return {
        "name": display_name,
        "provider_type": provider_type,
        "base_url": settings.LLM_BASE_URL,
        "api_key": settings.LLM_API_KEY,
        "model_name": settings.LLM_MODEL_NAME,
        "max_model_length": settings.LLM_MAX_MODEL_LENGTH,
    }


def _get_legacy_model_env_values() -> dict[str, str]:
    env_values = dict(dotenv_values(".env"))
    env_values.update(os.environ)
    legacy_keys = ("DEV_MODEL_NAME", "PROD_MODEL_NAME", "DEFAULT_MODEL")
    return {
        key: value
        for key in legacy_keys
        if (value := env_values.get(key))
    }


def _merge_model_changes(model: LLMModel, changes: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": changes.get("name", model.name),
        "provider_type": changes.get("provider_type", model.provider_type),
        "base_url": changes.get("base_url", model.base_url),
        "api_key": changes.get("api_key", model.api_key),
        "model_name": changes.get("model_name", model.model_name),
        "max_model_length": changes.get("max_model_length", model.max_model_length),
        "is_active": changes.get("is_active", model.is_active),
    }


def _build_candidate_model(data: dict[str, Any]) -> LLMModel:
    return LLMModel(
        name=data.get("name") or "Unnamed Model",
        provider_type=data.get("provider_type") or "openai_compatible",
        base_url=data.get("base_url"),
        api_key=data.get("api_key"),
        model_name=data.get("model_name"),
        max_model_length=data.get("max_model_length"),
        is_active=bool(data.get("is_active", False)),
    )
