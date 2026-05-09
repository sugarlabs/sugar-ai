"""
Database models and connection handling for Sugar-AI.
"""
import os
import datetime
from typing import Dict, Any, Generator

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# database connection
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sugar_ai.db")
engine_kwargs = {"connect_args": {"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# api key model
class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    name = Column(String)
    email = Column(String)
    can_change_model = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    request_reason = Column(Text, nullable=True)
    approved = Column(Boolean, default=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "email": self.email,
            "can_change_model": self.can_change_model,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "approved": self.approved
        }


class LLMModel(Base):
    __tablename__ = "llm_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    provider_type = Column(String, nullable=False, default="openai_compatible")
    base_url = Column(String, nullable=False)
    api_key = Column(String, nullable=True)
    model_name = Column(String, nullable=False)
    max_model_length = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
        nullable=False,
    )

    def to_dict(self, include_api_key: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "provider_type": self.provider_type,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "max_model_length": self.max_model_length,
            "is_active": self.is_active,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if include_api_key:
            data["api_key"] = self.api_key
        return data


def create_tables() -> None:
    """Create database tables if they don't exist"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Database dependency for FastAPI endpoints"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
