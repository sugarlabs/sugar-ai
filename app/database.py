"""
Database models and connection handling for Sugar-AI.
"""

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime
from typing import Dict, Any, Generator

# database connection
DATABASE_URL = "sqlite:///./sugar_ai.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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
            "approved": self.approved,
        }


class TokenUsage(Base):
    """Track token usage for cost optimization"""

    __tablename__ = "token_usage"

    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String, index=True)  # Which user
    user_name = Column(String)
    endpoint = Column(String)  # /ask, /debug, etc.
    question = Column(Text)  # The user's question

    # Token counts
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)

    # Model info
    model_name = Column(String)

    # Performance metrics
    response_time_seconds = Column(Integer)  # Changed to Integer for SQLite

    # Cost estimation (in cents)
    estimated_cost_cents = Column(Integer)  # Changed to Integer (store as cents)

    # Timestamp
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_name": self.user_name,
            "endpoint": self.endpoint,
            "question": (
                self.question[:100] + "..."
                if len(self.question) > 100
                else self.question
            ),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name,
            "response_time_seconds": self.response_time_seconds,
            "estimated_cost_cents": self.estimated_cost_cents
            / 100,  # Convert back to dollars
            "created_at": self.created_at.isoformat(),
        }


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
