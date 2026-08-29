"""
Database models and connection handling for Sugar-AI.
"""
from datetime import datetime, timezone
from typing import Dict, Any, Generator

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Date, DateTime, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    created_at = Column(DateTime, default=utc_now)
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


class APIQuota(Base):
    __tablename__ = "api_quotas"
    api_key = Column(String, primary_key=True)
    request_count = Column(Integer, default=0, nullable=False)
    quota_date = Column(Date, nullable=False)
    updated_at = Column(DateTime, default=utc_now)


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
