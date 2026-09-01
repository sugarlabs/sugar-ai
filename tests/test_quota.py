"""
Tests for persistent, session-safe API quota tracking.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import APIQuota, Base
from app.quota import check_and_increment_quota


def make_sessionmaker(db_path):
    """Create a sessionmaker bound to a shared-cache SQLite file so multiple
    engines/sessions (simulating separate connections/reconnects) can see
    the same persisted data."""
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return engine, sessionmaker(bind=engine)


def test_quota_persists_across_recreated_sessions(tmp_path):
    """Quota state must survive a brand new engine/session pointed at the
    same database, simulating a process restart or reconnect."""
    db_path = tmp_path / "quota.db"

    engine1, SessionLocal1 = make_sessionmaker(db_path)
    db1 = SessionLocal1()
    check_and_increment_quota("key1", db1)
    check_and_increment_quota("key1", db1)
    db1.close()
    engine1.dispose()

    # Recreate the engine/session from scratch, as would happen on restart.
    engine2, SessionLocal2 = make_sessionmaker(db_path)
    db2 = SessionLocal2()
    quota = check_and_increment_quota("key1", db2)
    db2.close()
    engine2.dispose()

    assert quota["remaining"] == settings.MAX_DAILY_REQUESTS - 3


def test_first_request_is_counted(tmp_path):
    engine, SessionLocal = make_sessionmaker(tmp_path / "quota.db")
    db = SessionLocal()

    quota = check_and_increment_quota("key1", db)

    assert quota["remaining"] == settings.MAX_DAILY_REQUESTS - 1
    assert quota["total"] == settings.MAX_DAILY_REQUESTS

    db.close()
    engine.dispose()


def test_quota_exceeded_raises_429(tmp_path):
    engine, SessionLocal = make_sessionmaker(tmp_path / "quota.db")
    db = SessionLocal()

    for _ in range(settings.MAX_DAILY_REQUESTS):
        check_and_increment_quota("key1", db)

    with pytest.raises(HTTPException) as exc_info:
        check_and_increment_quota("key1", db)

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail == "Daily request quota exceeded"

    db.close()
    engine.dispose()


def test_quota_resets_on_new_day(tmp_path):
    engine, SessionLocal = make_sessionmaker(tmp_path / "quota.db")
    db = SessionLocal()

    check_and_increment_quota("key1", db)
    row = db.query(APIQuota).filter(APIQuota.api_key == "key1").first()
    row.quota_date = date.today() - timedelta(days=1)
    row.request_count = settings.MAX_DAILY_REQUESTS
    db.commit()

    quota = check_and_increment_quota("key1", db)

    assert quota["remaining"] == settings.MAX_DAILY_REQUESTS - 1

    db.close()
    engine.dispose()


def test_concurrent_requests_never_exceed_quota(tmp_path):
    """Concurrent requests for the same key (as FastAPI's threadpool would
    dispatch for sync dependencies) must never be granted more than the
    daily limit, even when they race on the read-modify-write."""
    db_path = tmp_path / "quota.db"
    engine, SessionLocal = make_sessionmaker(db_path)

    attempts = settings.MAX_DAILY_REQUESTS * 3

    def worker(_):
        db = SessionLocal()
        try:
            check_and_increment_quota("key1", db)
            return True
        except HTTPException:
            return False
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(worker, range(attempts)))

    assert sum(results) == settings.MAX_DAILY_REQUESTS

    db = SessionLocal()
    row = db.query(APIQuota).filter(APIQuota.api_key == "key1").first()
    db.close()
    engine.dispose()

    assert row.request_count == settings.MAX_DAILY_REQUESTS
