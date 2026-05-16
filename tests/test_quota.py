from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import APIQuota, Base
from app.quota import check_and_increment_quota


def make_engine(db_name: str):
    return create_engine(
        f"sqlite:///file:{db_name}?mode=memory&cache=shared",
        connect_args={"check_same_thread": False, "uri": True},
        poolclass=StaticPool,
    )


def make_session(db_name: str):
    engine = make_engine(db_name)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    return engine, SessionLocal


def test_first_request_remaining_decrements():
    db_name = f"quota_{uuid.uuid4().hex}"
    engine, SessionLocal = make_session(db_name)
    db = SessionLocal()
    quota = check_and_increment_quota("key1", db)

    assert quota["remaining"] == settings.MAX_DAILY_REQUESTS - 1
    assert quota["total"] == settings.MAX_DAILY_REQUESTS

    db.close()
    engine.dispose()


def test_quota_exceeded_raises_429():
    db_name = f"quota_{uuid.uuid4().hex}"
    engine, SessionLocal = make_session(db_name)
    db = SessionLocal()

    for _ in range(settings.MAX_DAILY_REQUESTS):
        check_and_increment_quota("key1", db)

    with pytest.raises(HTTPException) as exc:
        check_and_increment_quota("key1", db)

    assert exc.value.status_code == 429

    db.close()
    engine.dispose()


def test_quota_resets_on_new_day():
    db_name = f"quota_{uuid.uuid4().hex}"
    engine, SessionLocal = make_session(db_name)
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


def test_quota_persists_after_engine_recreate():
    db_name = f"quota_{uuid.uuid4().hex}"
    engine1, SessionLocal1 = make_session(db_name)
    db1 = SessionLocal1()
    check_and_increment_quota("key1", db1)

    engine2 = make_engine(db_name)
    Base.metadata.create_all(bind=engine2)
    SessionLocal2 = sessionmaker(bind=engine2)
    db2 = SessionLocal2()

    row = db2.query(APIQuota).filter(APIQuota.api_key == "key1").first()
    assert row is not None

    db2.close()
    engine2.dispose()
    db1.close()
    engine1.dispose()


def test_concurrent_quota_increments():
    # File-based SQLite with NullPool: each thread gets its own connection
    # SQLite serializes writes internally but handles multiple connections
    import tempfile
    import os
    
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    try:
        from sqlalchemy.pool import NullPool
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False, "timeout": 30},
            poolclass=NullPool,  # Each thread gets its own connection
        )
        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)

        def worker():
            db = SessionLocal()
            try:
                return check_and_increment_quota("key1", db)
            finally:
                db.close()

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda _: worker(), range(10)))

        assert len(results) == 10
        
        # Verify DB count matches exactly
        db = SessionLocal()
        row = db.query(APIQuota).filter(APIQuota.api_key == "key1").first()
        db.close()
        assert row.request_count == 10, f"Expected 10, got {row.request_count}"

        engine.dispose()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
