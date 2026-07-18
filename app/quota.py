"""
Persistent API quota tracking for Sugar-AI.
"""
import datetime
from typing import Dict

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings


def check_and_increment_quota(api_key: str, db: Session) -> Dict[str, int]:
    """Check and increment a key's persisted daily request count.

    The check and the increment are done via a single conditional UPDATE
    statement so concurrent requests for the same key (FastAPI may run sync
    dependencies like this one on separate threads/DB connections) can't
    race on a read-modify-write and lose or double count updates.

    Raises HTTPException(429) if the key has exhausted today's quota.
    """
    today = datetime.date.today()
    max_requests = settings.MAX_DAILY_REQUESTS

    # Ensure a row exists for this key, resetting it if it's from a
    # previous day, before attempting the atomic increment below.
    db.execute(
        text(
            "INSERT INTO api_quotas (api_key, request_count, quota_date) "
            "VALUES (:api_key, 0, :today) "
            "ON CONFLICT(api_key) DO UPDATE SET "
            "request_count = CASE WHEN api_quotas.quota_date != :today THEN 0 "
            "ELSE api_quotas.request_count END, "
            "quota_date = :today"
        ),
        {"api_key": api_key, "today": today},
    )

    result = db.execute(
        text(
            "UPDATE api_quotas SET request_count = request_count + 1 "
            "WHERE api_key = :api_key AND quota_date = :today "
            "AND request_count < :max_requests"
        ),
        {"api_key": api_key, "today": today, "max_requests": max_requests},
    )
    db.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=429, detail="Daily request quota exceeded")

    request_count = db.execute(
        text("SELECT request_count FROM api_quotas WHERE api_key = :api_key"),
        {"api_key": api_key},
    ).scalar_one()

    return {"remaining": max_requests - request_count, "total": max_requests}
