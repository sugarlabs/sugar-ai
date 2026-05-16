"""
Quota management for Sugar-AI API usage.
"""
from datetime import date, datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError


from app.config import settings
from app.database import APIQuota


def check_and_increment_quota(api_key: str, db: Session) -> dict:
    """Thread-safe quota check + increment.

    Returns {"remaining": int, "total": int}
    Raises HTTPException(429) if exceeded.
    """
    today = date.today()
    max_req = settings.MAX_DAILY_REQUESTS

    # Atomic SQLite quota increment (avoids ORM read/modify/write races).
    # Requirement: keep response shape and 429 detail exact.

    from sqlalchemy import text

    now = datetime.now(timezone.utc)
    # Format dates as ISO strings for raw SQL (SQLite stores as TEXT)
    today_str = today.isoformat()
    now_str = now.isoformat()

    # 1) Ensure row exists for today.
    db.execute(
        text(
            """
            INSERT OR IGNORE INTO api_quotas (api_key, request_count, quota_date, updated_at)
            VALUES (:api_key, 0, :today, :updated_at)
            """
        ),
        {"api_key": api_key, "today": today_str, "updated_at": now_str},
    )

    # 2) Reset to today if date changed.
    db.execute(
        text(
            """
            UPDATE api_quotas
            SET request_count = 0, quota_date = :today, updated_at = :updated_at
            WHERE api_key = :api_key AND quota_date != :today
            """
        ),
        {"api_key": api_key, "today": today_str, "updated_at": now_str},
    )

    # 3) Atomically increment while quota remains.
    res = db.execute(
        text(
            """
            UPDATE api_quotas
            SET request_count = request_count + 1,
                updated_at = :updated_at
            WHERE api_key = :api_key
              AND quota_date = :today
              AND request_count < :max_req
            """
        ),
        {"api_key": api_key, "today": today_str, "updated_at": now_str, "max_req": max_req},
    )

    updated = res.rowcount if res.rowcount is not None else 0



    # 4) If increment didn't happen, check why.
    # - Normal/exhausted case: request_count is already >= max_req.
    # - Race/resets case: date changed; the earlier reset/insert may not be visible
    #   to this transaction yet. In that case, treat the key as fresh rather than
    #   incorrectly returning a 429 for a missing row.
    if updated == 0:
        row = db.query(APIQuota).filter(APIQuota.api_key == api_key).first()

        # If the row exists and quota is exhausted, raise exact 429.
        if row is not None and row.quota_date == today and row.request_count >= max_req:
            # Quota exhausted.
            db.commit()
            raise HTTPException(status_code=429, detail="Daily request quota exceeded")

        # A missing row here is a visibility/setup issue, not quota exhaustion.
        if row is None:
            db.commit()
            return {"remaining": max_req, "total": max_req}

        # Eagerly access request_count before commit to avoid lazy-load issues in concurrent tests
        remaining = max_req - row.request_count
        db.commit()
        return {"remaining": remaining, "total": max_req}


    # 5) Fetch updated row for remaining.
    row = db.query(APIQuota).filter(APIQuota.api_key == api_key).first()


    if row is None:
        db.commit()
        raise HTTPException(status_code=500, detail="Error processing request")

    db.commit()

    return {"remaining": max_req - row.request_count, "total": max_req}








