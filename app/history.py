"""
In-memory history storage for Sugar-AI.
Stores question-answer pairs per API key.
"""
import logging
from datetime import datetime

logger = logging.getLogger("sugar-ai")

_history: dict[str, list[dict]] = {}
MAX_HISTORY_PER_USER = 50


def add_to_history(api_key: str, question: str, answer: str, endpoint: str) -> None:
    """Save a Q&A pair for the given api_key."""
    if api_key not in _history:
        _history[api_key] = []

    entry = {
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoint": endpoint,
    }

    _history[api_key].append(entry)

    if len(_history[api_key]) > MAX_HISTORY_PER_USER:
        _history[api_key] = _history[api_key][-MAX_HISTORY_PER_USER:]

    logger.debug(f"History saved for api_key={api_key[:5]}... total={len(_history[api_key])}")


def get_history(api_key: str, limit: int = 50) -> list[dict]:
    """Return most recent `limit` entries for the given api_key."""
    entries = _history.get(api_key, [])
    return entries[-limit:]


def clear_history(api_key: str) -> int:
    """Delete all history for the given api_key. Returns count deleted."""
    entries = _history.pop(api_key, [])
    logger.info(f"Cleared {len(entries)} entries for api_key={api_key[:5]}...")
    return len(entries)