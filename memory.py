"""Two-tier memory.

Short-term memory is the recent conversation for a session. It is persisted
to PostgreSQL (Neon) in a "message_store" table keyed by session_id, so it
survives restarts. Writes go through PostgresChatMessageHistory; reads use a
direct SQL query that fetches only the last few messages.

Long-term memory uses Mem0 so facts persist across sessions and are recalled
automatically on each query.
"""
from __future__ import annotations

import logging

import httpx
import sqlalchemy
from langchain_core.messages import BaseMessage, messages_from_dict
from langchain_community.chat_message_histories import PostgresChatMessageHistory

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Postgres-backed short-term memory (via PostgresChatMessageHistory)
# ---------------------------------------------------------------------------

TABLE_NAME = "message_store"

# How many recent messages go into the prompt. Older turns stay in Postgres
# (and Mem0 keeps the long view); they just stop growing the prompt.
MAX_RECENT_MESSAGES = 10


def get_history(session_id: str) -> PostgresChatMessageHistory:
    """Return the Postgres-backed message history for a session.

    `PostgresChatMessageHistory` manages its own connection and creates the
    `message_store` table in the configured database on first use. Messages
    are keyed by session_id (one row per message, one user per session_id).
    """
    return PostgresChatMessageHistory(
        session_id=session_id,
        connection_string=settings.postgres_conn_string,
        table_name=TABLE_NAME,
    )


_engine = None
_table_ready = False


def _get_engine():
    """Shared SQLAlchemy engine (pooled) for direct read queries."""
    global _engine
    if _engine is None:
        _engine = sqlalchemy.create_engine(settings.postgres_conn_string)
    return _engine


def _ensure_table() -> None:
    """Guarantee the message_store table exists. Instantiating
    PostgresChatMessageHistory runs an idempotent CREATE TABLE IF NOT
    EXISTS, and it owns the schema, so it performs the creation."""
    global _table_ready
    if not _table_ready:
        get_history("__bootstrap__")
        _table_ready = True


def load_recent_messages(session_id: str) -> list[BaseMessage]:
    """Return the session's most recent messages, trimmed in the database.

    The SQL query itself fetches only the last MAX_RECENT_MESSAGES rows
    (ORDER BY id DESC LIMIT n), then the result is flipped back into
    conversation order and decoded into message objects.
    """
    if not settings.postgres_ready:
        return []
    try:
        _ensure_table()
        with _get_engine().connect() as conn:
            rows = conn.execute(
                sqlalchemy.text(
                    "SELECT message FROM message_store "
                    "WHERE session_id = :sid ORDER BY id DESC LIMIT :n"
                ),
                {"sid": session_id, "n": MAX_RECENT_MESSAGES},
            ).fetchall()
        # Rows come back newest-first; reverse into conversation order.
        return messages_from_dict([row[0] for row in reversed(rows)])
    except Exception as exc:  # noqa: BLE001 - memory must never break ask
        logger.warning("load history failed: %s", exc)
        return []


def save_exchange(session_id: str, question: str, answer: str) -> None:
    """Append one user question and its assistant answer to short-term memory."""
    if not settings.postgres_ready:
        return
    try:
        hist = get_history(session_id)
        hist.add_user_message(question)
        hist.add_ai_message(answer)
    except Exception as exc:  # noqa: BLE001
        logger.warning("save exchange failed: %s", exc)


def clear_short_term(session_id: str) -> None:
    """Delete a session's persisted short-term conversation."""
    if not settings.postgres_ready:
        return
    try:
        get_history(session_id).clear()
    except Exception as exc:  # noqa: BLE001
        logger.warning("clear short-term failed: %s", exc)


# ---------------------------------------------------------------------------
# Long-term memory via Mem0
# ---------------------------------------------------------------------------

_headers = None


def _mem0_headers() -> dict:
    global _headers
    if _headers is None:
        _headers = {
            "Authorization": f"Token {settings.mem0_api_key}",
            "Content-Type": "application/json",
        }
    return _headers


def _mem0_url() -> str:
    """Base URL for the Mem0 API. The trailing slash avoids a 301 redirect
    the API otherwise returns when calling collection endpoints without it."""
    return settings.mem0_api_url.rstrip("/") + "/v1/memories/"


def mem0_add(question: str, answer: str, session_id: str) -> dict | None:
    """Store a Q/A exchange in long-term memory for the given session."""
    if not settings.mem0_ready:
        return None
    url = _mem0_url()
    payload = {
        "messages": [{"role": "user", "content": question},
                     {"role": "assistant", "content": answer}],
        "user_id": session_id,
    }
    try:
        resp = httpx.post(url, json=payload, headers=_mem0_headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as exc:
        # Non-blocking: log-and-return so a memory failure never breaks ask.
        logger.warning("mem0 add failed: %s", exc)
        return None


def mem0_get(session_id: str) -> list:
    """Retrieve all long-term memories for a session."""
    if not settings.mem0_ready:
        return []
    url = _mem0_url()
    try:
        resp = httpx.get(
            url,
            params={"user_id": session_id},
            headers=_mem0_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # The hosted API returns a JSON list of memory objects directly.
        if isinstance(data, list):
            return data
        # Some versions wrap them in {"results": [...]}.
        return data.get("results", []) if isinstance(data, dict) else []
    except httpx.HTTPError as exc:
        logger.warning("mem0 get failed: %s", exc)
        return []


def mem0_clear(session_id: str) -> bool:
    """Delete all long-term memories for a session."""
    if not settings.mem0_ready:
        return False
    url = _mem0_url()
    try:
        resp = httpx.delete(
            url,
            params={"user_id": session_id},
            headers=_mem0_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        logger.warning("mem0 clear failed: %s", exc)
        return False