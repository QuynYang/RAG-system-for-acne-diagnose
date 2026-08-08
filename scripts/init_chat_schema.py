#!/usr/bin/env python3
"""
scripts/init_chat_schema.py
============================
Creates the chat_sessions and chat_messages tables in PostgreSQL.

Safety:
- Uses CREATE TABLE IF NOT EXISTS — will NOT drop or overwrite existing tables.
- Does NOT delete any existing data.
- Safe to run multiple times (idempotent).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("init_chat_schema")

from src.database.connection import normalize_async_database_url

DATABASE_URL = normalize_async_database_url(
    os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:password@localhost:5433/acne_agent_db",
    )
)

# ---------------------------------------------------------------------------
# SQL Statements
# ---------------------------------------------------------------------------

CREATE_CHAT_SESSIONS = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id              TEXT        PRIMARY KEY,
    user_id         TEXT,
    title           TEXT        NOT NULL DEFAULT 'Đoạn chat mới',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    hidden          BOOLEAN     NOT NULL DEFAULT false,
    metadata        JSONB
);
"""

CREATE_CHAT_MESSAGES = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id              TEXT        PRIMARY KEY,
    session_id      TEXT        NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role            TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT        NOT NULL,
    sources         JSONB,
    symptoms        JSONB,
    safety_flags    JSONB,
    graph_facts     JSONB,
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

CREATE_INDEX_MESSAGES_SESSION = """
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
    ON chat_messages (session_id);
"""

CREATE_INDEX_MESSAGES_CREATED = """
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at
    ON chat_messages (session_id, created_at);
"""

CREATE_INDEX_SESSIONS_UPDATED = """
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at
    ON chat_sessions (updated_at DESC);
"""

CREATE_INDEX_SESSIONS_HIDDEN = """
CREATE INDEX IF NOT EXISTS idx_chat_sessions_hidden
    ON chat_sessions (hidden);
"""


async def main() -> int:
    logger.info("=" * 60)
    logger.info("Chat History Schema Initialisation")
    logger.info("=" * 60)

    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        raise SystemExit(
            "Run: pip install sqlalchemy[asyncio] asyncpg"
        ) from exc

    engine = create_async_engine(DATABASE_URL, echo=(LOG_LEVEL == "DEBUG"))

    try:
        async with engine.begin() as conn:
            logger.info("Creating table chat_sessions (IF NOT EXISTS)...")
            await conn.execute(text(CREATE_CHAT_SESSIONS))
            logger.info("✓ chat_sessions ready.")

            logger.info("Creating table chat_messages (IF NOT EXISTS)...")
            await conn.execute(text(CREATE_CHAT_MESSAGES))
            logger.info("✓ chat_messages ready.")

            logger.info("Creating indexes...")
            await conn.execute(text(CREATE_INDEX_MESSAGES_SESSION))
            await conn.execute(text(CREATE_INDEX_MESSAGES_CREATED))
            await conn.execute(text(CREATE_INDEX_SESSIONS_UPDATED))
            await conn.execute(text(CREATE_INDEX_SESSIONS_HIDDEN))
            logger.info("✓ Indexes ready.")

        logger.info("=" * 60)
        logger.info("✅ Chat history schema initialisation completed successfully.")
        logger.info("=" * 60)
        return 0

    except Exception as exc:
        logger.error("Chat schema initialisation failed: %s", exc, exc_info=True)
        return 1

    finally:
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
