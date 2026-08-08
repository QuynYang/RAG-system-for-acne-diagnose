"""
src/database/connection.py – Database Engine & Session Factory
=============================================================
12-Factor App: Backing Services – treat database as an attached resource.
Connection URL comes entirely from the DATABASE_URL environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import AsyncGenerator

try:
    from dotenv import load_dotenv

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


def normalize_async_database_url(url: str) -> str:
    """Convert Railway/Heroku-style postgres URLs for SQLAlchemy asyncpg."""

    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    return url


def normalize_sync_database_url(url: str) -> str:
    """Derive psycopg2 sync URL from async or platform postgres URLs."""

    normalized = normalize_async_database_url(url)
    if normalized.startswith("postgresql+asyncpg://"):
        return "postgresql+psycopg2://" + normalized[len("postgresql+asyncpg://") :]
    if normalized.startswith("postgresql://"):
        return normalized.replace("postgresql://", "postgresql+psycopg2://", 1)
    return normalized


DATABASE_URL: str = normalize_async_database_url(
    os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://user:password@localhost:5433/acne_agent_db",
    )
)

# ---------------------------------------------------------------------------
# Engine – use NullPool in tests/scripts, pool in long-running services
# ---------------------------------------------------------------------------
_use_null_pool = os.getenv("DB_USE_NULL_POOL", "false").lower() == "true"

engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("LOG_LEVEL", "INFO").upper() == "DEBUG",
    pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
    pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
    poolclass=NullPool if _use_null_pool else None,
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager / FastAPI dependency that yields a DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
