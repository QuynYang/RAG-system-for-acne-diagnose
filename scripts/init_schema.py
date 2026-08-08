#!/usr/bin/env python3
"""
init_schema.py
==============
Acne Advisor AI – Database Schema Initialisation Script

Responsibilities
----------------
1. Load environment variables from .env.
2. Create patient_records table in PostgreSQL with JSONB profile.
3. Enable PostgreSQL extensions.
4. Create SQLAlchemy model tables if src.database.models exists.
5. Create Qdrant collection with named dense vector + sparse BM25 vector.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
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
logger = logging.getLogger("init_schema")

from src.database.connection import normalize_async_database_url, normalize_sync_database_url

DATABASE_URL = normalize_async_database_url(
    os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:password@localhost:5433/acne_agent_db",
    )
)

SYNC_DATABASE_URL = normalize_sync_database_url(
    os.getenv("SYNC_DATABASE_URL", DATABASE_URL)
)

VECTOR_DB_PROVIDER = os.getenv("VECTOR_DB_PROVIDER", "qdrant").lower()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "acne_knowledge")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "3072"))
FORCE_RECREATE_QDRANT_COLLECTION = os.getenv("FORCE_RECREATE_QDRANT_COLLECTION", "false").lower() in {"1", "true", "yes", "y"}


def _mask_password(url: str) -> str:
    return re.sub(r"(://[^:]+:)([^@]+)(@)", r"\1***\3", url)


def _raw_sql(statement: str):
    from sqlalchemy import text

    return text(statement)


def _create_patient_records_table() -> None:
    logger.info("[patient_records] Connecting via psycopg2...")
    logger.info("[patient_records] URL: %s", _mask_password(SYNC_DATABASE_URL))

    try:
        from sqlalchemy import Column, Integer, MetaData, Table, create_engine, text
        from sqlalchemy.dialects.postgresql import JSONB
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: pip install sqlalchemy psycopg2-binary"
        ) from exc

    engine = create_engine(
        SYNC_DATABASE_URL,
        echo=(LOG_LEVEL == "DEBUG"),
        pool_pre_ping=True,
    )

    metadata = MetaData()

    patient_records = Table(
        "patient_records",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column(
            "patient_profile",
            JSONB,
            nullable=False,
            server_default=text("'{}'::jsonb"),
        ),
        comment="Stores patient profile as flexible JSONB.",
    )

    try:
        patient_records.create(bind=engine, checkfirst=True)
        logger.info("✓ Table 'patient_records' is ready.")
    finally:
        engine.dispose()


async def _seed_reference_data(conn) -> None:
    logger.info("Seeding reference data...")
    logger.info("(no reference data to seed yet)")


async def _setup_postgres() -> None:
    logger.info("Connecting to PostgreSQL...")
    logger.info("URL: %s", _mask_password(DATABASE_URL))

    try:
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: pip install sqlalchemy[asyncio] asyncpg"
        ) from exc

    engine = create_async_engine(DATABASE_URL, echo=(LOG_LEVEL == "DEBUG"))

    async with engine.begin() as conn:
        logger.info("Enabling PostgreSQL extensions...")

        await conn.execute(_raw_sql('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))

        if VECTOR_DB_PROVIDER == "pgvector":
            await conn.execute(_raw_sql("CREATE EXTENSION IF NOT EXISTS vector;"))
            logger.info("✓ pgvector extension enabled.")

        logger.info("Creating SQLAlchemy model tables if available...")

        try:
            from src.database.models import metadata  # type: ignore

            await conn.run_sync(metadata.create_all)
            logger.info("✓ Tables created from src.database.models metadata.")
        except ImportError:
            logger.warning(
                "src.database.models not found yet. Skipping model table creation."
            )

        await _seed_reference_data(conn)

    await engine.dispose()
    logger.info("✓ PostgreSQL setup complete.")


async def _setup_qdrant() -> None:
    logger.info("Setting up Qdrant collection...")
    logger.info("URL: %s", QDRANT_URL)
    logger.info("Collection: %s", QDRANT_COLLECTION_NAME)
    logger.info("Dense dimensions: %d", EMBEDDING_DIMENSIONS)

    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, SparseVectorParams, VectorParams
        from src.database.vector_store import qdrant_client_kwargs
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install qdrant-client") from exc

    client = AsyncQdrantClient(**qdrant_client_kwargs())

    def get_named_config(config, name: str):
        if config is None:
            return None
        if isinstance(config, dict):
            return config.get(name)
        if hasattr(config, "get"):
            return config.get(name)
        return None

    try:
        collections = await client.get_collections()
        existing = {c.name for c in collections.collections}

        if QDRANT_COLLECTION_NAME in existing and FORCE_RECREATE_QDRANT_COLLECTION:
            logger.warning(
                "FORCE_RECREATE_QDRANT_COLLECTION=true → deleting existing Qdrant collection '%s'.",
                QDRANT_COLLECTION_NAME,
            )
            await client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
            existing.remove(QDRANT_COLLECTION_NAME)

        if QDRANT_COLLECTION_NAME not in existing:
            await client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(
                        size=EMBEDDING_DIMENSIONS,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams()
                },
            )
            logger.info(
                "✓ Qdrant collection '%s' created with dense + bm25.",
                QDRANT_COLLECTION_NAME,
            )
        else:
            info = await client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
            params = info.config.params
            if isinstance(params, dict):
                vectors_config = params.get("vectors")
                sparse_vectors_config = params.get("sparse_vectors")
            else:
                vectors_config = getattr(params, "vectors", None)
                sparse_vectors_config = getattr(params, "sparse_vectors", None)

            dense_config = get_named_config(vectors_config, "dense")
            bm25_config = get_named_config(sparse_vectors_config, "bm25")
            schema_errors: list[str] = []

            if dense_config is None:
                schema_errors.append("missing named dense vector 'dense'")
            else:
                dense_size = (
                    dense_config.get("size")
                    if isinstance(dense_config, dict)
                    else getattr(dense_config, "size", None)
                )
                if dense_size != EMBEDDING_DIMENSIONS:
                    schema_errors.append(
                        f"named vector 'dense' has size {dense_size}, "
                        f"expected {EMBEDDING_DIMENSIONS}"
                    )

            if bm25_config is None:
                schema_errors.append("missing sparse vector 'bm25'")

            if schema_errors:
                raise RuntimeError(
                    "Qdrant collection schema mismatch for "
                    f"'{QDRANT_COLLECTION_NAME}': "
                    + "; ".join(schema_errors)
                    + ". Set FORCE_RECREATE_QDRANT_COLLECTION=true only if you "
                    "intend to delete/recreate the collection, or migrate it manually."
                )

            logger.info(
                "✓ Qdrant collection '%s' already exists. Schema validated.",
                QDRANT_COLLECTION_NAME,
            )
            logger.info("  Existing Qdrant config: %s", params)

    finally:
        await client.close()


async def main() -> int:
    logger.info("=" * 60)
    logger.info("Acne Advisor AI – Schema Initialisation")
    logger.info("=" * 60)

    try:
        _create_patient_records_table()
    except Exception as exc:
        logger.error("patient_records table creation failed: %s", exc, exc_info=True)
        return 1

    try:
        await _setup_postgres()
    except Exception as exc:
        logger.error("PostgreSQL setup failed: %s", exc, exc_info=True)
        return 1

    if VECTOR_DB_PROVIDER == "qdrant":
        try:
            await _setup_qdrant()
        except Exception as exc:
            logger.error("Qdrant setup failed: %s", exc, exc_info=True)
            return 1

    logger.info("=" * 60)
    logger.info("✅ Schema initialisation completed successfully.")
    logger.info("=" * 60)
    return 0


def cli() -> int:
    """Synchronous console-script wrapper."""

    return asyncio.run(main())


if __name__ == "__main__":
    raise SystemExit(cli())
