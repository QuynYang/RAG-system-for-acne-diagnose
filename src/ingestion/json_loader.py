"""Load crawled web-page JSON datasets for offline ingestion."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MIN_RAW_TEXT_CHARS = 50
WRAPPER_RECORD_KEYS = ("records", "data", "items", "documents", "pages")


def load_web_json_documents(path: str | Path) -> list[dict[str, Any]]:
    """Load valid web JSON records as ingestion documents."""

    documents, _summary = load_web_json_documents_with_stats(path)
    return documents


def load_web_json_documents_with_stats(
    path: str | Path,
    *,
    min_raw_text_chars: int = DEFAULT_MIN_RAW_TEXT_CHARS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load web JSON records and return documents plus total/skipped counts."""

    json_path = Path(path)
    try:
        # utf-8-sig accepts files saved with or without a UTF-8 BOM (common on Windows).
        data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file at {json_path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read JSON file at {json_path}: {exc}") from exc

    records = _extract_records(data, json_path)
    documents: list[dict[str, Any]] = []
    skipped = 0

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            skipped += 1
            logger.warning(
                "[JSON LOADER] Skipping malformed record %d in %s: expected object",
                record_index,
                json_path.name,
            )
            continue

        raw_text = str(record.get("raw_text") or "").strip()
        if len(raw_text) < min_raw_text_chars:
            skipped += 1
            logger.warning(
                "[JSON LOADER] Skipping empty/short record %d in %s: %d chars",
                record_index,
                json_path.name,
                len(raw_text),
            )
            continue

        metadata = {
            "source_type": "web_json",
            "source_file": json_path.name,
            "source_path": _source_path_key(json_path),
            "seed_url": str(record.get("seed_url") or ""),
            "source_url": str(record.get("source_url") or ""),
            "record_index": record_index,
        }
        for optional_key in ("title", "document_id_hint"):
            if record.get(optional_key):
                metadata[optional_key] = str(record[optional_key])

        documents.append(
            {
                "text": raw_text,
                "metadata": metadata,
            }
        )

    return documents, {"total_records": len(records), "skipped_records": skipped}


def _extract_records(data: Any, path: Path) -> list[Any]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in WRAPPER_RECORD_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return value
        keys = ", ".join(WRAPPER_RECORD_KEYS)
        raise ValueError(
            f"Unsupported JSON structure at {path}: expected array or one of keys: {keys}"
        )

    raise ValueError(
        f"Unsupported JSON structure at {path}: expected array or object wrapper"
    )


def _source_path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


__all__ = [
    "DEFAULT_MIN_RAW_TEXT_CHARS",
    "load_web_json_documents",
    "load_web_json_documents_with_stats",
]
