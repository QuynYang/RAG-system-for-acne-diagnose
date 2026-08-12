"""Small adapter around the Google GenAI SDK.

The rest of the application should use this module instead of importing the
legacy Gemini SDK or configuring process-global SDK state.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

from google import genai
from google.genai import errors, types

from src.resilience.exceptions import (
    PermanentProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

GOOGLE_GENAI_SDK_VERSION = "google_genai_sdk_v1"


def get_google_api_key(api_key: str | None = None) -> str:
    key = (api_key if api_key is not None else os.getenv("GOOGLE_API_KEY", "")).strip()
    if not key:
        raise PermanentProviderError("GOOGLE_API_KEY is not configured.")
    return key


def build_google_genai_client(
    *,
    api_key: str | None = None,
    timeout_seconds: float | None = None,
) -> genai.Client:
    """Create an explicit Google GenAI client with SDK-level retry disabled."""

    retry_options = types.HttpRetryOptions(attempts=1)
    timeout_ms = None
    if timeout_seconds and timeout_seconds > 0:
        timeout_ms = max(1, int(timeout_seconds * 1000))
    http_options = types.HttpOptions(timeout=timeout_ms, retry_options=retry_options)
    return genai.Client(api_key=get_google_api_key(api_key), http_options=http_options)


def _generation_config(
    *,
    temperature: float,
    system_prompt: str | None = None,
) -> types.GenerateContentConfig:
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "http_options": types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
    }
    if system_prompt:
        kwargs["system_instruction"] = system_prompt
    return types.GenerateContentConfig(**kwargs)


async def generate_text_async(
    *,
    prompt: str,
    system_prompt: str | None,
    model_name: str,
    temperature: float,
    request_timeout: float | None = None,
    client: Any | None = None,
) -> str | None:
    """Generate text with the async Google GenAI client."""

    async def _call() -> Any:
        active_client = client or build_google_genai_client(timeout_seconds=request_timeout)
        return await active_client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=_generation_config(
                temperature=temperature,
                system_prompt=system_prompt,
            ),
        )

    try:
        if request_timeout and request_timeout > 0:
            async with asyncio.timeout(request_timeout):
                response = await _call()
        else:
            response = await _call()
        return extract_response_text(response)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise normalize_google_genai_exception(exc) from exc


def generate_text_sync(
    *,
    prompt: str,
    system_prompt: str | None,
    model_name: str,
    temperature: float,
    request_timeout: float | None = None,
    client: Any | None = None,
) -> str | None:
    """Generate text with the sync Google GenAI client."""

    try:
        active_client = client or build_google_genai_client(timeout_seconds=request_timeout)
        response = active_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=_generation_config(
                temperature=temperature,
                system_prompt=system_prompt,
            ),
        )
        return extract_response_text(response)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise normalize_google_genai_exception(exc) from exc


def extract_response_text(response: Any) -> str | None:
    """Extract text defensively from a Google GenAI response or test double."""

    try:
        text = getattr(response, "text")
    except Exception:
        return None
    if isinstance(text, str):
        return text
    return None


def embed_texts_sync(
    texts: list[str],
    *,
    model_name: str,
    task_type: str,
    expected_dimensions: int,
    api_key: str | None = None,
    client: Any | None = None,
) -> list[list[float]]:
    """Embed one or more texts and validate every returned vector."""

    if not texts:
        return []
    try:
        active_client = client or build_google_genai_client(api_key=api_key)
        config = types.EmbedContentConfig(task_type=task_type)
        # gemini-embedding-2 returns one vector per call even when contents is a list.
        vectors: list[list[float]] = []
        for text in texts:
            response = active_client.models.embed_content(
                model=model_name,
                contents=text,
                config=config,
            )
            vectors.extend(extract_embedding_vectors(response, expected_count=1))
        validate_embedding_vectors(vectors, expected_dimensions=expected_dimensions)
        return vectors
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        normalized = normalize_google_genai_exception(exc)
        if normalized is exc:
            raise
        raise normalized from exc


def extract_embedding_vectors(response: Any, *, expected_count: int) -> list[list[float]]:
    embeddings = _get_field(response, "embeddings")
    if embeddings is None:
        # Some tests may still use the old dict shape at this boundary.
        legacy = _get_field(response, "embedding")
        if legacy is not None:
            embeddings = legacy
    if embeddings is None:
        raise ValueError("Google GenAI embedding response is missing embeddings.")
    if not isinstance(embeddings, Sequence) or isinstance(embeddings, (str, bytes)):
        raise ValueError("Google GenAI embedding response has invalid embeddings shape.")
    if not embeddings:
        raise ValueError("Google GenAI embedding response returned no embeddings.")

    if embeddings and _looks_like_number_sequence(embeddings):
        vectors = [_coerce_vector(embeddings, index=0)]
    else:
        vectors = []
        for index, embedding in enumerate(embeddings):
            if isinstance(embedding, Mapping) and "values" in embedding:
                values = embedding.get("values")
            elif hasattr(embedding, "values"):
                values = getattr(embedding, "values")
            else:
                values = embedding
            vectors.append(_coerce_vector(values, index=index))

    if len(vectors) != expected_count:
        raise ValueError(
            f"Google GenAI embedding count mismatch: got {len(vectors)}, expected {expected_count}."
        )
    return vectors


def validate_embedding_vectors(
    vectors: list[list[float]],
    *,
    expected_dimensions: int,
) -> None:
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dimensions:
            raise ValueError(
                f"Google GenAI embedding dimension mismatch at index {index}: "
                f"got {len(vector)}, expected {expected_dimensions}."
            )


def normalize_google_genai_exception(exc: Exception) -> Exception:
    """Map Google GenAI SDK exceptions into runtime resilience exceptions."""

    if isinstance(exc, (PermanentProviderError, ProviderTimeoutError, ProviderUnavailableError)):
        return exc
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ProviderTimeoutError("Google GenAI request timed out.")
    if isinstance(exc, errors.APIError):
        code = int(getattr(exc, "code", 0) or 0)
        if code in {408, 504}:
            return ProviderTimeoutError(f"Google GenAI request timed out with HTTP {code}.")
        if code == 429 or code >= 500:
            return ProviderUnavailableError(
                f"Google GenAI provider returned retryable HTTP {code}."
            )
        if code in {400, 401, 403, 404, 422}:
            return PermanentProviderError(
                f"Google GenAI provider returned non-retryable HTTP {code}."
            )
    return exc


def _get_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _looks_like_number_sequence(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    return bool(value) and all(_is_real_number(item) for item in value)


def _coerce_vector(values: Any, *, index: int) -> list[float]:
    if values is None:
        raise ValueError(f"Google GenAI embedding at index {index} is missing values.")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"Google GenAI embedding at index {index} has invalid values.")
    if not values:
        raise ValueError(f"Google GenAI embedding at index {index} has empty values.")

    vector: list[float] = []
    for item in values:
        if not _is_real_number(item):
            raise ValueError(f"Google GenAI embedding at index {index} has non-finite values.")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"Google GenAI embedding at index {index} has non-finite values.")
        vector.append(number)
    return vector


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "GOOGLE_GENAI_SDK_VERSION",
    "build_google_genai_client",
    "embed_texts_sync",
    "extract_embedding_vectors",
    "extract_response_text",
    "generate_text_async",
    "generate_text_sync",
    "normalize_google_genai_exception",
    "validate_embedding_vectors",
]
