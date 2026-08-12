from __future__ import annotations

import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors

from src.agent.llm import provider as llm_provider
from src.integrations import google_genai
from src.resilience.exceptions import (
    PermanentProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


class _FakeAsyncModels:
    def __init__(self, capture: dict, response: object) -> None:
        self.capture = capture
        self.response = response

    async def generate_content(self, **kwargs: object) -> object:
        self.capture.update(kwargs)
        return self.response


class _FakeSyncModels:
    def __init__(self, capture: dict, response: object | None = None) -> None:
        self.capture = capture
        self.response = response

    def generate_content(self, **kwargs: object) -> object:
        self.capture.update(kwargs)
        return self.response or SimpleNamespace(text="sync text")

    def embed_content(self, **kwargs: object) -> object:
        self.capture.update(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, capture: dict, response: object) -> None:
        self.aio = SimpleNamespace(models=_FakeAsyncModels(capture, response))
        self.models = _FakeSyncModels(capture, response)


@pytest.mark.asyncio
async def test_generate_text_async_uses_google_genai_method_and_config() -> None:
    capture: dict = {}
    client = _FakeClient(capture, SimpleNamespace(text="Xin chào"))

    text = await google_genai.generate_text_async(
        prompt="Nội dung hỏi",
        system_prompt="System instruction",
        model_name="gemini-test",
        temperature=0.3,
        request_timeout=3,
        client=client,
    )

    assert text == "Xin chào"
    assert capture["model"] == "gemini-test"
    assert capture["contents"] == "Nội dung hỏi"
    assert capture["config"].temperature == 0.3
    assert capture["config"].system_instruction == "System instruction"
    assert capture["config"].http_options.retry_options.attempts == 1


def test_generate_text_sync_uses_google_genai_method_and_config() -> None:
    capture: dict = {}
    client = _FakeClient(capture, SimpleNamespace(text="sync text"))

    text = google_genai.generate_text_sync(
        prompt="Prompt",
        system_prompt=None,
        model_name="gemini-sync",
        temperature=0.1,
        request_timeout=5,
        client=client,
    )

    assert text == "sync text"
    assert capture["model"] == "gemini-sync"
    assert capture["contents"] == "Prompt"
    assert capture["config"].temperature == 0.1
    assert capture["config"].http_options.retry_options.attempts == 1


@pytest.mark.parametrize("response", [SimpleNamespace(text=None), object()])
def test_extract_response_text_invalid_response_returns_none(response: object) -> None:
    assert google_genai.extract_response_text(response) is None


def test_embed_texts_sync_extracts_and_validates_3072_vectors() -> None:
    capture: dict = {}
    vector_a = [0.0] * 3072
    vector_b = [1.0] * 3072
    client = SimpleNamespace(
        models=_FakeSyncModels(
            capture,
            SimpleNamespace(
                embeddings=[SimpleNamespace(values=vector_a)],
            ),
        )
    )

    vectors = google_genai.embed_texts_sync(
        ["a", "b"],
        model_name="models/gemini-embedding-2",
        task_type="retrieval_document",
        expected_dimensions=3072,
        api_key="test-key",
        client=client,
    )

    assert len(vectors) == 2
    assert vectors[0] == vector_a
    assert vectors[1] == vector_a
    assert capture["model"] == "models/gemini-embedding-2"
    assert capture["contents"] == "b"


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        (SimpleNamespace(), "missing embeddings"),
        (SimpleNamespace(embeddings=[]), "returned no embeddings"),
        (SimpleNamespace(embeddings=[SimpleNamespace(values=None)]), "missing values"),
        (SimpleNamespace(embeddings=[SimpleNamespace(values=[])]), "empty values"),
        (SimpleNamespace(embeddings=[SimpleNamespace(values=[0.0, "bad"])]), "non-finite"),
        (SimpleNamespace(embeddings=[SimpleNamespace(values=[math.nan] * 3072)]), "non-finite"),
        (SimpleNamespace(embeddings=[SimpleNamespace(values=[math.inf] * 3072)]), "non-finite"),
        (SimpleNamespace(embeddings=[SimpleNamespace(values=[0.0] * 3)]), "dimension mismatch"),
        (
            SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[0.0] * 3072),
                    SimpleNamespace(values=[0.0] * 3072),
                ]
            ),
            "count mismatch",
        ),
    ],
)
def test_embedding_extractor_rejects_invalid_vectors(response: object, expected_error: str) -> None:
    with pytest.raises(ValueError, match=expected_error):
        vectors = google_genai.extract_embedding_vectors(response, expected_count=1)
        google_genai.validate_embedding_vectors(vectors, expected_dimensions=3072)


def test_missing_api_key_is_permanent_and_not_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(PermanentProviderError):
        google_genai.build_google_genai_client()


@pytest.mark.parametrize(
    ("code", "expected_type"),
    [
        (401, PermanentProviderError),
        (403, PermanentProviderError),
        (429, ProviderUnavailableError),
        (500, ProviderUnavailableError),
        (504, ProviderTimeoutError),
    ],
)
def test_google_genai_api_errors_are_normalized(code: int, expected_type: type[Exception]) -> None:
    exc = errors.APIError(code, {"error": {"message": "redacted"}})

    assert isinstance(google_genai.normalize_google_genai_exception(exc), expected_type)


@pytest.mark.asyncio
async def test_provider_gemini_path_uses_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    async def fake_generate_text_async(**kwargs: object) -> str:
        captured.update(kwargs)
        return "Gemini answer"

    monkeypatch.setattr(llm_provider, "generate_text_async", fake_generate_text_async)

    result = await llm_provider._call_gemini(
        "Prompt",
        "System",
        "gemini-test",
        0.2,
        request_timeout=7,
    )

    assert result == "Gemini answer"
    assert captured["prompt"] == "Prompt"
    assert captured["system_prompt"] == "System"
    assert captured["model_name"] == "gemini-test"
    assert captured["temperature"] == 0.2
    assert captured["request_timeout"] == 7


def test_runtime_source_has_no_legacy_google_sdk_references() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = [
        "google" + "." + "generativeai",
        "google" + "-" + "generativeai",
        "Generative" + "Model",
        "generate" + "_content_async",
        "genai" + "." + "configure",
        "genai" + "." + "embed_content",
    ]
    paths = [
        path
        for folder in ("src", "scripts", "tests")
        for path in (root / folder).rglob("*.py")
        if "__pycache__" not in path.parts
    ]

    matches: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                matches.append(f"{path.relative_to(root)} contains {marker}")

    assert matches == []


def test_google_genai_version_marker() -> None:
    assert google_genai.GOOGLE_GENAI_SDK_VERSION == "google_genai_sdk_v1"
