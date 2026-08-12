"""
src/api/app.py
==============
FastAPI application for the Acne Advisor AI.
Exposes the LangGraph Agent via REST endpoints.
Includes chat history persistence to PostgreSQL.
"""

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

from src.agent.main import run_clinical_agent
from src.agent.source_presentation import build_source_metadata, display_names_for_sources
from src.agent.text_encoding import repair_mojibake
from src.observability.versioning import get_answer_cache_version
from src.resilience.exceptions import (
    AgentTimeoutError,
    CircuitOpenError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RetryExhaustedError,
    RuntimeResilienceError,
    StageTimeoutError,
)
from src.api.skin_chat_schemas import SkinChatRequest, SkinChatResponse
from src.api.skin_chat_helpers import (
    build_context_message,
    extract_recommendations,
    estimate_confidence,
    SKIN_CHAT_DISCLAIMER,
)

# Input Control Config
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", 500))
MAX_MESSAGE_WORDS = int(os.getenv("MAX_MESSAGE_WORDS", 120))
MAX_QUESTION_MARKS = int(os.getenv("MAX_QUESTION_MARKS", 3))
MAX_CONVERSATION_HISTORY_MESSAGES = int(os.getenv("MAX_CONVERSATION_HISTORY_MESSAGES", 10))
MAX_HISTORY_MESSAGE_CHARS = int(os.getenv("MAX_HISTORY_MESSAGE_CHARS", 1000))
CACHE_ANSWER_VERSION = get_answer_cache_version()

DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def parse_cors_origins(raw_value: str | None = None) -> list[str]:
    """Parse explicit CORS origins without falling back to unsafe wildcard."""

    raw = raw_value if raw_value is not None else os.getenv("CORS_ALLOW_ORIGINS", "")
    candidates = raw.split(",") if raw.strip() else list(DEFAULT_CORS_ORIGINS)
    origins: list[str] = []
    seen: set[str] = set()
    for origin in candidates:
        normalized = origin.strip().rstrip("/")
        if not normalized or normalized in seen:
            continue
        if normalized == "*":
            logger.warning("Ignoring wildcard CORS origin because credentials are enabled.")
            continue
        origins.append(normalized)
        seen.add(normalized)
    return origins or list(DEFAULT_CORS_ORIGINS)
RELEASE_READINESS_TEST_MODES = {"http_double", "deterministic"}

# In-memory lock for session requests
active_requests = set()


def _http_status_for_resilience_error(exc: RuntimeResilienceError) -> int:
    if isinstance(exc, (AgentTimeoutError, StageTimeoutError, ProviderTimeoutError)):
        return 504
    if isinstance(exc, (CircuitOpenError, ProviderUnavailableError, RetryExhaustedError)):
        return 503
    return 503


def _safe_resilience_detail(exc: RuntimeResilienceError) -> dict[str, Any]:
    code = getattr(exc, "error_code", "runtime_resilience_error")
    retryable = bool(getattr(exc, "retryable", True))
    if isinstance(exc, AgentTimeoutError):
        message = "Yêu cầu xử lý quá thời gian cho phép. Vui lòng thử lại sau ít phút."
    elif isinstance(exc, StageTimeoutError):
        message = "Một bước xử lý mất quá nhiều thời gian. Vui lòng thử lại sau."
    elif isinstance(exc, ProviderTimeoutError):
        message = "Dịch vụ tạo câu trả lời phản hồi quá chậm. Vui lòng thử lại."
    elif isinstance(exc, CircuitOpenError):
        message = "Dịch vụ tạo câu trả lời đang tạm ngưng do lỗi lặp lại. Vui lòng thử lại sau."
    else:
        message = "Dịch vụ tạo câu trả lời hiện chưa khả dụng. Vui lòng thử lại sau."
    return {
        "code": code,
        "message": message,
        "retryable": retryable,
        "error_type": exc.__class__.__name__,
    }


def _release_readiness_test_mode_enabled() -> bool:
    return os.getenv("RELEASE_READINESS_TEST_MODE", "").strip().lower() in RELEASE_READINESS_TEST_MODES


async def _run_release_readiness_agent_override(request: "ChatRequest", session_id: str) -> dict[str, Any]:
    """Deterministic HTTP-boundary doubles for release-readiness checks only."""

    if not _release_readiness_test_mode_enabled():
        raise RuntimeError("Release readiness test mode is not enabled.")

    message = request.message.strip()
    pipeline_manifest = {
        "phase": "phase2e",
        "answer_cache_version": CACHE_ANSWER_VERSION,
        "end_to_end_release_readiness_version": os.getenv(
            "END_TO_END_RELEASE_READINESS_VERSION",
            "end_to_end_release_readiness_v1",
        ),
    }

    if message == "__release_readiness_503__":
        raise ProviderUnavailableError("release readiness provider unavailable")
    if message == "__release_readiness_504__":
        raise AgentTimeoutError("release readiness timeout")

    fallback_applied = message == "__release_readiness_safe_fallback__"
    llm_fallback_used = message == "__release_readiness_fallback__"
    answer = (
        "Không có đủ bằng chứng đáng tin cậy trong kho tri thức để trả lời chắc chắn. "
        "Bạn nên hỏi bác sĩ da liễu nếu triệu chứng nặng lên."
        if fallback_applied
        else "Benzoyl peroxide không phải là kháng sinh. Đây là hoạt chất bôi trị mụn có tác dụng kháng khuẩn và hỗ trợ giảm bít tắc."
    )

    return {
        "answer": answer,
        "session_id": session_id,
        "sources": [] if fallback_applied else ["release_readiness_fixture"],
        "symptoms": [],
        "safety_flags": [],
        "graph_facts": [],
        "retrieval_status": "no_evidence" if fallback_applied else "release_readiness_double",
        "fallback_applied": fallback_applied,
        "fallback_type": "no_retrieval_evidence" if fallback_applied else "none",
        "fallback_reason": "release readiness no evidence" if fallback_applied else None,
        "fallback_cache_eligible": False if fallback_applied else True,
        "is_in_domain": True,
        "guardrail": "in_domain",
        "cache_checked": True,
        "cache_hit": False,
        "cache_reason": "safe_fallback_no_retrieval_evidence" if fallback_applied else "bypassed",
        "cache_metadata": {},
        "actual_provider": "ollama" if llm_fallback_used else ("system" if fallback_applied else "gemini"),
        "actual_model": os.getenv("OLLAMA_MODEL", "qwen3:8b") if llm_fallback_used else (None if fallback_applied else os.getenv("GOOGLE_MODEL", "gemini-3.5-flash")),
        "llm_fallback_used": llm_fallback_used,
        "fallback_provider": "ollama" if llm_fallback_used else None,
        "fallback_model": os.getenv("OLLAMA_MODEL", "qwen3:8b") if llm_fallback_used else None,
        "pipeline_fingerprint": "release_readiness_test_fingerprint",
        "pipeline_manifest": pipeline_manifest,
        "answer_quality_report": {"passed": True, "issues": []},
    }


def _repair_history_messages(messages: list["ChatHistoryMessage"]) -> list["ChatHistoryMessage"]:
    for msg in messages:
        msg.content = repair_mojibake(msg.content)
    return messages

# Initialize FastAPI app
app = FastAPI(
    title="Acne Advisor API",
    description="REST API for the Acne Advisor AI System",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---

class HealthResponse(BaseModel):
    status: str
    service: str
    postgres: Optional[str] = None
    qdrant: Optional[str] = None
    neo4j: Optional[str] = None
    redis: Optional[str] = None
    ollama: Optional[str] = None
    cache_enabled: Optional[bool] = None
    checks: Optional[dict[str, Any]] = None

class ChatHistoryMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_history: list[ChatHistoryMessage] = Field(default_factory=list)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    allow_model_fallback: bool = True
    bypass_cache: bool = False
    # Structured adapter payloads (e.g. /v1/skin-chat) embed catalog context and
    # are not end-user chat spam — skip the short free-text input caps.
    bypass_input_limits: bool = False

class ChatCacheMetadata(BaseModel):
    enabled: bool
    checked: bool
    hit: bool
    reason: Optional[str] = None
    similarity: Optional[float] = None
    answer_version: Optional[str] = None
    quality_checked: Optional[bool] = None
    quality_passed: Optional[bool] = None
    quality_reason: Optional[str] = None
    pipeline_fingerprint: Optional[str] = None

class ChatMetadata(BaseModel):
    provider: str
    model: Optional[str] = None
    requested_provider: Optional[str] = None
    requested_model: Optional[str] = None
    fallback_used: bool
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    fallback_chain: Optional[list[dict[str, Any]]] = None
    retrieval: str
    is_in_domain: Optional[bool] = None
    guardrail: Optional[str] = None
    ignored_out_of_domain_part: Optional[bool] = None
    used_retrieval: Optional[bool] = None
    domain_reason: Optional[str] = None
    llm_fallback: Optional[bool] = None
    fallback_reason: Optional[str] = None
    retrieval_status: Optional[str] = None
    fallback_applied: Optional[bool] = None
    fallback_type: Optional[str] = None
    fallback_cache_eligible: Optional[bool] = None
    cache: Optional[ChatCacheMetadata] = None
    cached_from_provider: Optional[str] = None
    cached_from_model: Optional[str] = None
    cached_at: Optional[str] = None
    phase2_debug: Optional[dict[str, Any]] = None
    response_origin: Optional[str] = None
    guardrail_applied: Optional[bool] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: Optional[str] = None
    sources: list[str] = Field(default_factory=list)
    source_metadata: list[dict[str, Any]] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    graph_facts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: ChatMetadata


def _chat_metadata_identity(
    result: dict[str, Any],
    request: ChatRequest,
    default_model: str,
) -> tuple[str, Optional[str]]:
    provider = result.get("actual_provider") or request.llm_provider or "gemini"
    if provider == "system" and result.get("actual_model") is None:
        return provider, None
    return provider, result.get("actual_model") or request.llm_model or default_model


def _response_origin(result: dict[str, Any], is_in_domain: Optional[bool]) -> str:
    if result.get("cache_hit"):
        return "cache"
    if _guardrail_applied(result, is_in_domain):
        return "guardrail"
    if result.get("fallback_applied"):
        return "safe_fallback"
    if result.get("actual_provider") == "system":
        return "deterministic"
    return "llm"


def _guardrail_applied(result: dict[str, Any], is_in_domain: Optional[bool]) -> bool:
    guardrail = result.get("guardrail")
    if is_in_domain is False:
        return True
    return bool(guardrail and guardrail not in {"in_domain", "in_domain_rule", "in_domain_followup_rule"})


class RetrieveResponse(BaseModel):
    query: str
    vector_contexts: list[dict[str, Any]] = Field(default_factory=list)
    graph_facts: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Chat History Models ---

class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    hidden: bool = False

class MessageResponse(BaseModel):
    id: Optional[str] = None
    role: str
    content: str
    sources: Optional[list] = None
    symptoms: Optional[list] = None
    safety_flags: Optional[list] = None
    graph_facts: Optional[list] = None
    metadata: Optional[dict] = None
    created_at: Optional[str] = None

class RenameRequest(BaseModel):
    title: str

class SyncMessagePayload(BaseModel):
    id: Optional[str] = None
    role: str
    content: str
    sources: Optional[list] = None
    symptoms: Optional[list] = None
    safety_flags: Optional[list] = None
    graph_facts: Optional[list] = None
    metadata: Optional[dict] = None
    created_at: Optional[float] = None  # JS timestamp (milliseconds)

class SyncSessionPayload(BaseModel):
    id: str
    title: str
    created_at: Optional[float] = None  # JS timestamp
    updated_at: Optional[float] = None
    hidden: bool = False
    messages: list[SyncMessagePayload] = Field(default_factory=list)

class SyncRequest(BaseModel):
    sessions: list[SyncSessionPayload]

class SyncResponse(BaseModel):
    synced: int
    skipped: int
    errors: int

class ClearChatHistoryResponse(BaseModel):
    ok: bool
    deleted_sessions: int
    deleted_messages: int
    deleted_redis_keys: int
    redis_key_patterns: list[str] = Field(default_factory=list)


# --- DB Helper ---

async def _get_db_session():
    """Get an async DB session. Returns None if DB is unavailable."""
    try:
        from src.database.connection import AsyncSessionLocal
        return AsyncSessionLocal()
    except Exception as e:
        logger.warning(f"Cannot create DB session: {e}")
        return None


async def _load_recent_history_from_db(session_id: str) -> list[dict[str, str]]:
    """Load recent chat history for a session when the client does not send it."""
    from src.database.repositories import chat_history as repo

    db_session = await _get_db_session()
    if db_session is None:
        return []

    try:
        async with db_session.begin():
            messages = await repo.get_messages(
                session=db_session,
                session_id=session_id,
                limit=MAX_CONVERSATION_HISTORY_MESSAGES,
            )
        return [
            {
                "role": str(msg.get("role", ""))[:20],
                "content": repair_mojibake(str(msg.get("content", "")))[:MAX_HISTORY_MESSAGE_CHARS],
            }
            for msg in messages
            if msg.get("role") in {"user", "assistant"} and msg.get("content")
        ]
    except Exception as exc:
        logger.warning("Could not load chat history from DB for session %s: %s", session_id, exc)
        return []
    finally:
        await db_session.close()


async def _delete_app_redis_cache_keys() -> tuple[int, list[str]]:
    """Delete only Acne Advisor answer-cache keys; never FLUSHALL."""
    from src.cache.redis_cache import get_redis

    patterns = ["cache:answer:*"]
    redis = await get_redis()
    if redis is None:
        return 0, patterns

    deleted = 0
    for pattern in patterns:
        cursor = 0
        while True:
            cursor, keys = await redis.scan(
                cursor=cursor,
                match=pattern,
                count=500,
            )
            if keys:
                deleted += int(await redis.delete(*keys))
            if cursor == 0:
                break
    return deleted, patterns


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if _release_readiness_test_mode_enabled() and os.getenv("RELEASE_READINESS_HEALTH_DOUBLE", "false").lower() == "true":
        checks = {
            "postgres": {"status": "ok"},
            "qdrant": {"status": "ok"},
            "neo4j": {"status": "ok"},
            "redis": {"status": "ok"},
            "ollama": {"status": "ok"},
        }
        return HealthResponse(
            status="ok",
            service="acne-advisor-api",
            postgres="ok",
            qdrant="ok",
            neo4j="ok",
            redis="ok",
            ollama="ok",
            cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            checks=checks,
        )

    from src.api.preflight import run_phase2_preflight

    preflight = await run_phase2_preflight()
    checks = preflight["checks"]
    cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    
    return HealthResponse(
        status=preflight["status"],
        service="acne-advisor-api",
        postgres=checks["postgres"]["status"],
        qdrant=checks["qdrant"]["status"],
        neo4j=checks["neo4j"]["status"],
        redis=checks["redis"]["status"],
        ollama=checks["ollama"]["status"],
        cache_enabled=cache_enabled,
        checks=checks,
    )


@app.get("/retrieve", response_model=RetrieveResponse)
async def retrieve_endpoint(q: str, top_k: int = 5):
    """Debug retrieval endpoint for Phase 2 Qdrant + Neo4j hybrid retrieval."""
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    top_k = max(1, min(top_k, 20))

    from src.database.retriever import HybridRetriever

    retriever = HybridRetriever()
    try:
        result = await retriever.retrieve(query=query, top_k=top_k)
        return RetrieveResponse(
            query=query,
            vector_contexts=result.vector_contexts,
            graph_facts=result.graph_facts,
            sources=result.sources,
            metadata=result.metadata,
        )
    except Exception as exc:
        logger.error("Retrieval endpoint failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Retrieval failed.")
    finally:
        await retriever.close()

@app.get("/models")
async def list_models():
    """List available LLM models."""
    from src.agent.llm.ollama_client import list_ollama_models
    from src.agent.llm.provider import DEFAULT_GEMINI_MODEL, parse_google_fallback_models
    
    ollama_models = await list_ollama_models()
    gemini_model = os.getenv("GOOGLE_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    if gemini_model == "gemini-1.5-flash":
        gemini_model = DEFAULT_GEMINI_MODEL
    gemini_fallback_models = parse_google_fallback_models(primary_model=gemini_model)
    qwen3_8b_available = "qwen3:8b" in ollama_models

    def model_entry(
        *,
        provider: str,
        model_id: str,
        display_name: str,
        model_type: str,
        available: bool,
        is_default: bool = False,
    ) -> dict[str, Any]:
        return {
            "provider": provider,
            "model": model_id,
            "model_id": model_id,
            "label": display_name,
            "display_name": display_name,
            "type": model_type,
            "available": available,
            "is_default": is_default,
        }
    
    models = [
        model_entry(
            provider="gemini",
            model_id=gemini_model,
            display_name=_display_name_for_model(gemini_model),
            model_type="cloud",
            available=True,
            is_default=True,
        ),
    ]
    models.extend(
        model_entry(
            provider="gemini",
            model_id=fallback_model,
            display_name=_display_name_for_model(fallback_model),
            model_type="cloud",
            available=True,
            is_default=False,
        )
        for fallback_model in gemini_fallback_models
    )
    models.append(
        model_entry(
            provider="ollama",
            model_id="qwen3:8b",
            display_name="Qwen3 8B Local",
            model_type="local",
            available=qwen3_8b_available,
        )
    )

    seen_model_keys: set[tuple[str, str]] = set()
    deduped_models: list[dict[str, Any]] = []
    for item in models:
        key = (item["provider"], item["model_id"])
        if key in seen_model_keys:
            continue
        seen_model_keys.add(key)
        deduped_models.append(item)

    return {
        "default_provider": "gemini",
        "default_model": gemini_model,
        "default_model_id": gemini_model,
        "models": deduped_models,
    }


def _display_name_for_model(model_id: str) -> str:
    aliases = {
        "gemini-3.5-flash": "Gemini 3.5 Flash",
        "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
        "gemini-2.5-flash": "Gemini 2.5 Flash",
    }
    return aliases.get(model_id, model_id)

@app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint to interact with the LangGraph Agent.
    After the agent responds, persists user message + assistant response
    to PostgreSQL (fire-and-forget: DB errors don't break the response).
    """
    request.message = repair_mojibake(request.message)
    request.conversation_history = _repair_history_messages(request.conversation_history)

    message_trimmed = request.message.strip() if request.message else ""
    if not message_trimmed:
        raise HTTPException(
            status_code=400,
            detail={"code": "empty_message", "message": "Câu hỏi không được để trống."}
        )

    if not request.bypass_input_limits:
        if len(message_trimmed) > MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=400,
                detail={"code": "message_too_long", "message": f"Câu hỏi của bạn hơi dài. Vui lòng rút gọn dưới {MAX_MESSAGE_CHARS} ký tự hoặc tách thành nhiều câu hỏi nhỏ."}
            )

        if len(message_trimmed.split()) > MAX_MESSAGE_WORDS:
            raise HTTPException(
                status_code=400,
                detail={"code": "too_many_words", "message": f"Câu hỏi của bạn chứa quá nhiều từ. Vui lòng rút gọn dưới {MAX_MESSAGE_WORDS} từ."}
            )

        if message_trimmed.count('?') > MAX_QUESTION_MARKS:
            raise HTTPException(
                status_code=400,
                detail={"code": "too_many_questions", "message": "Bạn đang hỏi quá nhiều ý cùng lúc. Vui lòng tách thành các câu hỏi riêng biệt để được tư vấn tốt nhất."}
            )
        
    # Determine session_id — use frontend's if provided, else generate one
    session_id = request.session_id or str(uuid.uuid4())[:12]
    
    # Cap conversation history. If client did not send it, load recent DB history
    # so same-session follow-ups work through API clients as well as the frontend.
    history = []
    if request.conversation_history:
        history = [
            {"role": msg.role, "content": msg.content[:MAX_HISTORY_MESSAGE_CHARS]}
            for msg in request.conversation_history[-MAX_CONVERSATION_HISTORY_MESSAGES:]
        ]
    elif request.session_id:
        history = await _load_recent_history_from_db(session_id)
    
    # Request locking to prevent concurrent processing for the same session
    if session_id in active_requests:
        raise HTTPException(
            status_code=409,
            detail={"code": "request_in_progress", "message": "Câu hỏi trước đang được xử lý. Vui lòng chờ hoàn tất rồi gửi câu tiếp theo."}
        )
        
    active_requests.add(session_id)
    
    try:
        logger.info(
            "Received chat request for session=%s, user_id_present=%s, message_chars=%d",
            session_id,
            bool(request.user_id),
            len(request.message),
        )
        if _release_readiness_test_mode_enabled():
            result = await _run_release_readiness_agent_override(request, session_id)
        else:
            result = await run_clinical_agent(
                message=request.message,
                user_id=request.user_id,
                session_id=session_id,
                conversation_history=history,
                llm_provider=request.llm_provider,
                llm_model=request.llm_model,
                allow_model_fallback=request.allow_model_fallback,
                bypass_cache=request.bypass_cache
            )
        
        # Check if the agent reported critical errors
        if result.get("errors"):
            logger.error(f"Agent errors: {result['errors']}")
            # We don't expose raw agent errors to the frontend for security reasons,
            # but we could choose to return a 500 or just a generic safe message.
            if not result.get("answer"):
                 raise Exception("Agent failed to produce an answer.")
        
        model_name = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash")
        if model_name == "gemini-1.5-flash":
            model_name = "gemini-3.5-flash"
            
        raw_graph_facts = result.get("graph_facts", [])
        
        # Determine prioritization keywords from user question and symptoms
        query_lower = request.message.lower()
        
        # Base keywords
        keywords = ["benzoyl peroxide", "bp", "topical retinoid", "acne", "mụn viêm", "trứng cá"]
        
        # Add blackhead-specific keywords if query mentions it
        if "mụn đầu đen" in query_lower or "comedone" in query_lower:
            keywords.extend(["mụn đầu đen", "comedone", "closed comedone", "open comedone", "skin care", "sữa rửa mặt", "dưỡng ẩm", "kem chống nắng", "salicylic acid", "bha"])
            
        symptoms = result.get("symptoms", [])
        keywords.extend([s.lower() for s in symptoms])
        
        # Add keywords present in query to prioritize them
        active_keywords = [kw for kw in keywords if kw in query_lower]
        if not active_keywords:
            active_keywords = keywords
            
        def score_fact(fact):
            score = 0
            text_to_search = (str(fact.get("entity", "")) + " " + 
                              str(fact.get("description", "")) + " " + 
                              str(fact.get("related_entity", ""))).lower()
            
            # Boost score if it contains active keywords
            for kw in active_keywords:
                if kw in text_to_search:
                    score += 2
                    
            # Penalize unrelated facts if blackhead is queried but antibiotics/hormones are returned
            if "mụn đầu đen" in query_lower and not any(k in query_lower for k in ["thuốc", "kháng sinh", "nội tiết", "antibiotic", "pill", "hormone"]):
                if any(k in text_to_search for k in ["ethinyl estradiol", "oral contraceptive", "pill", "antibiotic", "kháng sinh", "clindamycin"]):
                    score -= 5
                    
            return score
            
        sorted_facts = sorted(raw_graph_facts, key=score_fact, reverse=True)
        top_facts = sorted_facts[:10]
        
        def sanitize_dosage(text):
            if not text:
                return text
            # Remove dosage patterns like: 2.5%, 5%, 5 mg, 50 mcg, 2 lần/ngày, v.v...
            pattern = r'\s*\d+(\.\d+)?\s*(%|mg|mcg|g|lần/ngày|tuần/lần|viên|/ngày).*'
            return re.sub(pattern, '', text, flags=re.IGNORECASE).strip()

        safe_graph_facts = []
        for fact in top_facts:
            related_ent = sanitize_dosage(fact.get("related_entity", ""))
            
            # If after sanitization it becomes empty, skip it
            if not related_ent:
                continue
                
            safe_fact = {
                "entity": fact.get("entity"),
                "entity_type": fact.get("entity_type"),
                "relationship": fact.get("relationship"),
                "related_entity": related_ent,
                "related_type": fact.get("related_type"),
            }
            if fact.get("description"):
                safe_desc = sanitize_dosage(fact.get("description"))
                if safe_desc:
                    safe_fact["description"] = safe_desc
            safe_graph_facts.append(safe_fact)
            
        is_in_domain = result.get("is_in_domain")
        used_retrieval = is_in_domain is True
        retrieval_status = result.get("retrieval_status") or ("hybrid_qdrant_neo4j" if used_retrieval else "skipped")
        
        answer_text = repair_mojibake(result.get("answer", ""))
        raw_sources_list = result.get("sources", [])
        source_metadata = build_source_metadata(
            raw_sources_list,
            result.get("vector_contexts", []),
        )
        sources_list = display_names_for_sources(raw_sources_list, result.get("vector_contexts", []))
        symptoms_list = result.get("symptoms", [])
        safety_flags_list = result.get("safety_flags", [])
        answer_quality_report = result.get("answer_quality_report") or {}
        pipeline_fingerprint = result.get("pipeline_fingerprint")
        pipeline_manifest = result.get("pipeline_manifest") or {}
        phase2_debug_enabled = os.getenv("PHASE2_DEBUG_METADATA", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        phase2_debug = None
        if phase2_debug_enabled:
            phase2_debug = {
                "pipeline_fingerprint": pipeline_fingerprint,
                "pipeline_phase": pipeline_manifest.get("phase") if isinstance(pipeline_manifest, dict) else None,
                "observability_exported": result.get("observability_exported"),
                "runtime_resilience": result.get("runtime_resilience"),
                "answer_quality": {
                    "passed": answer_quality_report.get("passed") if isinstance(answer_quality_report, dict) else None,
                    "issue_count": len(answer_quality_report.get("issues", [])) if isinstance(answer_quality_report, dict) else 0,
                },
            }
        
        response_provider, response_model = _chat_metadata_identity(
            result,
            request,
            model_name,
        )
        response_origin = _response_origin(result, is_in_domain)
        guardrail_applied = _guardrail_applied(result, is_in_domain)

        # Build safe metadata dict for DB storage (no API keys, no raw exceptions)
        safe_db_metadata = {
            "provider": response_provider,
            "model": response_model,
            "requested_provider": result.get("requested_provider") or request.llm_provider or "gemini",
            "requested_model": result.get("requested_model") or request.llm_model or model_name,
            "fallback_used": result.get("llm_fallback_used", False),
            "fallback_provider": result.get("fallback_provider"),
            "fallback_model": result.get("fallback_model"),
            "fallback_reason": result.get("fallback_reason"),
            "fallback_chain": result.get("fallback_chain"),
            "retrieval": retrieval_status,
            "is_in_domain": is_in_domain,
            "guardrail": result.get("guardrail"),
            "response_origin": response_origin,
            "guardrail_applied": guardrail_applied,
            "source_metadata": source_metadata,
            "used_retrieval": used_retrieval,
            "pipeline_fingerprint": pipeline_fingerprint,
            "pipeline_manifest": {
                "phase": pipeline_manifest.get("phase") if isinstance(pipeline_manifest, dict) else None,
                "answer_cache_version": pipeline_manifest.get("answer_cache_version") if isinstance(pipeline_manifest, dict) else None,
                "rerank_provider": pipeline_manifest.get("rerank_provider") if isinstance(pipeline_manifest, dict) else None,
                "answer_guard_mode": pipeline_manifest.get("answer_guard_mode") if isinstance(pipeline_manifest, dict) else None,
            },
            "observability_exported": result.get("observability_exported"),
            "answer_quality": {
                "checked": bool(answer_quality_report),
                "passed": answer_quality_report.get("passed") if isinstance(answer_quality_report, dict) else None,
                "issue_count": len(answer_quality_report.get("issues", [])) if isinstance(answer_quality_report, dict) else 0,
                "critical_count": (
                    sum(
                        1
                        for issue in answer_quality_report.get("issues", [])
                        if isinstance(issue, dict) and issue.get("severity") == "critical"
                    )
                    if isinstance(answer_quality_report, dict)
                    else 0
                ),
                "guard_modified": result.get("answer_guard_modified"),
                "guard_mode": result.get("answer_guard_mode"),
            },
            "safe_fallback": {
                "retrieval_status": result.get("retrieval_status"),
                "fallback_applied": bool(result.get("fallback_applied")),
                "fallback_type": result.get("fallback_type"),
                "fallback_reason": result.get("fallback_reason"),
                "fallback_cache_eligible": result.get("fallback_cache_eligible"),
            },
            "cache": {
                "enabled": bool(result.get("cache_enabled", os.getenv("CACHE_ENABLED", "true").lower() == "true")),
                "checked": bool(result.get("cache_checked")),
                "hit": bool(result.get("cache_hit")),
                "reason": result.get("cache_reason") if result.get("cache_checked") or result.get("cache_reason") == "bypassed" else ("out_of_domain" if not is_in_domain else "skipped"),
                "answer_version": result.get("cache_metadata", {}).get("answer_version") if result.get("cache_hit") else CACHE_ANSWER_VERSION,
                "pipeline_fingerprint": result.get("cache_metadata", {}).get("pipeline_fingerprint") if result.get("cache_hit") else pipeline_fingerprint,
                "quality_checked": result.get("cache_metadata", {}).get("quality_checked") if result.get("cache_hit") else None,
                "quality_passed": result.get("cache_metadata", {}).get("quality_passed") if result.get("cache_hit") else None,
                "quality_reason": result.get("cache_metadata", {}).get("quality_reason") if result.get("cache_hit") else None
            },
            "runtime_resilience": result.get("runtime_resilience"),
        }
        
        # If cache hit, retrieve original model info
        cached_from_provider = None
        cached_from_model = None
        cached_at = None
        if result.get("cache_hit") and result.get("cache_metadata"):
            cached_from_provider = result["cache_metadata"].get("provider")
            cached_from_model = result["cache_metadata"].get("model")
            cached_at = result["cache_metadata"].get("created_at")
        
        # --- Fire-and-forget: persist to PostgreSQL ---
        if _release_readiness_test_mode_enabled():
            logger.info("Skipping DB persistence in release-readiness test mode.")
        else:
            try:
                logger.debug("Persisting chat exchange for session %s", session_id)
                await _persist_chat_to_db(
                    session_id=session_id,
                    user_id=request.user_id,
                    user_message=request.message,
                    assistant_answer=answer_text,
                    sources=sources_list,
                    symptoms=symptoms_list,
                    safety_flags=safety_flags_list,
                    graph_facts=safe_graph_facts,
                    db_metadata=safe_db_metadata,
                )
                logger.debug("Chat exchange persisted for session %s", session_id)
            except Exception as db_err:
                logger.warning(
                    "Failed to persist chat to DB for session %s (non-fatal): %s",
                    session_id,
                    db_err,
                )
        
        return ChatResponse(
            answer=answer_text,
            session_id=session_id,
            sources=sources_list,
            source_metadata=source_metadata,
            symptoms=symptoms_list,
            safety_flags=safety_flags_list,
            graph_facts=safe_graph_facts,
            metadata=ChatMetadata(
                provider=response_provider,
                model=response_model,
                requested_provider=result.get("requested_provider") or request.llm_provider or "gemini",
                requested_model=result.get("requested_model") or request.llm_model or model_name,
                fallback_used=result.get("llm_fallback_used", False),
                fallback_provider=result.get("fallback_provider"),
                fallback_model=result.get("fallback_model"),
                fallback_chain=result.get("fallback_chain"),
                retrieval=retrieval_status,
                is_in_domain=is_in_domain,
                guardrail=result.get("guardrail"),
                ignored_out_of_domain_part=result.get("ignored_out_of_domain_part"),
                used_retrieval=used_retrieval,
                domain_reason=result.get("domain_reason"),
                llm_fallback=result.get("llm_fallback"),
                fallback_reason=result.get("fallback_reason"),
                retrieval_status=result.get("retrieval_status"),
                fallback_applied=result.get("fallback_applied"),
                fallback_type=result.get("fallback_type"),
                fallback_cache_eligible=result.get("fallback_cache_eligible"),
                cache=ChatCacheMetadata(
                    enabled=bool(result.get("cache_enabled", os.getenv("CACHE_ENABLED", "true").lower() == "true")),
                    checked=bool(result.get("cache_checked")),
                    hit=bool(result.get("cache_hit")),
                    reason=result.get("cache_reason") if result.get("cache_checked") or result.get("cache_reason") == "bypassed" else ("out_of_domain" if not is_in_domain else "skipped"),
                    answer_version=result.get("cache_metadata", {}).get("answer_version") if result.get("cache_hit") else CACHE_ANSWER_VERSION,
                    pipeline_fingerprint=result.get("cache_metadata", {}).get("pipeline_fingerprint") if result.get("cache_hit") else pipeline_fingerprint,
                    quality_checked=result.get("cache_metadata", {}).get("quality_checked") if result.get("cache_hit") else None,
                    quality_passed=result.get("cache_metadata", {}).get("quality_passed") if result.get("cache_hit") else None,
                    quality_reason=result.get("cache_metadata", {}).get("quality_reason") if result.get("cache_hit") else None
                ),
                cached_from_provider=cached_from_provider,
                cached_from_model=cached_from_model,
                cached_at=cached_at,
                phase2_debug=phase2_debug,
                response_origin=response_origin,
                guardrail_applied=guardrail_applied,
            )
        )
        
    except asyncio.CancelledError:
        raise
    except RuntimeResilienceError as e:
        logger.warning("Runtime resilience error processing chat request: %s", e)
        raise HTTPException(
            status_code=_http_status_for_resilience_error(e),
            detail=_safe_resilience_detail(e),
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        # Return generic 500 error without leaking sensitive info
        raise HTTPException(status_code=500, detail="Internal server error processing the request.")
    finally:
        active_requests.discard(session_id)
@app.post("/v1/skin-chat", response_model=SkinChatResponse, response_model_exclude_none=True)
async def skin_chat_endpoint(request: SkinChatRequest):
    """
    Adapter cho glow_aura Group B integration (RagChatClient.cs).

    Nhận acne_data + user_profile, gộp thành 1 message giàu ngữ cảnh, rồi
    GỌI LẠI chat_endpoint() hiện có (không viết lại pipeline) để tái sử
    dụng nguyên vẹn retrieval, guardrail an toàn, cache và lưu lịch sử
    PostgreSQL. Cuối cùng chỉ map ChatResponse sang đúng shape mà client
    .NET đang chờ: recommendations / red_flags / confidence / disclaimer.
    """

    context_message = build_context_message(
        request.acne_data,
        request.user_profile,
        request.message,
        request.store_catalog,
    )

    inner_request = ChatRequest(
        message=context_message,
        session_id=request.session_id,
        allow_model_fallback=True,
        bypass_cache=False,
        bypass_input_limits=True,
    )

    chat_response = await chat_endpoint(inner_request)

    red_flags = list(chat_response.safety_flags)
    recommendations = extract_recommendations(chat_response.answer)
    confidence = estimate_confidence(
        is_in_domain=chat_response.metadata.is_in_domain,
        guardrail_applied=chat_response.metadata.guardrail_applied,
        fallback_used=chat_response.metadata.fallback_used,
        red_flags=red_flags,
    )

    return SkinChatResponse(
        session_id=chat_response.session_id or request.session_id or "",
        answer=chat_response.answer,
        recommendations=recommendations,
        red_flags=red_flags,
        safety_flags=chat_response.safety_flags,
        confidence=confidence,
        disclaimer=SKIN_CHAT_DISCLAIMER,
    )

async def _persist_chat_to_db(
    session_id: str,
    user_id: Optional[str],
    user_message: str,
    assistant_answer: str,
    sources: list,
    symptoms: list,
    safety_flags: list,
    graph_facts: list,
    db_metadata: dict,
):
    """
    Persist a chat exchange (user msg + assistant response) to PostgreSQL.
    This is called fire-and-forget from the chat endpoint.
    """
    from src.database.repositories import chat_history as repo
    
    db_session = await _get_db_session()
    if db_session is None:
        logger.warning("DB unavailable; skipping chat persistence for session %s.", session_id)
        return
    
    try:
        async with db_session.begin():
            # Create title from first 40 chars of user message
            title = user_message[:40] + ("..." if len(user_message) > 40 else "")
            
            # Upsert session
            await repo.create_or_update_session(
                session=db_session,
                session_id=session_id,
                title=title,
                user_id=user_id,
            )
            
            # Save user message
            await repo.save_message(
                session=db_session,
                session_id=session_id,
                role="user",
                content=user_message,
            )
            
            # Save assistant message
            await repo.save_message(
                session=db_session,
                session_id=session_id,
                role="assistant",
                content=assistant_answer,
                sources=sources,
                symptoms=symptoms,
                safety_flags=safety_flags,
                graph_facts=graph_facts,
                metadata=db_metadata,
            )
            
            # Touch session updated_at
            await repo.touch_session(session=db_session, session_id=session_id)
        
        logger.debug("Chat persisted to DB for session %s", session_id)
    except Exception as e:
        logger.warning("DB persistence error for session %s: %s", session_id, e)
        raise
    finally:
        await db_session.close()


# --- Chat History Endpoints ---

@app.get("/chat/sessions", response_model=list[SessionResponse])
async def get_chat_sessions(
    user_id: Optional[str] = None,
    include_hidden: bool = False,
):
    """Get all chat sessions, sorted by updated_at DESC."""
    from src.database.repositories import chat_history as repo
    
    db_session = await _get_db_session()
    if db_session is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    
    try:
        async with db_session.begin():
            sessions = await repo.get_sessions(
                session=db_session,
                user_id=user_id,
                include_hidden=include_hidden,
            )
        
        return [
            SessionResponse(
                id=s["id"],
                title=s["title"],
                created_at=s["created_at"].isoformat() if s.get("created_at") else "",
                updated_at=s["updated_at"].isoformat() if s.get("updated_at") else "",
                hidden=s.get("hidden", False),
            )
            for s in sessions
        ]
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch chat sessions.")
    finally:
        await db_session.close()


@app.delete("/chat/sessions", response_model=ClearChatHistoryResponse)
async def delete_all_chat_sessions():
    """
    Delete all persisted chat history and app-owned Redis answer cache.

    Safety:
    - Deletes chat_messages and chat_sessions rows only.
    - Does not drop tables.
    - Does not touch Qdrant, Neo4j, ingestion data, or Phase 1 stores.
    - Deletes Redis keys only under the app answer-cache prefix.
    """
    from src.database.repositories import chat_history as repo

    db_session = await _get_db_session()
    if db_session is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    try:
        async with db_session.begin():
            counts = await repo.delete_all_chat_history(session=db_session)

        deleted_redis_keys, patterns = await _delete_app_redis_cache_keys()

        return ClearChatHistoryResponse(
            ok=True,
            deleted_sessions=counts["deleted_sessions"],
            deleted_messages=counts["deleted_messages"],
            deleted_redis_keys=deleted_redis_keys,
            redis_key_patterns=patterns,
        )
    except Exception as exc:
        logger.error("Failed to delete chat history: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete chat history.")
    finally:
        await db_session.close()


@app.get("/chat/sessions/{session_id}/messages")
async def get_chat_messages(session_id: str, limit: int = 50):
    """Get messages for a specific chat session."""
    from src.database.repositories import chat_history as repo
    
    db_session = await _get_db_session()
    if db_session is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    
    try:
        async with db_session.begin():
            messages = await repo.get_messages(
                session=db_session,
                session_id=session_id,
                limit=limit,
            )
        
        return [
            MessageResponse(
                id=m.get("id"),
                role=m["role"],
                content=m["content"],
                sources=m.get("sources"),
                symptoms=m.get("symptoms"),
                safety_flags=m.get("safety_flags"),
                graph_facts=m.get("graph_facts"),
                metadata=m.get("metadata"),
                created_at=m["created_at"].isoformat() if m.get("created_at") else None,
            )
            for m in messages
        ]
    except Exception as e:
        logger.error(f"Error fetching messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch messages.")
    finally:
        await db_session.close()


@app.patch("/chat/sessions/{session_id}/rename")
async def rename_chat_session(session_id: str, body: RenameRequest):
    """Rename a chat session."""
    from src.database.repositories import chat_history as repo
    
    body.title = repair_mojibake(body.title)

    if not body.title or not body.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    
    db_session = await _get_db_session()
    if db_session is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    
    try:
        async with db_session.begin():
            updated = await repo.rename_session(
                session=db_session,
                session_id=session_id,
                title=body.title.strip(),
            )
        
        if not updated:
            raise HTTPException(status_code=404, detail="Session not found.")
        
        return {"status": "ok", "session_id": session_id, "title": body.title.strip()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming session: {e}")
        raise HTTPException(status_code=500, detail="Failed to rename session.")
    finally:
        await db_session.close()


@app.patch("/chat/sessions/{session_id}/hide")
async def hide_chat_session(session_id: str):
    """
    Hide a chat session by setting hidden=true.
    Does NOT delete any data from the database.
    """
    from src.database.repositories import chat_history as repo
    
    db_session = await _get_db_session()
    if db_session is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    
    try:
        async with db_session.begin():
            updated = await repo.hide_session(
                session=db_session,
                session_id=session_id,
            )
        
        if not updated:
            raise HTTPException(status_code=404, detail="Session not found.")
        
        return {"status": "ok", "session_id": session_id, "hidden": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error hiding session: {e}")
        raise HTTPException(status_code=500, detail="Failed to hide session.")
    finally:
        await db_session.close()


@app.post("/chat/sessions/sync", response_model=SyncResponse)
async def sync_sessions(body: SyncRequest):
    """
    Bulk import sessions + messages from localStorage to PostgreSQL.
    
    Safety:
    - Only imports sessions that don't already exist in DB.
    - For sessions that already exist, merges NEW messages (dedup by message ID).
    - Does NOT overwrite newer DB data with older localStorage data.
    - Does NOT delete localStorage on the client side.
    """
    from src.database.repositories import chat_history as repo
    
    db_session = await _get_db_session()
    if db_session is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")
    
    synced = 0
    skipped = 0
    errors = 0
    
    try:
        async with db_session.begin():
            for s_payload in body.sessions:
                try:
                    s_payload.title = repair_mojibake(s_payload.title)
                    exists = await repo.session_exists(
                        session=db_session,
                        session_id=s_payload.id,
                    )
                    
                    if not exists:
                        # New session — create it
                        await repo.create_or_update_session(
                            session=db_session,
                            session_id=s_payload.id,
                            title=s_payload.title,
                            hidden=s_payload.hidden,
                        )
                    
                    # Get existing message IDs to avoid duplicates
                    existing_msg_ids = await repo.get_message_ids_for_session(
                        session=db_session,
                        session_id=s_payload.id,
                    )
                    
                    for idx, msg in enumerate(s_payload.messages):
                        msg.content = repair_mojibake(msg.content)
                        # Generate deterministic message ID from session+index
                        # to avoid duplicates across multiple syncs
                        msg_id = msg.id or f"{s_payload.id}_msg_{idx}"
                        
                        if msg_id in existing_msg_ids:
                            continue  # Already exists, skip
                        
                        # Convert JS timestamp (ms) to datetime
                        created_at = None
                        if msg.created_at:
                            try:
                                created_at = datetime.fromtimestamp(
                                    msg.created_at / 1000, tz=timezone.utc
                                )
                            except (ValueError, OSError):
                                created_at = None
                        
                        await repo.save_message(
                            session=db_session,
                            session_id=s_payload.id,
                            role=msg.role,
                            content=msg.content,
                            message_id=msg_id,
                            sources=msg.sources,
                            symptoms=msg.symptoms,
                            safety_flags=msg.safety_flags,
                            graph_facts=msg.graph_facts,
                            metadata=msg.metadata,
                            created_at=created_at,
                        )
                    
                    synced += 1
                    
                except Exception as e:
                    logger.warning(f"Error syncing session {s_payload.id}: {e}")
                    errors += 1
        
        return SyncResponse(synced=synced, skipped=skipped, errors=errors)
    
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail="Sync failed.")
    finally:
        await db_session.close()
