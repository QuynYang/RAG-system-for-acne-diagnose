"""
src/api/skin_chat_helpers.py
==============================
Pure helper functions used by the /v1/skin-chat adapter endpoint.

Deliberately has no FastAPI/DB imports so it stays trivial to unit test
with plain pytest (no app/client fixtures needed).
"""

import re
from typing import Optional

from src.api.skin_chat_schemas import SkinChatAcneData, SkinChatUserProfile

SKIN_CHAT_DISCLAIMER = (
    "Thông tin trên chỉ mang tính tham khảo, không phải chẩn đoán y khoa. "
    "Vui lòng gặp bác sĩ da liễu nếu tình trạng nặng, kéo dài hoặc có dấu hiệu bất thường."
)

_SEVERITY_VI = {
    "none": "không có mụn đáng kể",
    "mild": "nhẹ",
    "moderate": "trung bình",
    "severe": "nặng",
}


def build_context_message(
    acne_data: SkinChatAcneData,
    user_profile: Optional[SkinChatUserProfile],
    user_message: Optional[str] = None,
    store_catalog: Optional[list[str]] = None,
) -> str:
    """Fold structured acne-detection + profile data into one free-text
    Vietnamese message, because the underlying /chat pipeline only accepts
    a `message` string — there is no structured patient_profile input on
    that endpoint today. This is the single most important piece of this
    adapter: without it, the RAG never sees the acne_data/user_profile at
    all (that was the original bug — see the mismatch you asked about)."""

    severity_vi = _SEVERITY_VI.get(acne_data.severity, acne_data.severity)

    parts = [
        "Phân tích tình trạng da dựa trên kết quả chụp ảnh sau và đưa ra lời khuyên chăm sóc:",
        f"- Tổng số nốt mụn phát hiện: {acne_data.total_acne}",
        f"- Mụn đầu đen: {acne_data.blackheads}",
        f"- Mụn đầu trắng: {acne_data.whiteheads}",
        f"- Mụn viêm/mụn mủ: {acne_data.pimples}",
        f"- Mức độ: {severity_vi}",
    ]

    if user_profile:
        if user_profile.age is not None:
            parts.append(f"- Tuổi: {user_profile.age}")
        if user_profile.skin_type and user_profile.skin_type != "unknown":
            parts.append(f"- Loại da: {user_profile.skin_type}")
        if user_profile.is_pregnant:
            parts.append("- Đang mang thai hoặc có khả năng mang thai: có")
        if user_profile.allergies:
            parts.append(f"- Dị ứng đã biết: {', '.join(user_profile.allergies)}")
        if user_profile.current_products:
            parts.append(f"- Sản phẩm đang dùng: {', '.join(user_profile.current_products)}")

    if store_catalog:
        parts.append("")
        parts.append(
            "Danh mục sản phẩm OTC có bán tại cửa hàng Glow Aura "
            "(ưu tiên gợi ý trong phạm vi này khi phù hợp, không bắt buộc kê thuốc kê đơn):"
        )
        for line in store_catalog[:15]:
            cleaned = line.strip()
            if cleaned:
                parts.append(f"  • {cleaned}")

    if user_message and user_message.strip():
        # Ignore the hardcoded default message the C# client currently sends
        # ("Phân tích tình trạng da dựa trên kết quả chụp ảnh...") so it does
        # not duplicate the line above; only append genuinely different text.
        cleaned = user_message.strip()
        if cleaned not in parts[0]:
            parts.append(f"- Ghi chú thêm từ người dùng: {cleaned}")

    return "\n".join(parts)


_BULLET_LINE = re.compile(
    r"^\s*(?:[-*•]|(?:bước|step)\s*\d+[:.]?|\d+[.)])\s+(.*\S)\s*$",
    re.IGNORECASE,
)


def extract_recommendations(answer_text: str, limit: int = 8) -> list[str]:
    """Pull ordered/bulleted list items out of the generated answer so the
    mobile UI (which renders `recommendations` as a step list) has content.

    Falls back to an empty list rather than guessing when the answer has no
    list structure — an empty UI section is safer than a fabricated
    recommendation that was never actually in the model's answer."""

    items: list[str] = []
    for raw_line in answer_text.splitlines():
        match = _BULLET_LINE.match(raw_line)
        if not match:
            continue
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", match.group(1)).strip()
        if text:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def estimate_confidence(
    is_in_domain: Optional[bool],
    guardrail_applied: Optional[bool],
    fallback_used: bool,
    red_flags: list[str],
) -> float:
    """Heuristic 0-1 confidence score derived from pipeline signals.

    Gemini does not return a calibrated probability, so this is a proxy
    built from signals the pipeline already exposes — NOT a statistically
    calibrated confidence. Treat it as a UI hint (e.g. show a "low
    confidence, please double check" badge below ~0.5), not as a precise
    accuracy metric."""

    if is_in_domain is False:
        return 0.35  # Question fell outside the acne knowledge base.

    score = 0.9
    if fallback_used:
        score -= 0.15  # Primary model failed; fallback provider was used.
    if guardrail_applied:
        score -= 0.1  # A deterministic safety template overrode the raw answer.
    if red_flags:
        score -= 0.15  # Encourage the user to verify with a professional.

    return round(max(0.3, min(score, 0.95)), 2)