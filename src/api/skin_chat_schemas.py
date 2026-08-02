"""
src/api/skin_chat_schemas.py
=============================
Pydantic schemas for the /v1/skin-chat adapter endpoint.

Field names here intentionally mirror the JSON contract already implemented
by glow_aura's RagChatClient.cs (Group B integration) — session_id, message,
acne_data{total_acne,blackheads,whiteheads,pimples,severity},
user_profile{age,skin_type,is_pregnant,allergies,current_products} on the
way in; session_id, answer, recommendations, red_flags, safety_flags,
confidence, disclaimer on the way out.

Because the names already match, the .NET side needs ZERO code changes —
only appsettings.json's GroupB.ChatEndpoint needs to point at /v1/skin-chat
instead of /chat.
"""

from typing import Optional

from pydantic import BaseModel, Field


class SkinChatAcneData(BaseModel):
    total_acne: int = 0
    blackheads: int = 0
    whiteheads: int = 0
    pimples: int = 0
    severity: str = "none"


class SkinChatUserProfile(BaseModel):
    age: Optional[int] = None
    skin_type: str = "unknown"
    is_pregnant: bool = False
    allergies: list[str] = Field(default_factory=list)
    current_products: list[str] = Field(default_factory=list)


class SkinChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    acne_data: SkinChatAcneData
    user_profile: Optional[SkinChatUserProfile] = None


class SkinChatResponse(BaseModel):
    session_id: str
    answer: str
    recommendations: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    disclaimer: str = ""