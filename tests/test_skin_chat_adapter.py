from src.api.skin_chat_helpers import build_context_message
from src.api.skin_chat_schemas import SkinChatAcneData, SkinChatUserProfile


def test_build_context_message_exceeds_public_chat_char_limit_with_catalog():
    acne = SkinChatAcneData(
        total_acne=5,
        blackheads=2,
        whiteheads=1,
        pimples=2,
        severity="mild",
    )
    profile = SkinChatUserProfile(age=25, skin_type="acne")
    catalog = [
        "La Roche-Posay Effaclar Duo (+) (La Roche-Posay) — Kem dưỡng — phù hợp da Mụn — thành phần: Niacinamide",
        "CeraVe Foaming Facial Cleanser (CeraVe) — Sữa rửa mặt — phù hợp da Dầu — thành phần: Ceramides",
    ]

    message = build_context_message(acne, profile, store_catalog=catalog)

    assert len(message) > 500
