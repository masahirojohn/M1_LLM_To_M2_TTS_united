from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from aituber_llm_tts.comment_ingest import (
    ADMIN_USER_ID,
    HIGH_VALUE_GIFT_THRESHOLD,
    NormalizedComment,
)
from .base import CommentSource, ensure_path


def _coerce_items(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        maybe = raw.get("comments", [])
        if isinstance(maybe, list):
            return [x for x in maybe if isinstance(x, dict)]
    return []


def _extract_gift_value(raw: Dict[str, Any]) -> int:
    if "gift_value" in raw:
        try:
            return int(raw.get("gift_value", 0) or 0)
        except Exception:
            return 0

    gift = raw.get("gift")
    if isinstance(gift, dict):
        for key in ("price", "amount", "value", "gift_value"):
            if key in gift:
                try:
                    return int(gift.get(key, 0) or 0)
                except Exception:
                    return 0

    return 0


def _normalize_onecomme_record(raw: Dict[str, Any]) -> NormalizedComment:
    service = str(raw.get("service", "unknown") or "unknown")
    name = str(raw.get("name", "") or "")
    user_id = str(raw.get("user_id", "") or "")
    comment = str(raw.get("comment", "") or "")
    timestamp = int(raw.get("timestamp", 0) or 0)

    has_gift = bool(raw.get("has_gift", False))
    gift_value = _extract_gift_value(raw)

    is_owner = bool(raw.get("is_owner", False))
    is_mod = bool(raw.get("is_mod", False))

    priority = "normal"
    if user_id == ADMIN_USER_ID or is_owner or is_mod:
        priority = "admin"
        if not user_id:
            user_id = "ONECOMME_ADMIN"
    elif has_gift and gift_value >= HIGH_VALUE_GIFT_THRESHOLD:
        priority = "high_value_gift"
    elif has_gift:
        priority = "gift"

    return NormalizedComment(
        service=service,
        name=name,
        user_id=user_id,
        comment=comment,
        timestamp=timestamp,
        has_gift=has_gift,
        gift_value=gift_value,
        priority=priority,
    )


class OneCommeJsonCommentSource(CommentSource):
    def __init__(self, path: str | Path) -> None:
        self.path = ensure_path(path)

    def load_comments(self) -> List[NormalizedComment]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        items = _coerce_items(raw)
        comments = [_normalize_onecomme_record(x) for x in items]
        comments.sort(key=lambda x: x.timestamp)
        return comments