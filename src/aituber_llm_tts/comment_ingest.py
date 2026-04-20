from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


ADMIN_USER_ID = "ADMIN_ID"
HIGH_VALUE_GIFT_THRESHOLD = 3000


@dataclass
class NormalizedComment:
    service: str
    name: str
    user_id: str
    comment: str
    timestamp: int
    has_gift: bool = False
    gift_value: int = 0
    priority: str = "normal"

    @property
    def is_admin(self) -> bool:
        return self.user_id == ADMIN_USER_ID

    @property
    def is_high_value_gift(self) -> bool:
        return self.has_gift and self.gift_value >= HIGH_VALUE_GIFT_THRESHOLD


def normalize_comment_record(raw: dict) -> NormalizedComment:
    service = str(raw.get("service", "unknown"))
    name = str(raw.get("name", ""))
    user_id = str(raw.get("user_id", ""))
    comment = str(raw.get("comment", ""))
    timestamp = int(raw.get("timestamp", 0))

    has_gift = bool(raw.get("has_gift", False))
    gift_value = int(raw.get("gift_value", 0))

    priority = "normal"
    if user_id == ADMIN_USER_ID:
        priority = "admin"
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


def _coerce_comment_items(raw: object) -> list[dict]:
    """
    comment JSON のルート揺れを吸収する。
    - 標準: list[dict]
    - 互換: {"comments": [...]}
    """
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]

    if isinstance(raw, dict):
        maybe = raw.get("comments", [])
        if isinstance(maybe, list):
            return [x for x in maybe if isinstance(x, dict)]

    return []


def load_comments_from_any_json(path: str | Path) -> List[NormalizedComment]:
    """
    JSONルートが配列直下でも {"comments": [...]} でも読める共通loader。
    dummy_json / onecomme_json の両方から使える最小共通層。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    raw_comments = _coerce_comment_items(raw)
    comments = [normalize_comment_record(x) for x in raw_comments]
    comments.sort(key=lambda x: x.timestamp)
    return comments


def load_comments(path: str | Path) -> List[NormalizedComment]:
    """
    既存互換API。
    内部的には揺れ吸収版 loader を使う。
    """
    return load_comments_from_any_json(path)


def calc_comment_rate_per_min(comments: List[NormalizedComment]) -> int:
    if not comments:
        return 0
    latest_ts = comments[-1].timestamp
    one_min_ago = latest_ts - 60_000
    return sum(1 for c in comments if c.timestamp >= one_min_ago)


def select_priority_comments(
    comments: List[NormalizedComment],
    max_comments: int = 5,
) -> List[NormalizedComment]:
    priority_order = {
        "admin": 0,
        "high_value_gift": 1,
        "gift": 2,
        "normal": 3,
    }
    sorted_comments = sorted(
        comments,
        key=lambda c: (priority_order.get(c.priority, 99), -c.timestamp),
    )
    return sorted_comments[:max_comments]