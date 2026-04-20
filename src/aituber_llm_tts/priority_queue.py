from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Optional

from aituber_llm_tts.comment_ingest import NormalizedComment


_PRIORITY_RANK = {
    "admin": 0,
    "high_value_gift": 1,
    "gift": 2,
    "normal": 3,
}


@dataclass(order=True)
class _QueueItem:
    sort_key: tuple = field(init=False, repr=False)
    priority_rank: int
    timestamp: int
    seq: int
    comment: NormalizedComment = field(compare=False)

    def __post_init__(self) -> None:
        # heapq は最小値優先。
        # 捨てる候補を探しやすいよう、弱いものほど後で見つけやすい形ではなく、
        # 単純保持用として使い、drop側は明示探索する。
        self.sort_key = (self.priority_rank, -self.timestamp, self.seq)


class BoundedPriorityCommentQueue:
    def __init__(self, max_size: int = 100) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be > 0")
        self.max_size = int(max_size)
        self._items: List[_QueueItem] = []
        self._seq = 0

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[NormalizedComment]:
        # 強い順 + 新しい順
        return [
            x.comment
            for x in sorted(
                self._items,
                key=lambda it: (it.priority_rank, -it.timestamp, it.seq),
            )
        ]

    def push(self, comment: NormalizedComment) -> bool:
        item = _QueueItem(
            priority_rank=_PRIORITY_RANK.get(comment.priority, 99),
            timestamp=int(comment.timestamp),
            seq=self._seq,
            comment=comment,
        )
        self._seq += 1

        if len(self._items) < self.max_size:
            heapq.heappush(self._items, item)
            return True

        drop_idx = self._find_drop_candidate_index(new_item=item)
        if drop_idx is None:
            return False

        self._items[drop_idx] = item
        heapq.heapify(self._items)
        return True

    def prune_expired_normals(self, now_ts: int, ttl_ms: int) -> int:
        kept: List[_QueueItem] = []
        removed = 0
        for it in self._items:
            is_expired_normal = (
                it.comment.priority == "normal"
                and int(now_ts) - int(it.comment.timestamp) > int(ttl_ms)
            )
            if is_expired_normal:
                removed += 1
            else:
                kept.append(it)
        if removed:
            self._items = kept
            heapq.heapify(self._items)
        return removed

    def select_for_llm(self, max_comments: int = 5) -> List[NormalizedComment]:
        return self.items()[:max_comments]

    def _find_drop_candidate_index(self, new_item: _QueueItem) -> Optional[int]:
        # まず normal の最古を捨てる
        normal_candidates = [
            (idx, it) for idx, it in enumerate(self._items)
            if it.comment.priority == "normal"
        ]
        if normal_candidates:
            return min(normal_candidates, key=lambda x: x[1].timestamp)[0]

        # 次に gift の最古
        gift_candidates = [
            (idx, it) for idx, it in enumerate(self._items)
            if it.comment.priority == "gift"
        ]
        if gift_candidates and new_item.priority_rank <= _PRIORITY_RANK["gift"]:
            return min(gift_candidates, key=lambda x: x[1].timestamp)[0]

        # high_value_gift を admin で押し出すのは許容
        hvg_candidates = [
            (idx, it) for idx, it in enumerate(self._items)
            if it.comment.priority == "high_value_gift"
        ]
        if hvg_candidates and new_item.priority_rank == _PRIORITY_RANK["admin"]:
            return min(hvg_candidates, key=lambda x: x[1].timestamp)[0]

        # admin 同士は古い admin を押し出してよい
        admin_candidates = [
            (idx, it) for idx, it in enumerate(self._items)
            if it.comment.priority == "admin"
        ]
        if admin_candidates and new_item.priority_rank == _PRIORITY_RANK["admin"]:
            return min(admin_candidates, key=lambda x: x[1].timestamp)[0]

        return None