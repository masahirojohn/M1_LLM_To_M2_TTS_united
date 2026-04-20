from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

from aituber_llm_tts.comment_ingest import (
    NormalizedComment,
    calc_comment_rate_per_min,
)
from aituber_llm_tts.priority_queue import BoundedPriorityCommentQueue
from aituber_llm_tts.priority_router import build_router_state, RouterState
from aituber_llm_tts.runtime_guard import RuntimeGuardConfig, validate_runtime_guard
from aituber_llm_tts.comment_store_jsonl import RuntimeStores


def calc_comment_rate_ratio_pct(
    *,
    comment_rate_per_min: int,
    baseline_comment_rate_per_min: float,
) -> Optional[float]:
    if baseline_comment_rate_per_min <= 0:
        return None
    return float(comment_rate_per_min) / float(baseline_comment_rate_per_min) * 100.0


class CommentRouter:
    def __init__(
        self,
        *,
        runtime_cfg: RuntimeGuardConfig,
        stores: RuntimeStores,
        baseline_comment_rate_per_min: float = 30.0,
        sleepy_threshold_pct_9_1: float = 20.0,
        sleepy_threshold_pct_9_2: float = 10.0,
    ) -> None:
        validate_runtime_guard(runtime_cfg)
        self.runtime_cfg = runtime_cfg
        self.stores = stores
        self.queue = BoundedPriorityCommentQueue(max_size=runtime_cfg.queue_max)

        self.baseline_comment_rate_per_min = float(baseline_comment_rate_per_min)
        self.sleepy_threshold_pct_9_1 = float(sleepy_threshold_pct_9_1)
        self.sleepy_threshold_pct_9_2 = float(sleepy_threshold_pct_9_2)

        # runtime state (最小)
        self.last_comment_ms: Optional[int] = None
        self.last_gift_ms: Optional[int] = None
        self.sleepy_start_ms: Optional[int] = None
        self.sleepy_duration_ms: int = 0

    def ingest_comments(
        self,
        comments: List[NormalizedComment],
        *,
        source: str,
    ) -> None:
        for c in comments:
            self.queue.push(c)
            self.stores.log_comment(source=source, comment=c)

            ts = int(c.timestamp)
            self.last_comment_ms = ts if self.last_comment_ms is None else max(self.last_comment_ms, ts)

            if getattr(c, "priority", None) in ("gift", "high_value_gift"):
                self.last_gift_ms = ts if self.last_gift_ms is None else max(self.last_gift_ms, ts)

    def build_state(self, comments: List[NormalizedComment]) -> RouterState:
        latest_ts = max((c.timestamp for c in comments), default=0)
        self.queue.prune_expired_normals(
            now_ts=latest_ts,
            ttl_ms=self.runtime_cfg.normal_ttl_ms,
        )
        selected = self.queue.select_for_llm(
            max_comments=self.runtime_cfg.llm_max_comments
        )
        rate = calc_comment_rate_per_min(comments)
        ratio_pct = calc_comment_rate_ratio_pct(
            comment_rate_per_min=rate,
            baseline_comment_rate_per_min=self.baseline_comment_rate_per_min,
        )

        state = build_router_state(
            selected_comments=selected,
            comment_rate_per_min=rate,
            comment_rate_ratio_pct=ratio_pct,
            sleepy_threshold_pct_9_1=self.sleepy_threshold_pct_9_1,
            sleepy_threshold_pct_9_2=self.sleepy_threshold_pct_9_2,
        )

        now_ts = int(latest_ts)
        sleepy_mode = bool(getattr(state, "sleepy_mode", False))

        if sleepy_mode:
            if self.sleepy_start_ms is None:
                self.sleepy_start_ms = now_ts
            self.sleepy_duration_ms = max(0, now_ts - self.sleepy_start_ms)
        else:
            self.sleepy_start_ms = None
            self.sleepy_duration_ms = 0

        setattr(state, "last_comment_ms", self.last_comment_ms)
        setattr(state, "last_gift_ms", self.last_gift_ms)
        setattr(state, "sleepy_start_ms", self.sleepy_start_ms)
        setattr(state, "sleepy_duration_ms", self.sleepy_duration_ms)

        return state

    def log_llm_payload(
        self,
        *,
        session_id: str,
        utt_id: str,
        state: RouterState,
        source: str,
    ) -> None:
        self.stores.log_llm_request({
            "session_id": session_id,
            "utt_id": utt_id,
            "source": source,
            "router_state": asdict(state),
            "runtime_state": {
                "last_comment_ms": getattr(state, "last_comment_ms", None),
                "last_gift_ms": getattr(state, "last_gift_ms", None),
                "sleepy_start_ms": getattr(state, "sleepy_start_ms", None),
                "sleepy_duration_ms": getattr(state, "sleepy_duration_ms", 0),
            },
        })

    def log_events(
        self,
        *,
        session_id: str,
        utt_id: str,
        events: list[dict],
    ) -> None:
        self.stores.log_event_batch(
            session_id=session_id,
            utt_id=utt_id,
            events=events,
        )