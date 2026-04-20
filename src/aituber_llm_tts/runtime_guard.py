from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeGuardConfig:
    queue_max: int = 100
    llm_max_comments: int = 5
    normal_ttl_ms: int = 30_000


def validate_runtime_guard(cfg: RuntimeGuardConfig) -> None:
    if cfg.queue_max <= 0:
        raise ValueError("queue_max must be > 0")
    if cfg.llm_max_comments <= 0:
        raise ValueError("llm_max_comments must be > 0")
    if cfg.normal_ttl_ms < 0:
        raise ValueError("normal_ttl_ms must be >= 0")