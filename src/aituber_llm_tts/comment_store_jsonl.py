from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


class JsonlStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, obj: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


class RuntimeStores:
    def __init__(self, out_root: str | Path) -> None:
        out_root = Path(out_root)
        runtime_dir = out_root / "runtime"
        self.comment_log = JsonlStore(runtime_dir / "comment_log.jsonl")
        self.llm_requests = JsonlStore(runtime_dir / "llm_requests.jsonl")
        self.event_log = JsonlStore(runtime_dir / "event_log.jsonl")

    def log_comment(self, *, source: str, comment: Any) -> None:
        self.comment_log.append({
            "kind": "comment",
            "source": source,
            "record": _to_jsonable(comment),
        })

    def log_llm_request(self, payload: Dict[str, Any]) -> None:
        self.llm_requests.append({
            "kind": "llm_request",
            **_to_jsonable(payload),
        })

    def log_event_batch(self, *, session_id: str, utt_id: str, events: list[dict]) -> None:
        self.event_log.append({
            "kind": "events",
            "session_id": session_id,
            "utt_id": utt_id,
            "events": _to_jsonable(events),
        })