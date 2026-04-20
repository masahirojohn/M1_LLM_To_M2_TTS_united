#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_STEP_MS = 40


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _calc_end_t_ms(audio_ms: int, step_ms: int) -> int:
    if audio_ms <= 0:
        raise ValueError("audio_ms must be > 0")
    if step_ms <= 0:
        raise ValueError("step_ms must be > 0")
    return ((audio_ms - 1) // step_ms) * step_ms


def _ensure_events_list(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = obj.get("events")
    if events is None:
        obj["events"] = []
        return obj["events"]

    if not isinstance(events, list):
        raise ValueError("input json field 'events' must be a list")

    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            raise ValueError(f"events[{i}] is not an object")
    return events


def _remove_existing_event_by_id(events: List[Dict[str, Any]], event_id: str) -> int:
    before = len(events)
    events[:] = [ev for ev in events if str(ev.get("event_id")) != str(event_id)]
    return before - len(events)


def add_auto_end_stream(
    *,
    in_json: Path,
    out_json: Path,
    audio_ms: int,
    step_ms: int = DEFAULT_STEP_MS,
    event_id: str = "sleepy_auto_end",
    trigger_source: str = "auto_sleepy_timeout",
    force_replace_same_event_id: bool = True,
) -> Dict[str, Any]:
    obj = _load_json(in_json)
    if not isinstance(obj, dict):
        raise ValueError("input unified json must be an object")

    events = _ensure_events_list(obj)
    removed = 0
    if force_replace_same_event_id:
        removed = _remove_existing_event_by_id(events, event_id)

    end_t_ms = _calc_end_t_ms(audio_ms, step_ms)

    end_event = {
        "type": "event",
        "event_id": event_id,
        "t_ms": int(end_t_ms),
        "duration_ms": 0,
        "event_mode": "end_stream",
        "trigger_source": trigger_source,
    }

    events.append(end_event)
    _dump_json(out_json, obj)

    stats = {
        "in_json": str(in_json),
        "out_json": str(out_json),
        "audio_ms": int(audio_ms),
        "step_ms": int(step_ms),
        "end_t_ms": int(end_t_ms),
        "event_id": event_id,
        "trigger_source": trigger_source,
        "removed_existing_same_event_id": int(removed),
        "events_n_after": int(len(events)),
    }
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Input m1_unified_output json")
    ap.add_argument("--out_json", required=True, help="Output json with end_stream appended")
    ap.add_argument("--audio_ms", required=True, type=int, help="SSOT audio length in ms")
    ap.add_argument("--step_ms", type=int, default=DEFAULT_STEP_MS, help="Grid step in ms (default: 40)")
    ap.add_argument("--event_id", default="sleepy_auto_end", help="end_stream event_id")
    ap.add_argument("--trigger_source", default="auto_sleepy_timeout", help="trigger_source value")
    ap.add_argument(
        "--no_force_replace_same_event_id",
        action="store_true",
        help="If set, do not remove existing events with the same event_id before append",
    )
    args = ap.parse_args()

    in_json = Path(args.in_json).resolve()
    out_json = Path(args.out_json).resolve()

    if not in_json.exists():
        raise FileNotFoundError(f"missing in_json: {in_json}")

    stats = add_auto_end_stream(
        in_json=in_json,
        out_json=out_json,
        audio_ms=int(args.audio_ms),
        step_ms=int(args.step_ms),
        event_id=str(args.event_id),
        trigger_source=str(args.trigger_source),
        force_replace_same_event_id=not bool(args.no_force_replace_same_event_id),
    )

    print("[add_auto_end_stream][OK]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
