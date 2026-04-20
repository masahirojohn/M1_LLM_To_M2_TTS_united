#!/usr/bin/env python3
import argparse
import json
import math
from copy import deepcopy
from pathlib import Path


DEFAULT_BLINK_INTERVAL_MS = 10000
SLEEPY_BLINK_MS = {
    "9_1": 8000,
    "9_2": 5000,
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def save_json(obj, path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def infer_emo_id(unified: dict, fallback: str | None) -> str:
    if fallback:
        return fallback

    speech = unified.get("speech", {})
    emo_id = speech.get("emo_id")
    if emo_id:
        return emo_id

    return "normal"



def build_expression_chunks(audio_ms: int, emo_id: str, step_ms: int):
    blink_interval_ms = SLEEPY_BLINK_MS.get(emo_id, DEFAULT_BLINK_INTERVAL_MS)

    chunks = []
    events = []

    t_ms = 0
    while t_ms < audio_ms:
        events.append({
            "t_ms": t_ms,
            "expression": "blink"
        })

        restore_t = t_ms + step_ms
        if restore_t < audio_ms:
            events.append({
                "t_ms": restore_t,
                "expression": "sad" if emo_id in ("9_1", "9_2") else "neutral"
            })

        t_ms += blink_interval_ms

    chunks.append({
        "chunk_start_ms": 0,
        "events": events
    })

    return {
        "schema": "expression_chunks.v1",
        "step_ms": step_ms,
        "chunks": chunks,
        "meta": {
            "emo_id": emo_id,
            "blink_interval_ms": blink_interval_ms,
        }
    }



def ensure_end_stream(unified: dict, audio_ms: int, step_ms: int):
    out = deepcopy(unified)
    events = out.setdefault("events", [])

    if any(
        e.get("event_mode") == "end_stream"
        for e in events
    ):
        return out

    end_t_ms = math.floor(audio_ms / step_ms) * step_ms

    events.append({
        "type": "event",
        "event_id": "sleepy_auto_end",
        "t_ms": end_t_ms,
        "duration_ms": 0,
        "event_mode": "end_stream",
        "trigger_source": "auto_sleepy_timeout",
    })

    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified_json", required=True)
    ap.add_argument("--audio_ms", type=int, required=True)
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--emo_id", default=None)
    ap.add_argument("--out_expr_chunks", required=True)
    ap.add_argument("--out_unified_json", required=True)
    args = ap.parse_args()

    unified = load_json(args.unified_json)
    emo_id = infer_emo_id(unified, args.emo_id)

    expr_chunks = build_expression_chunks(
        audio_ms=args.audio_ms,
        emo_id=emo_id,
        step_ms=args.step_ms,
    )

    unified_out = ensure_end_stream(
        unified=unified,
        audio_ms=args.audio_ms,
        step_ms=args.step_ms,
    )

    save_json(expr_chunks, args.out_expr_chunks)
    save_json(unified_out, args.out_unified_json)

    print("[build_sleepy_scenario_from_unified][OK]")
    print("  emo_id:", emo_id)
    print("  blink_interval_ms:", expr_chunks["meta"]["blink_interval_ms"])
    print("  expr_events:", len(expr_chunks["chunks"][0]["events"]))
    print("  out_expr_chunks:", args.out_expr_chunks)
    print("  out_unified_json:", args.out_unified_json)


if __name__ == "__main__":
    main()
