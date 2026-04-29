#!/usr/bin/env python3
from __future__ import annotations

"""
audio_tool_calls_to_expr_smoke.py

目的:
- audio_stream_bridge.py が保存した live_tool_calls.json から
  expression timeline を生成する単体スモーク。

入力:
- live_tool_calls.json

出力:
- expr.json
- audio_tool_calls_to_expr_smoke.summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Any

from m3p.live.expression_streamer import ExpressionStreamer, ExpressionStreamerConfig


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _default_emo_map() -> dict[str, str]:
    return {
        "1_0": "normal",
        "1_1": "smile",
        "1_2": "sad",
        "2_0": "angry",
        "9_1": "sad",
        "9_2": "sad",
    }


def _parse_emo_id(call: dict[str, Any]) -> str | None:
    args = call.get("args", {})
    if not isinstance(args, dict):
        return None

    emo_id = args.get("emo_id")
    if emo_id is None:
        return None

    s = str(emo_id).strip()
    return s if s else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="live_tool_calls.json -> expr.json smoke"
    )

    ap.add_argument("--session_id", default="sess_audio_tool_calls_to_expr_001")
    ap.add_argument("--tool_calls_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--default_expression", default="normal")
    ap.add_argument("--unknown_emo_to", default="normal")
    ap.add_argument("--emo_map_json", default=None)
    # ① CLI引数追加（audio_ms受け取り）
    ap.add_argument("--audio_meta_json", default=None)

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tool_calls_path = Path(args.tool_calls_json).resolve()
    tool_calls = _load_json(tool_calls_path)
    if not isinstance(tool_calls, list):
        raise RuntimeError("tool_calls_json must be a list")

    # ② audio_ms 読み込み
    audio_ms = None
    if args.audio_meta_json:
        meta = _load_json(Path(args.audio_meta_json).resolve())
        audio_ms = int(meta.get("audio_ms", 0) or 0)

    emo_map = _default_emo_map()
    if args.emo_map_json:
        emo_map = _load_json(Path(args.emo_map_json).resolve())

    streamer = ExpressionStreamer(
        ExpressionStreamerConfig(
            step_ms=int(args.step_ms),
            default_expression=str(args.default_expression),
            emo_map=emo_map,
            unknown_emo_to=str(args.unknown_emo_to),
        )
    )
    streamer.init_at(0, str(args.default_expression))

    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for call in tool_calls:
        if not isinstance(call, dict):
            skipped.append({"reason": "not_dict", "call": call})
            continue

        name = str(call.get("name", ""))
        if name != "set_emotion":
            skipped.append({"reason": "not_set_emotion", "call": call})
            continue

        emo_id = _parse_emo_id(call)
        if emo_id is None:
            skipped.append({"reason": "missing_emo_id", "call": call})
            continue

        # ③ clamp ロジック追加
        t_ms = int(call.get("t_ms", 0) or 0)

        if audio_ms is not None and audio_ms > 0:
            t_ms = min(t_ms, audio_ms - int(args.step_ms))
            t_ms = max(0, t_ms)

        ev = streamer.on_emo_id(
            emo_id,
            t_ms,
            source="audio_stream_bridge_tool_call",
        )

        accepted.append(
            {
                "t_ms": t_ms,
                "emo_id": emo_id,
                "event_added": ev is not None,
                "event": ev,
            }
        )

    events = streamer.to_json()

    expr_json = out_dir / "expr.json"
    summary_json = out_dir / "audio_tool_calls_to_expr_smoke.summary.json"

    expr_obj = {
        "schema_version": "session_expression_timeline_v0.1",
        "session_id": str(args.session_id),
        "step_ms": int(args.step_ms),
        "timeline": events,
        "meta": {
            "source": "audio_tool_calls_to_expr_smoke",
            "tool_calls_json": str(tool_calls_path),
            "auto_blink": False,
        },
    }

    _write_json(expr_json, expr_obj)

    summary = {
        "format": "audio_tool_calls_to_expr_smoke.summary.v0",
        "session_id": str(args.session_id),
        # ④ summary に audio_ms 追加
        "audio_ms": audio_ms,
        "tool_calls_json": str(tool_calls_path),
        "tool_calls_n": len(tool_calls),
        "accepted_n": len(accepted),
        "skipped_n": len(skipped),
        "events_n": len(events),
        "accepted": accepted,
        "skipped": skipped,
        "outputs": {
            "expr_json": str(expr_json),
        },
    }
    _write_json(summary_json, summary)

    print("[audio_tool_calls_to_expr_smoke][OK]")
    print(f"  tool_calls_n: {len(tool_calls)}")
    print(f"  accepted_n  : {len(accepted)}")
    print(f"  skipped_n   : {len(skipped)}")
    print(f"  events_n    : {len(events)}")
    print(f"  expr_json   : {expr_json}")
    print(f"  summary_json: {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())