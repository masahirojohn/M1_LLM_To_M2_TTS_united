#!/usr/bin/env python3
from __future__ import annotations

"""
Live Runtime chunk_orchestrator v0

目的:
- mouth_worker.ring.json
- expression_worker.latest.json
- pose_json
- audio_meta.json

から 400ms ごとの chunk input を生成する。

v0:
- M0 / M3.5 はまだ実行しない
- chunk JSON 生成のみ
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_STEP_MS = 40
DEFAULT_CHUNK_LEN_MS = 400
DEFAULT_FPS = 25


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _as_frames(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]

    if isinstance(raw, dict):
        for k in ("frames", "timeline"):
            v = raw.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

    raise ValueError("unknown timeline json shape")


def _wrap_like(raw: Any, frames_new: list[dict[str, Any]]) -> Any:
    if isinstance(raw, list):
        return frames_new

    if isinstance(raw, dict):
        out = dict(raw)
        if "frames" in out:
            out["frames"] = frames_new
            return out
        if "timeline" in out:
            out["timeline"] = frames_new
            return out
        out["frames"] = frames_new
        return out

    return {"frames": frames_new}


def _slice_shift_timeline(raw: Any, t0_ms: int, t1_ms: int) -> Any:
    frames = _as_frames(raw)

    inside = [
        fr
        for fr in frames
        if t0_ms <= int(fr.get("t_ms", 0)) < t1_ms
    ]

    prev = None
    for fr in frames:
        t = int(fr.get("t_ms", 0))
        if t < t0_ms:
            prev = fr
        else:
            break

    out: list[dict[str, Any]] = []

    if prev is not None:
        fr0 = dict(prev)
        fr0["t_ms"] = 0
        out.append(fr0)

    for fr in inside:
        fr2 = dict(fr)
        fr2["t_ms"] = int(fr2.get("t_ms", 0)) - int(t0_ms)
        out.append(fr2)

    return _wrap_like(raw, out)


def _read_audio_ms(audio_meta_json: Path) -> int:
    obj = _load_json(audio_meta_json)
    return int(obj["audio_ms"])


def _build_expr_chunk(
    *,
    latest_expr: dict[str, Any],
    session_id: str,
    step_ms: int,
) -> dict[str, Any]:
    latest = latest_expr.get("latest", latest_expr)

    expression = str(latest.get("expression", "normal"))
    emo_id = str(latest.get("emo_id", "1_0"))
    source = str(latest.get("source", "latest_expression"))

    return {
        "schema_version": "session_expression_timeline_v0.1",
        "session_id": session_id,
        "step_ms": int(step_ms),
        "timeline": [
            {
                "t_ms": 0,
                "expression": expression,
                "emo_id": emo_id,
                "source": source,
            }
        ],
        "meta": {
            "source_type": "latest_expression_snapshot",
            "auto_blink": False,
        },
    }


def _build_manifest(
    *,
    session_id: str,
    audio_ms: int,
    step_ms: int,
    chunk_len_ms: int,
    fps: int,
) -> dict[str, Any]:
    target_frames = int(math.ceil(audio_ms / step_ms))
    chunks_n = int(math.ceil(audio_ms / chunk_len_ms))

    chunks = []
    for i in range(chunks_n):
        t0 = i * chunk_len_ms
        t1 = min((i + 1) * chunk_len_ms, audio_ms)

        frame0 = int(math.floor(t0 / step_ms))
        frame1 = int(math.ceil(t1 / step_ms))

        chunks.append(
            {
                "chunk_id": i,
                "t0_ms": t0,
                "t1_ms": t1,
                "frame0": frame0,
                "frame1": frame1,
            }
        )

    return {
        "schema_version": "live_runtime_chunk_manifest.v0",
        "session_id": session_id,
        "audio_ms": int(audio_ms),
        "step_ms": int(step_ms),
        "chunk_len_ms": int(chunk_len_ms),
        "fps": int(fps),
        "target_frames": int(target_frames),
        "chunks": chunks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live Runtime chunk_orchestrator v0"
    )

    ap.add_argument("--session_id", default="sess_realtime_orchestrator_001")

    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--mouth_ring_json", required=True)
    ap.add_argument("--latest_expr_json", required=True)
    ap.add_argument("--audio_meta_json", required=True)

    ap.add_argument(
        "--out_dir",
        default="out/live_runtime_realtime/chunk_orchestrator_001",
    )

    ap.add_argument("--step_ms", type=int, default=DEFAULT_STEP_MS)
    ap.add_argument("--chunk_len_ms", type=int, default=DEFAULT_CHUNK_LEN_MS)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)

    args = ap.parse_args()

    if int(args.step_ms) != 40:
        raise ValueError("step_ms must be 40")
    if int(args.chunk_len_ms) != 400:
        raise ValueError("chunk_len_ms must be 400")

    session_id = str(args.session_id)

    pose_json = Path(args.pose_json).resolve()
    mouth_ring_json = Path(args.mouth_ring_json).resolve()
    latest_expr_json = Path(args.latest_expr_json).resolve()
    audio_meta_json = Path(args.audio_meta_json).resolve()
    out_dir = Path(args.out_dir).resolve()

    pose_raw = _load_json(pose_json)
    mouth_raw = _load_json(mouth_ring_json)
    latest_expr = _load_json(latest_expr_json)
    audio_ms = _read_audio_ms(audio_meta_json)

    manifest = _build_manifest(
        session_id=session_id,
        audio_ms=audio_ms,
        step_ms=int(args.step_ms),
        chunk_len_ms=int(args.chunk_len_ms),
        fps=int(args.fps),
    )

    manifest_json = out_dir / "manifest.realtime_chunks.json"
    _write_json(manifest_json, manifest)

    chunks_root = out_dir / "chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)

    for ch in manifest["chunks"]:
        cid = int(ch["chunk_id"])
        t0 = int(ch["t0_ms"])
        t1 = int(ch["t1_ms"])

        chunk_dir = chunks_root / f"{cid:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        pose_chunk = _slice_shift_timeline(pose_raw, t0, t1)
        mouth_chunk = _slice_shift_timeline(mouth_raw, t0, t1)
        expr_chunk = _build_expr_chunk(
            latest_expr=latest_expr,
            session_id=session_id,
            step_ms=int(args.step_ms),
        )

        _write_json(chunk_dir / "pose.chunk.json", pose_chunk)
        _write_json(chunk_dir / "mouth.chunk.json", mouth_chunk)
        _write_json(chunk_dir / "expr.chunk.json", expr_chunk)

        _write_json(
            chunk_dir / "chunk_meta.json",
            {
                "chunk_id": cid,
                "t0_ms": t0,
                "t1_ms": t1,
                "frame0": int(ch["frame0"]),
                "frame1": int(ch["frame1"]),
                "pose_json": str(chunk_dir / "pose.chunk.json"),
                "mouth_json": str(chunk_dir / "mouth.chunk.json"),
                "expr_json": str(chunk_dir / "expr.chunk.json"),
            },
        )

        mouth_frames = _as_frames(mouth_chunk)
        pose_frames = _as_frames(pose_chunk)

        print(
            f"[chunk_orchestrator] chunk={cid:06d} "
            f"t=[{t0},{t1}) "
            f"pose_frames={len(pose_frames)} "
            f"mouth_frames={len(mouth_frames)}"
        )

    summary_json = out_dir / "chunk_orchestrator.summary.json"
    _write_json(
        summary_json,
        {
            "schema_version": "chunk_orchestrator_summary.v0",
            "session_id": session_id,
            "audio_ms": int(audio_ms),
            "step_ms": int(args.step_ms),
            "chunk_len_ms": int(args.chunk_len_ms),
            "fps": int(args.fps),
            "chunks": len(manifest["chunks"]),
            "manifest_json": str(manifest_json),
            "chunks_root": str(chunks_root),
        },
    )

    print("[chunk_orchestrator][OK]")
    print(f"  audio_ms     : {audio_ms}")
    print(f"  chunks       : {len(manifest['chunks'])}")
    print(f"  manifest_json: {manifest_json}")
    print(f"  chunks_root  : {chunks_root}")
    print(f"  summary_json : {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())