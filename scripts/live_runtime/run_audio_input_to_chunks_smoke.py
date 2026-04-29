#!/usr/bin/env python3
from __future__ import annotations

"""
run_audio_input_to_chunks_smoke.py

目的:
- run_audio_input_smoke_pipeline.py の出力を使い、
  M0投入直前の chunk JSON を生成する。

入力:
- mouth.json
- expr.json
- audio_meta.json
- pose_json

出力:
- chunks/000000/pose.chunk.json
- chunks/000000/mouth.chunk.json
- chunks/000000/expr.chunk.json
- manifest.json
- run_audio_input_to_chunks_smoke.summary.json
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _as_frames(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]

    if isinstance(raw, dict):
        for k in ("frames", "timeline"):
            v = raw.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

    raise RuntimeError("unknown timeline shape")


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
        fr for fr in frames
        if t0_ms <= int(fr.get("t_ms", 0) or 0) < t1_ms
    ]

    prev = None
    for fr in frames:
        t = int(fr.get("t_ms", 0) or 0)
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
        fr2["t_ms"] = int(fr2.get("t_ms", 0) or 0) - int(t0_ms)
        out.append(fr2)

    return _wrap_like(raw, out)


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
        "schema_version": "audio_input_chunk_manifest.v0",
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
        description="audio input smoke outputs -> M0 chunk JSON smoke"
    )

    ap.add_argument("--session_id", default="sess_audio_input_chunks_001")
    ap.add_argument("--pipeline_dir", required=True)
    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--chunk_len_ms", type=int, default=400)
    ap.add_argument("--fps", type=int, default=25)

    args = ap.parse_args()

    pipeline_dir = Path(args.pipeline_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_in = pipeline_dir / "run_audio_input_smoke_pipeline.summary.json"
    summary_obj = _load_json(summary_in)

    mouth_json = Path(summary_obj["outputs"]["mouth_json"]).resolve()
    expr_json = Path(summary_obj["outputs"]["expr_json"]).resolve()
    audio_meta_json = Path(summary_obj["outputs"]["audio_meta_json"]).resolve()
    pose_json = Path(args.pose_json).resolve()

    mouth_obj = _load_json(mouth_json)
    expr_obj = _load_json(expr_json)
    audio_meta = _load_json(audio_meta_json)
    pose_obj = _load_json(pose_json)

    audio_ms = int(audio_meta["audio_ms"])
    step_ms = int(args.step_ms)
    chunk_len_ms = int(args.chunk_len_ms)
    fps = int(args.fps)

    manifest = _build_manifest(
        session_id=str(args.session_id),
        audio_ms=audio_ms,
        step_ms=step_ms,
        chunk_len_ms=chunk_len_ms,
        fps=fps,
    )

    manifest_json = out_dir / "manifest.json"
    chunks_root = out_dir / "chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)

    _write_json(manifest_json, manifest)

    written_chunks: list[dict[str, Any]] = []

    for ch in manifest["chunks"]:
        cid = int(ch["chunk_id"])
        t0 = int(ch["t0_ms"])
        t1 = int(ch["t1_ms"])

        cdir = chunks_root / f"{cid:06d}"
        cdir.mkdir(parents=True, exist_ok=True)

        pose_chunk = _slice_shift_timeline(pose_obj, t0, t1)
        mouth_chunk = _slice_shift_timeline(mouth_obj, t0, t1)
        expr_chunk = _slice_shift_timeline(expr_obj, t0, t1)

        pose_chunk_json = cdir / "pose.chunk.json"
        mouth_chunk_json = cdir / "mouth.chunk.json"
        expr_chunk_json = cdir / "expr.chunk.json"

        _write_json(pose_chunk_json, pose_chunk)
        _write_json(mouth_chunk_json, mouth_chunk)
        _write_json(expr_chunk_json, expr_chunk)

        written_chunks.append(
            {
                **ch,
                "dir": str(cdir),
                "pose_chunk_json": str(pose_chunk_json),
                "mouth_chunk_json": str(mouth_chunk_json),
                "expr_chunk_json": str(expr_chunk_json),
                "pose_frames_n": len(_as_frames(pose_chunk)),
                "mouth_frames_n": len(_as_frames(mouth_chunk)),
                "expr_events_n": len(_as_frames(expr_chunk)),
            }
        )

    summary = {
        "format": "audio_input_to_chunks_smoke.summary.v0",
        "session_id": str(args.session_id),
        "audio_ms": int(audio_ms),
        "step_ms": step_ms,
        "chunk_len_ms": chunk_len_ms,
        "fps": fps,
        "target_frames": manifest["target_frames"],
        "chunks_n": len(manifest["chunks"]),
        "inputs": {
            "pipeline_summary_json": str(summary_in),
            "pose_json": str(pose_json),
            "mouth_json": str(mouth_json),
            "expr_json": str(expr_json),
            "audio_meta_json": str(audio_meta_json),
        },
        "outputs": {
            "manifest_json": str(manifest_json),
            "chunks_root": str(chunks_root),
        },
        "chunks": written_chunks,
    }

    summary_json = out_dir / "run_audio_input_to_chunks_smoke.summary.json"
    _write_json(summary_json, summary)

    print("[run_audio_input_to_chunks_smoke][OK]")
    print(f"  session_id   : {args.session_id}")
    print(f"  audio_ms     : {audio_ms}")
    print(f"  target_frames: {manifest['target_frames']}")
    print(f"  chunks_n     : {len(manifest['chunks'])}")
    print(f"  chunks_root  : {chunks_root}")
    print(f"  manifest_json: {manifest_json}")
    print(f"  summary_json : {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())