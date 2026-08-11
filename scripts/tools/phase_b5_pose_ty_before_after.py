#!/usr/bin/env python3
"""Phase B5 Before/After: T2 pose.ty gap vs bg-aligned pose (B4 session geometry).

Uses B4 sample points from sess_phase11_subj_20260810_215344 (lock_bg=274).
Before pose_base=0; After ideal pose_base=274; After snap≈271 (chunk0 cursor).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _as_frames(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [f for f in raw if isinstance(f, dict)]
    if isinstance(raw, dict):
        for k in ("frames", "timeline", "poses"):
            v = raw.get(k)
            if isinstance(v, list):
                return [f for f in v if isinstance(f, dict)]
    return []


def _ty(fr: dict[str, Any]) -> float:
    for k in ("ty", "t_y", "y"):
        if k in fr and fr[k] is not None:
            return float(fr[k])
    tr = fr.get("transform") or fr.get("pose") or {}
    if isinstance(tr, dict):
        for k in ("ty", "t_y", "y"):
            if k in tr and tr[k] is not None:
                return float(tr[k])
    raise KeyError(f"no ty in pose frame keys={list(fr.keys())[:12]}")


def _nearest(frames: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    if not frames:
        raise RuntimeError("empty pose frames")
    i = max(0, min(int(idx), len(frames) - 1))
    return frames[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pose_json",
        default=r"C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json",
    )
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--lock_bg", type=int, default=274)
    args = ap.parse_args()

    pose_path = Path(args.pose_json).resolve()
    raw = json.loads(pose_path.read_text(encoding="utf-8"))
    frames = _as_frames(raw)
    if not frames:
        print(f"[B5] FAIL empty pose frames path={pose_path}", file=sys.stderr)
        return 2

    # B4 T2 samples (audio_ms, bg_pos)
    samples = [
        (0, 274),
        (250, 280),
        (1000, 299),
        (3000, 349),
        (4000, 374),
        (5600, 414),
    ]
    step = int(args.step_ms)
    lock = int(args.lock_bg)
    snap = max(0, lock - 3)  # T2 chunk0 cursor ≈271 vs RELOCK 274

    def gap(pose_base: int) -> tuple[float, float, list[tuple]]:
        rows = []
        abs_err = []
        for audio_ms, bg_pos in samples:
            use_i = int(pose_base) + int(audio_ms) // step
            p_use = _nearest(frames, use_i)
            p_bg = _nearest(frames, int(bg_pos))
            dty = _ty(p_use) - _ty(p_bg)
            abs_err.append(abs(dty))
            rows.append((audio_ms, bg_pos, use_i, dty))
        return (
            float(sum(abs_err) / len(abs_err)),
            float(max(abs_err)),
            rows,
        )

    before_mean, before_max, before_rows = gap(0)
    after_mean, after_max, after_rows = gap(lock)
    snap_mean, snap_max, snap_rows = gap(snap)

    print("[B5][Before/After] T2 pose.ty vs pose[bg_pos]")
    print(f"  pose_json={pose_path} n={len(frames)}")
    print(
        f"  BEFORE pose_base=0     absmean={before_mean:.2f} maxabs={before_max:.2f}"
    )
    print(
        f"  AFTER  pose_base={lock}   absmean={after_mean:.2f} maxabs={after_max:.2f}"
    )
    print(
        f"  AFTER  pose_base={snap}(snap) absmean={snap_mean:.2f} maxabs={snap_max:.2f}"
    )
    print("  rows(before): audio bg use_i dty")
    for audio_ms, bg_pos, use_i, dty in before_rows:
        print(f"    {audio_ms:5d} {bg_pos:4d} {use_i:4d} {dty:+7.2f}")
    print("  rows(after lock): audio bg use_i dty")
    for audio_ms, bg_pos, use_i, dty in after_rows:
        print(f"    {audio_ms:5d} {bg_pos:4d} {use_i:4d} {dty:+7.2f}")

    # Pass gate: after lock must collapse index gap; ty absmean << before.
    if after_mean > 1e-6:
        # identical indices ⇒ 0; tolerate float noise only
        if not math.isclose(after_mean, 0.0, abs_tol=1e-6):
            print(
                f"[B5] WARN after_lock absmean not zero ({after_mean}); "
                "check pose indexing",
                file=sys.stderr,
            )
    ok = after_mean < before_mean * 0.1 and snap_mean < before_mean * 0.5
    print(f"[B5][Before/After] {'PASS' if ok else 'FAIL'} improve_gate={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
