#!/usr/bin/env python3
"""Phase B5/B5hf/B5hf2: pose slice clock binds to PLAYING ideal_base (no devices).

B5hf:
- seq cursor → no freeze (turn_local provisional)
- audio cursor → freeze ideal_base≈bg−audio/step
- missing → last-good (same fo) or provisional (never silent absolute 0)

B5hf2:
- ok_audio freeze only when cursor.frame_offset == turn fo
- prior-turn PLAYING audio (stale fo) → fo_wait provisional (T3 fingerprint)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "live_runtime"))

from run_mic_input_obs_realtime_step1 import (  # noqa: E402
    _ensure_pose_base_frame,
    _pose_abs_window_ms,
    _read_bg_cursor_info,
    _read_bg_cursor_pose_base,
)


def _check_window() -> None:
    t0, t1 = _pose_abs_window_ms(
        t0_ms=0, t1_ms=120, pose_base_frame=274, step_ms=40
    )
    assert (t0, t1) == (274 * 40, 274 * 40 + 120), (t0, t1)
    t0b, t1b = _pose_abs_window_ms(
        t0_ms=1000, t1_ms=1120, pose_base_frame=274, step_ms=40
    )
    assert (t0b, t1b) == (274 * 40 + 1000, 274 * 40 + 1120), (t0b, t1b)
    audio_ms = 1000
    pose_i = (t0b + (audio_ms - 1000)) // 40
    assert pose_i == 274 + (audio_ms // 40), pose_i


def _write_cursor(path: Path, **payload: object) -> None:
    body = {"type": "bg_cursor", **payload}
    path.write_text(json.dumps(body), encoding="utf-8")


def _check_cursor_snapshot() -> None:
    with tempfile.TemporaryDirectory() as td:
        cur = Path(td) / "bg_cursor.json"
        # Legacy bg_pos-only still readable.
        _write_cursor(cur, bg_pos=271, bg_mode="seq")
        assert _read_bg_cursor_pose_base(cur) == 271
        info = _read_bg_cursor_info(cur)
        assert info is not None and info["ideal_base_frame"] == 271

        ref: dict = {
            "bg_cursor_file": cur,
            "pose_base_frame": None,
            "pose_clock_mode": None,
            "pose_base_pending_logged": False,
            "pose_base_ssot": {"last_good": None, "last_good_fo": None},
            "frame_offset": 138,
            "step_ms": 40,
        }
        # seq → provisional turn_local, no freeze
        b1, m1 = _ensure_pose_base_frame(ref)
        assert (b1, m1) == (0, "turn_local")
        assert ref.get("pose_clock_mode") is None
        assert ref.get("pose_base_frame") is None

        # audio + matching fo → freeze ideal_base = 110 - 60//40 = 109
        _write_cursor(
            cur,
            bg_pos=110,
            bg_mode="audio",
            audio_ms=60,
            step_ms=40,
            ideal_base_frame=109,
            frame_offset=138,
        )
        b2, m2 = _ensure_pose_base_frame(ref)
        b3, m3 = _ensure_pose_base_frame(ref)
        assert (b2, m2) == (109, "absolute")
        assert (b3, m3) == (109, "absolute")
        assert int(ref["pose_base_frame"]) == 109
        assert ref["pose_clock_mode"] == "absolute"
        assert int(ref["pose_base_ssot"]["last_good"]) == 109
        assert int(ref["pose_base_ssot"]["last_good_fo"]) == 138

        # Missing cursor + same-fo last-good → absolute.
        ref_miss: dict = {
            "bg_cursor_file": Path(td) / "missing.json",
            "pose_base_frame": None,
            "pose_clock_mode": None,
            "pose_base_pending_logged": False,
            "pose_base_ssot": {"last_good": 380, "last_good_fo": 504},
            "frame_offset": 504,
            "step_ms": 40,
        }
        b4, m4 = _ensure_pose_base_frame(ref_miss)
        assert (b4, m4) == (380, "absolute")
        assert ref_miss["pose_clock_mode"] == "absolute"

        # Missing + cross-fo last-good → provisional (not previous-turn absolute).
        ref_miss_fo: dict = {
            "bg_cursor_file": Path(td) / "missing2.json",
            "pose_base_frame": None,
            "pose_clock_mode": None,
            "pose_base_pending_logged": False,
            "pose_base_ssot": {"last_good": 380, "last_good_fo": 174},
            "frame_offset": 330,
            "step_ms": 40,
        }
        b5, m5 = _ensure_pose_base_frame(ref_miss_fo)
        assert (b5, m5) == (0, "turn_local")
        assert ref_miss_fo.get("pose_clock_mode") is None

        # Missing + no last-good → provisional turn_local (not absolute-at-0).
        ref_tl: dict = {
            "bg_cursor_file": Path(td) / "missing3.json",
            "pose_base_frame": None,
            "pose_clock_mode": None,
            "pose_base_pending_logged": False,
            "pose_base_ssot": {"last_good": None, "last_good_fo": None},
            "frame_offset": 0,
            "step_ms": 40,
        }
        b6, m6 = _ensure_pose_base_frame(ref_tl)
        assert (b6, m6) == (0, "turn_local")
        assert ref_tl.get("pose_clock_mode") is None


def _check_t3_stale_playing_fo() -> None:
    """162517 T3: prior-turn PLAYING audio must not freeze new fo.

    Before: ok_audio freeze pose_base=301 at fo=330 while cursor still old turn;
    later enter_playing RELOCK ideal≈457 → |gap|≈156.
    After: fo_wait provisional until cursor.fo==330 audio → freeze 457.
    """
    with tempfile.TemporaryDirectory() as td:
        cur = Path(td) / "bg_cursor.json"
        ref: dict = {
            "bg_cursor_file": cur,
            "pose_base_frame": None,
            "pose_clock_mode": None,
            "pose_base_pending_logged": False,
            "pose_base_ssot": {"last_good": 299, "last_good_fo": 174},
            "frame_offset": 330,
            "step_ms": 40,
        }
        # Stale prior-turn PLAYING cursor (fo=174, ideal≈301).
        _write_cursor(
            cur,
            bg_pos=400,
            bg_mode="audio",
            audio_ms=3960,
            step_ms=40,
            ideal_base_frame=301,
            frame_offset=174,
        )
        b1, m1 = _ensure_pose_base_frame(ref)
        assert (b1, m1) == (0, "turn_local")
        assert ref.get("pose_clock_mode") is None
        assert ref.get("pose_base_frame") is None

        # New-fo enter_playing RELOCK cursor.
        _write_cursor(
            cur,
            bg_pos=457,
            bg_mode="audio",
            audio_ms=0,
            step_ms=40,
            ideal_base_frame=457,
            frame_offset=330,
        )
        b2, m2 = _ensure_pose_base_frame(ref)
        assert (b2, m2) == (457, "absolute")
        assert int(ref["pose_base_frame"]) == 457
        gap = abs(457 - 457)
        assert gap < 2


def _check_t2_index_match() -> None:
    """B4 T2 table: after pose_base=274, pose_i == bg_pos at listed audio_ms."""
    samples = [
        (0, 274),
        (250, 280),
        (1000, 299),
        (3000, 349),
        (4000, 374),
        (5600, 414),
    ]
    pose_base = 274
    step = 40
    for audio_ms, bg_pos in samples:
        t0, _ = _pose_abs_window_ms(
            t0_ms=int(audio_ms),
            t1_ms=int(audio_ms) + step,
            pose_base_frame=pose_base,
            step_ms=step,
        )
        pose_i = int(t0 // step)
        assert pose_i == int(bg_pos), (audio_ms, pose_i, bg_pos)


def _check_t1_ideal_vs_idle_snap() -> None:
    """T1 fingerprint: idle snap 82 vs RELOCK ideal≈109 → old gap≈-27."""
    idle_snap = 82
    relock_bg = 110
    relock_a = 60
    ideal = relock_bg - relock_a // 40
    assert ideal == 109
    old_gap = (idle_snap + relock_a / 40.0) - float(relock_bg)
    new_gap = (ideal + relock_a / 40.0) - float(relock_bg)
    assert abs(old_gap) > 20
    assert abs(new_gap) < 2.0


def _check_legacy_audio_no_fo_turn0() -> None:
    """T1 compat: legacy audio cursor without fo may freeze only at turn_fo==0."""
    with tempfile.TemporaryDirectory() as td:
        cur = Path(td) / "bg_cursor.json"
        _write_cursor(
            cur,
            bg_pos=110,
            bg_mode="audio",
            audio_ms=60,
            step_ms=40,
            ideal_base_frame=109,
        )
        ref0: dict = {
            "bg_cursor_file": cur,
            "pose_base_frame": None,
            "pose_clock_mode": None,
            "pose_base_pending_logged": False,
            "pose_base_ssot": {"last_good": None, "last_good_fo": None},
            "frame_offset": 0,
            "step_ms": 40,
        }
        b0, m0 = _ensure_pose_base_frame(ref0)
        assert (b0, m0) == (109, "absolute")

        ref1: dict = {
            "bg_cursor_file": cur,
            "pose_base_frame": None,
            "pose_clock_mode": None,
            "pose_base_pending_logged": False,
            "pose_base_ssot": {"last_good": None, "last_good_fo": None},
            "frame_offset": 174,
            "step_ms": 40,
        }
        b1, m1 = _ensure_pose_base_frame(ref1)
        assert (b1, m1) == (0, "turn_local")
        assert ref1.get("pose_clock_mode") is None


def main() -> int:
    _check_window()
    _check_cursor_snapshot()
    _check_t3_stale_playing_fo()
    _check_t2_index_match()
    _check_t1_ideal_vs_idle_snap()
    _check_legacy_audio_no_fo_turn0()
    print("[phase_b5_pose_base_selfcheck] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
