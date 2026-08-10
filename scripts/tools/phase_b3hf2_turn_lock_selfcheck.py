#!/usr/bin/env python3
"""Phase B3hf2: Turn-frontier lock_audio_ms=0 falsy freeze (no devices).

Before (145032 fingerprint): turn RELOCK sets lock_audio_ms=0; desired uses
`lock or a_ms` so 0 is treated as missing → lock rebinds to current a_ms every
tick → desired stays at lock_bg while FG audio_ms advances (bg_pos stick).

After: lock_audio_ms is taken iff is not None (0 is valid) → bg_pos tracks.
Also models single post-adopt resolve (a_ms_bg == a_ms_fg).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "live_runtime"))

from run_virtualcam_persistent import (  # noqa: E402
    _b3_desired_bg_frame,
    _resolve_ssot_target,
)


def _fake_state(played_samples: int, state: str = "PLAYING") -> dict:
    return {
        "state": state,
        "played_samples": int(played_samples),
        "sample_rate": 24000,
        "player_local_ms": float(played_samples) * 1000.0 / 24000.0,
    }


def _sim_before_falsy_lock0(*, fps: float = 25.0, ticks: int = 80) -> dict:
    """Model turn RELOCK at audio_ms=0 with `lock or a_ms` bug."""
    period = 1000.0 / fps
    lock_audio = 0  # turn relative clock
    lock_bg = 337
    base = 212415
    bg_positions: list[int] = []
    audio_fg: list[int] = []
    for i in range(ticks):
        # played advances past turn base (FG audio_ms grows).
        played = int(base + round(i * period * 24.0))
        tgt = _resolve_ssot_target(
            playback_state=_fake_state(played),
            sync_meta={
                "frame_offset": 222,
                "base_played_samples": base,
                "step_ms": 40,
            },
            step_ms=40,
            frame_offset_cli=0,
        )
        a_ms = int(tgt["audio_ms"])
        audio_fg.append(a_ms)
        # BUG: 0 is falsy → lock rebound to a_ms every tick.
        lock_a = int(lock_audio or a_ms)
        desired = _b3_desired_bg_frame(
            audio_ms=a_ms,
            lock_audio_ms=lock_a,
            lock_bg_frame=lock_bg,
            frame_period_ms=period,
        )
        bg_positions.append(int(desired))
    unique_bg = len(set(bg_positions))
    audio_span = int(audio_fg[-1] - audio_fg[0]) if audio_fg else 0
    return {
        "unique_bg_pos": unique_bg,
        "bg_pos_first": bg_positions[0],
        "bg_pos_last": bg_positions[-1],
        "audio_span_ms": audio_span,
        "a_ms_fg_last": audio_fg[-1] if audio_fg else 0,
        "stalled": unique_bg == 1 and audio_span > 2000,
    }


def _sim_after_lock0_ok(*, fps: float = 25.0, ticks: int = 80) -> dict:
    """lock_audio_ms=0 kept; single resolve a_ms for BG+FG."""
    period = 1000.0 / fps
    lock_audio = 0
    lock_bg = 337
    base = 212415
    bg_positions: list[int] = []
    audio_shared: list[int] = []
    for i in range(ticks):
        played = int(base + round(i * period * 24.0))
        tgt = _resolve_ssot_target(
            playback_state=_fake_state(played),
            sync_meta={
                "frame_offset": 222,
                "base_played_samples": base,
                "step_ms": 40,
            },
            step_ms=40,
            frame_offset_cli=0,
        )
        a_ms = int(tgt["audio_ms"])
        audio_shared.append(a_ms)
        # FIX: 0 is a valid lock.
        lock_a = int(lock_audio) if lock_audio is not None else int(a_ms)
        desired = _b3_desired_bg_frame(
            audio_ms=a_ms,
            lock_audio_ms=lock_a,
            lock_bg_frame=lock_bg,
            frame_period_ms=period,
        )
        bg_positions.append(int(desired))
    unique_bg = len(set(bg_positions))
    audio_span = int(audio_shared[-1] - audio_shared[0]) if audio_shared else 0
    bg_span = int(bg_positions[-1] - bg_positions[0]) if bg_positions else 0
    return {
        "unique_bg_pos": unique_bg,
        "bg_pos_first": bg_positions[0],
        "bg_pos_last": bg_positions[-1],
        "bg_span": bg_span,
        "audio_span_ms": audio_span,
        "a_ms_fg_last": audio_shared[-1] if audio_shared else 0,
        "stalled": unique_bg == 1 and audio_span > 2000,
    }


def main() -> int:
    before = _sim_before_falsy_lock0()
    after = _sim_after_lock0_ok()
    print("[B3hf2][before]", before)
    print("[B3hf2][after]", after)
    ok = (
        before["stalled"] is True
        and before["bg_pos_first"] == before["bg_pos_last"] == 337
        and after["stalled"] is False
        and after["bg_span"] >= 50
        and after["unique_bg_pos"] >= 50
        and after["audio_span_ms"] > 2000
        and after["bg_pos_first"] == 337
    )
    print("[B3hf2][selfcheck]", "PASS" if ok else "FAIL")
    print(
        "[B3hf2][delta_table]",
        f"T2_stick: before=bg_pos={before['bg_pos_first']} unique={before['unique_bg_pos']}"
        f" audio_span={before['audio_span_ms']}ms;",
        f"after=bg_span={after['bg_span']} unique={after['unique_bg_pos']}"
        f" audio_span={after['audio_span_ms']}ms",
    )
    # Boundary note (not a fail): PLAYING-through wait with reason=turn is separate.
    print(
        "[B3hf2][boundary_note]",
        "T1->T2 wait may RELOCK reason=turn while still PLAYING (no seq gap);",
        "out of scope for this Hotfix - track separately from T2 head stick.",
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
