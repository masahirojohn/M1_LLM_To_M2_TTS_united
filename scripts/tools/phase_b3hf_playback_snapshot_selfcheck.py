#!/usr/bin/env python3
"""Phase B3hf: dual-read stall vs single-snapshot (no devices).

Before (B3 bug): BG tick sees None/UNKNOWN → freeze a_ms; FG re-reads PLAYING
→ audio_ms advances, bg_pos sticks at lock frame.

After (B3hf): one snapshot (+retry) shared by BG/FG; UNKNOWN uses last-good
PLAYING dict for both → bg_pos tracks audio_ms while bg_mode=audio.
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


def _sim_before_dual_read(*, fps: float = 25.0, ticks: int = 45) -> dict:
    """Model torn first-read None + FG second-read PLAYING (Turn2 fingerprint)."""
    period = 1000.0 / fps
    lock_audio = 0
    lock_bg = 283
    bg_pos = lock_bg
    last_playing_a_ms = 0
    bg_positions: list[int] = []
    audio_fg: list[int] = []
    for i in range(ticks):
        played = int(round(i * period * 24.0))  # 24 samples/ms @ 24kHz
        # BG: first read None → UNKNOWN → freeze last_playing_a_ms
        a_bg = int(last_playing_a_ms)
        desired = _b3_desired_bg_frame(
            audio_ms=a_bg,
            lock_audio_ms=lock_audio,
            lock_bg_frame=lock_bg,
            frame_period_ms=period,
        )
        bg_pos = int(desired)
        bg_positions.append(bg_pos)
        # FG: second read succeeds
        tgt = _resolve_ssot_target(
            playback_state=_fake_state(played),
            sync_meta={"frame_offset": 126, "base_played_samples": 0, "step_ms": 40},
            step_ms=40,
            frame_offset_cli=0,
        )
        audio_fg.append(int(tgt["audio_ms"]))
        # last_playing only updates on PLAYING first-read — never in this model
    unique_bg = len(set(bg_positions))
    audio_span = int(audio_fg[-1] - audio_fg[0]) if audio_fg else 0
    return {
        "unique_bg_pos": unique_bg,
        "bg_pos_first": bg_positions[0],
        "bg_pos_last": bg_positions[-1],
        "audio_span_ms": audio_span,
        "stalled": unique_bg == 1 and audio_span > 500,
    }


def _sim_after_single_snapshot(*, fps: float = 25.0, ticks: int = 45) -> dict:
    """Single snapshot: retry yields PLAYING; BG+FG share same audio_ms."""
    period = 1000.0 / fps
    lock_audio = 0
    lock_bg = 283
    last_good: dict | None = None
    bg_positions: list[int] = []
    audio_shared: list[int] = []
    for i in range(ticks):
        played = int(round(i * period * 24.0))
        # Simulate torn first attempt then retry success (shared).
        raw = None
        retry = _fake_state(played)
        snap = retry if raw is None else raw
        if str(snap.get("state")) == "PLAYING":
            last_good = dict(snap)
        effective = snap
        if (
            effective is None
            and last_good is not None
        ):
            effective = last_good
        tgt = _resolve_ssot_target(
            playback_state=effective,
            sync_meta={"frame_offset": 126, "base_played_samples": 0, "step_ms": 40},
            step_ms=40,
            frame_offset_cli=0,
        )
        a_ms = int(tgt["audio_ms"])
        desired = _b3_desired_bg_frame(
            audio_ms=a_ms,
            lock_audio_ms=lock_audio,
            lock_bg_frame=lock_bg,
            frame_period_ms=period,
        )
        bg_positions.append(int(desired))
        audio_shared.append(a_ms)
    unique_bg = len(set(bg_positions))
    audio_span = int(audio_shared[-1] - audio_shared[0]) if audio_shared else 0
    bg_span = int(bg_positions[-1] - bg_positions[0]) if bg_positions else 0
    return {
        "unique_bg_pos": unique_bg,
        "bg_pos_first": bg_positions[0],
        "bg_pos_last": bg_positions[-1],
        "bg_span": bg_span,
        "audio_span_ms": audio_span,
        "stalled": unique_bg == 1 and audio_span > 500,
    }


def main() -> int:
    before = _sim_before_dual_read()
    after = _sim_after_single_snapshot()
    print("[B3hf][before]", before)
    print("[B3hf][after]", after)
    ok = (
        before["stalled"] is True
        and after["stalled"] is False
        and after["bg_span"] >= 20
        and after["unique_bg_pos"] >= 20
        and after["audio_span_ms"] > 500
    )
    print("[B3hf][selfcheck]", "PASS" if ok else "FAIL")
    print(
        "[B3hf][delta_table]",
        f"bg_pos_stall_ticks: before=unique{before['unique_bg_pos']}"
        f"(pos={before['bg_pos_first']}) audio_span={before['audio_span_ms']}ms;",
        f"after=unique{after['unique_bg_pos']}"
        f"(span={after['bg_span']}) audio_span={after['audio_span_ms']}ms",
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
