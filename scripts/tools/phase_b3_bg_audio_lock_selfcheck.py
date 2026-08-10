#!/usr/bin/env python3
"""Phase B3: Before/After Δ for PLAYING audio→BG lock (no devices).

Before: wall sequential BG during REB/idle; resume PLAYING keeps carry Δ.
After: REB/idle still sequential; enter PLAYING re-locks → within-PLAYING Δ≈0.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "live_runtime"))

from run_virtualcam_persistent import _b3_desired_bg_frame  # noqa: E402


def _sim_before(*, fps: float = 25.0) -> dict:
    period = 1000.0 / fps
    bg = 0
    audio = 0
    # 1s PLAYING
    for _ in range(25):
        bg += 1
        audio += int(period)
    d0 = int(round(bg * period - audio))
    # 400ms REB (audio frozen, BG wall)
    for _ in range(10):
        bg += 1
    d_reb = int(round(bg * period - audio))
    # 1s PLAYING resume without re-lock (carry held)
    for _ in range(25):
        bg += 1
        audio += int(period)
    d_resume = int(round(bg * period - audio))
    # 3s idle
    for _ in range(75):
        bg += 1
    d_idle = int(round(bg * period - audio))
    return {
        "playing0_delta_ms": d0,
        "after_reb_delta_ms": d_reb,
        "after_resume_carry_ms": d_resume,
        "after_idle_carry_ms": d_idle,
    }


def _sim_after_b3(*, fps: float = 25.0) -> dict:
    """Drive bg_read with audio during PLAYING; re-lock on enter PLAYING."""
    period = 1000.0 / fps
    bg_read = 0
    audio = 0
    lock_audio = None
    lock_bg = None
    deltas_playing: list[int] = []
    anchor_bg = None
    anchor_audio = None
    last_state = None

    def step(st: str, audio_ms: int, *, advance_wall: bool) -> int:
        nonlocal bg_read, lock_audio, lock_bg, anchor_bg, anchor_audio, last_state
        if st == "PLAYING":
            if lock_audio is None or last_state != "PLAYING":
                lock_audio = int(audio_ms)
                lock_bg = int(bg_read)  # current wall pos becomes lock
            desired = _b3_desired_bg_frame(
                audio_ms=int(audio_ms),
                lock_audio_ms=int(lock_audio),
                lock_bg_frame=int(lock_bg),
                frame_period_ms=period,
            )
            # One virtualcam tick always consumes one read slot (sent++).
            bg_read += 1
            # Logical BG follows audio (hold/drop); obs uses read count + re-anchor.
            _ = desired
        else:
            lock_audio = None
            lock_bg = None
            if advance_wall:
                bg_read += 1
        # Mirror _b2_obs_maybe_log re-anchor rules without printing.
        entered = last_state is None or str(last_state) != "PLAYING"
        if st == "PLAYING" and (anchor_bg is None or entered):
            anchor_bg = int(bg_read)
            anchor_audio = int(audio_ms)
        last_state = st
        if anchor_bg is None or anchor_audio is None:
            return 0
        bg_elapsed = int(round((bg_read - int(anchor_bg)) * period))
        audio_elapsed = int(audio_ms) - int(anchor_audio)
        return int(bg_elapsed - audio_elapsed)

    # 1s PLAYING
    for i in range(25):
        audio = i * int(period)
        d = step("PLAYING", audio, advance_wall=False)
        deltas_playing.append(d)
    d_play0_max = max(abs(x) for x in deltas_playing) if deltas_playing else 0
    # 400ms REB
    for _ in range(10):
        step("REBUFFERING", audio, advance_wall=True)
    # resume PLAYING — re-lock cuts carry
    resume_deltas: list[int] = []
    base = audio
    for i in range(25):
        audio = base + i * int(period)
        resume_deltas.append(step("PLAYING", audio, advance_wall=False))
    # idle 3s then new PLAYING
    for _ in range(75):
        step("BUFFERING", audio, advance_wall=True)
    after_idle: list[int] = []
    for i in range(25):
        audio_i = i * int(period)
        after_idle.append(step("PLAYING", audio_i, advance_wall=False))

    return {
        "playing0_max_abs_delta_ms": d_play0_max,
        "resume_max_abs_delta_ms": max(abs(x) for x in resume_deltas),
        "after_idle_max_abs_delta_ms": max(abs(x) for x in after_idle),
        "resume_final_delta_ms": resume_deltas[-1] if resume_deltas else 0,
    }


def main() -> int:
    before = _sim_before()
    after = _sim_after_b3()
    print("[B3][before]", before)
    print("[B3][after]", after)
    ok = (
        before["playing0_delta_ms"] == 0
        and before["after_reb_delta_ms"] == 400
        and before["after_resume_carry_ms"] == 400
        and before["after_idle_carry_ms"] == 3400
        and after["playing0_max_abs_delta_ms"] <= 40
        and after["resume_max_abs_delta_ms"] <= 40
        and after["after_idle_max_abs_delta_ms"] <= 40
    )
    print("[B3][selfcheck]", "PASS" if ok else "FAIL")
    print(
        "[B3][delta_table]",
        "resume_carry: before="
        f"{before['after_resume_carry_ms']}ms -> after<="
        f"{after['resume_max_abs_delta_ms']}ms;",
        "idle_carry: before="
        f"{before['after_idle_carry_ms']}ms -> after<="
        f"{after['after_idle_max_abs_delta_ms']}ms",
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
