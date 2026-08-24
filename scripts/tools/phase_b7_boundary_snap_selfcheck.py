#!/usr/bin/env python3
"""Phase B7 Method C: boundary snap only (no devices).

After: idle enter / enter_playing / turn RELOCK rewind display BG to pose.
Steady PLAYING stays on B3 audio lock. Mid-idle sequential still advances.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "live_runtime"))

from run_virtualcam_persistent import (  # noqa: E402
    _b3_desired_bg_frame,
    _b7_boundary_reason,
    _b7_relock_bg_frame,
    _b7_snap_target,
)


def main() -> int:
    assert _b7_boundary_reason(relock_reason="enter_playing", idle_enter=False) == (
        "enter_playing"
    )
    assert _b7_boundary_reason(relock_reason="turn", idle_enter=False) == "turn"
    assert _b7_boundary_reason(relock_reason=None, idle_enter=True) == "idle_enter"
    assert _b7_boundary_reason(relock_reason=None, idle_enter=False) is None
    assert _b7_boundary_reason(relock_reason="other", idle_enter=False) is None

    # Steady PLAYING: no snap even if Δ is large (B3 stays in charge).
    assert (
        _b7_snap_target(reason=None, pose_idx=10, display_bg_idx=400) is None
    )
    # Mid-idle is not a boundary.
    assert (
        _b7_snap_target(reason=None, pose_idx=10, display_bg_idx=80) is None
    )
    # Boundary + BG ahead → rewind to pose.
    assert (
        _b7_snap_target(reason="idle_enter", pose_idx=10, display_bg_idx=80) == 10
    )
    assert (
        _b7_snap_target(reason="enter_playing", pose_idx=10, display_bg_idx=80)
        == 10
    )
    assert _b7_snap_target(reason="turn", pose_idx=200, display_bg_idx=450) == 200
    # Already aligned: no seek.
    assert (
        _b7_snap_target(reason="idle_enter", pose_idx=10, display_bg_idx=10)
        is None
    )
    assert _b7_snap_target(reason="turn", pose_idx=None, display_bg_idx=80) is None

    assert _b7_relock_bg_frame(pose_idx=12, fallback_bg=400) == 12
    assert _b7_relock_bg_frame(pose_idx=None, fallback_bg=400) == 400
    assert _b7_relock_bg_frame(pose_idx=-1, fallback_bg=7) == 7

    period = 40.0
    # Idle enter snap, then IDLE_BG_ADVANCE continues (not sucked every tick).
    pose = 20
    bg = 80
    snap = _b7_snap_target(reason="idle_enter", pose_idx=pose, display_bg_idx=bg)
    assert snap == 20
    bg = int(snap)
    idle_seq = [bg]
    for _ in range(8):
        assert (
            _b7_snap_target(reason=None, pose_idx=pose, display_bg_idx=idle_seq[-1])
            is None
        )
        idle_seq.append(idle_seq[-1] + 1)
    assert idle_seq[-1] == 28
    assert idle_seq[-1] != pose

    # enter_playing / turn: lock to pose, then B3 follows audio.
    lock_a = 0
    lock_bg = _b7_relock_bg_frame(pose_idx=pose, fallback_bg=idle_seq[-1])
    assert lock_bg == pose
    playing = []
    for i in range(6):
        a_ms = i * 40
        desired = _b3_desired_bg_frame(
            audio_ms=a_ms,
            lock_audio_ms=lock_a,
            lock_bg_frame=lock_bg,
            frame_period_ms=period,
        )
        # Steady PLAYING must not snap (would disable B3).
        assert (
            _b7_snap_target(reason=None, pose_idx=pose + i, display_bg_idx=desired)
            is None
        )
        playing.append(int(desired))
    assert playing == [20, 21, 22, 23, 24, 25]

    # fo↑ during REB/idle is ターン開始 (snap even before enter_playing).
    assert _b7_boundary_reason(relock_reason="turn", idle_enter=True) == "turn"
    reb_bg = 1497
    new_pose = 0
    assert (
        _b7_snap_target(reason="turn", pose_idx=new_pose, display_bg_idx=reb_bg)
        == 0
    )

    # turn RELOCK: do not keep the advanced idle/wall bg.
    turn_lock = _b7_relock_bg_frame(pose_idx=100, fallback_bg=500)
    assert turn_lock == 100
    d0 = _b3_desired_bg_frame(
        audio_ms=0,
        lock_audio_ms=0,
        lock_bg_frame=turn_lock,
        frame_period_ms=period,
    )
    assert d0 == 100

    print("[B7][selfcheck] PASS")
    print(
        "[B7][delta_table]",
        "idle_enter snap 80→20 then seq +8 (IDLE_BG_ADVANCE);",
        "enter_playing lock=pose B3 20..25;",
        "turn lock=100 not 500",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
