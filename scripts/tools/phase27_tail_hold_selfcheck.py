#!/usr/bin/env python3
"""Offline selfcheck for Phase27 turn-end mouth_tail hold-extend arithmetic."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "live_runtime"))

from run_mic_input_obs_realtime_session_loop import (  # noqa: E402
    _tail_hold_extend_live_mouth,
)
from run_mic_input_obs_realtime_step1 import _as_frames  # noqa: E402


def _case(last_t: int, until: int, step: int = 40) -> None:
    mouth_ref: dict = {
        "obj": {
            "frames": [
                {"t_ms": t, "mouth_id": 3}
                for t in range(0, last_t + 1, step)
            ]
        }
    }
    before = len(_as_frames(mouth_ref["obj"]))
    mouth_cov = last_t + step
    assert mouth_cov < until, (mouth_cov, until)
    added = _tail_hold_extend_live_mouth(
        mouth_ref, until_t1_ms=until, step_ms=step
    )
    assert added > 0, (last_t, until, added)
    fr = _as_frames(mouth_ref["obj"])
    assert len(fr) == before + added
    new_last = int(fr[-1]["t_ms"])
    new_cov = new_last + step
    assert new_cov >= until, (new_last, new_cov, until)
    assert all(f.get("src") == "mouth_hold_extend" for f in fr[-added:])


def main() -> int:
    # Before multi 140224 fingerprints (gap 10–20ms, mouth_ready stuck at 0).
    _case(last_t=5080, until=5130)
    _case(last_t=8320, until=8370)
    _case(last_t=8760, until=8820)
    _case(last_t=10760, until=10810)
    _case(last_t=7440, until=7500)
    # No-op when mouth already reaches until+step exclusive bound.
    mouth_ref = {"obj": {"frames": [{"t_ms": 140, "mouth_id": 1}]}}
    assert _tail_hold_extend_live_mouth(mouth_ref, until_t1_ms=140, step_ms=40) == 0
    print("PASS phase27 tail hold-extend arithmetic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
