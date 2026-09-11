#!/usr/bin/env python3
"""Offline self-check for VirtualCam SSOT catch-up Hotfix."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


def main() -> int:
    script = (
        Path(__file__).resolve().parents[1]
        / "live_runtime"
        / "run_virtualcam_persistent.py"
    )
    spec = importlib.util.spec_from_file_location("vcam_ssot", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with tempfile.TemporaryDirectory() as td:
        fg = Path(td)
        for i in (0, 1, 2, 10, 11, 44):
            (fg / f"{i:08d}.png").write_bytes(b"x")

        hit = mod._find_latest_fg_at_or_before(
            fg_dir=fg, target_frame=45, hint_frame=18
        )
        assert hit is not None and hit[1] == 44, hit

        hit2 = mod._find_latest_fg_at_or_before(
            fg_dir=fg, target_frame=11, hint_frame=10
        )
        assert hit2 is not None and hit2[1] == 11, hit2

        miss = mod._find_latest_fg_at_or_before(
            fg_dir=fg, target_frame=-1, hint_frame=None
        )
        assert miss is None

        # Never returns a frame ahead of target.
        hit3 = mod._find_latest_fg_at_or_before(
            fg_dir=fg, target_frame=3, hint_frame=0
        )
        assert hit3 is not None and hit3[1] == 2, hit3

        # Phase28: holes from parallel M0 must not freeze at the gap edge.
        for i in (100, 101, 105, 106, 110):
            (fg / f"{i:08d}.png").write_bytes(b"x")
        hit4 = mod._find_latest_fg_at_or_before(
            fg_dir=fg, target_frame=112, hint_frame=100
        )
        assert hit4 is not None and hit4[1] == 110, hit4

        # Phase28: defer adopting a newer frame_offset until new-turn FG exists.
        applied = {
            "frame_offset": 0,
            "base_played_samples": 0,
            "step_ms": 40,
            "playback_origin_ms": 0,
        }
        candidate = {
            "frame_offset": 210,
            "base_played_samples": 100000,
            "step_ms": 40,
            "playback_origin_ms": 0,
        }
        assert (
            mod._should_adopt_sync_meta(
                fg_dir=fg, applied=applied, candidate=candidate
            )
            is False
        )
        (fg / f"{210:08d}.png").write_bytes(b"x")
        assert (
            mod._should_adopt_sync_meta(
                fg_dir=fg, applied=applied, candidate=candidate
            )
            is True
        )

        # L2: future base (played+pending at first enqueue) must follow
        # player_local_ms, not clamp audio_ms to 0 at frame_offset.
        seam = mod._resolve_ssot_target(
            playback_state={
                "state": "PLAYING",
                "played_samples": 467232,
                "player_local_ms": 19468.0,
                "sample_rate": 24000,
            },
            sync_meta={
                "frame_offset": 567,
                "base_played_samples": 541455,
                "step_ms": 40,
                "playback_origin_ms": 0,
            },
            step_ms=40,
            frame_offset_cli=0,
        )
        assert int(seam["audio_ms"]) < 0, seam
        assert int(seam["target_frame"]) < 567, seam
        caught = mod._resolve_ssot_target(
            playback_state={
                "state": "PLAYING",
                "played_samples": 541455,
                "player_local_ms": 541455 * 1000.0 / 24000.0,
                "sample_rate": 24000,
            },
            sync_meta={
                "frame_offset": 567,
                "base_played_samples": 541455,
                "step_ms": 40,
                "playback_origin_ms": 0,
            },
            step_ms=40,
            frame_offset_cli=0,
        )
        assert int(caught["audio_ms"]) == 0, caught
        assert int(caught["target_frame"]) == 567, caught
        # Talkover-scale gap after player reaches base: same as old formula.
        after = mod._resolve_ssot_target(
            playback_state={
                "state": "PLAYING",
                "played_samples": 59520 + 4800,
                "player_local_ms": (59520 + 4800) * 1000.0 / 24000.0,
                "sample_rate": 24000,
            },
            sync_meta={
                "frame_offset": 66,
                "base_played_samples": 59520,
                "step_ms": 40,
                "playback_origin_ms": 0,
            },
            step_ms=40,
            frame_offset_cli=0,
        )
        assert int(after["audio_ms"]) == 200, after
        assert int(after["target_frame"]) == 71, after

    print("OK phase7_ssot_catchup_selfcheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
