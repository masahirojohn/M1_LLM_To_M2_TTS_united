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

    print("OK phase7_ssot_catchup_selfcheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
