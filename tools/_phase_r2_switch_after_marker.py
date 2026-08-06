#!/usr/bin/env python3
"""Wait for a log marker, then write vad_profile_live.txt (Phase R2 switch helper)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--value", type=int, default=250)
    ap.add_argument("--marker", default="[resp_timing] activity_end")
    ap.add_argument("--timeout_s", type=float, default=180.0)
    ap.add_argument("--delay_s", type=float, default=0.3)
    args = ap.parse_args()

    log_path = Path(args.log)
    profile_path = Path(args.profile)
    marker = str(args.marker)
    deadline = time.time() + float(args.timeout_s)

    while time.time() < deadline:
        if log_path.is_file():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if marker in text:
                time.sleep(float(args.delay_s))
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                profile_path.write_text(f"{int(args.value)}\n", encoding="utf-8")
                print(
                    f"[phase_r2_switch] wrote {int(args.value)} after marker={marker!r}",
                    flush=True,
                )
                return 0
        time.sleep(0.2)

    print(f"[phase_r2_switch] TIMEOUT waiting for marker={marker!r}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
