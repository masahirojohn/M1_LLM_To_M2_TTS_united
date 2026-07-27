#!/usr/bin/env python3
"""Play Cable synth when session_loop opens each mic_send window (Phase10 load helper)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _count_mic_begin(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return text.count("[mic_send][BEGIN]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="session_loop log path being written")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--device", type=int, default=6)
    ap.add_argument("--voice_s", type=float, default=2.2)
    ap.add_argument("--tail_silence_s", type=float, default=1.5)
    ap.add_argument("--poll_s", type=float, default=0.25)
    ap.add_argument("--timeout_s", type=float, default=600.0)
    ap.add_argument(
        "--synth_tool",
        default=str(Path(__file__).resolve().parent / "phase1_play_synth_speech_to_cable.py"),
    )
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    log_path = Path(args.log)
    py = str(args.python)
    tool = str(args.synth_tool)
    seen = 0
    t0 = time.time()
    print(
        f"[phase10_follow] waiting turns={args.turns} log={log_path} voice_s={args.voice_s}",
        flush=True,
    )
    while seen < int(args.turns):
        if time.time() - t0 > float(args.timeout_s):
            print(f"[phase10_follow][TIMEOUT] seen={seen}/{args.turns}", flush=True)
            return 2
        n = _count_mic_begin(log_path)
        while n > seen and seen < int(args.turns):
            seen += 1
            # New mic window opened; small settle then play.
            time.sleep(0.4)
            print(
                f"[phase10_follow][FIRE] turn={seen} mic_begin_count={n}",
                flush=True,
            )
            subprocess.run(
                [
                    py,
                    tool,
                    "--device",
                    str(int(args.device)),
                    "--delay_s",
                    "0.05",
                    "--voice_s",
                    str(float(args.voice_s)),
                    "--tail_silence_s",
                    str(float(args.tail_silence_s)),
                ],
                check=False,
            )
        if seen >= int(args.turns):
            break
        time.sleep(float(args.poll_s))
    print(f"[phase10_follow][DONE] fired={seen}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
