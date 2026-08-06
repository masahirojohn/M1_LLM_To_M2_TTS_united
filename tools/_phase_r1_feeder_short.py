#!/usr/bin/env python3
"""Phase R1 short-utterance feeder (hesitation-ish: short voice + mid silence)."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def wait_log(path: Path, needle: str, timeout_s: float, min_count: int = 1) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if "\x00" in text[:200]:
                text = path.read_text(encoding="utf-16", errors="replace")
        except Exception:
            time.sleep(0.2)
            continue
        if text.count(needle) >= min_count:
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--synth_device", type=int, default=6)
    ap.add_argument("--py", default=sys.executable)
    args = ap.parse_args()

    log = Path(args.log)
    synth = str(
        Path(__file__).resolve().parents[1]
        / "tools"
        / "phase1_play_synth_speech_to_cable.py"
    )
    py = args.py

    # turn1: very short; turn2: short + longer tail (言いよどみ後続の余地)
    plans = [
        ("0.55", "0.9"),
        ("0.70", "1.1"),
    ][: int(args.turns)]

    for i, (voice_s, tail_s) in enumerate(plans):
        need = i + 1
        print(f"[feeder_short] wait mic BEGIN count>={need}", flush=True)
        ok = wait_log(log, "[mic_send][BEGIN]", 90.0, need)
        print(f"[feeder_short] begin_ok={ok}", flush=True)
        subprocess.run(
            [
                py,
                synth,
                "--device",
                str(args.synth_device),
                "--delay_s",
                "0.35",
                "--voice_s",
                voice_s,
                "--tail_silence_s",
                tail_s,
            ],
            check=False,
        )
        got = wait_log(log, "turn_first_audio_detected", 55.0, need)
        print(f"[feeder_short] first_audio={got} turn={need}", flush=True)
        time.sleep(1.0)

    print("[feeder_short] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
