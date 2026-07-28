#!/usr/bin/env python3
"""Phase11 event test feeder: synth turns + event_runtime trigger."""
from __future__ import annotations

import argparse
import json
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
            time.sleep(0.25)
            continue
        if text.count(needle) >= min_count:
            return True
        time.sleep(0.25)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--event_file", required=True)
    ap.add_argument("--event_id", default="evt_001")
    ap.add_argument("--synth_device", type=int, default=16)
    ap.add_argument("--py", default=sys.executable)
    args = ap.parse_args()

    log = Path(args.log)
    event_file = Path(args.event_file)
    synth = str(Path(__file__).resolve().parents[1] / "tools" / "phase1_play_synth_speech_to_cable.py")
    py = args.py

    print("[feeder_event] wait mic BEGIN", flush=True)
    wait_log(log, "[mic_send][BEGIN]", 90.0, 1)
    print("[feeder_event] synth turn1", flush=True)
    subprocess.run(
        [py, synth, "--device", str(args.synth_device), "--delay_s", "0.4", "--voice_s", "2.2", "--tail_silence_s", "1.0"],
        check=False,
    )

    print("[feeder_event] wait first_audio", flush=True)
    wait_log(log, "turn_first_audio_detected", 55.0, 1)
    time.sleep(2.0)

    print("[feeder_event] trigger event", flush=True)
    event_file.write_text(
        json.dumps({"type": "event", "event_id": args.event_id}, ensure_ascii=False),
        encoding="utf-8",
    )

    # wait bg_override + restore evidence
    wait_log(log, "[event_runtime][bg_override_written]", 20.0, 1)
    wait_log(log, "[virtualcam_persistent][bg_override]", 20.0, 1)
    wait_log(log, "[virtualcam_persistent][bg_restore]", 20.0, 1)
    wait_log(log, "[event_runtime][mic_gate_open]", 20.0, 1)

    # one recovery turn
    print("[feeder_event] wait 2nd mic BEGIN", flush=True)
    wait_log(log, "[mic_send][BEGIN]", 45.0, 2)
    subprocess.run(
        [py, synth, "--device", str(args.synth_device), "--delay_s", "0.3", "--voice_s", "2.0", "--tail_silence_s", "1.0"],
        check=False,
    )
    print("[feeder_event] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
