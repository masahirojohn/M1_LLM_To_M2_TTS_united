#!/usr/bin/env python3
"""Phase11 talkover test feeder: synth to CABLE + interrupt file write."""
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
            # Tee-Object may write UTF-16; also try utf-16-le
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
    ap.add_argument("--interrupt_file", required=True)
    ap.add_argument("--synth_device", type=int, default=16)
    ap.add_argument("--py", default=sys.executable)
    args = ap.parse_args()

    log = Path(args.log)
    intr = Path(args.interrupt_file)
    synth = str(Path(__file__).resolve().parents[1] / "tools" / "phase1_play_synth_speech_to_cable.py")
    py = args.py

    print("[feeder] wait mic_send BEGIN", flush=True)
    if not wait_log(log, "[mic_send][BEGIN]", 90.0, 1):
        print("[feeder][WARN] mic_send BEGIN timeout; synth anyway", flush=True)

    print("[feeder] synth turn1", flush=True)
    subprocess.run(
        [py, synth, "--device", str(args.synth_device), "--delay_s", "0.4", "--voice_s", "2.4", "--tail_silence_s", "1.0"],
        check=False,
    )

    print("[feeder] wait first_audio", flush=True)
    got_audio = wait_log(log, "turn_first_audio_detected", 55.0, 1)
    print(f"[feeder] first_audio={got_audio}", flush=True)

    print("[feeder] interrupt", flush=True)
    time.sleep(1.5)
    payload = {
        "type": "interrupt",
        "priority": "battle",
        "text": "今すぐ短く被せてツッコメ",
        "expire_sec": 60.0,
    }
    intr.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("[feeder] interrupt wrote", flush=True)
    time.sleep(3.0)
    intr.write_text("", encoding="utf-8")

    for i in range(3):
        need = i + 2  # 2nd, 3rd, 4th BEGIN
        print(f"[feeder] wait mic BEGIN count>={need}", flush=True)
        wait_log(log, "[mic_send][BEGIN]", 45.0, need)
        print(f"[feeder] synth followup {i}", flush=True)
        subprocess.run(
            [py, synth, "--device", str(args.synth_device), "--delay_s", "0.3", "--voice_s", "2.0", "--tail_silence_s", "1.0"],
            check=False,
        )

    print("[feeder] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
