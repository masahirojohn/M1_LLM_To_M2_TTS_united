from __future__ import annotations

import argparse
import numpy as np
import sounddevice as sd
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcm", required=True)
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--device", default=None)
    # 後で numeric string を int に変換
    args = ap.parse_args()

    pcm_path = Path(args.pcm).resolve()
    if not pcm_path.exists():
        raise FileNotFoundError(pcm_path)

    # PCM16 → float32
    pcm_bytes = pcm_path.read_bytes()
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    print("[audio] length_sec=", len(audio) / args.sr)

    device = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
           pass

    # 再生（デバイス指定）
    sd.play(audio, samplerate=args.sr, device=device)
    sd.wait()

    print("[audio][DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())