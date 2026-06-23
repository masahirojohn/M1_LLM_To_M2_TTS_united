#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--pcm", required=True)
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--device", default=None)

    ap.add_argument("--chunk_ms", type=int, default=400)
    ap.add_argument("--start_chunk", type=int, default=0)

    args = ap.parse_args()

    pcm_path = Path(args.pcm).resolve()
    if not pcm_path.exists():
        raise FileNotFoundError(pcm_path)

    raw = pcm_path.read_bytes()

    audio = np.frombuffer(raw, dtype=np.int16)

    sr = int(args.sr)
    chunk_ms = int(args.chunk_ms)

    samples_per_chunk = int(sr * chunk_ms / 1000)

    start = int(args.start_chunk) * samples_per_chunk
    end = min(len(audio), start + samples_per_chunk)

    chunk_audio = audio[start:end]

    if len(chunk_audio) == 0:
        print("[pcm_chunk_player] empty chunk")
        return 0



    device = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass

    print(
        f"[pcm_chunk_player] "
        f"chunk={args.start_chunk} "
        f"samples={len(chunk_audio)} "
        f"device={device}",
        flush=True,
    )

    sd.play(chunk_audio, samplerate=sr, device=device, blocking=True)


    return 0


if __name__ == "__main__":
    raise SystemExit(main())