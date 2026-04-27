#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import cv2

"""
Realtime PIPE version v1

FG PNG → ffmpeg stdin → MP4

・M3.5は使わない（重要）
・BGなし（次フェーズで追加）
・リアルタイム動画生成の基礎
"""

def start_ffmpeg(out_mp4: Path, width: int, height: int, fps: int):
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgra",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",  # stdin
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]

    print("[FFMPEG START]")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--session_id", required=True)
    ap.add_argument("--fg_dir", required=True)
    ap.add_argument("--out_mp4", required=True)

    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)

    args = ap.parse_args()

    fg_dir = Path(args.fg_dir)
    out_mp4 = Path(args.out_mp4)

    pngs = sorted(fg_dir.glob("*.png"))

    if not pngs:
        raise RuntimeError("no FG png found")

    print("[INFO] frames:", len(pngs))

    ff = start_ffmpeg(out_mp4, args.width, args.height, args.fps)

    try:
        for i, p in enumerate(pngs):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

            if img is None:
                print("[WARN] skip:", p)
                continue

            if img.shape[2] != 4:
                raise RuntimeError("not BGRA image")

            ff.stdin.write(img.tobytes())

            if i % 10 == 0:
                print("[PIPE]", i)

    finally:
        ff.stdin.close()
        ff.wait()

    print("[OK] realtime MP4:", out_mp4)


if __name__ == "__main__":
    main()