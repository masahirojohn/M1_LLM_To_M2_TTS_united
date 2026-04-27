#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2


def start_ffmpeg(
    *,
    bg_video: Path,
    out_mp4: Path,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
):
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop", "-1",
        "-i", str(bg_video),
        "-f", "rawvideo",
        "-pix_fmt", "bgra",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-filter_complex",
        (
            f"[0:v]scale={width}:{height},format=rgba[bg];"
            "[1:v]format=rgba[fg];"
            "[bg][fg]overlay=0:0:format=auto:shortest=1,format=yuv420p[v]"
        ),
        "-map", "[v]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-frames:v", str(num_frames),
        "-shortest",
        str(out_mp4),
    ]

    print("[FFMPEG START]", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg_dir", required=True)
    ap.add_argument("--bg_video", required=True)
    ap.add_argument("--out_mp4", required=True)

    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--realtime_sleep", action="store_true")

    args = ap.parse_args()

    fg_dir = Path(args.fg_dir).resolve()
    bg_video = Path(args.bg_video).resolve()
    out_mp4 = Path(args.out_mp4).resolve()

    if not fg_dir.exists():
        raise FileNotFoundError(f"missing fg_dir: {fg_dir}")
    if not bg_video.exists():
        raise FileNotFoundError(f"missing bg_video: {bg_video}")

    pngs = sorted(fg_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"no png found: {fg_dir}")

    print(f"[INFO] frames: {len(pngs)}")

    ff = start_ffmpeg(
        bg_video=bg_video,
        out_mp4=out_mp4,
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_frames=len(pngs),
    )

    frame_s = 1.0 / float(args.fps)
    t0 = time.time()

    try:
        assert ff.stdin is not None

        for i, p in enumerate(pngs):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"failed to read png: {p}")
            if len(img.shape) != 3 or img.shape[2] != 4:
                raise RuntimeError(f"not BGRA png: {p} shape={img.shape}")

            if img.shape[1] != args.width or img.shape[0] != args.height:
                img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

            ff.stdin.write(img.tobytes())

            if i % 10 == 0:
                print(f"[PIPE_BG] {i}")

            if args.realtime_sleep:
                target = t0 + ((i + 1) * frame_s)
                sleep_s = target - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)

    finally:
        if ff.stdin:
            ff.stdin.close()
        rc = ff.wait()

    if rc != 0:
        raise RuntimeError(f"ffmpeg failed rc={rc}")

    print(f"[OK] realtime BG MP4: {out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())