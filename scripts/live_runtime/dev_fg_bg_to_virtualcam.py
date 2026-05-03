from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyvirtualcam


def _read_bgra_png(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to read png: {path}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise RuntimeError(f"FG must be BGRA png: {path} shape={img.shape}")
    return img


def _alpha_overlay_bgra_on_bgr(bg_bgr: np.ndarray, fg_bgra: np.ndarray) -> np.ndarray:
    h, w = bg_bgr.shape[:2]

    if fg_bgra.shape[1] != w or fg_bgra.shape[0] != h:
        fg_bgra = cv2.resize(fg_bgra, (w, h), interpolation=cv2.INTER_LINEAR)

    fg_bgr = fg_bgra[:, :, :3].astype(np.float32)
    alpha = fg_bgra[:, :, 3:4].astype(np.float32) / 255.0

    bg = bg_bgr.astype(np.float32)
    out = fg_bgr * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _get_fg_pngs(fg_dir: Path) -> list[Path]:
    return sorted(fg_dir.glob("*.png"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="M3.5-like FG+BG composite -> virtual camera"
    )

    ap.add_argument("--fg_dir", required=True)
    ap.add_argument("--bg_video", required=True)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--start_timeout_s", type=float, default=10.0)
    ap.add_argument("--frame_timeout_s", type=float, default=2.0)

    args = ap.parse_args()

    fg_dir = Path(args.fg_dir).resolve()
    bg_video = Path(args.bg_video).resolve()

    if not fg_dir.exists():
        raise FileNotFoundError(f"missing fg_dir: {fg_dir}")
    if not bg_video.exists():
        raise FileNotFoundError(f"missing bg_video: {bg_video}")

    cap = cv2.VideoCapture(str(bg_video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open bg_video: {bg_video}")

    width = int(args.width)
    height = int(args.height)
    fps = int(args.fps)

    sent = 0
    idx = 0
    t_start = time.time()

    with pyvirtualcam.Camera(width=width, height=height, fps=fps, backend="unitycapture") as cam:
        print(f"[virtualcam][OK] device={cam.device} {width}x{height}@{fps}")
        print(f"[virtualcam] fg_dir={fg_dir}")
        print(f"[virtualcam] bg_video={bg_video}")

        while True:
            pngs = _get_fg_pngs(fg_dir)

            if idx >= len(pngs):
                if args.follow:
                    if sent == 0 and (time.time() - t_start) > args.start_timeout_s:
                        raise RuntimeError("timeout waiting for first FG png")
                    time.sleep(0.02)
                    continue

                if args.loop:
                    idx = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                break

            fg_path = pngs[idx]

            ok, bg_bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bg_bgr = cap.read()
                if not ok:
                    raise RuntimeError("failed to read bg frame")

            bg_bgr = cv2.resize(bg_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
            fg_bgra = _read_bgra_png(fg_path)

            comp_bgr = _alpha_overlay_bgra_on_bgr(bg_bgr, fg_bgra)
            comp_rgb = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)

            cam.send(comp_rgb)
            cam.sleep_until_next_frame()

            sent += 1
            idx += 1

            if sent % 25 == 0:
                print(f"[virtualcam] sent={sent}", flush=True)

    cap.release()
    print(f"[virtualcam][DONE] sent={sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())