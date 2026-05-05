#!/usr/bin/env python3
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


def _overlay(bg_bgr: np.ndarray, fg_bgra: np.ndarray) -> np.ndarray:
    h, w = bg_bgr.shape[:2]

    if fg_bgra.shape[:2] != (h, w):
        fg_bgra = cv2.resize(fg_bgra, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha = fg_bgra[:, :, 3:4].astype(np.float32) / 255.0
    fg = fg_bgra[:, :, :3].astype(np.float32)
    bg = bg_bgr.astype(np.float32)

    return np.clip(fg * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)


def _get_pngs(fg_dir: Path) -> list[Path]:
    if not fg_dir.exists():
        return []
    return sorted(fg_dir.glob("*.png"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Persistent virtualcam: watch FG PNG dir and stream to OBS"
    )

    ap.add_argument("--fg_dir", required=True)
    ap.add_argument("--bg_video", required=True)

    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--poll_s", type=float, default=0.02)
    ap.add_argument("--idle_hold", action="store_true")
    ap.add_argument("--loop_bg", action="store_true")
    ap.add_argument("--loop_fg", action="store_true")

    args = ap.parse_args()

    fg_dir = Path(args.fg_dir).resolve()
    bg_video = Path(args.bg_video).resolve()

    if not bg_video.exists():
        raise FileNotFoundError(f"missing bg_video: {bg_video}")

    cap = cv2.VideoCapture(str(bg_video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open bg_video: {bg_video}")

    width = int(args.width)
    height = int(args.height)
    fps = int(args.fps)

    idx = 0
    sent = 0
    last_rgb = None

    print("[virtualcam_persistent][START]", flush=True)
    print(f"  fg_dir  : {fg_dir}", flush=True)
    print(f"  bg_video: {bg_video}", flush=True)

    with pyvirtualcam.Camera(
        width=width,
        height=height,
        fps=fps,
        backend="unitycapture",
    ) as cam:
        print(f"[virtualcam_persistent][OK] device={cam.device}", flush=True)

        while True:
            pngs = _get_pngs(fg_dir)

            # --- 修正箇所：置換ブロック ---
            if idx >= len(pngs):
                if args.loop_fg and len(pngs) > 0:
                    idx = 0
                    continue
                elif args.idle_hold and last_rgb is not None:
                    cam.send(last_rgb)
                    cam.sleep_until_next_frame()
                    continue
                else:
                    time.sleep(float(args.poll_s))
                    continue
            # ------------------------------

            fg_path = pngs[idx]

            ok, bg = cap.read()
            if not ok:
                if args.loop_bg:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, bg = cap.read()
                if not ok:
                    raise RuntimeError("failed to read bg frame")

            bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_LINEAR)
            fg = _read_bgra_png(fg_path)

            comp_bgr = _overlay(bg, fg)
            comp_rgb = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)

            cam.send(comp_rgb)
            cam.sleep_until_next_frame()

            last_rgb = comp_rgb
            sent += 1
            idx += 1

            if sent % 25 == 0:
                print(f"[virtualcam_persistent] sent={sent}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())