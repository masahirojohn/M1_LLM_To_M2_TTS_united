#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _open_bg_capture(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"missing bg_video: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open bg_video: {path}")

    return cap


def _read_bg_override(path: Path | None, last_mtime: float) -> tuple[dict | None, float]:
    if path is None:
        return None, last_mtime

    if not path.exists():
        return None, last_mtime

    stat = path.stat()
    mtime = float(stat.st_mtime)

    if mtime <= last_mtime:
        return None, last_mtime

    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        return None, mtime

    obj = json.loads(raw)
    if not isinstance(obj, dict):
        return None, mtime

    if str(obj.get("type", "")).strip() != "bg_override":
        return None, mtime

    return obj, mtime


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
    ap.add_argument(
        "--bg_override_file",
        default=None,
        help="JSON file for temporary BG video override.",
    )

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
    normal_bg_video = bg_video
    bg_override_file = (
        Path(args.bg_override_file).resolve()
        if args.bg_override_file
        else None
    )

    cap = _open_bg_capture(bg_video)

    override_until_t = 0.0
    override_active = False
    override_last_mtime = 0.0

    width = int(args.width)
    height = int(args.height)
    fps = int(args.fps)

    idx = 0
    sent = 0
    last_rgb = None
    last_fg = None

    print("[virtualcam_persistent][START]", flush=True)
    print(f"  fg_dir  : {fg_dir}", flush=True)
    print(f"  bg_video: {bg_video}", flush=True)
    if bg_override_file is not None:
        print(f"  bg_override_file: {bg_override_file}", flush=True)

    with pyvirtualcam.Camera(
        width=width,
        height=height,
        fps=fps,
        backend="unitycapture",
    ) as cam:
        print(f"[virtualcam_persistent][OK] device={cam.device}", flush=True)

        while True:
            now_t = time.monotonic()

            if override_active and now_t >= override_until_t:
                try:
                    cap.release()
                except Exception:
                    pass

                bg_video = normal_bg_video
                cap = _open_bg_capture(bg_video)
                override_active = False

                print(
                    "[virtualcam_persistent][bg_restore]",
                    f"bg_video={bg_video}",
                    flush=True,
                )

            try:
                override_obj, override_last_mtime = _read_bg_override(
                    bg_override_file,
                    override_last_mtime,
                )
            except Exception as e:
                override_obj = None
                print(
                    f"[virtualcam_persistent][bg_override_warn] {type(e).__name__}: {e}",
                    flush=True,
                )

            if override_obj:
                next_bg = Path(str(override_obj.get("bg_video", ""))).resolve()
                duration_s = float(override_obj.get("duration_s", 0.0) or 0.0)

                try:
                    next_cap = _open_bg_capture(next_bg)

                    try:
                        cap.release()
                    except Exception:
                        pass

                    cap = next_cap
                    bg_video = next_bg
                    override_active = duration_s > 0
                    override_until_t = time.monotonic() + max(0.0, duration_s)

                    print(
                        "[virtualcam_persistent][bg_override]",
                        f"bg_video={bg_video}",
                        f"duration_s={duration_s:.3f}",
                        flush=True,
                    )

                    if bg_override_file is not None:
                        bg_override_file.write_text("", encoding="utf-8")

                except Exception as e:
                    print(
                        f"[virtualcam_persistent][bg_override_error] {type(e).__name__}: {e}",
                        flush=True,
                    )

            pngs = _get_pngs(fg_dir)

            # --- 修正箇所：置換ブロック ---
            if idx >= len(pngs):
                if override_active:
                    ok, bg = cap.read()
                    if not ok:
                        if args.loop_bg:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ok, bg = cap.read()
                        if not ok:
                            print("[virtualcam_persistent][bg_override_frame_warn] failed to read bg frame", flush=True)
                            time.sleep(float(args.poll_s))
                            continue

                    bg = cv2.resize(bg, (width, height), interpolation=cv2.INTER_LINEAR)
                    comp_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

                    cam.send(comp_rgb)
                    cam.sleep_until_next_frame()

                    last_rgb = comp_rgb
                    sent += 1

                    if sent % 25 == 0:
                        print(f"[virtualcam_persistent] sent={sent}", flush=True)

                    continue

                elif args.loop_fg and len(pngs) > 0:
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

            if override_active:
                comp_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                cam.send(comp_rgb)
                cam.sleep_until_next_frame()

                last_rgb = comp_rgb
                sent += 1

                if sent % 25 == 0:
                    print(f"[virtualcam_persistent] sent={sent}", flush=True)

                continue

            fg = None
            last_err = None

            for _ in range(5):
                try:
                    fg = _read_bgra_png(fg_path)
                    break
                except RuntimeError as e:
                    last_err = e
                    time.sleep(0.02)

            if fg is None:
                if last_fg is not None:
                    fg = last_fg
                    print(
                        f"[virtualcam_persistent][WARN] failed to read fg, using last_fg: {fg_path} err={last_err}",
                        flush=True,
                    )
                elif last_rgb is not None:
                    cam.send(last_rgb)
                    cam.sleep_until_next_frame()
                    print(
                        f"[virtualcam_persistent][WARN] failed to read fg, using last_rgb: {fg_path} err={last_err}",
                        flush=True,
                    )
                    continue
                else:
                    print(
                        f"[virtualcam_persistent][WARN] failed to read fg, skip frame: {fg_path} err={last_err}",
                        flush=True,
                    )
                    time.sleep(float(args.poll_s))
                    continue
            else:
                last_fg = fg

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