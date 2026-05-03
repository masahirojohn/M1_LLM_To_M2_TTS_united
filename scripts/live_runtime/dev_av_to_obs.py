from __future__ import annotations

import argparse
import threading
from pathlib import Path

import cv2
import numpy as np
import pyvirtualcam
import sounddevice as sd


def play_audio(pcm_path: Path, sr: int, device):
    pcm = pcm_path.read_bytes()
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    sd.play(audio, samplerate=sr, device=device)
    sd.wait()


def read_fg(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        raise RuntimeError(f"invalid FG png: {path}")
    return img


def overlay(bg_bgr: np.ndarray, fg_bgra: np.ndarray) -> np.ndarray:
    h, w = bg_bgr.shape[:2]
    if fg_bgra.shape[:2] != (h, w):
        fg_bgra = cv2.resize(fg_bgra, (w, h))

    alpha = fg_bgra[:, :, 3:4].astype(np.float32) / 255.0
    fg = fg_bgra[:, :, :3].astype(np.float32)
    bg = bg_bgr.astype(np.float32)

    return np.clip(fg * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fg_dir", required=True)
    ap.add_argument("--bg_video", required=True)
    ap.add_argument("--pcm", required=True)
    ap.add_argument("--audio_sr", type=int, default=24000)
    ap.add_argument("--audio_device", default="15")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    fg_dir = Path(args.fg_dir).resolve()
    bg_video = Path(args.bg_video).resolve()
    pcm_path = Path(args.pcm).resolve()

    audio_device = int(args.audio_device)

    pngs = sorted(fg_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"no FG pngs: {fg_dir}")

    cap = cv2.VideoCapture(str(bg_video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open bg_video: {bg_video}")

    audio_thread = threading.Thread(
        target=play_audio,
        args=(pcm_path, args.audio_sr, audio_device),
        daemon=True,
    )

    width = args.width
    height = args.height
    fps = args.fps

    with pyvirtualcam.Camera(
        width=width,
        height=height,
        fps=fps,
        backend="unitycapture",
    ) as cam:
        print(f"[obs_av][OK] virtualcam={cam.device}")
        print(f"[obs_av] audio_device={audio_device}")
        print(f"[obs_av] frames={len(pngs)}")

        audio_thread.start()

        for i, fg_path in enumerate(pngs):
            ok, bg = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, bg = cap.read()
                if not ok:
                    raise RuntimeError("failed to read bg frame")

            bg = cv2.resize(bg, (width, height))
            fg = read_fg(fg_path)
            comp = overlay(bg, fg)

            rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
            cam.send(rgb)
            cam.sleep_until_next_frame()

            if i % 25 == 0:
                print(f"[obs_av] sent={i}")

        audio_thread.join(timeout=2.0)

    cap.release()
    print("[obs_av][DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())