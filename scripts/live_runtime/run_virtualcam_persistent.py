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


def _read_json_file(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8-sig").strip()
        if not raw:
            return None
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _resolve_ssot_target(
    *,
    playback_state: dict | None,
    sync_meta: dict | None,
    step_ms: int,
    frame_offset_cli: int,
) -> dict:
    """Map player played_samples → mouth/M0 audio_ms → target PNG frame index."""
    step = int(step_ms)
    if sync_meta is not None and int(sync_meta.get("step_ms", 0) or 0) > 0:
        step = int(sync_meta.get("step_ms") or step)

    frame_offset = int(frame_offset_cli)
    origin_ms = 0
    base_samples = 0
    if sync_meta is not None:
        frame_offset = int(sync_meta.get("frame_offset", frame_offset) or frame_offset)
        origin_ms = int(sync_meta.get("playback_origin_ms", 0) or 0)
        base_samples = int(sync_meta.get("base_played_samples", 0) or 0)

    state = "UNKNOWN"
    played_samples = 0
    player_local_ms = 0.0
    sample_rate = 24000
    if playback_state is not None:
        state = str(playback_state.get("state", "UNKNOWN") or "UNKNOWN")
        played_samples = int(playback_state.get("played_samples", 0) or 0)
        sample_rate = int(playback_state.get("sample_rate", 24000) or 24000)
        if sample_rate <= 0:
            sample_rate = 24000
        if "player_local_ms" in playback_state:
            player_local_ms = float(playback_state.get("player_local_ms") or 0.0)
        else:
            player_local_ms = float(played_samples) * 1000.0 / float(sample_rate)

    rel_samples = max(0, int(played_samples) - int(base_samples))
    audio_ms = int(rel_samples * 1000.0 / float(sample_rate))
    target_t_ms = int(origin_ms) + int(audio_ms)
    target_frame = int(frame_offset) + int(target_t_ms // max(1, step))

    return {
        "state": state,
        "played_samples": int(played_samples),
        "player_local_ms": float(player_local_ms),
        "audio_ms": int(audio_ms),
        "target_t_ms": int(target_t_ms),
        "target_frame": int(target_frame),
        "frame_offset": int(frame_offset),
        "step_ms": int(step),
        "base_played_samples": int(base_samples),
    }


def _fg_png_path(fg_dir: Path, frame_idx: int) -> Path:
    return fg_dir / f"{int(frame_idx):08d}.png"


def _find_latest_fg_at_or_before(
    *,
    fg_dir: Path,
    target_frame: int,
    hint_frame: int | None,
    max_scan: int = 256,
) -> tuple[Path, int] | None:
    """Latest existing PNG with index <= target_frame (never ahead of audio_ms).

    Phase7 Hotfix: avoid idle_hold freeze on missing exact target by catching up
    to the newest ready frame at/before audio. hint_frame accelerates the scan.
    """
    hi = int(target_frame)
    if hi < 0:
        return None

    # Fast path: previously displayed / known high-water still valid.
    if hint_frame is not None:
        hint = int(hint_frame)
        if 0 <= hint <= hi:
            p = _fg_png_path(fg_dir, hint)
            if p.exists():
                # Try to walk forward from hint toward target (small gap catch-up).
                best_i = hint
                best_p = p
                fwd_lim = min(hi, hint + max(1, int(max_scan)))
                for i in range(hint + 1, fwd_lim + 1):
                    cand = _fg_png_path(fg_dir, i)
                    if cand.exists():
                        best_i = i
                        best_p = cand
                    else:
                        break
                return best_p, int(best_i)

    lo = max(0, hi - max(0, int(max_scan)) + 1)
    for i in range(hi, lo - 1, -1):
        p = _fg_png_path(fg_dir, i)
        if p.exists():
            return p, int(i)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Persistent virtualcam: audio_ms SSOT FG select → OBS"
    )

    ap.add_argument("--fg_dir", required=True)
    ap.add_argument("--bg_video", required=True)
    ap.add_argument(
        "--bg_override_file",
        default=None,
        help="JSON file for temporary BG video override.",
    )
    ap.add_argument(
        "--playback_state_file",
        default=None,
        help="Player-published JSON with played_samples / player_local_ms.",
    )
    ap.add_argument(
        "--sync_meta_file",
        default=None,
        help="session_loop JSON: frame_offset / base_played_samples / step_ms.",
    )
    ap.add_argument(
        "--step_ms",
        type=int,
        default=40,
        help="Mouth/M0 frame step (ms). Must match session_loop --step_ms.",
    )
    ap.add_argument(
        "--frame_offset",
        type=int,
        default=0,
        help="Fallback frame_offset if sync_meta_file absent.",
    )

    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--poll_s", type=float, default=0.02)
    ap.add_argument("--idle_hold", action="store_true")
    ap.add_argument("--loop_bg", action="store_true")
    ap.add_argument(
        "--loop_fg",
        action="store_true",
        help="Legacy flag (ignored in audio_ms SSOT mode).",
    )

    args = ap.parse_args()

    fg_dir = Path(args.fg_dir).resolve()
    bg_video = Path(args.bg_video).resolve()
    normal_bg_video = bg_video
    bg_override_file = (
        Path(args.bg_override_file).resolve()
        if args.bg_override_file
        else None
    )
    playback_state_file = (
        Path(args.playback_state_file).resolve()
        if args.playback_state_file
        else None
    )
    sync_meta_file = (
        Path(args.sync_meta_file).resolve() if args.sync_meta_file else None
    )

    cap = _open_bg_capture(bg_video)

    override_until_t = 0.0
    override_active = False
    override_last_mtime = 0.0

    width = int(args.width)
    height = int(args.height)
    fps = int(args.fps)

    sent = 0
    last_rgb = None
    last_fg = None
    last_displayed_frame: int | None = None
    max_existing_frame: int | None = None
    last_logged_target = None
    ssot_enabled = playback_state_file is not None

    print("[virtualcam_persistent][START]", flush=True)
    print(f"  fg_dir  : {fg_dir}", flush=True)
    print(f"  bg_video: {bg_video}", flush=True)
    if bg_override_file is not None:
        print(f"  bg_override_file: {bg_override_file}", flush=True)
    print(
        "[virtualcam_persistent][ssot]",
        f"mode={'audio_ms' if ssot_enabled else 'idle_only_no_playback_state'}",
        f"playback_state_file={playback_state_file}",
        f"sync_meta_file={sync_meta_file}",
        f"step_ms={int(args.step_ms)}",
        "sequential_idx=disabled",
        flush=True,
    )

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

            ok, bg = cap.read()
            if not ok:
                if args.loop_bg:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, bg = cap.read()
                if not ok:
                    if override_active:
                        print(
                            "[virtualcam_persistent][bg_override_frame_warn] failed to read bg frame",
                            flush=True,
                        )
                        time.sleep(float(args.poll_s))
                        continue
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

            fg_path: Path | None = None
            target = None
            missing_png = False
            display_frame_idx: int | None = None

            if ssot_enabled:
                playback_state = _read_json_file(playback_state_file)
                sync_meta = _read_json_file(sync_meta_file)
                target = _resolve_ssot_target(
                    playback_state=playback_state,
                    sync_meta=sync_meta,
                    step_ms=int(args.step_ms),
                    frame_offset_cli=int(args.frame_offset),
                )
                # Follow audio only while actually playing; hold otherwise.
                if str(target["state"]) == "PLAYING":
                    target_i = int(target["target_frame"])
                    cand = _fg_png_path(fg_dir, target_i)
                    if cand.exists():
                        fg_path = cand
                        display_frame_idx = target_i
                        max_existing_frame = (
                            target_i
                            if max_existing_frame is None
                            else max(int(max_existing_frame), target_i)
                        )
                    else:
                        # Phase7 Hotfix: catch up to newest PNG <= audio target.
                        # Never select a frame ahead of audio_ms (Sync SSOT preserved).
                        missing_png = True
                        hint = max_existing_frame
                        if last_displayed_frame is not None:
                            hint = (
                                int(last_displayed_frame)
                                if hint is None
                                else max(int(hint), int(last_displayed_frame))
                            )
                        fb = _find_latest_fg_at_or_before(
                            fg_dir=fg_dir,
                            target_frame=target_i,
                            hint_frame=hint,
                        )
                        if fb is not None:
                            fg_path, fb_i = fb
                            display_frame_idx = int(fb_i)
                            max_existing_frame = (
                                int(fb_i)
                                if max_existing_frame is None
                                else max(int(max_existing_frame), int(fb_i))
                            )
                            log_key = (
                                int(target["audio_ms"]),
                                target_i,
                                "catchup",
                                int(fb_i),
                            )
                            if last_logged_target != log_key:
                                print(
                                    "[sync][virtualcam][SSOT_CATCHUP]",
                                    f"audio_ms={int(target['audio_ms'])}",
                                    f"player_local_ms={float(target['player_local_ms']):.1f}",
                                    f"target_frame={target_i}",
                                    f"fallback_frame={int(fb_i)}",
                                    f"displayed_frame={int(fb_i)}",
                                    f"state={target['state']}",
                                    flush=True,
                                )
                                last_logged_target = log_key
                        else:
                            log_key = (
                                int(target["audio_ms"]),
                                target_i,
                                "missing",
                            )
                            if last_logged_target != log_key:
                                print(
                                    "[sync][virtualcam][SSOT_WAIT]",
                                    f"audio_ms={int(target['audio_ms'])}",
                                    f"player_local_ms={float(target['player_local_ms']):.1f}",
                                    f"target_frame={target_i}",
                                    f"displayed_frame={last_displayed_frame}",
                                    f"state={target['state']}",
                                    flush=True,
                                )
                                last_logged_target = log_key

            if fg_path is None:
                if args.idle_hold and last_rgb is not None:
                    cam.send(last_rgb)
                    cam.sleep_until_next_frame()
                    sent += 1
                    if sent % 25 == 0:
                        print(f"[virtualcam_persistent] sent={sent}", flush=True)
                    continue
                time.sleep(float(args.poll_s))
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
            if target is not None:
                shown = (
                    int(display_frame_idx)
                    if display_frame_idx is not None
                    else int(target["target_frame"])
                )
                last_displayed_frame = int(shown)
                max_existing_frame = (
                    int(shown)
                    if max_existing_frame is None
                    else max(int(max_existing_frame), int(shown))
                )
                log_key = (
                    int(target["audio_ms"]),
                    int(target["target_frame"]),
                    "ok",
                    int(shown),
                )
                if last_logged_target != log_key and (
                    sent % 5 == 0 or missing_png or last_logged_target is None
                ):
                    print(
                        "[sync][virtualcam]",
                        f"audio_ms={int(target['audio_ms'])}",
                        f"player_local_ms={float(target['player_local_ms']):.1f}",
                        f"target_frame={int(target['target_frame'])}",
                        f"displayed_frame={int(shown)}",
                        f"frame_offset={int(target['frame_offset'])}",
                        f"step_ms={int(target['step_ms'])}",
                        f"state={target['state']}",
                        flush=True,
                    )
                    last_logged_target = log_key

            if sent % 25 == 0:
                print(f"[virtualcam_persistent] sent={sent}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
