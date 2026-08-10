#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
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


def _read_bgra_raw(path: Path) -> np.ndarray:
    """Phase 9 raw FG: magic 'M0BG' + uint32 w/h LE + BGRA bytes."""
    data = Path(path).read_bytes()
    if len(data) < 12:
        raise RuntimeError(f"fg bgra too short: {path}")
    magic, w, h = struct.unpack_from("<4sII", data, 0)
    if magic != b"M0BG":
        raise RuntimeError(f"fg bgra bad magic: {path} magic={magic!r}")
    need = 12 + int(w) * int(h) * 4
    if len(data) < need:
        raise RuntimeError(f"fg bgra truncated: {path} have={len(data)} need={need}")
    arr = np.frombuffer(data, dtype=np.uint8, offset=12, count=int(w) * int(h) * 4)
    return arr.reshape((int(h), int(w), 4)).copy()


def _read_fg_frame(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".bgra":
        return _read_bgra_raw(path)
    return _read_bgra_png(path)


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


def _bg_frame_count(cap) -> int:
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    except Exception:
        n = 0
    return n if n > 0 else 0


def _read_bg_sequential(cap, *, loop_bg: bool) -> tuple[bool, object, int]:
    """Wall-pace sequential decode (IDLE_BG_ADVANCE / non-PLAYING)."""
    ok, bg = cap.read()
    if not ok and loop_bg:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, bg = cap.read()
    if not ok:
        return False, None, -1
    try:
        # POS_FRAMES is next-to-read; last delivered ≈ max(0, pos-1).
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        idx = max(0, pos - 1)
    except Exception:
        idx = -1
    return True, bg, idx


def _bg_delivered_pos(cap) -> int:
    """POS_FRAMES is next-to-read; last delivered ≈ max(0, pos-1)."""
    try:
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        return max(0, pos - 1)
    except Exception:
        return -1


def _read_bg_at_frame(
    cap,
    frame_idx: int,
    *,
    loop_bg: bool,
    total: int,
    seek_fail_log: bool = True,
) -> tuple[bool, object, int]:
    """Seek+read one BG frame (PLAYING audio lock). loop_bg wraps when total>0.

    Verifies delivered POS after seek; on miss, one retry then accept actual
    frame (log [B3_BG_SEEK_FAIL]) so bg_pos cannot silently stick on request idx.
    """
    idx = max(0, int(frame_idx))
    if total > 0:
        if loop_bg:
            idx = int(idx % total)
        else:
            idx = min(idx, total - 1)

    def _seek_read(want: int) -> tuple[bool, object, int]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(want))
        ok_i, bg_i = cap.read()
        if not ok_i and loop_bg:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_i, bg_i = cap.read()
            if ok_i:
                return True, bg_i, 0
        if not ok_i:
            return False, None, -1
        actual = _bg_delivered_pos(cap)
        return True, bg_i, int(actual if actual >= 0 else want)

    ok, bg, actual = _seek_read(idx)
    if not ok:
        return False, None, -1
    # Allow ±1 frame decoder slack; larger gap ⇒ retry once.
    if total > 0 and abs(int(actual) - int(idx)) > 1:
        ok2, bg2, actual2 = _seek_read(idx)
        if ok2 and abs(int(actual2) - int(idx)) <= 1:
            return True, bg2, int(idx)
        if seek_fail_log:
            print(
                "[sync][virtualcam][B3_BG_SEEK_FAIL]",
                f"want={int(idx)}",
                f"got={int(actual2 if ok2 else actual)}",
                f"total={int(total)}",
                flush=True,
            )
        if ok2:
            return True, bg2, int(actual2)
        return True, bg, int(actual)
    # Prefer requested idx when within slack (stable bg_pos series).
    return True, bg, int(idx)


def _b3_desired_bg_frame(
    *,
    audio_ms: int,
    lock_audio_ms: int,
    lock_bg_frame: int,
    frame_period_ms: float,
) -> int:
    """Map played audio elapsed since lock → BG frame index (relative lock)."""
    period = float(frame_period_ms) if float(frame_period_ms) > 0 else 40.0
    elapsed = int(audio_ms) - int(lock_audio_ms)
    advance = int(round(float(elapsed) / period))
    return int(lock_bg_frame) + int(advance)


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


def _read_json_file_retry(path: Path | None, *, attempts: int = 3) -> dict | None:
    """Read JSON with short retries (Windows replace→PermissionError→truncate race)."""
    last = None
    n = max(1, int(attempts))
    for i in range(n):
        last = _read_json_file(path)
        if last is not None:
            return last
        if i + 1 < n:
            # ~2ms: enough for publisher fallback write to finish; not a pace sleep.
            time.sleep(0.002)
    return last


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


def _fg_frame_path(fg_dir: Path, frame_idx: int) -> Path | None:
    """Prefer Phase9 raw .bgra, fall back to .png (disk FG boundary)."""
    idx = int(frame_idx)
    bgra = fg_dir / f"{idx:08d}.bgra"
    if bgra.exists():
        return bgra
    png = fg_dir / f"{idx:08d}.png"
    if png.exists():
        return png
    return None


def _find_latest_fg_at_or_before(
    *,
    fg_dir: Path,
    target_frame: int,
    hint_frame: int | None,
    max_scan: int = 256,
) -> tuple[Path, int] | None:
    """Latest existing FG (.bgra/.png) with index <= target_frame (never ahead of audio_ms).

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
            p = _fg_frame_path(fg_dir, hint)
            if p is not None:
                # Walk forward from hint toward target. Skip holes: parallel M0 can
                # write later frames before earlier ones; breaking on the first gap
                # freezes display on a stale frame while audio_ms advances (Phase28).
                best_i = hint
                best_p = p
                fwd_lim = min(hi, hint + max(1, int(max_scan)))
                for i in range(hint + 1, fwd_lim + 1):
                    cand = _fg_frame_path(fg_dir, i)
                    if cand is not None:
                        best_i = i
                        best_p = cand
                return best_p, int(best_i)

    lo = max(0, hi - max(0, int(max_scan)) + 1)
    for i in range(hi, lo - 1, -1):
        p = _fg_frame_path(fg_dir, i)
        if p is not None:
            return p, int(i)
    return None


def _fg_ready_at_or_after(
    *,
    fg_dir: Path,
    frame_idx: int,
    max_scan: int = 8,
) -> bool:
    """True if any FG exists at frame_idx .. frame_idx+max_scan-1."""
    start = max(0, int(frame_idx))
    for i in range(start, start + max(1, int(max_scan))):
        if _fg_frame_path(fg_dir, i) is not None:
            return True
    return False


def _b2_obs_maybe_log(
    *,
    state: str | None,
    audio_ms: int | None,
    frame_offset: int | None,
    bg_read_n: int,
    sent: int,
    frame_period_ms: float,
    force: bool,
    anchor_bg_read: int | None,
    anchor_audio_ms: int | None,
    anchor_frame_offset: int | None,
    last_state: str | None,
    last_log_key,
    bg_mode: str | None = None,
    bg_pos: int | None = None,
) -> tuple[int | None, int | None, int | None, str | None, object]:
    """Phase B2/B3: emit simultaneous BG/audio series. Observation only (no control)."""
    st = str(state or "UNKNOWN")
    a_ms = int(audio_ms or 0)
    f_off = int(frame_offset or 0)
    mode = str(bg_mode or "seq")
    # Re-anchor on first PLAYING, REB/idle→PLAYING, and turn frontier (frame_offset↑).
    if st == "PLAYING":
        entered_playing = last_state is None or str(last_state) != "PLAYING"
        turn_frontier = (
            anchor_frame_offset is not None and f_off > int(anchor_frame_offset)
        )
        if anchor_bg_read is None or entered_playing or turn_frontier:
            anchor_bg_read = int(bg_read_n)
            anchor_audio_ms = int(a_ms)
            anchor_frame_offset = int(f_off)
    bg_est_ms = int(round(float(bg_read_n) * float(frame_period_ms)))
    if anchor_bg_read is not None and anchor_audio_ms is not None:
        bg_elapsed = int(
            round((int(bg_read_n) - int(anchor_bg_read)) * float(frame_period_ms))
        )
        audio_elapsed = int(a_ms) - int(anchor_audio_ms)
        delta_ms = int(bg_elapsed - audio_elapsed)
    else:
        bg_elapsed = 0
        audio_elapsed = 0
        delta_ms = 0
    state_changed = last_state is not None and st != str(last_state)
    log_key = (st, int(a_ms) // 200, int(bg_read_n) // 25, int(f_off), mode)
    if force or state_changed or last_log_key != log_key:
        print(
            "[sync][virtualcam][B2_OBS]",
            f"state={st}",
            f"audio_ms={a_ms}",
            f"bg_read_n={int(bg_read_n)}",
            f"bg_est_ms={bg_est_ms}",
            f"bg_elapsed_ms={bg_elapsed}",
            f"audio_elapsed_ms={audio_elapsed}",
            f"delta_ms={delta_ms}",
            f"bg_mode={mode}",
            f"bg_pos={int(bg_pos) if bg_pos is not None else -1}",
            f"frame_offset={f_off}",
            f"sent={int(sent)}",
            flush=True,
        )
        last_log_key = log_key
    return (
        anchor_bg_read,
        anchor_audio_ms,
        anchor_frame_offset,
        st,
        last_log_key,
    )


def _b3hf2_snap_maybe_log(
    *,
    a_ms_bg: int | None,
    desired: int | None,
    bg_pos: int | None,
    a_ms_fg: int | None,
    bg_mode: str | None,
    force: bool,
    last_log_key,
) -> object:
    """Phase B3hf2: close a_ms_bg vs a_ms_fg / desired vs bg_pos branch (obs only)."""
    mode = str(bg_mode or "seq")
    if mode != "audio":
        return last_log_key
    ab = int(a_ms_bg) if a_ms_bg is not None else -1
    af = int(a_ms_fg) if a_ms_fg is not None else -1
    des = int(desired) if desired is not None else -1
    pos = int(bg_pos) if bg_pos is not None else -1
    diverge = ab >= 0 and af >= 0 and abs(ab - af) > 40
    stuck = des >= 0 and pos >= 0 and des == pos and af > ab + 200
    log_key = (ab // 200, des // 25, pos // 25, af // 200, int(diverge), int(stuck))
    if force or diverge or stuck or last_log_key != log_key:
        print(
            "[sync][virtualcam][B3hf2_SNAP]",
            f"a_ms_bg={ab}",
            f"desired={des}",
            f"bg_pos={pos}",
            f"a_ms_fg={af}",
            f"bg_mode={mode}",
            flush=True,
        )
        return log_key
    return last_log_key


def _should_adopt_sync_meta(
    *,
    fg_dir: Path,
    applied: dict | None,
    candidate: dict | None,
) -> bool:
    """Adopt new turn sync_meta only after new-turn FG exists at frame_offset.

    Phase28: session_loop writes the next frame_offset/base while previous-turn
    PCM may still be draining from the player. Applying meta early remaps that
    leftover audio onto missing new-turn frames → SSOT_CATCHUP sticky freeze.
    """
    if candidate is None:
        return False
    if applied is None:
        return True

    new_off = int(candidate.get("frame_offset", 0) or 0)
    old_off = int(applied.get("frame_offset", 0) or 0)
    if new_off <= old_off:
        # Same turn (or base-only refresh): always take the newer meta.
        return True

    return _fg_ready_at_or_after(fg_dir=fg_dir, frame_idx=new_off, max_scan=8)


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
    bg_read_n = 0
    bg_pos = -1
    bg_total = _bg_frame_count(cap)
    last_rgb = None
    last_fg = None
    last_displayed_frame: int | None = None
    max_existing_frame: int | None = None
    last_logged_target = None
    applied_sync_meta: dict | None = None
    last_meta_defer_key = None
    # Phase B2 obs only: anchors for Δ(bg_est, audio_ms). No sync control.
    b2_anchor_bg_read: int | None = None
    b2_anchor_audio_ms: int | None = None
    b2_anchor_frame_offset: int | None = None
    b2_last_state: str | None = None
    b2_last_log_key = None
    b3hf2_last_snap_key = None
    # Phase B3: PLAYING audio→BG lock (relative). Cleared on explicit non-PLAYING.
    b3_lock_audio_ms: int | None = None
    b3_lock_bg_frame: int | None = None
    b3_lock_frame_offset: int | None = None
    b3_last_drive_state: str | None = None
    b3_last_playing_audio_ms: int | None = None
    b3_last_playing_frame_offset: int | None = None
    # Phase B3hf: last good PLAYING playback_state dict (BG+FG share on UNKNOWN).
    b3_last_good_playback_state: dict | None = None
    ssot_enabled = playback_state_file is not None
    frame_period_ms = 1000.0 / float(max(1, fps))

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
                bg_total = _bg_frame_count(cap)
                bg_pos = -1
                b3_lock_audio_ms = None
                b3_lock_bg_frame = None
                b3_lock_frame_offset = None
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
                    bg_total = _bg_frame_count(cap)
                    bg_pos = -1
                    b3_lock_audio_ms = None
                    b3_lock_bg_frame = None
                    b3_lock_frame_offset = None
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

            # Phase B3/B3hf/B3hf2: one playback_state + sync_meta snapshot per tick.
            # Adopt meta (META_DEFER rules) BEFORE resolve; BG+FG share one target.
            # B3hf: never re-read for FG alone. B3hf2: never resolve BG on pre-adopt
            # meta while FG resolves post-adopt (Turn frontier a_ms split).
            playback_state = None
            sync_meta_candidate = None
            tick_target = None
            b3_desired_bg: int | None = None
            b3_a_ms_bg: int | None = None
            st_raw = "UNKNOWN"
            if ssot_enabled and not override_active:
                playback_state = _read_json_file_retry(playback_state_file)
                sync_meta_candidate = _read_json_file_retry(sync_meta_file)
                if (
                    playback_state is not None
                    and str(playback_state.get("state", "") or "") == "PLAYING"
                ):
                    b3_last_good_playback_state = dict(playback_state)
                st_raw = (
                    str(playback_state.get("state", "UNKNOWN") or "UNKNOWN")
                    if playback_state is not None
                    else "UNKNOWN"
                )
                # Effective state shared by BG drive + FG select.
                effective_playback_state = playback_state
                if (
                    st_raw == "UNKNOWN"
                    and b3_last_drive_state == "PLAYING"
                    and b3_last_good_playback_state is not None
                ):
                    effective_playback_state = b3_last_good_playback_state
                    st_raw = "PLAYING"
                playback_state = effective_playback_state

                # B3hf2: adopt sync_meta once up-front (same DEFER gate as before).
                if _should_adopt_sync_meta(
                    fg_dir=fg_dir,
                    applied=applied_sync_meta,
                    candidate=sync_meta_candidate,
                ):
                    if (
                        applied_sync_meta is not None
                        and sync_meta_candidate is not None
                        and int(sync_meta_candidate.get("frame_offset", 0) or 0)
                        > int(applied_sync_meta.get("frame_offset", 0) or 0)
                    ):
                        # New turn FG frontier: do not hint from previous-turn indices.
                        new_off = int(sync_meta_candidate.get("frame_offset", 0) or 0)
                        if (
                            last_displayed_frame is not None
                            and int(last_displayed_frame) < new_off
                        ):
                            last_displayed_frame = None
                        if (
                            max_existing_frame is not None
                            and int(max_existing_frame) < new_off
                        ):
                            max_existing_frame = None
                    applied_sync_meta = sync_meta_candidate
                elif (
                    sync_meta_candidate is not None
                    and applied_sync_meta is not None
                    and int(sync_meta_candidate.get("frame_offset", 0) or 0)
                    > int(applied_sync_meta.get("frame_offset", 0) or 0)
                ):
                    defer_key = (
                        int(sync_meta_candidate.get("frame_offset", 0) or 0),
                        int(sync_meta_candidate.get("base_played_samples", 0) or 0),
                        int((playback_state or {}).get("played_samples", 0) or 0)
                        // 4800,
                    )
                    if last_meta_defer_key != defer_key:
                        print(
                            "[sync][virtualcam][SSOT_META_DEFER]",
                            f"pending_offset={int(sync_meta_candidate.get('frame_offset', 0) or 0)}",
                            f"applied_offset={int(applied_sync_meta.get('frame_offset', 0) or 0)}",
                            f"state={(playback_state or {}).get('state')}",
                            flush=True,
                        )
                        last_meta_defer_key = defer_key

                tick_target = _resolve_ssot_target(
                    playback_state=effective_playback_state,
                    sync_meta=applied_sync_meta,
                    step_ms=int(args.step_ms),
                    frame_offset_cli=int(args.frame_offset),
                )

            st_bg = (
                str(tick_target.get("state") or "UNKNOWN")
                if tick_target is not None
                else "UNKNOWN"
            )
            # Transient JSON miss/UNKNOWN must not drop PLAYING lock (spurious relock).
            if st_bg == "PLAYING":
                use_audio_bg = True
            elif st_raw == "UNKNOWN" and b3_last_drive_state == "PLAYING":
                # No last-good yet: hold audio mode with frozen a_ms (brief).
                use_audio_bg = True
            else:
                use_audio_bg = False

            b3_bg_mode = "seq"
            if use_audio_bg:
                if st_bg == "PLAYING" and tick_target is not None:
                    a_ms = int(tick_target.get("audio_ms", 0) or 0)
                    f_off = int(tick_target.get("frame_offset", 0) or 0)
                    b3_last_playing_audio_ms = int(a_ms)
                    b3_last_playing_frame_offset = int(f_off)
                else:
                    a_ms = (
                        int(b3_last_playing_audio_ms)
                        if b3_last_playing_audio_ms is not None
                        else 0
                    )
                    f_off = (
                        int(b3_last_playing_frame_offset)
                        if b3_last_playing_frame_offset is not None
                        else 0
                    )
                relock_reason = None
                if b3_lock_audio_ms is None or b3_last_drive_state != "PLAYING":
                    relock_reason = "enter_playing"
                elif (
                    b3_lock_frame_offset is not None
                    and f_off > int(b3_lock_frame_offset)
                ):
                    relock_reason = "turn"
                if relock_reason is not None:
                    b3_lock_audio_ms = int(a_ms)
                    b3_lock_bg_frame = int(bg_pos) if int(bg_pos) >= 0 else 0
                    b3_lock_frame_offset = int(f_off)
                    print(
                        "[sync][virtualcam][B3_BG_RELOCK]",
                        f"reason={relock_reason}",
                        f"audio_ms={int(b3_lock_audio_ms)}",
                        f"bg_frame={int(b3_lock_bg_frame)}",
                        f"frame_offset={int(b3_lock_frame_offset)}",
                        flush=True,
                    )
                # B3hf2: lock_audio_ms=0 is valid (turn relative clock). Never use
                # `lock or a_ms` — falsy 0 rebinds lock to current a_ms and freezes
                # desired at lock_bg (T2 bg_pos stick fingerprint).
                lock_a = (
                    int(b3_lock_audio_ms)
                    if b3_lock_audio_ms is not None
                    else int(a_ms)
                )
                lock_bg = (
                    int(b3_lock_bg_frame) if b3_lock_bg_frame is not None else 0
                )
                desired = _b3_desired_bg_frame(
                    audio_ms=int(a_ms),
                    lock_audio_ms=int(lock_a),
                    lock_bg_frame=int(lock_bg),
                    frame_period_ms=frame_period_ms,
                )
                b3_a_ms_bg = int(a_ms)
                b3_desired_bg = int(desired)
                ok, bg, bg_pos = _read_bg_at_frame(
                    cap,
                    desired,
                    loop_bg=bool(args.loop_bg),
                    total=int(bg_total),
                )
                b3_last_drive_state = "PLAYING"
                b3_bg_mode = "audio"
            else:
                ok, bg, bg_pos = _read_bg_sequential(
                    cap, loop_bg=bool(args.loop_bg)
                )
                # Clear lock only on explicit non-PLAYING (keep IDLE_BG_ADVANCE).
                if st_bg != "UNKNOWN" and st_raw != "UNKNOWN":
                    b3_lock_audio_ms = None
                    b3_lock_bg_frame = None
                    b3_lock_frame_offset = None
                    b3_last_drive_state = st_bg
                    b3_last_good_playback_state = None
                else:
                    b3_last_drive_state = "SEQ"
                b3_bg_mode = "seq"

            if not ok:
                if override_active:
                    print(
                        "[virtualcam_persistent][bg_override_frame_warn] failed to read bg frame",
                        flush=True,
                    )
                    time.sleep(float(args.poll_s))
                    continue
                raise RuntimeError("failed to read bg frame")
            bg_read_n += 1

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
                # B3hf2: reuse tick snapshot only (no second playback_state / meta
                # read, no second resolve).
                target = tick_target
                if target is None:
                    time.sleep(float(args.poll_s))
                    continue
                # Follow audio only while actually playing; hold otherwise.
                if str(target["state"]) == "PLAYING":
                    target_i = int(target["target_frame"])
                    cand = _fg_frame_path(fg_dir, target_i)
                    if cand is not None:
                        fg_path = cand
                        display_frame_idx = target_i
                        max_existing_frame = (
                            target_i
                            if max_existing_frame is None
                            else max(int(max_existing_frame), target_i)
                        )
                    else:
                        # Phase7 Hotfix: catch up to newest FG <= audio target.
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
                elif args.idle_hold:
                    # Phase 14: non-PLAYING (REBUFFERING/BUFFERING) still prefer FG
                    # at frozen audio_ms / last FG over IDLE_BG-only. Does not
                    # advance audio clock — display order 2 only.
                    target_i = int(target["target_frame"])
                    cand = _fg_frame_path(fg_dir, target_i)
                    if cand is not None:
                        fg_path = cand
                        display_frame_idx = target_i
                        max_existing_frame = (
                            target_i
                            if max_existing_frame is None
                            else max(int(max_existing_frame), target_i)
                        )
                    else:
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
            if fg_path is None:
                if args.idle_hold and (last_fg is not None or last_rgb is not None):
                    # Phase12/14: do not re-send frozen last_rgb composite.
                    # Advance BGV; hold last FG (or BG-only) only when no FG exists
                    # at/before frozen audio_ms. Non-PLAYING may still select FG
                    # via the branch above (Phase 14 last-FG preference).
                    try:
                        if last_fg is not None:
                            comp_bgr = _overlay(bg, last_fg)
                            comp_rgb = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)
                        else:
                            comp_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                        cam.send(comp_rgb)
                        last_rgb = comp_rgb
                    except Exception:
                        if last_rgb is not None:
                            cam.send(last_rgb)
                    cam.sleep_until_next_frame()
                    sent += 1
                    if sent % 25 == 0:
                        print(f"[virtualcam_persistent] sent={sent}", flush=True)
                        if target is not None and str(target.get("state")) != "PLAYING":
                            # Phase 14: log once per frozen audio_ms (avoid same-ms spam).
                            idle_key = (
                                "idle_bg",
                                int(target.get("audio_ms", 0) or 0),
                                str(target.get("state")),
                            )
                            if last_logged_target != idle_key:
                                print(
                                    "[sync][virtualcam][IDLE_BG_ADVANCE]",
                                    f"state={target.get('state')}",
                                    f"audio_ms={int(target.get('audio_ms', 0) or 0)}",
                                    f"sent={sent}",
                                    flush=True,
                                )
                                last_logged_target = idle_key
                    if target is not None:
                        (
                            b2_anchor_bg_read,
                            b2_anchor_audio_ms,
                            b2_anchor_frame_offset,
                            b2_last_state,
                            b2_last_log_key,
                        ) = _b2_obs_maybe_log(
                            state=str(target.get("state")),
                            audio_ms=int(target.get("audio_ms", 0) or 0),
                            frame_offset=int(target.get("frame_offset", 0) or 0),
                            bg_read_n=bg_read_n,
                            sent=sent,
                            frame_period_ms=frame_period_ms,
                            force=(sent % 25 == 0),
                            anchor_bg_read=b2_anchor_bg_read,
                            anchor_audio_ms=b2_anchor_audio_ms,
                            anchor_frame_offset=b2_anchor_frame_offset,
                            last_state=b2_last_state,
                            last_log_key=b2_last_log_key,
                            bg_mode=b3_bg_mode,
                            bg_pos=bg_pos,
                        )
                        b3hf2_last_snap_key = _b3hf2_snap_maybe_log(
                            a_ms_bg=b3_a_ms_bg,
                            desired=b3_desired_bg,
                            bg_pos=bg_pos,
                            a_ms_fg=int(target.get("audio_ms", 0) or 0),
                            bg_mode=b3_bg_mode,
                            force=(sent % 25 == 0),
                            last_log_key=b3hf2_last_snap_key,
                        )
                    continue
                time.sleep(float(args.poll_s))
                continue

            fg = None
            last_err = None

            for _ in range(5):
                try:
                    fg = _read_fg_frame(fg_path)
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
                (
                    b2_anchor_bg_read,
                    b2_anchor_audio_ms,
                    b2_anchor_frame_offset,
                    b2_last_state,
                    b2_last_log_key,
                ) = _b2_obs_maybe_log(
                    state=str(target.get("state")),
                    audio_ms=int(target.get("audio_ms", 0) or 0),
                    frame_offset=int(target.get("frame_offset", 0) or 0),
                    bg_read_n=bg_read_n,
                    sent=sent,
                    frame_period_ms=frame_period_ms,
                    force=(sent % 25 == 0),
                    anchor_bg_read=b2_anchor_bg_read,
                    anchor_audio_ms=b2_anchor_audio_ms,
                    anchor_frame_offset=b2_anchor_frame_offset,
                    last_state=b2_last_state,
                    last_log_key=b2_last_log_key,
                    bg_mode=b3_bg_mode,
                    bg_pos=bg_pos,
                )
                b3hf2_last_snap_key = _b3hf2_snap_maybe_log(
                    a_ms_bg=b3_a_ms_bg,
                    desired=b3_desired_bg,
                    bg_pos=bg_pos,
                    a_ms_fg=int(target.get("audio_ms", 0) or 0),
                    bg_mode=b3_bg_mode,
                    force=(sent % 25 == 0),
                    last_log_key=b3hf2_last_snap_key,
                )

            if sent % 25 == 0:
                print(f"[virtualcam_persistent] sent={sent}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
