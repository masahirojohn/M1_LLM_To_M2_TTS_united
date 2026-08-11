#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
from threading import Event, Lock

import yaml

RESPONSE_PREFIX = "__M0_WORKER_RESPONSE__ "

# Phase 2: abnormal M0 hang only (normal ~250ms/chunk must not be cut short).
_M0_PIPELINE_HANG_TIMEOUT_MS = 3000

# Phase 6 observability: m0_ms breakdown keys (sum ≈ wall-clock m0_ms).
# wait_mouth: coverage wait in session_loop;
# lock: advance wall residual (peer pool wait / cond); often ≈ peer png_wait when N=1;
# slice: timeline slice (bisect; suspicion B); disk: json/yaml IO; req_send: worker send;
# png_wait: worker response (= PNG gen+arrive; Phase 9 if dominant);
# verify: post PNG exist + pipeline verify; other: remainder at log time.
# Phase 13: multi-M0 pool (chunk/job dispatch). Distinct from legacy fast/slow workers.
_M0_BREAKDOWN_KEYS = (
    "m0_wait_mouth_ms",
    "m0_lock_ms",
    "m0_slice_ms",
    "m0_disk_ms",
    "m0_req_send_ms",
    "m0_png_wait_ms",
    "m0_verify_ms",
)


def _m0_timing_add(acc: dict[str, float] | None, key: str, ms: float) -> None:
    if acc is None:
        return
    acc[key] = float(acc.get(key, 0.0)) + float(ms)


def _m0_timing_empty() -> dict[str, float]:
    return {k: 0.0 for k in _M0_BREAKDOWN_KEYS}


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(obj, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    # Phase 9: raw .bgra or legacy .png
    n_bgra = len(list(path.glob("*.bgra")))
    n_png = len(list(path.glob("*.png")))
    return max(n_bgra, n_png)


def _fg_frame_exists(fg_dir: Path, frame_idx: int) -> bool:
    idx = int(frame_idx)
    return (fg_dir / f"{idx:08d}.bgra").exists() or (fg_dir / f"{idx:08d}.png").exists()


def _safe_clean_dir(path: Path, *, required_parent: Path) -> None:
    path = path.resolve()
    required_parent = required_parent.resolve()

    if required_parent not in path.parents and path != required_parent:
        raise ValueError(f"refuse to clean unsafe path: {path}")

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _build_env(m1_repo: Path, m3_repo: Path, m35_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(m1_repo / "src"),
            str(m3_repo / "src"),
            str(m35_repo),
            env.get("PYTHONPATH", ""),
        ]
    )
    return env


def _play_audio_async(
    *,
    py: Path,
    audio_script: Path,
    pcm: Path,
    chunk_id: int,
    audio_device: str,
    cwd: Path,
    env: dict[str, str],
) -> threading.Thread:
    def _target() -> None:
        if not pcm.exists():
            raise FileNotFoundError(f"missing pcm: {pcm}")

        _run(
            [
                str(py),
                str(audio_script),
                "--pcm",
                str(pcm),
                "--sr",
                "24000",
                "--device",
                str(audio_device),
                "--chunk_ms",
                "400",
                "--start_chunk",
                str(chunk_id),
            ],
            cwd=cwd,
            env=env,
        )

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    return th


def _m0_worker_tcp_request(
    *,
    host: str,
    port: int,
    req: dict[str, Any],
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    with socket.create_connection((host, int(port)), timeout=float(timeout_s)) as sock:
        sock.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        f = sock.makefile("r", encoding="utf-8", newline="\n")
        line = f.readline()
    if not line:
        raise RuntimeError(f"empty response from m0 tcp worker {host}:{port}")
    res = json.loads(line)
    if not res.get("ok"):
        raise RuntimeError(res)
    return res


def _m0_pool_reset_tcp(*, host: str, ports: list[int]) -> None:
    """Phase 13: turn-boundary flush/reset on every local-VAD turn."""
    for port in ports:
        try:
            _m0_worker_tcp_request(
                host=str(host),
                port=int(port),
                req={"cmd": "reset"},
                timeout_s=3.0,
            )
            print(
                f"[m0_pool][reset_ok] host={host} port={int(port)}",
                flush=True,
            )
        except Exception as e:
            print(
                f"[m0_pool][reset_error] host={host} port={int(port)} "
                f"{type(e).__name__}: {e}",
                flush=True,
            )


def _run_m0_worker_render_tcp(
    *,
    host: str,
    port: int,
    cfg_path: Path,
    timing_acc: dict[str, float] | None = None,
) -> None:
    req = {
        "cmd": "render",
        "config": str(cfg_path.resolve()),
    }

    t_send0 = time.perf_counter()
    with socket.create_connection((host, port), timeout=30.0) as sock:
        sock.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        _m0_timing_add(timing_acc, "m0_req_send_ms", (time.perf_counter() - t_send0) * 1000.0)
        t_wait0 = time.perf_counter()
        f = sock.makefile("r", encoding="utf-8", newline="\n")
        line = f.readline()
        _m0_timing_add(timing_acc, "m0_png_wait_ms", (time.perf_counter() - t_wait0) * 1000.0)

    if not line:
        raise RuntimeError("empty response from m0 tcp worker")

    res = json.loads(line)
    if not res.get("ok"):
        raise RuntimeError(res)


def _run_m0_worker_render(
    *,
    m0_worker_proc: subprocess.Popen,
    cfg_path: Path,
    timing_acc: dict[str, float] | None = None,
) -> None:
    if m0_worker_proc.poll() is not None:
        raise RuntimeError(f"m0 persistent worker already exited: rc={m0_worker_proc.returncode}")

    if m0_worker_proc.stdin is None or m0_worker_proc.stdout is None:
        raise RuntimeError("m0 persistent worker stdin/stdout is not available")

    req = {
        "cmd": "render",
        "config": str(cfg_path.resolve()),
    }

    t_send0 = time.perf_counter()
    m0_worker_proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    m0_worker_proc.stdin.flush()
    _m0_timing_add(timing_acc, "m0_req_send_ms", (time.perf_counter() - t_send0) * 1000.0)

    t_wait0 = time.perf_counter()
    while True:
        line = m0_worker_proc.stdout.readline()
        if not line:
            raise RuntimeError("m0 persistent worker closed stdout")

        line = line.rstrip()

        if line.startswith(RESPONSE_PREFIX):
            res = json.loads(line[len(RESPONSE_PREFIX):])
            _m0_timing_add(timing_acc, "m0_png_wait_ms", (time.perf_counter() - t_wait0) * 1000.0)
            if not res.get("ok"):
                raise RuntimeError(res)
            return

        print(line, flush=True)


def _run_m0_one_chunk(
    *,
    py: Path,
    m0_repo: Path,
    m1_repo: Path,
    m3_repo: Path,
    base_cfg: dict[str, Any],
    chunk: dict[str, Any],
    chunks_summary: dict[str, Any],
    work_dir: Path,
    watch_fg_dir: Path,
    env: dict[str, str],
    frame_offset: int,
    m0_worker_proc: subprocess.Popen | None,
    m0_worker_host: str,
    m0_worker_port: int | None,
    timing_acc: dict[str, float] | None = None,
) -> int:
    t_chunk0 = time.perf_counter()
    cid = int(chunk["chunk_id"])
    frame0 = int(chunk["frame0"])
    frame1 = int(chunk["frame1"])
    expected_frames = frame1 - frame0

    pose_json = Path(chunk["pose_chunk_json"]).resolve()
    mouth_json = Path(chunk["mouth_chunk_json"]).resolve()
    expr_json = Path(chunk["expr_chunk_json"]).resolve()

    run_dir = work_dir / f"chunk_{cid:06d}"
    local_fg_dir = watch_fg_dir

    local_fg_dir.mkdir(parents=True, exist_ok=True)

    t_disk0 = time.perf_counter()
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("io", {})
    cfg.setdefault("video", {})
    cfg.setdefault("render", {})
    cfg.setdefault("inputs", {})
    cfg.setdefault("atlas", {})

    cfg["io"]["assets_dir"] = str((m0_repo / "assets" / "sprites").resolve())
    cfg["io"]["out_dir"] = str(run_dir.resolve())
    cfg["io"]["exp_name"] = "realtime_step1_chunk"

    cfg["video"]["fps"] = int(chunks_summary["fps"])
    cfg["video"]["duration_s"] = float(expected_frames / int(chunks_summary["fps"]))

    cfg["render"]["dump_fg_png"] = True
    cfg["render"]["fg_png_dir"] = str(local_fg_dir.resolve())
    cfg["render"]["fg_frame_offset"] = int(frame_offset + frame0)
    cfg["render"]["write_mp4"] = False
    # Phase 9: raw BGRA FG (~1ms/frame) — VirtualCam reads .bgra (PNG fallback kept).
    cfg["render"]["fg_format"] = "bgra"

    cfg["atlas"]["atlas_json"] = str((m0_repo / "assets" / "atlas.min.json").resolve())
    cfg["atlas"]["affine_points_yaml"] = str((m0_repo / "configs" / "affine_points.yaml").resolve())
    cfg["render"]["affine_points_yaml_rel"] = str((m0_repo / "configs" / "affine_points.yaml").resolve())

    cfg["inputs"]["pose_timeline"] = str(pose_json)
    cfg["inputs"]["mouth_timeline"] = str(mouth_json)
    cfg["inputs"]["expression_timeline"] = str(expr_json)

    cfg_path = run_dir / "m0_chunk_config.yaml"
    _write_yaml(cfg_path, cfg)
    _m0_timing_add(timing_acc, "m0_disk_ms", (time.perf_counter() - t_disk0) * 1000.0)

    t_render0 = time.perf_counter()
    if m0_worker_proc is None and m0_worker_port is None:
        t_send0 = time.perf_counter()
        # Subprocess path: request send ≈ 0; whole run is PNG wait.
        _m0_timing_add(timing_acc, "m0_req_send_ms", (time.perf_counter() - t_send0) * 1000.0)
        t_wait0 = time.perf_counter()
        _run(
            [
                str(py),
                str((m0_repo / "src" / "m0_runner.py").resolve()),
                "--config",
                str(cfg_path),
            ],
            cwd=m0_repo,
            env=env,
        )
        _m0_timing_add(timing_acc, "m0_png_wait_ms", (time.perf_counter() - t_wait0) * 1000.0)
    elif m0_worker_port is not None:
        _run_m0_worker_render_tcp(
            host=m0_worker_host,
            port=int(m0_worker_port),
            cfg_path=cfg_path,
            timing_acc=timing_acc,
        )
    else:
        _run_m0_worker_render(
            m0_worker_proc=m0_worker_proc,
            cfg_path=cfg_path,
            timing_acc=timing_acc,
        )
    m0_render_sec = time.perf_counter() - t_render0
    

    t_verify0 = time.perf_counter()
    missing = []
    for i in range(expected_frames):
        idx = int(frame_offset + frame0 + i)
        if not _fg_frame_exists(local_fg_dir, idx):
            missing.append(str(local_fg_dir / f"{idx:08d}.bgra|.png"))
    _m0_timing_add(timing_acc, "m0_verify_ms", (time.perf_counter() - t_verify0) * 1000.0)

    if missing:
        raise RuntimeError(
            f"chunk {cid:06d} missing direct fg frames: "
            f"expected={expected_frames} missing={len(missing)} first={missing[0]}"
        )

    t_copy0 = time.perf_counter()
    copied = expected_frames
    copy_sec = time.perf_counter() - t_copy0


    chunk_total_sec = time.perf_counter() - t_chunk0

    print(
        f"[perf][m0_chunk] idx={cid} "
        f"render_sec={m0_render_sec:.3f} "
        f"copy_sec={copy_sec:.3f} "
        f"total_sec={chunk_total_sec:.3f}",
        flush=True,
    )

    print(
        f"[realtime_step1][m0_chunk_done] "
        f"idx={cid} frames={copied} "
        f"global_range=[{frame_offset + frame0},{frame_offset + frame1})",
        flush=True,
    )

    return copied

AUDIO_RESPONSE_PREFIX = "__AUDIO_PLAYER_RESPONSE__ "


def _start_audio_player(
    *,
    py: Path,
    audio_script: Path,
    audio_device: str,
    cwd: Path,
    env: dict[str, str],
    initial_buffer_ms: int = 300,
    start_fallback_ms: int = 1000,
    rebuffer_target_ms: int = 240,
    min_start_pcm_ms: int = 20,
    playback_state_file: Path | None = None,
) -> subprocess.Popen:
    # Phase 3: data-amount jitter knobs (distinct from M0 hang timeout).
    cmd = [
        str(py),
        str(audio_script),
        "--device",
        str(audio_device),
        "--sr",
        "24000",
        "--chunk_ms",
        "400",
        "--initial_buffer_ms",
        str(int(initial_buffer_ms)),
        "--start_fallback_ms",
        str(int(start_fallback_ms)),
        "--rebuffer_target_ms",
        str(int(rebuffer_target_ms)),
        "--min_start_pcm_ms",
        str(int(min_start_pcm_ms)),
    ]
    if playback_state_file is not None:
        cmd.extend(
            [
                "--playback_state_file",
                str(Path(playback_state_file).resolve()),
            ]
        )
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("audio player closed before READY")
        line = line.rstrip()
        print(line, flush=True)
        if "[audio_chunk_player_persistent][READY]" in line:
            break

    return proc


def _make_audio_playback_state_ref(*, sample_rate: int = 24000) -> dict[str, Any]:
    return {
        "lock": Lock(),
        "playback_origin_ms": 0,
        "response_playback_base_samples": 0,
        "played_samples": 0,
        "pending_ms": 0.0,
        "sample_rate": int(sample_rate),
        "pipeline_audio_end_ms": 0,
    }


def _update_audio_playback_state_ref(
    audio_playback_state_ref: dict[str, Any] | None,
    res: dict[str, Any] | None,
) -> None:
    if audio_playback_state_ref is None or not isinstance(res, dict):
        return
    with audio_playback_state_ref["lock"]:
        if "played_samples" in res:
            audio_playback_state_ref["played_samples"] = int(res.get("played_samples") or 0)
        if "pending_ms" in res:
            try:
                audio_playback_state_ref["pending_ms"] = float(res.get("pending_ms") or 0.0)
            except Exception:
                pass


def _send_audio_chunk(
    *,
    audio_player_proc: subprocess.Popen,
    pcm: Path,
    chunk_id: int,
    audio_device: str,
    single_chunk: bool = False,
    audio_playback_state_ref: dict[str, Any] | None = None,
) -> None:
    if audio_player_proc.stdin is None:
        raise RuntimeError("audio player stdin is not available")

    req = {
        "cmd": "play",
        "pcm": str(pcm),
        "sr": 24000,
        "device": str(audio_device),
        "chunk_ms": 400,
        "start_chunk": int(chunk_id),
        "single_chunk": bool(single_chunk),
    }


    audio_player_proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    audio_player_proc.stdin.flush()

    print(f"[realtime_step1] queued audio chunk idx={chunk_id}", flush=True)

    if audio_player_proc.stdout is None:
        raise RuntimeError("audio player stdout is not available")

    while True:
        line = audio_player_proc.stdout.readline()
        if not line:
            raise RuntimeError("audio player closed stdout")

        line = line.rstrip()
        print(line, flush=True)

        if line.startswith(AUDIO_RESPONSE_PREFIX):
            res = json.loads(line[len(AUDIO_RESPONSE_PREFIX):])
            if not res.get("ok"):
                raise RuntimeError(res)
            _update_audio_playback_state_ref(audio_playback_state_ref, res)
            break


def _watch_stream_pcm_chunks(
    *,
    audio_player_proc: subprocess.Popen,
    pcm_stream_chunks_dir: Path,
    audio_device: str,
    stop_event: Event,
    poll_s: float = 0.02,
) -> threading.Thread:
    def _target() -> None:
        next_idx = 0
        while not stop_event.is_set():
            p = pcm_stream_chunks_dir / f"chunk_{next_idx:06d}.pcm"
            if p.exists():
                _send_audio_chunk(
                    audio_player_proc=audio_player_proc,
                    pcm=p,
                    chunk_id=next_idx,
                    audio_device=audio_device,
                    single_chunk=True,
                )
                next_idx += 1
                continue

            time.sleep(float(poll_s))

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    return th


def _as_frames(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        for k in ("frames", "timeline"):
            v = raw.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _wrap_like(raw: Any, frames_new: list[dict[str, Any]]) -> Any:
    if isinstance(raw, list):
        return frames_new
    if isinstance(raw, dict):
        out = dict(raw)
        if "frames" in out:
            out["frames"] = frames_new
            return out
        if "timeline" in out:
            out["timeline"] = frames_new
            return out
        out["frames"] = frames_new
        return out
    return {"frames": frames_new}


def _frame_t_ms(fr: dict[str, Any]) -> int:
    return int(fr.get("t_ms", 0) or 0)


def _bisect_frames_left(frames: list[dict[str, Any]], t_ms: int) -> int:
    """First index with t_ms >= target. Assumes frames sorted by t_ms."""
    lo = 0
    hi = len(frames)
    target = int(t_ms)
    while lo < hi:
        mid = (lo + hi) // 2
        if _frame_t_ms(frames[mid]) < target:
            lo = mid + 1
        else:
            hi = mid
    return int(lo)


def _pose_abs_window_ms(
    *,
    t0_ms: int,
    t1_ms: int,
    pose_base_frame: int,
    step_ms: int,
) -> tuple[int, int]:
    """Phase B5: map turn-local chunk window → absolute pose t_ms via BGV base."""
    step = max(1, int(step_ms))
    base_ms = int(pose_base_frame) * int(step)
    return int(base_ms + int(t0_ms)), int(base_ms + int(t1_ms))


def _read_bg_cursor_pose_base(path: Path | None) -> int | None:
    """Compat: bg_pos only. Prefer _read_bg_cursor_info for B5hf."""
    info = _read_bg_cursor_info(path)
    if info is None:
        return None
    pos = info.get("bg_pos")
    return int(pos) if pos is not None and int(pos) >= 0 else None


def _read_bg_cursor_info(path: Path | None) -> dict[str, Any] | None:
    """Read VirtualCam bg_cursor.json (bg_pos/mode/ideal_base/fo). None if unavailable."""
    if path is None:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8-sig").strip()
        if not raw:
            return None
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None
        pos = int(obj.get("bg_pos", -1))
        if pos < 0:
            return None
        mode = str(obj.get("bg_mode") or "seq")
        step = max(1, int(obj.get("step_ms", 40) or 40))
        a_raw = obj.get("audio_ms", None)
        if a_raw is None:
            audio_ms = None
        else:
            audio_ms = int(a_raw)
            if audio_ms < 0:
                audio_ms = None
        if "ideal_base_frame" in obj and obj.get("ideal_base_frame") is not None:
            ideal = int(obj.get("ideal_base_frame"))
        elif audio_ms is not None:
            ideal = int(pos) - int(audio_ms) // int(step)
        else:
            ideal = int(pos)
        fo_raw = obj.get("frame_offset", None)
        if fo_raw is None:
            cursor_fo = None
        else:
            cursor_fo = int(fo_raw)
            if cursor_fo < 0:
                cursor_fo = None
        return {
            "bg_pos": int(pos),
            "bg_mode": str(mode),
            "audio_ms": audio_ms,
            "step_ms": int(step),
            "ideal_base_frame": int(ideal),
            "frame_offset": cursor_fo,
        }
    except Exception:
        return None


def _read_bg_cursor_info_retry(
    path: Path | None,
    *,
    attempts: int = 5,
    sleep_s: float = 0.002,
) -> dict[str, Any] | None:
    """Retry brief torn-read / replace races. Never invents pose_base=0."""
    last: dict[str, Any] | None = None
    n = max(1, int(attempts))
    for i in range(n):
        last = _read_bg_cursor_info(path)
        if last is not None:
            return last
        if i + 1 < n:
            time.sleep(float(sleep_s))
    return last


def _ensure_pose_base_frame(m0_pipeline_ref: dict[str, Any]) -> tuple[int, str]:
    """Snapshot PLAYING/RELOCK ideal_base once per turn for pose slice clock.

    B5hf:
    - Do not freeze on idle/seq cursor (avoids T1 constant Δ vs enter_playing RELOCK).
    - Never silent-freeze pose_base=0 on missing cursor; retry → last-good → turn_local.
    B5hf2:
    - ok_audio freeze only when cursor.frame_offset matches this turn's fo
      (rejects prior-turn PLAYING cursor before new-fo enter_playing/turn RELOCK).
    - fo↑ / fo mismatch → provisional turn_local (no absolute freeze yet).
    Returns (pose_base_frame, mode) with mode in {"absolute", "turn_local"}.
    """
    existing_mode = m0_pipeline_ref.get("pose_clock_mode")
    existing = m0_pipeline_ref.get("pose_base_frame")
    if existing_mode == "absolute" and existing is not None:
        return int(existing), "absolute"
    if existing_mode == "turn_local":
        return 0, "turn_local"

    turn_fo = int(m0_pipeline_ref.get("frame_offset", 0) or 0)

    def _freeze_absolute(pose_base: int, *, cursor_tag: str) -> tuple[int, str]:
        m0_pipeline_ref["pose_base_frame"] = int(pose_base)
        m0_pipeline_ref["pose_clock_mode"] = "absolute"
        ssot = m0_pipeline_ref.get("pose_base_ssot")
        if isinstance(ssot, dict):
            ssot["last_good"] = int(pose_base)
            ssot["last_good_fo"] = int(turn_fo)
        print(
            "[sync][B5_POSE_BASE]",
            f"pose_base_frame={int(pose_base)}",
            f"bg_cursor={cursor_tag}",
            f"mode=absolute",
            f"frame_offset={int(turn_fo)}",
            f"step_ms={int(m0_pipeline_ref.get('step_ms', 40) or 40)}",
            flush=True,
        )
        return int(pose_base), "absolute"

    def _pending(tag: str, *, info: dict[str, Any] | None = None) -> tuple[int, str]:
        if not m0_pipeline_ref.get("pose_base_pending_logged"):
            m0_pipeline_ref["pose_base_pending_logged"] = True
            parts = [
                "[sync][B5_POSE_BASE]",
                "pose_base_frame=pending",
                f"bg_cursor={tag}",
                "mode=turn_local",
            ]
            if info is not None:
                cfo = info.get("frame_offset")
                cfo_s = str(int(cfo)) if cfo is not None else "missing"
                parts.extend(
                    [
                        f"bg_pos={int(info['bg_pos'])}",
                        f"ideal_base_frame={int(info['ideal_base_frame'])}",
                        f"cursor_fo={cfo_s}",
                    ]
                )
            parts.extend(
                [
                    f"frame_offset={int(turn_fo)}",
                    f"step_ms={int(m0_pipeline_ref.get('step_ms', 40) or 40)}",
                ]
            )
            print(*parts, flush=True)
        return 0, "turn_local"

    def _cursor_fo_aligned(info: dict[str, Any]) -> bool:
        """True only when cursor fo is known and equals this turn fo.

        Legacy cursor without fo: allow only turn_fo==0 (T1) so fo↑ turns
        never freeze on stale prior-turn PLAYING audio.
        """
        cfo = info.get("frame_offset")
        if cfo is None:
            return int(turn_fo) == 0
        return int(cfo) == int(turn_fo)

    def _snapshot_unlocked() -> tuple[int, str]:
        mode_i = m0_pipeline_ref.get("pose_clock_mode")
        existing_i = m0_pipeline_ref.get("pose_base_frame")
        if mode_i == "absolute" and existing_i is not None:
            return int(existing_i), "absolute"
        if mode_i == "turn_local":
            return 0, "turn_local"

        info = _read_bg_cursor_info_retry(m0_pipeline_ref.get("bg_cursor_file"))
        if info is not None:
            bg_mode = str(info.get("bg_mode") or "seq")
            # B5hf2: prior-turn PLAYING audio must not freeze the new fo.
            if not _cursor_fo_aligned(info):
                return _pending("fo_wait", info=info)
            # Freeze only on PLAYING/audio cursor (= RELOCK-aligned ideal_base).
            if bg_mode == "audio":
                return _freeze_absolute(
                    int(info["ideal_base_frame"]),
                    cursor_tag="ok_audio",
                )
            # Idle/seq: do not freeze; provisional turn-local until audio RELOCK.
            return _pending("seq_wait", info=info)

        # Missing after retry: same-fo last-good only; else provisional (never silent 0).
        ssot = m0_pipeline_ref.get("pose_base_ssot")
        last_good = None
        last_good_fo = None
        if isinstance(ssot, dict) and ssot.get("last_good") is not None:
            try:
                last_good = int(ssot.get("last_good"))
            except Exception:
                last_good = None
            try:
                if ssot.get("last_good_fo") is not None:
                    last_good_fo = int(ssot.get("last_good_fo"))
            except Exception:
                last_good_fo = None
        if (
            last_good is not None
            and last_good >= 0
            and last_good_fo is not None
            and int(last_good_fo) == int(turn_fo)
        ):
            return _freeze_absolute(int(last_good), cursor_tag="missing_last_good")
        # Cross-turn / unknown-fo missing: keep provisional so later ok_audio can freeze.
        return _pending("missing_wait")

    # N>1: first chunk race — serialize snapshot (render runs outside claim lock).
    lock = m0_pipeline_ref.get("lock")
    if lock is not None:
        with lock:
            return _snapshot_unlocked()
    return _snapshot_unlocked()


def _slice_shift_timeline(raw: Any, t0_ms: int, t1_ms: int) -> Any:
    """Slice [t0_ms, t1_ms) and shift to local t=0.

    Assumes frames/timeline are sorted by t_ms (mouth/pose producers guarantee this).
    Uses bisect so cost is O(log N + K) for window size K, not O(N) full scans.
    """
    frames = _as_frames(raw)
    if not frames:
        return _wrap_like(raw, [])

    t0 = int(t0_ms)
    t1 = int(t1_ms)
    i0 = _bisect_frames_left(frames, t0)
    i1 = _bisect_frames_left(frames, t1)
    prev = frames[i0 - 1] if i0 > 0 else None

    out: list[dict[str, Any]] = []
    if prev is not None:
        fr0 = dict(prev)
        fr0["t_ms"] = 0
        out.append(fr0)

    for fr in frames[i0:i1]:
        fr2 = dict(fr)
        fr2["t_ms"] = _frame_t_ms(fr2) - t0
        out.append(fr2)

    return _wrap_like(raw, out)


def _inline_emo_id_to_expression(emo_id: str | None) -> str:
    mapping = {
        "1_1": "happy",
        "1_2": "happy",
        "2_0": "surprised",
        "9_1": "sad",
        "9_2": "sad",
    }
    return mapping.get(str(emo_id or "1_1"), "happy")


def _default_expr_chunk(
    *,
    session_id: str,
    step_ms: int,
    inline_emo_id: str | None = None,
) -> dict[str, Any]:
    emo_to_expression = {
        "1_0": "normal",
        "1_1": "happy",
        "1_2": "happy",
        "2_0": "surprised",
        "9_1": "sad",
        "9_2": "sad",
    }

    expression = emo_to_expression.get(str(inline_emo_id), "normal")

    return {
        "schema_version": "session_expression_timeline_v0.1",
        "session_id": str(session_id),
        "step_ms": int(step_ms),
        "timeline": [
            {
                "t_ms": 0,
                "expression": expression,
                "source": "inline_emo_tag_mode" if inline_emo_id else "stream_mouth_m0_default",
                "emo_id": inline_emo_id,
            }
        ],
        "meta": {
            "source": "inline_emo_tag_mode" if inline_emo_id else "stream_mouth_m0_default",
            "auto_blink": False,
        },
    }


def _expr_chunk_from_live_emo_events(
    *,
    session_id: str,
    step_ms: int,
    fallback_emo_id: str | None,
    chunk_start_ms: int,
    chunk_end_ms: int,
    live_emo_events: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    events = list(live_emo_events or [])

    effective_emo_id = fallback_emo_id

    # chunk開始時点で最後に有効だったemo_idを引き継ぐ
    for ev in events:
        try:
            ev_t = int(ev.get("t_ms", 0))
        except Exception:
            continue

        if ev_t <= int(chunk_start_ms):
            effective_emo_id = str(ev.get("emo_id") or effective_emo_id)

    timeline = [
        {
            "t_ms": 0,
            "expression": _inline_emo_id_to_expression(effective_emo_id),
            "source": "live_emo_events" if events else "inline_emo_tag_mode",
            "emo_id": str(effective_emo_id or "1_1"),
        }
    ]

    # chunk内で新しく来たemo_idを相対t_msで追加
    for ev in events:
        try:
            ev_t = int(ev.get("t_ms", 0))
        except Exception:
            continue

        if int(chunk_start_ms) <= ev_t < int(chunk_end_ms):
            emo_id = str(ev.get("emo_id") or effective_emo_id)
            rel_t = max(0, ev_t - int(chunk_start_ms))

            if rel_t == 0 and timeline and timeline[0].get("emo_id") == emo_id:
                continue

            timeline.append(
                {
                    "t_ms": rel_t,
                    "expression": _inline_emo_id_to_expression(emo_id),
                    "source": "live_emo_events",
                    "emo_id": emo_id,
                }
            )

    timeline = sorted(timeline, key=lambda x: int(x.get("t_ms", 0)))

    return {
        "schema_version": "session_expression_timeline_v0.1",
        "session_id": str(session_id),
        "step_ms": int(step_ms),
        "timeline": timeline,
        "meta": {
            "source": "live_emo_events",
            "auto_blink": False,
            "chunk_start_ms": int(chunk_start_ms),
            "chunk_end_ms": int(chunk_end_ms),
        },
    }


def _m0_pipeline_global_frame_range(
    *,
    m0_pipeline_ref: dict[str, Any],
    t0_ms: int,
    t1_ms: int,
) -> tuple[int, int]:
    step_ms = int(m0_pipeline_ref["step_ms"])
    frame_offset = int(m0_pipeline_ref.get("frame_offset", 0))
    frame0 = int(t0_ms // step_ms)
    frame1 = int(t1_ms // step_ms)
    return int(frame_offset + frame0), int(frame_offset + frame1)


def _m0_pipeline_verify_pngs_exist(
    *,
    m0_pipeline_ref: dict[str, Any],
    global_frame0: int,
    global_frame1: int,
) -> bool:
    watch_fg_dir = m0_pipeline_ref["watch_fg_dir"]
    if int(global_frame1) <= int(global_frame0):
        return False
    for i in range(int(global_frame1) - int(global_frame0)):
        if not _fg_frame_exists(watch_fg_dir, int(global_frame0) + i):
            return False
    return True


def _m0_pipeline_enqueue_timeline_end_ms(
    audio_playback_state_ref: dict[str, Any] | None,
    chunk_pcm_bytes: bytes,
) -> int:
    """Reserve / compute timeline end ms for this playback chunk at push time."""
    if not chunk_pcm_bytes:
        return 0

    chunk_samples = len(chunk_pcm_bytes) // 2
    sr = 24000
    if audio_playback_state_ref is None:
        return int(chunk_samples * 1000.0 / float(sr))

    with audio_playback_state_ref["lock"]:
        sr = int(audio_playback_state_ref.get("sample_rate", 24000) or 24000)
        origin_ms = int(audio_playback_state_ref.get("playback_origin_ms", 0) or 0)
        end_ms = int(audio_playback_state_ref.get("pipeline_audio_end_ms", 0) or 0)
        if end_ms <= 0 and origin_ms:
            end_ms = int(origin_ms)
        chunk_ms = chunk_samples * 1000.0 / float(sr)
        new_end = int(end_ms + chunk_ms)
        audio_playback_state_ref["pipeline_audio_end_ms"] = int(new_end)
        return int(new_end)


def _m0_pipeline_rendered_end_ms(
    m0_pipeline_ref: dict[str, Any],
    audio_playback_state_ref: dict[str, Any] | None,
) -> int:
    origin_ms = 0
    if audio_playback_state_ref is not None:
        with audio_playback_state_ref["lock"]:
            origin_ms = int(audio_playback_state_ref.get("playback_origin_ms", 0) or 0)
    with m0_pipeline_ref["lock"]:
        rendered = int(m0_pipeline_ref.get("rendered_chunks", 0) or 0)
        chunk_len_ms = int(m0_pipeline_ref.get("chunk_len_ms", 120) or 120)
    return int(origin_ms + rendered * chunk_len_ms)


def _attach_mouth_closed_dummy_frames(
    mouth_obj_ref: dict[str, Any],
    *,
    t0_ms: int,
    t1_ms: int,
    step_ms: int,
    close_mouth_id: int = 0,
) -> int:
    obj = mouth_obj_ref.get("obj")
    if not isinstance(obj, dict):
        return 0

    mouth_frames = list(_as_frames(obj))
    frame0 = int(t0_ms // int(step_ms))
    frame1 = int(t1_ms // int(step_ms))
    added = 0

    for i in range(max(len(mouth_frames), frame0), frame1):
        mouth_frames.append(
            {
                "t_ms": int(i * int(step_ms)),
                "mouth_id": int(close_mouth_id),
                "src": "mouth_closed_dummy",
            }
        )
        added += 1

    if added > 0:
        mouth_obj_ref["obj"] = _wrap_like(obj, mouth_frames)

    return int(added)


def _mouth_obj_with_hold_extend(
    mouth_obj: dict[str, Any],
    *,
    t1_ms: int,
    step_ms: int,
) -> tuple[dict[str, Any], int]:
    """Return a shallow mouth copy with last mouth_id held up to t1 (render-only).

    Phase22: M0 claim windows are chunk_len (default 120ms) while playback until
    advances on ~40ms PCM. Gating mouth_ready on full t1 forces wait_mouth on
    future PCM and collapses player pending. Hold-extend covers only the
    claim overhang beyond current until for this render; does not mutate the
    live mouth_obj_ref and does not use mouth_closed fill.
    """
    mouth_frames = list(_as_frames(mouth_obj))
    if not mouth_frames:
        return mouth_obj, 0
    step = max(1, int(step_ms))
    frame1 = int(t1_ms // step)
    last = mouth_frames[-1]
    last_t = int(last.get("t_ms", 0) or 0)
    last_id = int(last.get("mouth_id", 0) or 0)
    # Next index after last dense-or-sparse frame time.
    next_i = int(last_t // step) + 1
    added = 0
    out = list(mouth_frames)
    for i in range(max(next_i, 0), frame1):
        out.append(
            {
                "t_ms": int(i * step),
                "mouth_id": int(last_id),
                "src": "mouth_hold_extend",
            }
        )
        added += 1
    if added <= 0:
        return mouth_obj, 0
    return _wrap_like(mouth_obj, out), int(added)


def _normalize_m0_worker_ports(
    *,
    m0_worker_port: int | None,
    m0_worker_ports: list[int] | None,
) -> list[int]:
    if m0_worker_ports:
        return [int(p) for p in m0_worker_ports]
    if m0_worker_port is not None:
        return [int(m0_worker_port)]
    return []


def _create_m0_pipeline_ref(
    *,
    py: Path,
    m0_repo: Path,
    m1_repo: Path,
    m3_repo: Path,
    base_cfg: dict[str, Any],
    pose_json: Path,
    session_id: str,
    work_dir: Path,
    watch_fg_dir: Path,
    env: dict[str, str],
    frame_offset: int,
    step_ms: int,
    chunk_len_ms: int,
    fps: int,
    m0_worker_proc: subprocess.Popen | None,
    m0_worker_host: str,
    m0_worker_port: int | None,
    m0_worker_ports: list[int] | None = None,
    inline_emo_id: str | None = None,
    close_mouth_id: int = 0,
    bg_cursor_file: Path | None = None,
    pose_base_ssot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunks_root = work_dir / "stream_chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)

    ports = _normalize_m0_worker_ports(
        m0_worker_port=m0_worker_port,
        m0_worker_ports=m0_worker_ports,
    )
    n_workers = max(1, len(ports)) if ports else 1
    # Phase 13: free worker indices for chunk/job dispatch (N=1 ≡ legacy serial).
    pool_free = set(range(len(ports))) if ports else set()

    lock = threading.Lock()
    return {
        "lock": lock,
        "cond": threading.Condition(lock),
        # Legacy single-owner field kept for N=1 introspection; pool supersedes it.
        "worker_owner": None,
        "rendered_chunks": 0,
        "next_claim_cid": 0,
        "completed_cids": set(),
        "completed_meta": {},
        "m0_worker_ports": ports,
        "m0_worker_n": int(n_workers),
        "pool_free": pool_free,
        "total_frames": 0,
        "py": py,
        "m0_repo": m0_repo,
        "m1_repo": m1_repo,
        "m3_repo": m3_repo,
        "base_cfg": base_cfg,
        "pose_obj": _load_json(pose_json),
        "session_id": str(session_id),
        "work_dir": work_dir,
        "watch_fg_dir": watch_fg_dir,
        "env": env,
        "frame_offset": int(frame_offset),
        "step_ms": int(step_ms),
        "chunk_len_ms": int(chunk_len_ms),
        "fps": int(fps),
        "chunks_root": chunks_root,
        "m0_worker_proc": m0_worker_proc,
        "m0_worker_host": str(m0_worker_host),
        # Primary/base port (compat); render uses claimed port from pool.
        "m0_worker_port": (int(ports[0]) if ports else m0_worker_port),
        "inline_emo_id": inline_emo_id,
        "close_mouth_id": int(close_mouth_id),
        # Phase B5/B5hf2: fo-aligned PLAYING ideal_base freeze; else provisional.
        "bg_cursor_file": Path(bg_cursor_file).resolve() if bg_cursor_file else None,
        "pose_base_frame": None,
        "pose_clock_mode": None,
        "pose_base_pending_logged": False,
        "pose_base_ssot": pose_base_ssot if isinstance(pose_base_ssot, dict) else {"last_good": None},
    }


def _m0_pipeline_render_one_chunk_sync(
    *,
    m0_pipeline_ref: dict[str, Any],
    mouth_obj: dict[str, Any],
    cid: int,
    t0_ms: int,
    t1_ms: int,
    live_emo_id_getter: Callable[[], str | None] | None = None,
    live_emo_events_getter: Callable[[], list[dict[str, Any]] | None] | None = None,
    timing_acc: dict[str, float] | None = None,
    m0_worker_port: int | None = None,
) -> int:
    step_ms = int(m0_pipeline_ref["step_ms"])
    frame0 = int(t0_ms // step_ms)
    frame1 = int(t1_ms // step_ms)
    chunks_root = m0_pipeline_ref["chunks_root"]
    cdir = chunks_root / f"{int(cid):06d}"
    cdir.mkdir(parents=True, exist_ok=True)

    pose_chunk_json = cdir / "pose.chunk.json"
    mouth_chunk_json = cdir / "mouth.chunk.json"
    expr_chunk_json = cdir / "expr.chunk.json"

    t_slice0 = time.perf_counter()
    # Phase B5/B5hf2: pose follows fo-aligned PLAYING ideal_base; mouth turn-local.
    pose_base, pose_clock_mode = _ensure_pose_base_frame(m0_pipeline_ref)
    if pose_clock_mode == "turn_local":
        pose_t0_ms, pose_t1_ms = int(t0_ms), int(t1_ms)
    else:
        pose_t0_ms, pose_t1_ms = _pose_abs_window_ms(
            t0_ms=int(t0_ms),
            t1_ms=int(t1_ms),
            pose_base_frame=int(pose_base),
            step_ms=int(step_ms),
        )
    pose_chunk = _slice_shift_timeline(
        m0_pipeline_ref["pose_obj"], pose_t0_ms, pose_t1_ms
    )
    mouth_chunk = _slice_shift_timeline(mouth_obj, t0_ms, t1_ms)

    effective_emo_id = m0_pipeline_ref.get("inline_emo_id")
    if live_emo_id_getter is not None:
        try:
            live_emo_id = live_emo_id_getter()
        except Exception:
            live_emo_id = None
        if live_emo_id:
            effective_emo_id = str(live_emo_id)

    live_emo_events = None
    if live_emo_events_getter is not None:
        try:
            live_emo_events = live_emo_events_getter()
        except Exception:
            live_emo_events = None

    if live_emo_events:
        expr_chunk = _expr_chunk_from_live_emo_events(
            session_id=str(m0_pipeline_ref["session_id"]),
            step_ms=step_ms,
            fallback_emo_id=effective_emo_id,
            chunk_start_ms=int(t0_ms),
            chunk_end_ms=int(t1_ms),
            live_emo_events=live_emo_events,
        )
    else:
        expr_chunk = _default_expr_chunk(
            session_id=str(m0_pipeline_ref["session_id"]),
            step_ms=step_ms,
            inline_emo_id=effective_emo_id,
        )
    _m0_timing_add(timing_acc, "m0_slice_ms", (time.perf_counter() - t_slice0) * 1000.0)

    t_disk0 = time.perf_counter()
    # Compact JSON: hot-path chunk IO (indent was pure overhead for M0 worker).
    _json_dump_kw = {"ensure_ascii": False, "separators": (",", ":")}
    pose_chunk_json.write_text(
        json.dumps(pose_chunk, **_json_dump_kw),
        encoding="utf-8",
    )
    mouth_chunk_json.write_text(
        json.dumps(mouth_chunk, **_json_dump_kw),
        encoding="utf-8",
    )
    expr_chunk_json.write_text(
        json.dumps(expr_chunk, **_json_dump_kw),
        encoding="utf-8",
    )
    _m0_timing_add(timing_acc, "m0_disk_ms", (time.perf_counter() - t_disk0) * 1000.0)

    ch = {
        "chunk_id": int(cid),
        "t0_ms": int(t0_ms),
        "t1_ms": int(t1_ms),
        "frame0": int(frame0),
        "frame1": int(frame1),
        "pose_chunk_json": str(pose_chunk_json),
        "mouth_chunk_json": str(mouth_chunk_json),
        "expr_chunk_json": str(expr_chunk_json),
    }
    chunks_summary = {
        "fps": int(m0_pipeline_ref["fps"]),
        "step_ms": int(step_ms),
        "chunk_len_ms": int(m0_pipeline_ref["chunk_len_ms"]),
    }

    port = m0_worker_port
    if port is None:
        port = m0_pipeline_ref.get("m0_worker_port")

    copied = _run_m0_one_chunk(
        py=m0_pipeline_ref["py"],
        m0_repo=m0_pipeline_ref["m0_repo"],
        m1_repo=m0_pipeline_ref["m1_repo"],
        m3_repo=m0_pipeline_ref["m3_repo"],
        base_cfg=m0_pipeline_ref["base_cfg"],
        chunk=ch,
        chunks_summary=chunks_summary,
        work_dir=m0_pipeline_ref["work_dir"] / "m0_stream_work",
        watch_fg_dir=m0_pipeline_ref["watch_fg_dir"],
        env=m0_pipeline_ref["env"],
        frame_offset=int(m0_pipeline_ref["frame_offset"]),
        m0_worker_proc=m0_pipeline_ref["m0_worker_proc"],
        m0_worker_host=str(m0_pipeline_ref["m0_worker_host"]),
        m0_worker_port=port,
        timing_acc=timing_acc,
    )
    return int(copied)


def _m0_pipeline_render_with_hang_guard(
    *,
    m0_pipeline_ref: dict[str, Any],
    mouth_obj: dict[str, Any],
    cid: int,
    t0_ms: int,
    t1_ms: int,
    live_emo_id_getter: Callable[[], str | None] | None,
    live_emo_events_getter: Callable[[], list[dict[str, Any]] | None] | None,
    hang_timeout_ms: int = _M0_PIPELINE_HANG_TIMEOUT_MS,
    timing_acc: dict[str, float] | None = None,
    m0_worker_port: int | None = None,
) -> tuple[int, bool, float]:
    """Normal path blocks until M0 completes. Hang guard only on abnormal stall."""
    if int(hang_timeout_ms) <= 0:
        t0 = time.perf_counter()
        copied = _m0_pipeline_render_one_chunk_sync(
            m0_pipeline_ref=m0_pipeline_ref,
            mouth_obj=mouth_obj,
            cid=int(cid),
            t0_ms=int(t0_ms),
            t1_ms=int(t1_ms),
            live_emo_id_getter=live_emo_id_getter,
            live_emo_events_getter=live_emo_events_getter,
            timing_acc=timing_acc,
            m0_worker_port=m0_worker_port,
        )
        return int(copied), False, (time.perf_counter() - t0) * 1000.0

    render_box: dict[str, Any] = {
        "copied": 0,
        "error": None,
        "timing": _m0_timing_empty() if timing_acc is not None else None,
    }

    def _target() -> None:
        try:
            render_box["copied"] = _m0_pipeline_render_one_chunk_sync(
                m0_pipeline_ref=m0_pipeline_ref,
                mouth_obj=mouth_obj,
                cid=int(cid),
                t0_ms=int(t0_ms),
                t1_ms=int(t1_ms),
                live_emo_id_getter=live_emo_id_getter,
                live_emo_events_getter=live_emo_events_getter,
                timing_acc=render_box["timing"],
                m0_worker_port=m0_worker_port,
            )
        except BaseException as e:
            render_box["error"] = e

    t_start = time.perf_counter()
    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout=max(0.001, float(hang_timeout_ms) / 1000.0))
    wait_ms = (time.perf_counter() - t_start) * 1000.0

    if th.is_alive():
        print(
            "[sync][pipeline_chunk][M0_HANG]",
            f"cid={int(cid)}",
            f"t0_ms={int(t0_ms)}",
            f"t1_ms={int(t1_ms)}",
            f"wait_ms={wait_ms:.0f}",
            flush=True,
        )
        # Hang: wall wait has no completed breakdown; attribute to png_wait.
        _m0_timing_add(timing_acc, "m0_png_wait_ms", float(wait_ms))
        return 0, True, float(wait_ms)

    thread_timing = render_box.get("timing")
    if isinstance(thread_timing, dict) and timing_acc is not None:
        for k in _M0_BREAKDOWN_KEYS:
            _m0_timing_add(timing_acc, k, float(thread_timing.get(k, 0.0)))

    if render_box["error"] is not None:
        raise render_box["error"]

    return int(render_box["copied"]), False, float(wait_ms)


def _m0_pipeline_advance_sync(
    *,
    m0_pipeline_ref: dict[str, Any],
    mouth_obj_ref: dict[str, Any],
    audio_playback_state_ref: dict[str, Any] | None,
    live_emo_id_getter: Callable[[], str | None] | None = None,
    live_emo_events_getter: Callable[[], list[dict[str, Any]] | None] | None = None,
    hang_timeout_ms: int = _M0_PIPELINE_HANG_TIMEOUT_MS,
    max_chunks: int = 256,
    until_t1_ms: int | None = None,
) -> dict[str, Any]:
    """Advance M0 render coverage up to until_t1_ms.

    Phase 13 lock policy (multi-M0 pool, chunk/job dispatch):
    - Up to N workers render distinct cids in parallel (ports from pool_free).
    - Contiguous watermark `rendered_chunks` advances only in cid order so
      peers can enqueue when their until_t1_ms is covered.
    - N=1 reduces to legacy serial behavior (one free slot).
    - State lock is released during slice/disk/png_wait.
    """
    result = {
        "chunks_rendered": 0,
        "m0_ms_total": 0.0,
        "hang_used": False,
        "hang_chunks": 0,
        "last_cid": -1,
        "last_global_frame1": -1,
        "png_verified": True,
        "m0_breakdown": _m0_timing_empty(),
    }
    timing_acc = result["m0_breakdown"]

    obj = mouth_obj_ref.get("obj")
    if not isinstance(obj, dict):
        return result

    origin_ms = 0
    if audio_playback_state_ref is not None:
        with audio_playback_state_ref["lock"]:
            origin_ms = int(audio_playback_state_ref.get("playback_origin_ms", 0) or 0)

    chunk_len_ms = int(m0_pipeline_ref["chunk_len_ms"])
    step_ms = int(m0_pipeline_ref["step_ms"])
    close_mouth_id = int(m0_pipeline_ref.get("close_mouth_id", 0))

    lock = m0_pipeline_ref["lock"]
    cond = m0_pipeline_ref.get("cond")
    if not isinstance(cond, threading.Condition):
        cond = threading.Condition(lock)
        m0_pipeline_ref["cond"] = cond

    ports = list(m0_pipeline_ref.get("m0_worker_ports") or [])
    if not ports and m0_pipeline_ref.get("m0_worker_port") is not None:
        ports = [int(m0_pipeline_ref["m0_worker_port"])]
        m0_pipeline_ref["m0_worker_ports"] = ports
    if "pool_free" not in m0_pipeline_ref or m0_pipeline_ref["pool_free"] is None:
        m0_pipeline_ref["pool_free"] = set(range(len(ports)))
    if "completed_cids" not in m0_pipeline_ref or m0_pipeline_ref["completed_cids"] is None:
        m0_pipeline_ref["completed_cids"] = set()
    if "completed_meta" not in m0_pipeline_ref or m0_pipeline_ref["completed_meta"] is None:
        m0_pipeline_ref["completed_meta"] = {}
    if "next_claim_cid" not in m0_pipeline_ref:
        m0_pipeline_ref["next_claim_cid"] = int(m0_pipeline_ref.get("rendered_chunks", 0) or 0)

    # stdio single-proc fallback: treat as one logical slot (no TCP ports).
    use_stdio = bool(m0_pipeline_ref.get("m0_worker_proc") is not None and not ports)
    if use_stdio and not m0_pipeline_ref["pool_free"]:
        m0_pipeline_ref["pool_free"] = {0}

    def _coverage_t0() -> int:
        cid = int(m0_pipeline_ref["rendered_chunks"])
        return int(origin_ms + cid * chunk_len_ms)

    def _advance_watermark_locked() -> None:
        completed_cids = m0_pipeline_ref["completed_cids"]
        completed_meta = m0_pipeline_ref["completed_meta"]
        while int(m0_pipeline_ref["rendered_chunks"]) in completed_cids:
            cid_done = int(m0_pipeline_ref["rendered_chunks"])
            completed_cids.discard(cid_done)
            meta = completed_meta.pop(cid_done, {}) or {}
            copied = int(meta.get("copied", 0) or 0)
            if copied > 0 and bool(meta.get("ok", False)):
                m0_pipeline_ref["total_frames"] = int(
                    m0_pipeline_ref.get("total_frames", 0)
                ) + int(copied)
            m0_pipeline_ref["rendered_chunks"] = int(cid_done) + 1

    claims_this_call = 0

    while int(claims_this_call) < int(max_chunks):
        worker_idx: int | None = None
        claim_cid = -1
        t0_ms = 0
        t1_ms = 0
        claimed = False

        with cond:
            while True:
                t0_cov = _coverage_t0()
                if until_t1_ms is not None and int(t0_cov) >= int(until_t1_ms):
                    return result

                obj = mouth_obj_ref.get("obj")
                if not isinstance(obj, dict):
                    return result

                claim_cid = int(m0_pipeline_ref["next_claim_cid"])
                t0_ms = int(origin_ms + claim_cid * chunk_len_ms)
                t1_ms = int(t0_ms + chunk_len_ms)
                need_more_claims = (
                    until_t1_ms is None or int(t0_ms) < int(until_t1_ms)
                )
                mouth_frames = _as_frames(obj)
                # Prefer last t_ms so sparse / non-dense frame lists still gate correctly.
                # Phase22: when until falls inside (t0,t1), gate on until — not full t1
                # overhang — so equal-rate enqueue does not wait on future PCM.
                mouth_need_t1 = int(t1_ms)
                if (
                    until_t1_ms is not None
                    and int(until_t1_ms) > int(t0_ms)
                    and int(until_t1_ms) < int(t1_ms)
                ):
                    mouth_need_t1 = int(until_t1_ms)
                if mouth_frames:
                    last_t = int(mouth_frames[-1].get("t_ms", 0) or 0)
                    mouth_ready = last_t + int(step_ms) >= int(mouth_need_t1)
                else:
                    last_t = -1
                    mouth_ready = False
                pool_free = m0_pipeline_ref["pool_free"]
                in_flight = int(m0_pipeline_ref["next_claim_cid"]) > int(
                    m0_pipeline_ref["rendered_chunks"]
                )

                if (
                    need_more_claims
                    and mouth_ready
                    and pool_free
                    and int(claims_this_call) < int(max_chunks)
                ):
                    worker_idx = int(pool_free.pop())
                    m0_pipeline_ref["next_claim_cid"] = int(claim_cid) + 1
                    # Legacy introspection: any holder of a slot.
                    m0_pipeline_ref["worker_owner"] = threading.get_ident()
                    claimed = True
                    break

                # Covered by peers finishing while we waited.
                _advance_watermark_locked()
                t0_cov = _coverage_t0()
                if until_t1_ms is not None and int(t0_cov) >= int(until_t1_ms):
                    return result

                if not need_more_claims and in_flight:
                    cond.wait(timeout=0.05)
                    _advance_watermark_locked()
                    continue

                if not mouth_ready and not in_flight:
                    # session_loop will wait_mouth and retry.
                    # Phase17 observability only (no behavior change): mouth vs claim gate.
                    _last_t = int(last_t) if mouth_frames else -1
                    mouth_cov_ms = (_last_t + int(step_ms)) if _last_t >= 0 else 0
                    gap_ms = int(mouth_need_t1) - int(mouth_cov_ms)
                    # Always log B-suspect (gap<=0); else ~every 8th A sample.
                    _fc = int(m0_pipeline_ref.get("_p17_frontier_log_i", 0) or 0) + 1
                    m0_pipeline_ref["_p17_frontier_log_i"] = _fc
                    if gap_ms <= 0 or (_fc % 8) == 1:
                        print(
                            "[sync][pipeline_chunk][mouth_claim_frontier]",
                            f"mouth_last_t_ms={_last_t}",
                            f"mouth_cov_ms={int(mouth_cov_ms)}",
                            f"next_claim_t1_ms={int(t1_ms)}",
                            f"mouth_need_t1_ms={int(mouth_need_t1)}",
                            f"until_ms={int(until_t1_ms) if until_t1_ms is not None else -1}",
                            f"rendered_end_ms={int(t0_cov)}",
                            f"gap_ms={int(gap_ms)}",
                            f"mouth_ready={1 if mouth_ready else 0}",
                            f"need_more={1 if need_more_claims else 0}",
                            flush=True,
                        )
                    return result

                if in_flight or (need_more_claims and mouth_ready and not pool_free):
                    cond.wait(timeout=0.05)
                    _advance_watermark_locked()
                    continue

                # Mouth not ready but peers may still complete our range.
                if not mouth_ready and in_flight:
                    cond.wait(timeout=0.05)
                    _advance_watermark_locked()
                    continue

                # need_more + mouth_ready + pool_free but claims_this_call hit max:
                # return so caller retries (do not silently drop coverage).
                return result

        if not claimed or worker_idx is None:
            break

        claims_this_call += 1
        port: int | None
        if ports:
            port = int(ports[int(worker_idx)])
        else:
            port = m0_pipeline_ref.get("m0_worker_port")

        hung = False
        chunk_ok = True
        copied = 0
        m0_ms = 0.0
        global_f0 = 0
        global_f1 = 0
        print(
            "[m0_pool][claim]",
            f"cid={int(claim_cid)}",
            f"worker_idx={int(worker_idx)}",
            f"port={port}",
            f"n={int(m0_pipeline_ref.get('m0_worker_n', 1) or 1)}",
            flush=True,
        )
        try:
            global_f0, global_f1 = _m0_pipeline_global_frame_range(
                m0_pipeline_ref=m0_pipeline_ref,
                t0_ms=int(t0_ms),
                t1_ms=int(t1_ms),
            )

            render_obj = obj
            hold_added = 0
            _mf = _as_frames(obj)
            if _mf:
                _last_t = int(_mf[-1].get("t_ms", 0) or 0)
                if _last_t + int(step_ms) < int(t1_ms):
                    render_obj, hold_added = _mouth_obj_with_hold_extend(
                        obj, t1_ms=int(t1_ms), step_ms=int(step_ms)
                    )
                    if hold_added > 0:
                        print(
                            "[sync][pipeline_chunk][mouth_hold_extend]",
                            f"cid={int(claim_cid)}",
                            f"t0_ms={int(t0_ms)}",
                            f"t1_ms={int(t1_ms)}",
                            f"until_ms={int(until_t1_ms) if until_t1_ms is not None else -1}",
                            f"mouth_last_t_ms={int(_last_t)}",
                            f"hold_frames={int(hold_added)}",
                            flush=True,
                        )

            try:
                copied, hung, m0_ms = _m0_pipeline_render_with_hang_guard(
                    m0_pipeline_ref=m0_pipeline_ref,
                    mouth_obj=render_obj,
                    cid=int(claim_cid),
                    t0_ms=int(t0_ms),
                    t1_ms=int(t1_ms),
                    live_emo_id_getter=live_emo_id_getter,
                    live_emo_events_getter=live_emo_events_getter,
                    hang_timeout_ms=int(hang_timeout_ms),
                    timing_acc=timing_acc,
                    m0_worker_port=port,
                )
            except BaseException:
                chunk_ok = False
                copied = 0
                m0_ms = 0.0

            result["m0_ms_total"] = float(result["m0_ms_total"]) + float(m0_ms)
            result["last_cid"] = int(claim_cid)
            result["last_global_frame1"] = int(global_f1)

            if hung or not chunk_ok:
                result["hang_used"] = True
                result["hang_chunks"] = int(result["hang_chunks"]) + 1
                _attach_mouth_closed_dummy_frames(
                    mouth_obj_ref,
                    t0_ms=int(t0_ms),
                    t1_ms=int(t1_ms),
                    step_ms=int(step_ms),
                    close_mouth_id=int(close_mouth_id),
                )
                obj = mouth_obj_ref.get("obj")
                if isinstance(obj, dict):
                    try:
                        copied, hung2, dummy_ms = _m0_pipeline_render_with_hang_guard(
                            m0_pipeline_ref=m0_pipeline_ref,
                            mouth_obj=obj,
                            cid=int(claim_cid),
                            t0_ms=int(t0_ms),
                            t1_ms=int(t1_ms),
                            live_emo_id_getter=live_emo_id_getter,
                            live_emo_events_getter=live_emo_events_getter,
                            hang_timeout_ms=int(hang_timeout_ms),
                            timing_acc=timing_acc,
                            m0_worker_port=port,
                        )
                        result["m0_ms_total"] = (
                            float(result["m0_ms_total"]) + float(dummy_ms)
                        )
                        if hung2:
                            result["hang_chunks"] = int(result["hang_chunks"]) + 1
                        else:
                            hung = False
                            chunk_ok = True
                    except BaseException:
                        copied = 0

            t_verify0 = time.perf_counter()
            if not _m0_pipeline_verify_pngs_exist(
                m0_pipeline_ref=m0_pipeline_ref,
                global_frame0=int(global_f0),
                global_frame1=int(global_f1),
            ):
                result["png_verified"] = False
                print(
                    "[sync][pipeline_chunk][PNG_MISSING]",
                    f"cid={int(claim_cid)}",
                    f"global_range=[{int(global_f0)},{int(global_f1)})",
                    flush=True,
                )
            _m0_timing_add(
                timing_acc, "m0_verify_ms", (time.perf_counter() - t_verify0) * 1000.0
            )
        finally:
            with cond:
                m0_pipeline_ref["completed_cids"].add(int(claim_cid))
                m0_pipeline_ref["completed_meta"][int(claim_cid)] = {
                    "copied": int(copied),
                    "ok": bool((not hung) and chunk_ok and int(copied) > 0),
                }
                m0_pipeline_ref["pool_free"].add(int(worker_idx))
                n_workers = int(m0_pipeline_ref.get("m0_worker_n", 1) or 1)
                if len(m0_pipeline_ref["pool_free"]) >= n_workers:
                    m0_pipeline_ref["worker_owner"] = None
                _advance_watermark_locked()
                result["chunks_rendered"] = int(result["chunks_rendered"]) + 1
                cond.notify_all()

        obj = mouth_obj_ref.get("obj")
        if not isinstance(obj, dict):
            break

    return result


def _watch_stream_mouth_and_render_m0(
    *,
    py: Path,
    m0_repo: Path,
    m1_repo: Path,
    m3_repo: Path,
    base_cfg: dict[str, Any],
    pose_json: Path,
    mouth_json: Path,
    session_id: str,
    work_dir: Path,
    watch_fg_dir: Path,
    env: dict[str, str],
    frame_offset: int,
    step_ms: int,
    chunk_len_ms: int,
    fps: int,
    turn_start_perf: float | None,
    stop_event: Event,
    producer_done_event: Event,
    m0_worker_proc: subprocess.Popen | None,
    m0_worker_host: str,
    m0_worker_port: int | None,
    poll_s: float = 0.02,
    mouth_updated_event: Event | None = None,
    inline_emo_id: str | None = None,
    live_emo_id_getter: Callable[[], str | None] | None = None,
    live_emo_events_getter: Callable[[], list[dict[str, Any]] | None] | None = None,
) -> dict[str, Any]:
    pose_obj = _load_json(pose_json)

    frames_per_chunk = int(chunk_len_ms // step_ms)
    rendered_chunks = 0
    total_frames = 0
    first_m0_done_sec = None
    first_mouth_json_seen_logged = False
    first_mouth_frames_ready_logged = False
    first_m0_render_start_logged = False
    live_emo_extend_done = False
    t0 = time.perf_counter()

    chunks_root = work_dir / "stream_chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)

    while not stop_event.is_set():
        if not mouth_json.exists():
            if producer_done_event.is_set():
                break

            if mouth_updated_event is not None:
                mouth_updated_event.wait(timeout=float(poll_s))
                mouth_updated_event.clear()
            else:
                time.sleep(float(poll_s))

            continue

        if not first_mouth_json_seen_logged:
            now_perf = time.perf_counter()
            sec_from_stream = now_perf - t0
            print(
                f"[perf][first_mouth_json_seen_from_stream_start_sec] {sec_from_stream:.3f}",
                flush=True,
            )

            if turn_start_perf is not None:
                sec_from_turn = now_perf - float(turn_start_perf)
                print(
                    f"[perf][first_mouth_json_seen_from_turn_start_sec] {sec_from_turn:.3f}",
                    flush=True,
                )

            first_mouth_json_seen_logged = True

        try:
            mouth_obj = _load_json(mouth_json)
        except Exception:
            time.sleep(float(poll_s))
            continue

        mouth_frames = _as_frames(mouth_obj)

        needed_frames = (rendered_chunks + 1) * frames_per_chunk

        # live emo event が mouth 終了より後ろにある場合、
        # expression 切替だけを描画するため、mouth の最終frameを複製して延長する。
        if (
            len(mouth_frames) < needed_frames
            and producer_done_event.is_set()
            and not live_emo_extend_done
        ):
            live_event_max_t_ms = None

            if live_emo_events_getter is not None:
                try:
                    live_events_for_extend = live_emo_events_getter() or []
                except Exception:
                    live_events_for_extend = []

                for ev in live_events_for_extend:
                    try:
                        ev_t = int(ev.get("t_ms", 0))
                    except Exception:
                        continue

                    if live_event_max_t_ms is None or ev_t > live_event_max_t_ms:
                        live_event_max_t_ms = ev_t

            extend_until_ms = None
            if live_event_max_t_ms is not None:
                extend_until_ms = int(live_event_max_t_ms) + int(chunk_len_ms)

            if extend_until_ms is not None:
                extend_needed_frames = int(extend_until_ms // int(step_ms)) + 1
                target_frames = min(
                    max(int(needed_frames), int(extend_needed_frames)),
                    len(mouth_frames) + int(frames_per_chunk) * 8,
                )

                if mouth_frames:
                    last_frame = dict(mouth_frames[-1])
                    start_n = len(mouth_frames)

                    for i in range(start_n, target_frames):
                        fr = dict(last_frame)
                        fr["t_ms"] = int(i * int(step_ms))
                        fr["src"] = "mouth_hold_for_live_emo_events"
                        mouth_frames.append(fr)

                    mouth_obj = _wrap_like(mouth_obj, mouth_frames)

                    # 次ループで mouth_json を再読込しても延長済みframesが消えないように保存する。
                    mouth_json.write_text(
                        json.dumps(mouth_obj, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    print(
                        f"[stream_mouth_m0][extend_for_live_emo] "
                        f"rendered_chunks={rendered_chunks} "
                        f"needed_frames={needed_frames} "
                        f"extended_frames={len(mouth_frames)} "
                        f"live_event_max_t_ms={live_event_max_t_ms} "
                        f"extend_until_ms={extend_until_ms}",
                        flush=True,
                    )

                    live_emo_extend_done = True

        if len(mouth_frames) < needed_frames:
            if producer_done_event.is_set():
                break

            if mouth_updated_event is not None:
                mouth_updated_event.wait(timeout=float(poll_s))
                mouth_updated_event.clear()
            else:
                time.sleep(float(poll_s))

            continue

        if not first_mouth_frames_ready_logged:
            now_perf = time.perf_counter()
            sec_from_stream = now_perf - t0
            print(
                f"[perf][first_mouth_frames_ready_from_stream_start_sec] {sec_from_stream:.3f}",
                flush=True,
            )

            if turn_start_perf is not None:
                sec_from_turn = now_perf - float(turn_start_perf)
                print(
                    f"[perf][first_mouth_frames_ready_from_turn_start_sec] {sec_from_turn:.3f}",
                    flush=True,
                )

            first_mouth_frames_ready_logged = True

        cid = int(rendered_chunks)
        t0_ms = cid * chunk_len_ms
        t1_ms = t0_ms + chunk_len_ms
        frame0 = cid * frames_per_chunk
        frame1 = frame0 + frames_per_chunk

        cdir = chunks_root / f"{cid:06d}"
        cdir.mkdir(parents=True, exist_ok=True)

        pose_chunk_json = cdir / "pose.chunk.json"
        mouth_chunk_json = cdir / "mouth.chunk.json"
        expr_chunk_json = cdir / "expr.chunk.json"

        pose_chunk = _slice_shift_timeline(pose_obj, t0_ms, t1_ms)
        mouth_chunk = _slice_shift_timeline(mouth_obj, t0_ms, t1_ms)

        effective_emo_id = inline_emo_id

        if live_emo_id_getter is not None:
            try:
                live_emo_id = live_emo_id_getter()
            except Exception:
                live_emo_id = None

            if live_emo_id:
                effective_emo_id = str(live_emo_id)

                if effective_emo_id != inline_emo_id:
                    print(
                        f"[stream_mouth_m0][live_emo_override] "
                        f"fallback={inline_emo_id} live={effective_emo_id} chunk={cid}",
                        flush=True,
                    )

        live_emo_events = None

        if live_emo_events_getter is not None:
            try:
                live_emo_events = live_emo_events_getter()
            except Exception:
                live_emo_events = None

        if live_emo_events:
            expr_chunk = _expr_chunk_from_live_emo_events(
                session_id=session_id,
                step_ms=step_ms,
                fallback_emo_id=effective_emo_id,
                chunk_start_ms=t0_ms,
                chunk_end_ms=t1_ms,
                live_emo_events=live_emo_events,
            )
            expr_timeline = expr_chunk.get("timeline", [])
            expr_tail = expr_timeline[-3:] if isinstance(expr_timeline, list) else []

            print(
                f"[stream_mouth_m0][live_emo_events] "
                f"chunk={cid} "
                f"chunk_ms=[{t0_ms},{t1_ms}) "
                f"events_total={len(live_emo_events)} "
                f"expr_timeline_n={len(expr_timeline) if isinstance(expr_timeline, list) else -1} "
                f"expr_tail={expr_tail}",
                flush=True,
            )
        else:
            expr_chunk = _default_expr_chunk(
                session_id=session_id,
                step_ms=step_ms,
                inline_emo_id=effective_emo_id,
            )

        pose_chunk_json.write_text(json.dumps(pose_chunk, ensure_ascii=False, indent=2), encoding="utf-8")
        mouth_chunk_json.write_text(json.dumps(mouth_chunk, ensure_ascii=False, indent=2), encoding="utf-8")
        expr_chunk_json.write_text(json.dumps(expr_chunk, ensure_ascii=False, indent=2), encoding="utf-8")

        ch = {
            "chunk_id": cid,
            "t0_ms": t0_ms,
            "t1_ms": t1_ms,
            "frame0": frame0,
            "frame1": frame1,
            "pose_chunk_json": str(pose_chunk_json),
            "mouth_chunk_json": str(mouth_chunk_json),
            "expr_chunk_json": str(expr_chunk_json),
        }

        chunks_summary = {
            "fps": int(fps),
            "step_ms": int(step_ms),
            "chunk_len_ms": int(chunk_len_ms),
        }

        if not first_m0_render_start_logged:
            now_perf = time.perf_counter()
            sec_from_stream = now_perf - t0
            print(
                f"[perf][first_m0_render_start_from_stream_start_sec] {sec_from_stream:.3f}",
                flush=True,
            )

            if turn_start_perf is not None:
                sec_from_turn = now_perf - float(turn_start_perf)
                print(
                    f"[perf][first_m0_render_start_from_turn_start_sec] {sec_from_turn:.3f}",
                    flush=True,
                )

            first_m0_render_start_logged = True

        copied = _run_m0_one_chunk(
            py=py,
            m0_repo=m0_repo,
            m1_repo=m1_repo,
            m3_repo=m3_repo,
            base_cfg=base_cfg,
            chunk=ch,
            chunks_summary=chunks_summary,
            work_dir=work_dir / "m0_stream_work",
            watch_fg_dir=watch_fg_dir,
            env=env,
            frame_offset=int(frame_offset),
            m0_worker_proc=m0_worker_proc,
            m0_worker_host=m0_worker_host,
            m0_worker_port=m0_worker_port,
        )

        total_frames += int(copied)
        rendered_chunks += 1

        if first_m0_done_sec is None:
            now_perf = time.perf_counter()
            first_m0_done_sec = now_perf - t0
            print(
                f"[perf][stream_first_m0_done_from_stream_start_sec] {first_m0_done_sec:.3f}",
                flush=True,
            )

            if turn_start_perf is not None:
                first_m0_done_from_turn_sec = now_perf - float(turn_start_perf)
                print(
                    f"[perf][stream_first_m0_done_from_turn_start_sec] {first_m0_done_from_turn_sec:.3f}",
                    flush=True,
                )

        print(
            f"[stream_mouth_m0][chunk_done] idx={cid} total_frames={total_frames}",
            flush=True,
        )

    return {
        "rendered_chunks": int(rendered_chunks),
        "total_frames": int(total_frames),
        "first_m0_done_sec": first_m0_done_sec,
    }


def _stop_audio_player(audio_player_proc: subprocess.Popen | None) -> None:
    if audio_player_proc is None:
        return
    if audio_player_proc.poll() is not None:
        return

    try:
        if audio_player_proc.stdin is not None:
            audio_player_proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            audio_player_proc.stdin.flush()
        audio_player_proc.wait(timeout=10)
    except Exception:
        audio_player_proc.kill()


def _stop_m0_worker(m0_worker_proc: subprocess.Popen | None) -> None:
    if m0_worker_proc is None:
        return

    if m0_worker_proc.poll() is not None:
        return

    try:
        if m0_worker_proc.stdin is not None:
            m0_worker_proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            m0_worker_proc.stdin.flush()
        m0_worker_proc.wait(timeout=5)
    except Exception:
        m0_worker_proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Realtime step1: mic pipeline -> chunks -> M0 per chunk -> persistent OBS"
    )

    ap.add_argument("--session_id", required=True)

    ap.add_argument("--m1_repo_root", required=True)
    ap.add_argument("--m3_repo_root", required=True)
    ap.add_argument("--m0_repo_root", required=True)
    ap.add_argument("--m35_repo_root", required=True)

    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--bg_video", required=True)

    ap.add_argument("--duration_s", type=float, default=5.0)
    ap.add_argument("--mic_send_max_s", type=float, default=None)
    ap.add_argument("--audio_device", default="15")

    ap.add_argument("--model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--api_version", default="v1alpha")
    ap.add_argument("--response_trigger", default="短く返答してください。返答前にset_emotionを1回呼んでください")
    ap.add_argument("--early_response_trigger_s", type=float, default=None)
    ap.add_argument("--stream_mouth", action="store_true")
    ap.add_argument("--stream_mouth_m0", action="store_true")
    ap.add_argument("--stream_mouth_knn_script", default=None)
    ap.add_argument("--stream_mouth_gt_glob", default=None)

    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--chunk_len_ms", type=int, default=400)
    ap.add_argument("--stream_mouth_m0_chunk_len_ms", type=int, default=None)
    ap.add_argument("--stream_mouth_knn_min_interval_s", type=float, default=0.2)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--m0_base_config", default=None)
    ap.add_argument("--watch_fg_dir", default=None)
    ap.add_argument("--clean", action="store_true")

    ap.add_argument("--external_virtualcam", action="store_true")
    ap.add_argument("--frame_offset", type=int, default=0)
    ap.add_argument("--use_m0_persistent", action="store_true")
    ap.add_argument("--m0_worker_host", default="127.0.0.1")
    ap.add_argument("--m0_worker_port", type=int, default=None)

    ap.add_argument("--chunk_display_gap_s", type=float, default=0.0)


    args = ap.parse_args()

    if int(args.step_ms) != 40:
        raise ValueError("step_ms must be 40")
    if int(args.chunk_len_ms) != 400:
        raise ValueError("chunk_len_ms must be 400")

    if args.stream_mouth_m0_chunk_len_ms is not None:
        if int(args.stream_mouth_m0_chunk_len_ms) not in (80, 120, 200, 400):
            raise ValueError("stream_mouth_m0_chunk_len_ms must be 120, 200, or 400")
    if int(args.fps) != 25:
        raise ValueError("fps must be 25")

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    py = m1_repo / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        raise FileNotFoundError(f"missing python exe: {py}")

    pose_json = Path(args.pose_json).resolve()
    bg_video = Path(args.bg_video).resolve()

    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")
    if not bg_video.exists():
        raise FileNotFoundError(f"missing bg_video: {bg_video}")

    env = _build_env(m1_repo, m3_repo, m35_repo)

    out_root = m1_repo / "out" / "obs_realtime_step1" / args.session_id
    pipeline_dir = out_root / "01_audio_input_smoke_pipeline"
    chunks_dir = out_root / "02_audio_input_to_chunks"
    m0_work_dir = out_root / "03_m0_chunks_work"

    obs_stream_root = m1_repo / "out" / "obs_stream_step1"
    watch_fg_dir = (
        Path(args.watch_fg_dir).resolve()
        if args.watch_fg_dir
        else obs_stream_root / "fg"
    )

    if args.clean:
        _safe_clean_dir(out_root, required_parent=m1_repo / "out")
        _safe_clean_dir(watch_fg_dir, required_parent=m1_repo / "out")

    pipeline_script = m1_repo / "scripts" / "live_runtime" / "run_audio_input_smoke_pipeline.py"
    chunks_script = m1_repo / "scripts" / "live_runtime" / "run_audio_input_to_chunks_smoke.py"
    audio_bridge_script = m1_repo / "scripts" / "live_runtime" / "audio_stream_bridge.py"
    mouth_script = m1_repo / "scripts" / "live_runtime" / "audio_response_to_mouth_smoke.py"
    expr_script = m1_repo / "scripts" / "live_runtime" / "audio_tool_calls_to_expr_smoke.py"
    knn_script = m3_repo / "tools" / "knn_from_formant_raw_to_mouth_timeline.py"
    virtualcam_script = m1_repo / "scripts" / "live_runtime" / "run_virtualcam_persistent.py"
    audio_script = m1_repo / "scripts" / "live_runtime" / "dev_audio_chunk_player_persistent.py"



    for p in [
        pipeline_script,
        chunks_script,
        virtualcam_script,
        audio_script,
        audio_bridge_script,
        mouth_script,
        expr_script,
        knn_script,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"missing script: {p}")




    base_cfg_path = (
        Path(args.m0_base_config).resolve()
        if args.m0_base_config
        else m0_repo / "configs" / "smoke_pose_improved.yaml"
    )
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"missing m0_base_config: {base_cfg_path}")

    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    cam_proc = None
    m0_worker_proc = None
    audio_player_proc = None
    stream_audio_stop_event = Event()
    stream_audio_thread = None

    if not args.external_virtualcam:
        print("[realtime_step1] start virtualcam persistent", flush=True)
        cam_proc = subprocess.Popen(
            [
                str(py),
                str(virtualcam_script),
                "--fg_dir",
                str(watch_fg_dir),
                "--bg_video",
                str(bg_video),
                "--fps",
                str(int(args.fps)),
                "--width",
                str(int(args.width)),
                "--height",
                str(int(args.height)),
                "--idle_hold",
                "--loop_bg",
            ],
            cwd=str(m1_repo),
            env=env,
        )
    else:
        print("[realtime_step1] external virtualcam mode", flush=True)

    if args.use_m0_persistent and args.m0_worker_port is None:
        worker_script = m0_repo / "src" / "m0_persistent_worker.py"
        if not worker_script.exists():
            raise FileNotFoundError(f"missing m0 persistent worker: {worker_script}")

        print("[realtime_step1] start m0 persistent worker", flush=True)
        m0_worker_proc = subprocess.Popen(
            [str(py), str(worker_script)],
            cwd=str(m0_repo),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )

    try:
        t_turn0 = time.perf_counter()
        first_m0_done_sec = None
        first_audio_queued_sec = None

        time.sleep(1.0)



        bridge_dir = pipeline_dir / "01_audio_stream_bridge"
        pcm_stream_chunks_dir = bridge_dir / "pcm_stream_chunks"

        # stream watcher が過去runの pcm/mouth を拾わないように、必ずbridge_dirをcleanする
        if bridge_dir.exists():
            shutil.rmtree(bridge_dir)
        bridge_dir.mkdir(parents=True, exist_ok=True)

        audio_player_proc = _start_audio_player(
            py=py,
            audio_script=audio_script,
            audio_device=str(args.audio_device),
            cwd=m1_repo,
            env=env,
        )

        stream_audio_thread = _watch_stream_pcm_chunks(
            audio_player_proc=audio_player_proc,
            pcm_stream_chunks_dir=pcm_stream_chunks_dir,
            audio_device=str(args.audio_device),
            stop_event=stream_audio_stop_event,
        )



        print("[realtime_step1] run audio bridge", flush=True)
        t_audio0 = time.perf_counter()

        bridge_dir = pipeline_dir / "01_audio_stream_bridge"
        mouth_dir = pipeline_dir / "02_audio_response_to_mouth"
        expr_dir = pipeline_dir / "03_tool_calls_to_expr"

        
            # --- ここから置き換え ---
        bridge_cmd = [
            str(py),
            str(audio_bridge_script),
            "--session_id", str(args.session_id),
            "--out_dir", str(bridge_dir),
            "--model", str(args.model),
            "--stream_mouth_knn_min_interval_s",
            str(float(args.stream_mouth_knn_min_interval_s)),
            "--api_version", str(args.api_version),
            "--duration_s", str(float(args.duration_s)),
            "--chunk_ms", str(int(args.step_ms)),
            "--mode", "mic",
            "--response_trigger", str(args.response_trigger),
        ]

        if args.mic_send_max_s is not None:
            bridge_cmd += [
                "--mic_send_max_s",
                str(float(args.mic_send_max_s)),
            ]

        if args.early_response_trigger_s is not None:
            bridge_cmd += [
                "--early_response_trigger_s",
                str(float(args.early_response_trigger_s)),
            ]

        if args.stream_mouth or args.stream_mouth_m0:
            bridge_cmd.append("--stream_mouth")

        if args.stream_mouth_m0:
            bridge_cmd += [
                "--stream_mouth_knn",
                "--stream_mouth_knn_script",
                str(
                    Path(args.stream_mouth_knn_script).resolve()
                    if args.stream_mouth_knn_script
                    else (m3_repo / "tools" / "knn_from_formant_raw_to_mouth_timeline.py").resolve()
                ),
                "--stream_mouth_gt_glob",
                str(
                    args.stream_mouth_gt_glob
                    if args.stream_mouth_gt_glob
                    else (m3_repo / "data" / "knn_db" / "*.f1f2.json")
                ),
            ]

        if args.stream_mouth_m0:
            print("[realtime_step1] run audio bridge with stream_mouth_m0", flush=True)

            bridge_proc = subprocess.Popen(
                bridge_cmd,
                cwd=str(m1_repo),
                env=env,
            )

            stream_mouth_dir = bridge_dir / "stream_mouth"
            stream_mouth_json = stream_mouth_dir / "mouth.json"

            m0_stream_stop_event = Event()
            producer_done_event = Event()
            m0_stream_box: dict[str, Any] = {}
            m0_stream_error: list[BaseException] = []

            stream_mouth_m0_chunk_len_ms = (
                int(args.stream_mouth_m0_chunk_len_ms)
                if args.stream_mouth_m0_chunk_len_ms is not None
                else int(args.chunk_len_ms)
            )

            def _m0_stream_target() -> None:
                try:
                    m0_stream_box["result"] = _watch_stream_mouth_and_render_m0(
                        py=py,
                        m0_repo=m0_repo,
                        m1_repo=m1_repo,
                        m3_repo=m3_repo,
                        base_cfg=base_cfg,
                        pose_json=Path(args.pose_json).resolve(),
                        mouth_json=stream_mouth_json,
                        session_id=str(args.session_id),
                        work_dir=out_root / "03_stream_mouth_m0",
                        watch_fg_dir=watch_fg_dir,
                        env=env,
                        frame_offset=int(args.frame_offset),
                        step_ms=int(args.step_ms),
                        chunk_len_ms=int(stream_mouth_m0_chunk_len_ms),
                        fps=int(args.fps),
                        turn_start_perf=t_turn0,
                        stop_event=m0_stream_stop_event,
                        producer_done_event=producer_done_event,
                        m0_worker_proc=m0_worker_proc,
                        m0_worker_host=str(args.m0_worker_host),
                        m0_worker_port=args.m0_worker_port,
                    )
                except BaseException as e:
                    m0_stream_error.append(e)

            m0_stream_thread = threading.Thread(target=_m0_stream_target, daemon=True)
            m0_stream_thread.start()

            bridge_rc = bridge_proc.wait()
            producer_done_event.set()

            m0_stream_thread.join(timeout=10.0)
            if m0_stream_thread.is_alive():
                m0_stream_stop_event.set()
                m0_stream_thread.join(timeout=3.0)

            if m0_stream_error:
                raise m0_stream_error[0]

            m0_stream_result = m0_stream_box.get(
                "result",
                {"rendered_chunks": 0, "total_frames": 0, "first_m0_done_sec": None},
            )

            if bridge_rc != 0:
                raise RuntimeError(f"audio bridge failed: rc={bridge_rc}")

            print(
                f"[stream_mouth_m0][OK] chunks={m0_stream_result['rendered_chunks']} "
                f"frames={m0_stream_result['total_frames']}",
                flush=True,
            )

            total_frames = int(m0_stream_result["total_frames"])

            # stream_mouth_m0では従来batch後段をスキップ
            turn_total_sec = time.perf_counter() - t_turn0
            print(f"[perf][turn_total_sec] {turn_total_sec:.3f}", flush=True)

            summary_json = out_root / "run_mic_input_obs_realtime_step1.summary.json"
            summary = {
                "session_id": str(args.session_id),
                "chunks_n": int(m0_stream_result["rendered_chunks"]),
                "total_frames": int(total_frames),
                "frame_offset": int(args.frame_offset),
                "watch_fg_dir": str(watch_fg_dir),
                "use_m0_persistent": bool(args.use_m0_persistent),
                "stream_mouth_m0": True,
                "stream_mouth_m0_chunk_len_ms": (
                    int(args.stream_mouth_m0_chunk_len_ms)
                    if args.stream_mouth_m0_chunk_len_ms is not None
                    else int(args.chunk_len_ms)
                ),
            }
            summary_json.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print("[run_mic_input_obs_realtime_step1][OK]", flush=True)
            print(f"  session_id        : {args.session_id}", flush=True)
            print(f"  chunks_n          : {summary['chunks_n']}", flush=True)
            print(f"  total_frames      : {total_frames}", flush=True)
            print(f"  watch_fg_dir      : {watch_fg_dir}", flush=True)
            print(f"  stream_mouth_m0   : True", flush=True)

            return 0

        else:
            _run(
                bridge_cmd,
                cwd=m1_repo,
                env=env,
            )
        # --- ここまで ---



        pcm = bridge_dir / "audio_response.pcm"
        audio_meta_json = mouth_dir / "audio_meta.json"
        mouth_raw_json = mouth_dir / "mouth_timeline.formant.raw.json"
        mouth_json = mouth_dir / "mouth.json"
        tool_calls_json = bridge_dir / "live_tool_calls.json"

        _run(
            [
                str(py),
                str(mouth_script),
                "--session_id",
                str(args.session_id),
                "--input_pcm",
                str(pcm),
                "--out_dir",
                str(mouth_dir),
                "--input_sr",
                "24000",
                "--step_ms",
                str(int(args.step_ms)),
                "--chunk_ms",
                str(int(args.step_ms)),
            ],
            cwd=m1_repo,
            env=env,
        )

        _run(
            [
                str(py),
                str(knn_script),
                "--raw",
                str(mouth_raw_json),
                "--gt_glob",
                "data/knn_db/*.f1f2.json",
                "--out",
                str(mouth_json),
                "--audio",
                str(pcm),
                "--step_ms",
                str(int(args.step_ms)),
            ],
            cwd=m3_repo,
            env=env,
        )

        _run(
            [
                str(py),
                str(expr_script),
                "--session_id",
                str(args.session_id),
                "--tool_calls_json",
                str(tool_calls_json),
                "--audio_meta_json",
                str(audio_meta_json),
                "--out_dir",
                str(expr_dir),
                "--step_ms",
                str(int(args.step_ms)),
            ],
            cwd=m1_repo,
            env=env,
        )

        audio_pipeline_sec = time.perf_counter() - t_audio0
        print(f"[perf][audio_pipeline_sec] {audio_pipeline_sec:.3f}", flush=True)



                # 互換summary作成:
        # run_audio_input_to_chunks_smoke.py は従来の
        # run_audio_input_smoke_pipeline.summary.json を読むため、
        # pipeline分解後も同じ最小schemaを生成する。
        bridge_summary_json = bridge_dir / "audio_stream_bridge.summary.json"
        mouth_summary_json = mouth_dir / "audio_response_to_mouth_smoke.summary.json"
        expr_summary_json = expr_dir / "audio_tool_calls_to_expr_smoke.summary.json"

        bridge_summary = _load_json(bridge_summary_json)
        mouth_summary = _load_json(mouth_summary_json)
        expr_summary = _load_json(expr_summary_json)

        pipeline_summary_json = pipeline_dir / "run_audio_input_smoke_pipeline.summary.json"

        expr_json = expr_dir / "expr.json"

        pipeline_summary = {
            "format": "audio_input_smoke_pipeline.summary.v0",
            "session_id": str(args.session_id),
            "step_ms": int(args.step_ms),
            "bridge": {
                "input_audio_ms": bridge_summary.get("input_audio_ms"),
                "response_audio_bytes": bridge_summary.get("response_audio_bytes"),
                "response_audio_chunks": bridge_summary.get("response_audio_chunks"),
                "tool_calls": bridge_summary.get("tool_calls"),
            },
            "mouth": {
                "audio_ms": mouth_summary.get("audio_ms"),
                "target_frames": mouth_summary.get("target_frames"),
                "frames_n": int(mouth_summary.get("frames_n", mouth_summary.get("mouth_frames_n", 0))),
                "voiced_frames_n": mouth_summary.get("voiced_frames_n"),
            },
            "expr": {
                "tool_calls_n": expr_summary.get("tool_calls_n"),
                "accepted_n": expr_summary.get("accepted_n"),
                "events_n": expr_summary.get("events_n"),
            },
            "outputs": {
                "audio_response_pcm": str(pcm),
                "mouth_raw_json": str(mouth_raw_json),
                "mouth_json": str(mouth_json),
                "audio_meta_json": str(audio_meta_json),
                "expr_json": str(expr_json),
            },
        }



        pipeline_summary_json.write_text(
            json.dumps(pipeline_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[realtime_step1] wrote compat pipeline summary: {pipeline_summary_json}", flush=True)




        
        # Step-D:
        # 音声は watcher 先行再生ではなく、M0 chunk完了後に同期再生する。
        # stream_audio_thread = _watch_stream_pcm_chunks(...)

        print("[realtime_step1] run chunks build", flush=True)
        t_chunks0 = time.perf_counter()
        _run(
            [
                str(py),
                str(chunks_script),
                "--session_id",
                str(args.session_id),
                "--pipeline_dir",
                str(pipeline_dir),
                "--pose_json",
                str(pose_json),
                "--out_dir",
                str(chunks_dir),
                "--step_ms",
                str(int(args.step_ms)),
                "--chunk_len_ms",
                str(int(args.chunk_len_ms)),
                "--fps",
                str(int(args.fps)),
            ],
            cwd=m1_repo,
            env=env,
        )

        chunks_build_sec = time.perf_counter() - t_chunks0
        print(f"[perf][chunks_build_sec] {chunks_build_sec:.3f}", flush=True)

        chunks_summary_json = chunks_dir / "run_audio_input_to_chunks_smoke.summary.json"
        chunks_summary = _load_json(chunks_summary_json)

        audio_thread = None
        total_frames = 0

        for ch in chunks_summary["chunks"]:
            total_frames += _run_m0_one_chunk(
                py=py,
                m0_repo=m0_repo,
                m1_repo=m1_repo,
                m3_repo=m3_repo,
                base_cfg=base_cfg,
                chunk=ch,
                chunks_summary=chunks_summary,
                work_dir=m0_work_dir,
                watch_fg_dir=watch_fg_dir,
                env=env,
                frame_offset=int(args.frame_offset),
                m0_worker_proc=m0_worker_proc,
                m0_worker_host=str(args.m0_worker_host),
                m0_worker_port=args.m0_worker_port,
            )

            if first_m0_done_sec is None:
                first_m0_done_sec = time.perf_counter() - t_turn0
                print(
                    f"[perf][first_m0_done_from_start_sec] {first_m0_done_sec:.3f}",
                    flush=True,
                )



            if first_audio_queued_sec is None:
                first_audio_queued_sec = time.perf_counter() - t_turn0
                print(
                    f"[perf][first_audio_queued_from_start_sec] {first_audio_queued_sec:.3f}",
                    flush=True,
                )



            # Step-C-C:
            # 音声は pcm_stream_chunks watcher が受信直後に再生する。
            # ここでは映像chunk生成のみ行う。



            if float(args.chunk_display_gap_s) > 0:
                print(
                    f"[realtime_step1] chunk display gap {args.chunk_display_gap_s}s",
                    flush=True,
                )
                time.sleep(float(args.chunk_display_gap_s))


        turn_total_sec = time.perf_counter() - t_turn0
        print(f"[perf][turn_total_sec] {turn_total_sec:.3f}", flush=True)

        summary_json = out_root / "run_mic_input_obs_realtime_step1.summary.json"
        summary = {
            "session_id": str(args.session_id),
            "chunks_n": len(chunks_summary["chunks"]),
            "total_frames": int(total_frames),
            "frame_offset": int(args.frame_offset),
            "watch_fg_dir": str(watch_fg_dir),
            "use_m0_persistent": bool(args.use_m0_persistent),
        }
        summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("[run_mic_input_obs_realtime_step1][OK]", flush=True)
        print(f"  session_id        : {args.session_id}", flush=True)
        print(f"  chunks_n          : {len(chunks_summary['chunks'])}", flush=True)
        print(f"  total_frames      : {total_frames}", flush=True)
        print(f"  watch_fg_dir      : {watch_fg_dir}", flush=True)
        print(f"  use_m0_persistent : {bool(args.use_m0_persistent)}", flush=True)

        time.sleep(2.0)

    finally:
        stream_audio_stop_event.set()
        if stream_audio_thread is not None:
            stream_audio_thread.join(timeout=2.0)

        _stop_audio_player(audio_player_proc)
        _stop_m0_worker(m0_worker_proc)

        if cam_proc is not None and cam_proc.poll() is None:
            print("[realtime_step1] terminate virtualcam", flush=True)
            cam_proc.terminate()
            try:
                cam_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cam_proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())