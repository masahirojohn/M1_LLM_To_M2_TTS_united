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
from threading import Event

import yaml

RESPONSE_PREFIX = "__M0_WORKER_RESPONSE__ "


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
    return len(list(path.glob("*.png"))) if path.exists() else 0


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


def _run_m0_worker_render_tcp(
    *,
    host: str,
    port: int,
    cfg_path: Path,
) -> None:
    req = {
        "cmd": "render",
        "config": str(cfg_path.resolve()),
    }

    with socket.create_connection((host, port), timeout=30.0) as sock:
        sock.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        f = sock.makefile("r", encoding="utf-8", newline="\n")
        line = f.readline()

    if not line:
        raise RuntimeError("empty response from m0 tcp worker")

    res = json.loads(line)
    if not res.get("ok"):
        raise RuntimeError(res)


def _run_m0_worker_render(
    *,
    m0_worker_proc: subprocess.Popen,
    cfg_path: Path,
) -> None:
    if m0_worker_proc.poll() is not None:
        raise RuntimeError(f"m0 persistent worker already exited: rc={m0_worker_proc.returncode}")

    if m0_worker_proc.stdin is None or m0_worker_proc.stdout is None:
        raise RuntimeError("m0 persistent worker stdin/stdout is not available")

    req = {
        "cmd": "render",
        "config": str(cfg_path.resolve()),
    }

    m0_worker_proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    m0_worker_proc.stdin.flush()

    while True:
        line = m0_worker_proc.stdout.readline()
        if not line:
            raise RuntimeError("m0 persistent worker closed stdout")

        line = line.rstrip()

        if line.startswith(RESPONSE_PREFIX):
            res = json.loads(line[len(RESPONSE_PREFIX):])
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

    cfg["atlas"]["atlas_json"] = str((m0_repo / "assets" / "atlas.min.json").resolve())
    cfg["atlas"]["affine_points_yaml"] = str((m0_repo / "configs" / "affine_points.yaml").resolve())
    cfg["render"]["affine_points_yaml_rel"] = str((m0_repo / "configs" / "affine_points.yaml").resolve())

    cfg["inputs"]["pose_timeline"] = str(pose_json)
    cfg["inputs"]["mouth_timeline"] = str(mouth_json)
    cfg["inputs"]["expression_timeline"] = str(expr_json)

    cfg_path = run_dir / "m0_chunk_config.yaml"
    _write_yaml(cfg_path, cfg)

    t_render0 = time.perf_counter()
    if m0_worker_proc is None and m0_worker_port is None:
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
    elif m0_worker_port is not None:
        _run_m0_worker_render_tcp(
            host=m0_worker_host,
            port=int(m0_worker_port),
            cfg_path=cfg_path,
        )
    else:
        _run_m0_worker_render(
            m0_worker_proc=m0_worker_proc,
            cfg_path=cfg_path,
        )
    m0_render_sec = time.perf_counter() - t_render0
    

    missing = []
    for i in range(expected_frames):
        p = local_fg_dir / f"{frame_offset + frame0 + i:08d}.png"
        if not p.exists():
            missing.append(str(p))

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
) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            str(py),
            str(audio_script),
            "--device",
            str(audio_device),
            "--sr",
            "24000",
            "--chunk_ms",
            "400",
        ],
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


def _send_audio_chunk(
    *,
    audio_player_proc: subprocess.Popen,
    pcm: Path,
    chunk_id: int,
    audio_device: str,
    single_chunk: bool = False,
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


def _slice_shift_timeline(raw: Any, t0_ms: int, t1_ms: int) -> Any:
    frames = _as_frames(raw)

    inside = [
        fr for fr in frames
        if t0_ms <= int(fr.get("t_ms", 0) or 0) < t1_ms
    ]

    prev = None
    for fr in frames:
        t = int(fr.get("t_ms", 0) or 0)
        if t < t0_ms:
            prev = fr
        else:
            break

    out: list[dict[str, Any]] = []

    if prev is not None:
        fr0 = dict(prev)
        fr0["t_ms"] = 0
        out.append(fr0)

    for fr in inside:
        fr2 = dict(fr)
        fr2["t_ms"] = int(fr2.get("t_ms", 0) or 0) - int(t0_ms)
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