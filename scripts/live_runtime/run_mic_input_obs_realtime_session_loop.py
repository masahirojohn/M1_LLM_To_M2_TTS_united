#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

import yaml
from google import genai
from google.genai import types

_THIS_DIR = Path(__file__).resolve().parent
_M1_ROOT = _THIS_DIR.parent.parent

_DEFAULT_M3_ROOT = Path(
    os.environ.get(
        "M3_REPO_ROOT",
        r"C:\dev\M3_Live_API_1_united",
    )
).resolve()


_M3_PATH_CANDIDATES = [
    _DEFAULT_M3_ROOT,
    _DEFAULT_M3_ROOT / "src",
]

for _p in _M3_PATH_CANDIDATES:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from m3p.live.mouth_streamer_oc import MouthStreamerOC, MouthOCConfig

from run_mic_input_obs_realtime_step1 import (
    _start_audio_player,
    _stop_audio_player,
    _watch_stream_pcm_chunks,
    _watch_stream_mouth_and_render_m0,
)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _decode_audio_bytes(data: bytes) -> bytes:
    if not data:
        return b""
    try:
        import base64

        s = data.decode("ascii")
        if len(s) >= 16:
            decoded = base64.b64decode(s, validate=True)
            if decoded:
                return decoded
    except Exception:
        pass
    return data


def _extract_audio_bytes(msg: Any) -> bytes | None:
    server_content = _safe_getattr(msg, "server_content", None)
    model_turn = _safe_getattr(server_content, "model_turn", None)
    parts = _safe_getattr(model_turn, "parts", None)

    if parts:
        for part in parts:
            inline_data = _safe_getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            data = _safe_getattr(inline_data, "data", None)
            if data:
                return _decode_audio_bytes(bytes(data))

    data = _safe_getattr(msg, "data", None)
    if data:
        return _decode_audio_bytes(bytes(data))

    return None


def _dump_live_msg_raw(msg: Any, *, limit_chars: int = 2000) -> str:
    parts: list[str] = []

    try:
        parts.append(f"type={type(msg)}")
    except Exception:
        pass

    try:
        attrs = [
            x for x in dir(msg)
            if not x.startswith("_")
        ]
        parts.append(f"dir={attrs}")
    except Exception as e:
        parts.append(f"dir_error={type(e).__name__}: {e}")

    for method_name in ("model_dump_json", "to_json", "json"):
        try:
            fn = getattr(msg, method_name, None)
            if callable(fn):
                s = fn()
                if s:
                    parts.append(f"{method_name}={str(s)[:limit_chars]}")
                    break
        except Exception as e:
            parts.append(f"{method_name}_error={type(e).__name__}: {e}")

    try:
        parts.append(f"str={str(msg)[:limit_chars]}")
    except Exception as e:
        parts.append(f"str_error={type(e).__name__}: {e}")

    return "\n".join(parts)


def _load_inline_emo_queue_jsonl(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}

    if not path.exists():
        raise FileNotFoundError(f"inline emo queue jsonl not found: {path}")
    
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except Exception as e:
            raise ValueError(f"invalid jsonl at {path}:{line_no}: {e}") from e

        turn = obj.get("turn")
        emo_id = obj.get("emo_id")

        if turn is None or emo_id is None:
            raise ValueError(f"missing turn/emo_id at {path}:{line_no}: {obj}")

        result[int(turn)] = str(emo_id)

    return result


def _extract_emo_id_from_transcription(text: str) -> str | None:
    m = re.search(r"\[emo:([0-9]+_[0-9]+)\]", str(text))
    if not m:
        return None
    return m.group(1)


def _extract_tool_calls(msg: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tool_call = _safe_getattr(msg, "tool_call", None)
    function_calls = _safe_getattr(tool_call, "function_calls", None)

    if function_calls:
        for fc in function_calls:
            out.append(
                {
                    "id": _safe_getattr(fc, "id", None),
                    "name": _safe_getattr(fc, "name", None),
                    "args": _safe_getattr(fc, "args", None),
                }
            )
    return out


async def _send_tool_response_for_call(
    *,
    session: Any,
    call: dict[str, Any],
) -> None:
    name = call.get("name")
    call_id = call.get("id")
    args = call.get("args")

    response = {
        "ok": True,
        "handled": True,
        "name": name,
        "args": args if isinstance(args, dict) else {},
    }

    await session.send_tool_response(
        function_responses=[
            types.FunctionResponse(
                id=call_id,
                name=name,
                response=response,
            )
        ]
    )

    print(
        f"[tool_response] id={call_id} name={name} ok=True",
        flush=True,
    )


def _build_live_config(
    system_instruction: str,
    *,
    enable_tools: bool = True,
    output_audio_transcription: bool = False,
) -> types.LiveConnectConfig:
    extra_kwargs: dict[str, Any] = {}

    if output_audio_transcription:
        extra_kwargs["output_audio_transcription"] = {}

    if not enable_tools:
        return types.LiveConnectConfig(
            system_instruction=system_instruction,
            response_modalities=["AUDIO"],
            **extra_kwargs,
        )

    set_emotion = types.FunctionDeclaration(
        name="set_emotion",
        description="Set current emotion id for avatar expression control.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"emo_id": types.Schema(type=types.Type.STRING)},
            required=["emo_id"],
        ),
    )

    return types.LiveConnectConfig(
        system_instruction=system_instruction,
        tools=[types.Tool(function_declarations=[set_emotion])],
        response_modalities=["AUDIO"],
        **extra_kwargs,
    )


def _project_streamer_to_raw(streamer_json_path: Path) -> dict[str, Any]:
    data = json.loads(streamer_json_path.read_text(encoding="utf-8"))
    frames_in = data.get("frames", []) if isinstance(data, dict) else []

    frames_out: list[dict[str, Any]] = []
    for fr in frames_in:
        if not isinstance(fr, dict):
            continue
        frames_out.append(
            {
                "t_ms": fr.get("t_ms"),
                "vad_active": int(fr.get("vad_active", 0) or 0),
                "f1_hz": fr.get("f1_hz"),
                "f2_hz": fr.get("f2_hz"),
                "src": "session_loop_stream_mouth",
            }
        )

    return {
        "version": "m3p.mouth.timeline.v1",
        "step_ms": int(data.get("step_ms", 40)),
        "frames": frames_out,
        "meta": data.get("meta", {}),
    }


_KNN_FUNC_CACHE: dict[str, Any] = {}


def _run_knn_in_process(
    *,
    knn_script: Path,
    raw_json: Path,
    out_json: Path,
    gt_glob: str,
    step_ms: int,
) -> float:
    t0 = time.perf_counter()
    key = str(knn_script.resolve())

    if key not in _KNN_FUNC_CACHE:
        spec = importlib.util.spec_from_file_location(
            "knn_from_formant_raw_to_mouth_timeline_runtime",
            str(knn_script.resolve()),
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load knn module: {knn_script}")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fn = getattr(mod, "run_knn_from_raw_obj", None)
        if fn is None:
            raise RuntimeError(f"missing run_knn_from_raw_obj(): {knn_script}")

        _KNN_FUNC_CACHE[key] = fn

    raw_obj = json.loads(raw_json.read_text(encoding="utf-8"))

    out_obj = _KNN_FUNC_CACHE[key](
        raw_obj=raw_obj,
        gt_glob=str(gt_glob),
        step_ms=int(step_ms),
        k=5,
        fallback_id_active=2,
        min_conf_ratio=1.0,
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return time.perf_counter() - t0


async def _send_mic_once(
    *,
    session: Any,
    duration_s: float,
    input_sr: int,
    chunk_ms: int,
) -> int:
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError("sounddevice / numpy required") from e

    block_samples = int(input_sr * chunk_ms / 1000.0)
    total_blocks = max(1, int((duration_s * 1000.0) / chunk_ms))

    q: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        x = indata[:, 0]
        x = np.clip(x, -1.0, 1.0)
        pcm = (x * 32767.0).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(q.put_nowait, pcm)

    sent_bytes = 0

    with sd.InputStream(
        samplerate=int(input_sr),
        channels=1,
        dtype="float32",
        blocksize=block_samples,
        callback=callback,
    ):
        for _ in range(total_blocks):
            chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            await session.send_realtime_input(
                audio=types.Blob(
                    data=chunk,
                    mime_type=f"audio/pcm;rate={int(input_sr)}",
                )
            )
            sent_bytes += len(chunk)

    return sent_bytes


async def _receive_loop(
    *,
    session: Any,
    stop_event: asyncio.Event,
    pcm_stream_chunks_dir: Path,
    audio_response_pcm: Path,
    mouth_streamer: MouthStreamerOC,
    mouth_streamer_json: Path,
    mouth_raw_json: Path,
    mouth_json: Path,
    knn_script: Path,
    gt_glob: str,
    step_ms: int,
    turn_state: dict[str, Any],
    mouth_updated_event: Event,
    drop_initial_audio_ms: int = 0,
    debug_receive: bool = False,
    debug_receive_raw: bool = False,
) -> None:
    audio_response_pcm.parent.mkdir(parents=True, exist_ok=True)
    pcm_stream_chunks_dir.mkdir(parents=True, exist_ok=True)

    audio_chunk_idx = 0
    playback_chunk_idx = 0
    drop_initial_audio_bytes_remaining = max(
        0,
        int(round(24000 * 2 * int(drop_initial_audio_ms) / 1000.0)),
    )

    with audio_response_pcm.open("ab") as audio_f:
        while not stop_event.is_set():
            got_any = False

            async for msg in session.receive():
                if stop_event.is_set():
                    break

                got_any = True

                if debug_receive_raw:
                    raw_count = int(turn_state.get("debug_receive_raw_count", 0))

                    if raw_count < 5:
                        print(
                            "[debug][receive_raw]\n"
                            + _dump_live_msg_raw(msg, limit_chars=2500),
                            flush=True,
                        )
                        turn_state["debug_receive_raw_count"] = raw_count + 1

                audio = _extract_audio_bytes(msg)
                calls = _extract_tool_calls(msg)

                text_value = None

                try:
                    server_content = getattr(msg, "server_content", None)

                    if server_content is not None:
                        model_turn = getattr(server_content, "model_turn", None)

                        if model_turn is not None:
                            parts = getattr(model_turn, "parts", None) or []

                            for p in parts:
                                t = getattr(p, "text", None)

                                if t:
                                    text_value = str(t)
                                    break
                except Exception:
                    pass

                if debug_receive and text_value:
                    print(f"[debug][text_chunk] {text_value[:120]}", flush=True)

                transcription_text = None

                try:
                    server_content = getattr(msg, "server_content", None)

                    if server_content is not None:
                        output_transcription = getattr(
                            server_content,
                            "output_transcription",
                            None,
                        )

                        if output_transcription is not None:
                            transcription_text = getattr(output_transcription, "text", None)
                except Exception:
                    pass

                if transcription_text:
                    print(
                        f"[transcription][output] {str(transcription_text)[:160]}",
                        flush=True,
                    )

                    live_emo_id = _extract_emo_id_from_transcription(str(transcription_text))

                    if live_emo_id:
                        turn_state["live_emo_id"] = live_emo_id
                        print(
                            f"[transcription][emo_id] active_turn={turn_state.get('active_turn')} emo_id={live_emo_id}",
                            flush=True,
                        )

                if debug_receive:
                    print(
                        f"[debug][receive_msg] "
                        f"active_turn={turn_state.get('active_turn')} "
                        f"has_audio={audio is not None} "
                        f"audio_bytes={len(audio) if audio else 0} "
                        f"tool_calls_n={len(calls)}",
                        flush=True,
                    )

                if audio:
                    now = time.perf_counter()

                    turn_state["last_audio_perf"] = now

                    active_turn = turn_state.get("active_turn")

                    audio_chunk_idx_log = int(turn_state.get("audio_chunk_idx_log", 0))

                    sec_from_turn = None
                    if turn_state.get("turn_start_perf") is not None:
                        sec_from_turn = (
                            now - float(turn_state["turn_start_perf"])
                        )

                    print(
                        (
                            "[perf][session_audio_chunk] "
                            f"idx={audio_chunk_idx_log} "
                            f"active_turn={active_turn} "
                            f"sec_from_turn="
                            f"{sec_from_turn:.3f}" if sec_from_turn is not None else "None"
                        ),
                        flush=True,
                    )

                    turn_state["audio_chunk_idx_log"] = audio_chunk_idx_log + 1

                    if active_turn is not None and turn_state.get("first_audio_sec") is None:
                        first_audio = now - float(turn_state["turn_start_perf"])
                        turn_state["first_audio_sec"] = first_audio
                        print(
                            f"[perf][turn{active_turn}_first_response_audio_chunk_sec] {first_audio:.3f}",
                            flush=True,
                        )

                    audio_f.write(audio)
                    audio_f.flush()

                    # audio_response_pcm / mouth には full audio を使う。
                    # audio_player 用の pcm_stream_chunks だけ、冒頭N msをdropする。
                    playback_audio = audio

                    if drop_initial_audio_bytes_remaining > 0:
                        if len(playback_audio) <= drop_initial_audio_bytes_remaining:
                            drop_initial_audio_bytes_remaining -= len(playback_audio)
                            playback_audio = b""
                        else:
                            playback_audio = playback_audio[drop_initial_audio_bytes_remaining:]
                            drop_initial_audio_bytes_remaining = 0

                    if playback_audio:
                        chunk_path = pcm_stream_chunks_dir / f"chunk_{playback_chunk_idx:06d}.pcm"
                        chunk_path.write_bytes(playback_audio)
                        playback_chunk_idx += 1

                    audio_chunk_idx += 1

                    emitted = mouth_streamer.push_pcm16_mono(audio, input_sr=24000)
                    mouth_streamer.flush()

                    raw_obj = _project_streamer_to_raw(mouth_streamer_json)
                    mouth_raw_json.write_text(
                        json.dumps(raw_obj, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    if emitted > 0:
                        if not turn_state.get("first_mouth_json_logged", False):
                            active_turn = turn_state.get("active_turn")

                            if active_turn is not None:
                                dt = time.perf_counter() - float(turn_state["turn_start_perf"])

                                print(
                                    f"[perf][session_first_stream_mouth_json_written_from_turn_start_sec] {dt:.3f}",
                                    flush=True,
                                )

                                turn_state["first_mouth_json_logged"] = True

                        _run_knn_in_process(
                            knn_script=knn_script,
                            raw_json=mouth_raw_json,
                            out_json=mouth_json,
                            gt_glob=gt_glob,
                            step_ms=step_ms,
                        )
                        mouth_updated_event.set()

                for call in calls:
                    print(f"[tool_call] {call}", flush=True)

                    try:
                        await _send_tool_response_for_call(
                            session=session,
                            call=call,
                        )
                    except BaseException as e:
                        print(
                            f"[tool_response][WARN] {type(e).__name__}: {e}",
                            flush=True,
                        )

            print("[session_loop][receive_loop] receive() ended; restart", flush=True)
            await asyncio.sleep(0.05)

            if not got_any:
                await asyncio.sleep(0.1)


def _start_virtualcam(
    *,
    py: Path,
    script: Path,
    fg_dir: Path,
    bg_video: Path,
    fps: int,
    width: int,
    height: int,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            str(py),
            str(script),
            "--fg_dir",
            str(fg_dir),
            "--bg_video",
            str(bg_video),
            "--fps",
            str(int(fps)),
            "--width",
            str(int(width)),
            "--height",
            str(int(height)),
            "--idle_hold",
            "--loop_bg",
        ],
        cwd=str(cwd),
        env=env,
    )


def _start_m0_worker_tcp(
    *,
    py: Path,
    m0_repo: Path,
    host: str,
    port: int,
    env: dict[str, str],
) -> subprocess.Popen:
    worker_script = m0_repo / "src" / "m0_persistent_worker.py"
    if not worker_script.exists():
        raise FileNotFoundError(f"missing m0 worker: {worker_script}")

    return subprocess.Popen(
        [
            str(py),
            str(worker_script),
            "--tcp",
            "--host",
            str(host),
            "--port",
            str(int(port)),
        ],
        cwd=str(m0_repo),
        env=env,
    )


def _stop_m0_worker_tcp(proc: subprocess.Popen | None, host: str, port: int) -> None:
    if proc is None or proc.poll() is not None:
        return

    try:
        with socket.create_connection((host, int(port)), timeout=3.0) as sock:
            sock.sendall((json.dumps({"cmd": "quit"}) + "\n").encode("utf-8"))
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def _call_watch_stream_mouth_and_render_m0(kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(_watch_stream_mouth_and_render_m0)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return _watch_stream_mouth_and_render_m0(**filtered)


async def _run(args: argparse.Namespace) -> int:
    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    py = m1_repo / ".venv" / "Scripts" / "python.exe"

    out_root = m1_repo / "out" / "obs_realtime_session_loop" / str(args.session_id)
    pipeline_dir = out_root / "01_audio_input_smoke_pipeline"

    # turnごとに作るため、ここでは初期化しない
    bridge_dir = None
    stream_mouth_dir = None
    pcm_stream_chunks_dir = None
    m0_stream_dir = None

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    watch_fg_dir = (
        Path(args.watch_fg_dir).resolve()
        if args.watch_fg_dir
        else m1_repo / "out" / "obs_stream_session_loop" / "fg"
    )
    if args.clean_fg and watch_fg_dir.exists():
        shutil.rmtree(watch_fg_dir)
    watch_fg_dir.mkdir(parents=True, exist_ok=True)

    virtualcam_script = m1_repo / "scripts" / "live_runtime" / "run_virtualcam_persistent.py"
    audio_script = m1_repo / "scripts" / "live_runtime" / "dev_audio_chunk_player_persistent.py"
    knn_script = m3_repo / "tools" / "knn_from_formant_raw_to_mouth_timeline.py"

    base_cfg_path = (
        Path(args.m0_base_config).resolve()
        if args.m0_base_config
        else m0_repo / "configs" / "smoke_pose_improved.yaml"
    )
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    env = os.environ.copy()

    cam_proc = None
    m0_proc = None
    audio_player_proc = None
    # turnごとに作るため、ここでは初期化しない
    audio_stop_event: Event | None = None
    m0_stop_event: Event | None = None
    producer_done_event: Event | None = None
    mouth_updated_event: Event | None = None
    audio_thread: Thread | None = None

    try:
        print("[session_loop] start virtualcam", flush=True)
        cam_proc = _start_virtualcam(
            py=py,
            script=virtualcam_script,
            fg_dir=watch_fg_dir,
            bg_video=Path(args.bg_video).resolve(),
            fps=int(args.fps),
            width=int(args.width),
            height=int(args.height),
            cwd=m1_repo,
            env=env,
        )

        print("[session_loop] start m0 tcp worker", flush=True)
        m0_proc = _start_m0_worker_tcp(
            py=py,
            m0_repo=m0_repo,
            host=str(args.m0_worker_host),
            port=int(args.m0_worker_port),
            env=env,
        )

        time.sleep(1.0)

        print("[session_loop] start audio player", flush=True)
        audio_player_proc = _start_audio_player(
            py=py,
            audio_script=audio_script,
            audio_device=str(args.audio_device),
            cwd=m1_repo,
            env=env,
        )

        mouth_streamer = None

        m0_result_box: dict[str, Any] = {}
        m0_error_box: list[BaseException] = []
        m0_thread: Thread | None = None
        # turnごとに作るため、ここでは初期化しない
        m0_stream_dir_current: Path | None = None
        mouth_json_current: Path | None = None
        turn_start_perf_current: float | None = None
        frame_offset_current = 0
        next_frame_offset = 0

        inline_emo_queue: dict[int, str] = {}

        if args.inline_emo_queue_jsonl:
            inline_emo_queue = _load_inline_emo_queue_jsonl(
                Path(args.inline_emo_queue_jsonl).resolve()
            )
            print(
                f"[session_loop][inline_emo_queue] loaded={len(inline_emo_queue)} "
                f"path={Path(args.inline_emo_queue_jsonl).resolve()}",
                flush=True,
            )

        def _m0_target() -> None:
            try:
                kwargs = dict(
                    py=py,
                    m0_repo=m0_repo,
                    m1_repo=m1_repo,
                    m3_repo=m3_repo,
                    base_cfg=base_cfg,
                    pose_json=Path(args.pose_json).resolve(),
                    mouth_json=mouth_json_current,
                    session_id=str(args.session_id),
                    work_dir=m0_stream_dir_current,
                    watch_fg_dir=watch_fg_dir,
                    env=env,
                    frame_offset=int(frame_offset_current),
                    step_ms=int(args.step_ms),
                    chunk_len_ms=int(args.stream_mouth_m0_chunk_len_ms),
                    fps=int(args.fps),
                    turn_start_perf=turn_start_perf_current,
                    stop_event=m0_stop_event,
                    producer_done_event=producer_done_event,
                    mouth_updated_event=mouth_updated_event,
                    inline_emo_id=(
                        str(inline_emo_id_for_turn)
                        if bool(args.inline_emo_tag_mode)
                        else None
                    ),
                    live_emo_id_getter=(
                        (lambda: turn_state.get("live_emo_id"))
                        if bool(args.inline_emo_tag_mode)
                        else None
                    ),
                    m0_worker_proc=None,
                    m0_worker_host=str(args.m0_worker_host),
                    m0_worker_port=int(args.m0_worker_port),
                )
                m0_result_box["result"] = _call_watch_stream_mouth_and_render_m0(kwargs)
            except BaseException as e:
                m0_error_box.append(e)

        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing env var: {args.api_key_env}")

        client = genai.Client(
            api_key=api_key,
            http_options={"api_version": str(args.api_version)},
        )

        if args.inline_emo_tag_mode:
            system_instruction = (
                "あなたは感情豊かな猫キャラです。"
                "ユーザーの音声に短く自然な日本語で返答してください。"
                "必ず音声で返答してください。"

                "返答の冒頭に、必ず次の形式で感情タグを1つだけ出力してください。"
                "[emo:<emo_id>]"

                "この [emo:...] はシステム制御用タグです。"
                "絶対に音声として読み上げてはいけません。"
                "タグ直後に短い自然な返答本文を開始してください。"

                "使用できるemo_idは以下のみです。"
                "1_1: normal / calm / friendly / default。通常・穏やか・軽い肯定。"
                "1_2: happy / cheerful。嬉しい・明るい・楽しい。"
                "2_0: surprised。驚き・反応が強い。"
                "9_1: sleepy / sad / low-energy。眠い・寂しい・弱い反応。"
                "9_2: very sleepy / very sad / ending mood。とても眠い・終了間近・かなり弱い反応。"

                "選択ルール:"
                "通常の短い返答では [emo:1_1] を選んでください。"
                "楽しい、褒められた、嬉しい内容では [emo:1_2] を選んでください。"
                "驚いた、予想外、びっくりする内容では [emo:2_0] を選んでください。"
                "眠い、疲れた、寂しい、しょんぼりした内容では [emo:9_1] を選んでください。"
                "非常に眠い、もう寝そう、配信終了に近い内容では [emo:9_2] を選んでください。"

                "重要:"
                "毎回 [emo:1_1] に固定してはいけません。"
                "ユーザー発話の意味と会話文脈に応じて最適なemo_idを選んでください。"
                "返答は短く、1文程度にしてください。"
            )
        else:
            system_instruction = (
                "あなたは感情豊かな猫キャラです。"
                "ユーザーの音声に短く自然な日本語で返答してください。"
                "必ず音声で返答してください。"
                "返答前に set_emotion を1回呼んでください。"
            )

        config = _build_live_config(
            system_instruction,
            enable_tools=not bool(args.inline_emo_tag_mode),
            output_audio_transcription=bool(args.output_audio_transcription),
        )

        turn_state: dict[str, Any] = {}
        recv_stop = asyncio.Event()

        print(f"[session_loop] connect model={args.model}", flush=True)

        async with client.aio.live.connect(model=args.model, config=config) as session:
            try:
                for i in range(int(args.turns)):
                    turn_no = i + 1
                    print(f"[session_loop] turn={turn_no}", flush=True)

                    inline_emo_id_for_turn = str(args.inline_emo_id)

                    # Priority:
                    # 1. JSONL queue/probe input
                    # 2. comma-separated --inline_emo_ids
                    # 3. single --inline_emo_id
                    if int(turn_no) in inline_emo_queue:
                        inline_emo_id_for_turn = str(inline_emo_queue[int(turn_no)])
                    elif args.inline_emo_ids:
                        inline_emo_ids = [
                            x.strip()
                            for x in str(args.inline_emo_ids).split(",")
                            if x.strip()
                        ]

                        if inline_emo_ids:
                            idx = min(i, len(inline_emo_ids) - 1)
                            inline_emo_id_for_turn = inline_emo_ids[idx]

                    print(
                        f"[session_loop][inline_emo] turn={turn_no} emo_id={inline_emo_id_for_turn}",
                        flush=True,
                    )

                    turn_dir = out_root / f"turn_{turn_no:03d}"
                    bridge_dir = turn_dir / "01_audio_stream_bridge"
                    stream_mouth_dir = bridge_dir / "stream_mouth"
                    pcm_stream_chunks_dir = bridge_dir / "pcm_stream_chunks"
                    m0_stream_dir = turn_dir / "03_stream_mouth_m0"

                    bridge_dir.mkdir(parents=True, exist_ok=True)
                    stream_mouth_dir.mkdir(parents=True, exist_ok=True)
                    pcm_stream_chunks_dir.mkdir(parents=True, exist_ok=True)

                    mouth_streamer_json = stream_mouth_dir / "mouth_streamer.json"
                    mouth_raw_json = stream_mouth_dir / "mouth_timeline.formant.raw.json"
                    mouth_json = stream_mouth_dir / "mouth.json"
                    audio_response_pcm = bridge_dir / "audio_response.pcm"

                    mouth_streamer = MouthStreamerOC(
                        out_json=str(mouth_streamer_json),
                        session_id=f"{args.session_id}_turn{turn_no:03d}",
                        cfg=MouthOCConfig(
                            step_ms=int(args.step_ms),
                            window_ms=int(args.mouth_window_ms),
                            analysis_sr=int(args.mouth_analysis_sr),
                            input_sr_default=24000,
                            rms_thr=float(args.mouth_rms_thr),
                            vad_energy_thr=float(args.mouth_vad_energy_thr),
                            vad_min_speech_ms=int(args.mouth_vad_min_speech_ms),
                            vad_min_silence_ms=int(args.mouth_vad_min_silence_ms),
                            open_id=int(args.mouth_open_id),
                            close_id=int(args.mouth_close_id),
                            flush_every_frames=int(args.mouth_flush_every_frames),
                            max_buffer_s=float(args.mouth_max_buffer_s),
                            vowel_mode=str(args.mouth_vowel_mode),
                            formant_window_ms=int(args.mouth_formant_window_ms),
                            formant_max_hz=int(args.mouth_formant_max_hz),
                        ),
                    )

                    turn_state.clear()
                    turn_state["active_turn"] = None
                    turn_state["first_audio_sec"] = None
                    turn_state["first_mouth_json_logged"] = False
                    turn_state["last_audio_perf"] = None
                    turn_state["audio_chunk_idx_log"] = 0
                    turn_state["live_emo_id"] = None
                    turn_state["debug_receive_raw_count"] = 0

                    sent_bytes = await _send_mic_once(
                        session=session,
                        duration_s=float(args.mic_send_max_s),
                        input_sr=int(args.input_sr),
                        chunk_ms=int(args.step_ms),
                    )

                    if args.audio_stream_end_per_turn:
                        await session.send_realtime_input(audio_stream_end=True)
                        print(
                            f"[session_loop][audio_stream_end_sent] turn={turn_no}",
                            flush=True,
                        )

                    # mic送信完了後、response_trigger直前からこのturnのfirst_audio計測を開始する
                    turn_state["active_turn"] = turn_no
                    turn_state["turn_start_perf"] = time.perf_counter()
                    turn_state["first_audio_sec"] = None

                    # このturn用に M0 watcher を起動
                    m0_stop_event = Event()
                    producer_done_event = Event()
                    mouth_updated_event = Event()
                    m0_stream_dir_current = m0_stream_dir
                    mouth_json_current = mouth_json
                    turn_start_perf_current = float(turn_state["turn_start_perf"])
                    frame_offset_current = int(next_frame_offset)

                    m0_thread = Thread(target=_m0_target, daemon=True)
                    m0_thread.start()

                    # active_turn 設定後に audio watcher を起動
                    turn_audio_stop_event = Event()
                    audio_thread = _watch_stream_pcm_chunks(
                        audio_player_proc=audio_player_proc,
                        pcm_stream_chunks_dir=pcm_stream_chunks_dir,
                        audio_device=str(args.audio_device),
                        stop_event=turn_audio_stop_event,
                    )

                    # active_turn 設定後に receiver を起動
                    recv_stop = asyncio.Event()
                    recv_task = asyncio.create_task(
                        _receive_loop(
                            session=session,
                            stop_event=recv_stop,
                            pcm_stream_chunks_dir=pcm_stream_chunks_dir,
                            audio_response_pcm=audio_response_pcm,
                            mouth_streamer=mouth_streamer,
                            mouth_streamer_json=mouth_streamer_json,
                            mouth_raw_json=mouth_raw_json,
                            mouth_json=mouth_json,
                            knn_script=knn_script,
                            gt_glob=str(m3_repo / "data" / "knn_db" / "*.f1f2.json"),
                            step_ms=int(args.step_ms),
                            turn_state=turn_state,
                            mouth_updated_event=mouth_updated_event,
                            drop_initial_audio_ms=(
                                int(args.drop_initial_audio_ms)
                                if bool(args.inline_emo_tag_mode)
                                else 0
                            ),
                            debug_receive=bool(args.debug_receive),
                            debug_receive_raw=bool(args.debug_receive_raw),
                        )
                    )

                    response_trigger = str(args.response_trigger)

                    if args.response_triggers:
                        trigger_list = [
                            x.strip()
                            for x in str(args.response_triggers).split("|")
                            if x.strip()
                        ]

                        if trigger_list:
                            idx_trigger = min(i, len(trigger_list) - 1)
                            response_trigger = trigger_list[idx_trigger]

                    print(
                        f"[session_loop][response_trigger] "
                        f"turn={turn_no} text={response_trigger}",
                        flush=True,
                    )

                    await session.send_realtime_input(text=response_trigger)

                    input_audio_ms = int(round((sent_bytes // 2) * 1000.0 / int(args.input_sr)))
                    print(
                        f"[session_loop][turn_sent] turn={turn_no} input_audio_ms={input_audio_ms}",
                        flush=True,
                    )

                    # first_audio 到着待ち
                    wait_t0 = time.perf_counter()

                    while True:
                        if turn_state.get("first_audio_sec") is not None:
                            print(
                                f"[session_loop][turn_first_audio_detected] turn={turn_no}",
                                flush=True,
                            )
                            break

                        wait_sec = time.perf_counter() - wait_t0

                        if wait_sec >= float(args.turn_first_audio_timeout_s):
                            print(
                                f"[session_loop][WARN] first_audio timeout turn={turn_no}",
                                flush=True,
                            )
                            break

                        await asyncio.sleep(0.02)

                    # このturnの応答音声が止まるまで待つ
                    drain_t0 = time.perf_counter()
                    last_seen_audio_perf = turn_state.get("last_audio_perf")

                    while True:
                        latest_audio_perf = turn_state.get("last_audio_perf")

                        # まだ1回も音声が来ていない場合も、短時間は待つ
                        if latest_audio_perf is None:
                            if time.perf_counter() - drain_t0 >= float(args.turn_idle_wait_s):
                                break
                            await asyncio.sleep(0.05)
                            continue

                        idle_sec = time.perf_counter() - float(latest_audio_perf)
                        if idle_sec >= float(args.turn_idle_wait_s):
                            break

                        # 安全上限
                        if time.perf_counter() - drain_t0 >= 3.0:
                            break

                        await asyncio.sleep(0.05)

                    recv_stop.set()
                    recv_task.cancel()
                    try:
                        await recv_task
                    except asyncio.CancelledError:
                        pass
                    except BaseException:
                        pass

                    # このturn用 audio watcher を停止
                    turn_audio_stop_event.set()
                    if audio_thread is not None:
                        audio_thread.join(timeout=2.0)
                        audio_thread = None

                    # このturn用 M0 watcher を停止
                    if producer_done_event is not None:
                        producer_done_event.set()
                    if m0_stop_event is not None:
                        m0_stop_event.set()
                    if m0_thread is not None:
                        m0_thread.join(timeout=5)
                        m0_thread = None

                    # 次turnのFGファイル名が 00000000.png に戻らないように、
                    # turnごとのM0出力フレーム数を累積する。
                    turn_m0_result = m0_result_box.get("result")
                    if isinstance(turn_m0_result, dict):
                        next_frame_offset += int(turn_m0_result.get("total_frames", 0) or 0)
                        print(
                            f"[session_loop][frame_offset] next_frame_offset={next_frame_offset}",
                            flush=True,
                        )

                    m0_result_box.clear()

            finally:
                pass

        if m0_error_box:
            raise m0_error_box[0]

        total_frames = 0
        chunks_n = 0
        if "result" in m0_result_box:
            total_frames = int(m0_result_box["result"].get("total_frames", 0))
            chunks_n = int(m0_result_box["result"].get("rendered_chunks", 0))

        summary = {
            "session_id": str(args.session_id),
            "turns": int(args.turns),
            "watch_fg_dir": str(watch_fg_dir),
            "chunks_n": chunks_n,
            "total_frames": total_frames,
            "stream_mouth_m0_chunk_len_ms": int(args.stream_mouth_m0_chunk_len_ms),
        }
        summary_json = out_root / "run_mic_input_obs_realtime_session_loop.summary.json"
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print("[run_mic_input_obs_realtime_session_loop][OK]", flush=True)
        print(f"  summary_json : {summary_json}", flush=True)
        print(f"  chunks_n     : {chunks_n}", flush=True)
        print(f"  total_frames : {total_frames}", flush=True)
        print(f"  watch_fg_dir : {watch_fg_dir}", flush=True)

        return 0

    finally:
        if audio_stop_event is not None:
            audio_stop_event.set()
        if audio_player_proc is not None:
            _stop_audio_player(audio_player_proc)

        _stop_m0_worker_tcp(m0_proc, str(args.m0_worker_host), int(args.m0_worker_port))

        if cam_proc is not None and cam_proc.poll() is None:
            cam_proc.terminate()
            try:
                cam_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cam_proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--session_id", required=True)
    ap.add_argument("--m1_repo_root", required=True)
    ap.add_argument("--m3_repo_root", required=True)
    ap.add_argument("--m0_repo_root", required=True)
    ap.add_argument("--m35_repo_root", required=True)

    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--bg_video", required=True)

    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--gap_s", type=float, default=0.8)
    ap.add_argument("--turn_idle_wait_s", type=float, default=0.8)
    ap.add_argument("--turn_first_audio_timeout_s", type=float, default=5.0)
    ap.add_argument("--debug_receive", action="store_true")
    ap.add_argument("--debug_receive_raw", action="store_true")
    ap.add_argument("--output_audio_transcription", action="store_true")
    ap.add_argument("--mic_send_max_s", type=float, default=0.6)

    ap.add_argument("--audio_device", default="15")
    ap.add_argument("--input_sr", type=int, default=16000)

    ap.add_argument("--model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--api_version", default="v1alpha")
    ap.add_argument("--api_key_env", default="GEMINI_API_KEY")

    ap.add_argument("--response_trigger", default="短く返答してください。返答前にset_emotionを1回呼んでください。")
    ap.add_argument(
        "--response_triggers",
        default=None,
        help=(
            "Per-turn response triggers separated by '|'. "
            "Example: 'sleepy|normal|very sleepy'"
        ),
    )
    ap.add_argument("--inline_emo_tag_mode", action="store_true")
    ap.add_argument("--inline_emo_id", default="1_1")
    ap.add_argument(
        "--inline_emo_ids",
        default=None,
        help="Comma-separated emo_id list per turn, e.g. 9_1,1_1,9_2",
    )
    ap.add_argument(
        "--inline_emo_queue_jsonl",
        default=None,
        help="JSONL fallback/probe input. Each line: {\"turn\":1,\"emo_id\":\"9_1\"}",
    )
    ap.add_argument("--drop_initial_audio_ms", type=int, default=120)
    ap.add_argument("--audio_stream_end_per_turn", action="store_true")

    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--stream_mouth_m0_chunk_len_ms", type=int, default=120)

    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--watch_fg_dir", default=None)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--clean_fg", action="store_true")

    ap.add_argument("--m0_worker_host", default="127.0.0.1")
    ap.add_argument("--m0_worker_port", type=int, default=39390)
    ap.add_argument("--m0_base_config", default=None)

    ap.add_argument("--mouth_window_ms", type=int, default=240)
    ap.add_argument("--mouth_analysis_sr", type=int, default=16000)
    ap.add_argument("--mouth_rms_thr", type=float, default=0.015)
    ap.add_argument("--mouth_vad_energy_thr", type=float, default=0.0004)
    ap.add_argument("--mouth_vad_min_speech_ms", type=int, default=80)
    ap.add_argument("--mouth_vad_min_silence_ms", type=int, default=120)
    ap.add_argument("--mouth_open_id", type=int, default=1)
    ap.add_argument("--mouth_close_id", type=int, default=0)
    ap.add_argument("--mouth_flush_every_frames", type=int, default=1)
    ap.add_argument("--mouth_max_buffer_s", type=float, default=10.0)
    ap.add_argument("--mouth_vowel_mode", choices=["formant", "simple"], default="formant")
    ap.add_argument("--mouth_formant_window_ms", type=int, default=200)
    ap.add_argument("--mouth_formant_max_hz", type=int, default=5500)

    args = ap.parse_args()

    if int(args.stream_mouth_m0_chunk_len_ms) not in (120, 200, 400):
        raise ValueError("stream_mouth_m0_chunk_len_ms must be 120, 200, or 400")

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())