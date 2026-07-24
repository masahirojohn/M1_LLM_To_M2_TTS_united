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


ALLOWED_EMO_IDS = {
    "1_0",
    "1_1",
    "1_2",
    "2_0",
    "9_1",
    "9_2",
}


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


def _parse_dev_live_emo_events_csv(s: str | None) -> list[dict[str, Any]]:
    """
    例:
      1_1@600,2_0@1000,9_1@1400
    """
    out: list[dict[str, Any]] = []

    if not s:
        return out

    for item in str(s).split(","):
        item = item.strip()
        if not item:
            continue

        emo_id, t = item.split("@", 1)
        out.append(
            {
                "t_ms": int(t),
                "emo_id": str(emo_id),
                "source": "dev_live_emo_events_csv",
            }
        )

    return out


def _extract_emo_id_from_transcription(text: str) -> str | None:
    m = re.search(r"\[emo:([0-9]+_[0-9]+)\]", str(text))
    if not m:
        return None
    return m.group(1)


def _extract_emo_ids_from_transcription(text: str) -> list[str]:
    return [
        m.group(1)
        for m in re.finditer(r"\[emo:([0-9]+_[0-9]+)\]", str(text))
    ]


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


def _normalize_emo_id(raw_emo_id: str | None) -> str:
    emo_id = str(raw_emo_id or "").strip()

    if emo_id in ALLOWED_EMO_IDS:
        return emo_id

    print(
        "[emo_id][fallback]",
        f"raw={emo_id}",
        "fallback=1_1",
        flush=True,
    )

    return "1_1"


async def _send_tool_response_for_call(
    *,
    session: Any,
    call: dict[str, Any],
) -> None:
    name = call.get("name")
    call_id = call.get("id")
    args = call.get("args")

    tool_args = args if isinstance(args, dict) else {}

    if name == "set_emotion":
        emo_id = str(tool_args.get("emo_id", "")).strip()

        emo_id = _normalize_emo_id(emo_id)

        print(
            "[emo_id][accepted]",
            f"emo_id={emo_id}",
            flush=True,
        )

        active_emo_id = emo_id

        tool_args = dict(tool_args)
        tool_args["emo_id"] = active_emo_id

    response = {
        "ok": True,
        "handled": True,
        "name": name,
        "args": tool_args,
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


DEFAULT_PROMPT_FILES = [
    "00_base_system.txt",
    "10_cat_profile.txt",
    "20_normal_prompt.txt",
    "30_battle_prompt.txt",
    "40_sleepy_prompt.txt",
    "50_zoom_prompt.txt",
    "60_tiktok_prompt.txt",
    "70_event_prompt.txt",
    "80_opponent_meta.txt",
]


def _load_prompt_text(path: Path) -> str:
    if not path.exists():
        print(f"[prompt_loader][skip_missing] path={path}", flush=True)
        return ""

    text = path.read_text(encoding="utf-8-sig").strip()

    print(
        "[prompt_loader][loaded]",
        f"path={path}",
        f"chars={len(text)}",
        flush=True,
    )

    return text


def _build_system_instruction_from_prompt_dir(
    *,
    prompt_dir: Path,
    prompt_files: list[str] | None = None,
    inline_emo_tag_mode: bool,
    audio_priority_mode: bool,
) -> str:
    files = prompt_files or DEFAULT_PROMPT_FILES

    parts: list[str] = []

    for name in files:
        text = _load_prompt_text(prompt_dir / name)
        if text:
            parts.append(f"\n\n# {name}\n{text}")

    if inline_emo_tag_mode:
        parts.append(
            "\n\n# inline_emo_tag_mode\n"
            "返答の冒頭に、必ず次の形式で感情タグを1つだけ出力してください。\n"
            "[emo:<emo_id>]\n"
            "この [emo:.] はシステム制御用タグです。\n"
            "絶対に音声として読み上げてはいけません。\n"
            "タグ直後に短い自然な返答本文を開始してください。\n"
        )
    else:
        parts.append(
            "\n\n# set_emotion_mode\n"
            "返答前に set_emotion を1回呼んでください。\n"
        )

    if audio_priority_mode:
        parts.append(
            "\n\n# audio_priority_mode\n"
            "最重要: 必ず音声で返答してください。\n"
            "テキスト説明だけで終わってはいけません。\n"
            "返答は短く、1文以内にしてください。\n"
            "長い説明、箇条書き、内部状態の説明は禁止です。\n"
            "音声生成を最優先し、すぐに話し始めてください。\n"
        )

    system_instruction = "\n".join(parts).strip()

    if not system_instruction:
        raise RuntimeError(f"empty system_instruction: prompt_dir={prompt_dir}")

    print(
        "[prompt_loader][system_instruction]",
        f"prompt_dir={prompt_dir}",
        f"files={len(files)}",
        f"chars={len(system_instruction)}",
        flush=True,
    )

    return system_instruction


def _build_live_config(
    system_instruction: str,
    *,
    enable_tools: bool = True,
    output_audio_transcription: bool = False,
) -> types.LiveConnectConfig:
    extra_kwargs: dict[str, Any] = {}

    if output_audio_transcription:
        extra_kwargs["output_audio_transcription"] = {}

    # Phase 1 / 方式2: server VAD を明示 OFF。クライアントが activity_start/end を送る。
    extra_kwargs["realtime_input_config"] = types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            disabled=True,
        ),
    )

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
    device: int | str | None = None,
    stop_event: asyncio.Event | None = None,
    mic_gate_ref: dict[str, str] | None = None,
    # --- Phase 1: client / local RMS VAD（口形用 mouth_vad_* とは別）---
    mic_vad_end_enabled: bool = True,
    mic_vad_rms_threshold: float = 0.015,
    mic_vad_end_rms_threshold: float | None = None,
    mic_vad_min_voice_ms: int = 200,
    mic_vad_silence_ms: int = 700,
    mic_vad_min_listen_ms: int = 800,
    mic_vad_debug: bool = False,
    send_activity_signals: bool = True,
) -> int:
    """
    Mic PCM を Live API へ送信する。

    automatic_activity_detection=disabled 時は、クライアント VAD に合わせて
    activity_start / activity_end を送る（audio_stream_end は使わない）。
    """
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError("sounddevice / numpy required") from e

    block_samples = int(input_sr * chunk_ms / 1000.0)
    total_blocks = max(1, int((duration_s * 1000.0) / chunk_ms))

    actual_chunk_ms = (
        block_samples / float(input_sr) * 1000.0
        if input_sr > 0
        else float(chunk_ms)
    )
    min_voice_blocks = max(
        1,
        int(round(float(mic_vad_min_voice_ms) / actual_chunk_ms)),
    )
    silence_blocks_limit = max(
        1,
        int(round(float(mic_vad_silence_ms) / actual_chunk_ms)),
    )
    min_listen_blocks = max(
        1,
        int(round(float(mic_vad_min_listen_ms) / actual_chunk_ms)),
    )

    voice_blocks = 0
    silence_blocks = 0
    has_spoken = False
    activity_started = False
    vad_ended = False

    vad_start_threshold = float(mic_vad_rms_threshold)
    vad_end_threshold = (
        float(mic_vad_end_rms_threshold)
        if mic_vad_end_rms_threshold is not None
        else vad_start_threshold
    )

    if mic_vad_debug or mic_vad_end_enabled:
        print(
            "[mic_vad][INIT]",
            f"enabled={bool(mic_vad_end_enabled)}",
            f"chunk_ms={actual_chunk_ms:.2f}",
            f"voice_blocks={min_voice_blocks}",
            f"silence_blocks={silence_blocks_limit}",
            f"min_listen_blocks={min_listen_blocks}",
            f"start_threshold={vad_start_threshold:.5f}",
            f"end_threshold={vad_end_threshold:.5f}",
            f"send_activity_signals={bool(send_activity_signals)}",
            flush=True,
        )

    q: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        x = indata[:, 0]
        x = np.clip(x, -1.0, 1.0)
        pcm = (x * 32767.0).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(q.put_nowait, pcm)

    sent_bytes = 0
    t0 = time.perf_counter()

    print(
        "[mic_send][BEGIN]",
        f"duration_s={float(duration_s):.3f}",
        f"chunk_ms={int(chunk_ms)}",
        f"device={device}",
        flush=True,
    )

    input_device = device
    if input_device is not None:
        try:
            input_device = int(input_device)
        except (TypeError, ValueError):
            input_device = str(input_device)

    async def _emit_activity_start(i: int) -> None:
        nonlocal activity_started
        if activity_started or not send_activity_signals:
            return
        await session.send_realtime_input(activity_start=types.ActivityStart())
        activity_started = True
        print(
            "[mic_vad][ACTIVITY_START]",
            f"i={i}",
            f"elapsed_s={time.perf_counter() - t0:.3f}",
            flush=True,
        )

    async def _emit_activity_end(reason: str) -> None:
        if not activity_started or not send_activity_signals:
            return
        await session.send_realtime_input(activity_end=types.ActivityEnd())
        print(
            "[mic_vad][ACTIVITY_END]",
            f"reason={reason}",
            f"elapsed_s={time.perf_counter() - t0:.3f}",
            flush=True,
        )

    # VAD OFF（固定長）でも AAD disabled のため、先頭で activity_start が必要。
    if send_activity_signals and not mic_vad_end_enabled:
        await _emit_activity_start(0)

    try:
        with sd.InputStream(
            samplerate=int(input_sr),
            channels=1,
            dtype="float32",
            blocksize=block_samples,
            callback=callback,
            device=input_device,
        ):
            for i in range(total_blocks):
                if stop_event is not None and stop_event.is_set():
                    print(
                        "[mic_send][CUT_IN_STOP]",
                        f"i={i}",
                        f"elapsed_s={time.perf_counter() - t0:.3f}",
                        flush=True,
                    )
                    break

                try:
                    chunk = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    print(
                        f"[session_loop][WARN] mic chunk timeout "
                        f"i={i} total_blocks={total_blocks}; stop sending mic audio",
                        flush=True,
                    )
                    break

                if stop_event is not None and stop_event.is_set():
                    print(
                        "[mic_send][CUT_IN_STOP_BEFORE_SEND]",
                        f"i={i}",
                        f"elapsed_s={time.perf_counter() - t0:.3f}",
                        flush=True,
                    )
                    break

                # --- client RMS VAD（発話開始 / 無音終了）---
                if mic_vad_end_enabled:
                    samples = np.frombuffer(chunk, dtype=np.int16)
                    if samples.size > 0:
                        rms = float(
                            np.sqrt(
                                np.mean(
                                    (samples.astype(np.float32) / 32768.0) ** 2
                                )
                            )
                        )
                    else:
                        rms = 0.0

                    if not has_spoken:
                        if rms >= vad_start_threshold:
                            voice_blocks += 1
                            if voice_blocks >= min_voice_blocks:
                                has_spoken = True
                                silence_blocks = 0
                                if mic_vad_debug:
                                    print(
                                        "[mic_vad][VOICE_START]",
                                        f"i={i}",
                                        f"rms={rms:.5f}",
                                        f"start_threshold={vad_start_threshold:.5f}",
                                        flush=True,
                                    )
                                await _emit_activity_start(i)
                        else:
                            voice_blocks = 0
                    else:
                        if rms >= vad_end_threshold:
                            if mic_vad_debug and silence_blocks > 0:
                                print(
                                    "[mic_vad][SILENCE_RESET]",
                                    f"i={i}",
                                    f"rms={rms:.5f}",
                                    f"silence_blocks={silence_blocks}/{silence_blocks_limit}",
                                    flush=True,
                                )
                            silence_blocks = 0
                        else:
                            silence_blocks += 1
                            if mic_vad_debug and (
                                silence_blocks == 1
                                or silence_blocks % 5 == 0
                                or silence_blocks >= silence_blocks_limit
                            ):
                                print(
                                    "[mic_vad][SILENCE]",
                                    f"i={i}",
                                    f"rms={rms:.5f}",
                                    f"silence_blocks={silence_blocks}/{silence_blocks_limit}",
                                    f"min_listen_ok={i >= min_listen_blocks}",
                                    flush=True,
                                )

                        if (
                            i >= min_listen_blocks
                            and silence_blocks >= silence_blocks_limit
                        ):
                            vad_ended = True
                            if mic_vad_debug:
                                print(
                                    "[mic_vad][END]",
                                    f"i={i}",
                                    f"rms={rms:.5f}",
                                    flush=True,
                                )
                            # 終端チャンクは送らず、activity_end でターン完結
                            break

                mic_gate_state = (
                    str(mic_gate_ref.get("value", "open")).strip().lower()
                    if mic_gate_ref is not None
                    else "open"
                )

                if mic_gate_state == "mute":
                    if i == 0:
                        print(
                            "[battle_mic_gate][MUTED]",
                            f"i={i}",
                            flush=True,
                        )
                    continue

                # VAD ON で未発話の間は PCM を送らない（activity_start 前の先行送信を防ぐ）
                if mic_vad_end_enabled and not has_spoken:
                    continue

                await session.send_realtime_input(
                    audio=types.Blob(
                        data=chunk,
                        mime_type=f"audio/pcm;rate={int(input_sr)}",
                    )
                )
                sent_bytes += len(chunk)

                print(
                    "[mic_send][CHUNK]",
                    f"i={i}",
                    f"bytes={len(chunk)}",
                    f"elapsed_s={time.perf_counter() - t0:.3f}",
                    flush=True,
                )
    finally:
        if stop_event is not None and stop_event.is_set():
            end_reason = "cut_in"
        elif vad_ended:
            end_reason = "vad_silence"
        elif mic_vad_end_enabled and not has_spoken:
            end_reason = "no_speech"
        else:
            end_reason = "max_duration"
        await _emit_activity_end(end_reason)

    print(
        "[mic_send][DONE]",
        f"sent_bytes={sent_bytes}",
        f"elapsed_s={time.perf_counter() - t0:.3f}",
        f"activity_started={activity_started}",
        f"has_spoken={has_spoken}",
        flush=True,
    )

    return sent_bytes


# --- [ADD] Battle Runtime: admin CLI interrupt helpers ---
def _build_battle_interrupt_prompt(raw_text: str) -> str:
    raw_text = str(raw_text).strip()
    return (
        "【管理者割り込み指示】"
        "相手の発話終了を待たず、今すぐ短く被せて話してください。"
        "人気ライバーのように、テンポよく、1文で返してください。"
        f"指示内容: {raw_text}"
    )


# --- [ADD] Battle Runtime: queue file loader ---
def _read_battle_interrupt_queue_file(path: Path) -> list[str]:
    path = Path(path).resolve()
    if not path.exists():
        return []

    for enc in ("utf-8-sig", "cp932", "utf-16"):
        try:
            return [
                line.strip()
                for line in path.read_text(encoding=enc).splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        except UnicodeDecodeError:
            continue

    return []


def _clear_audio_player_queue(audio_player_proc: subprocess.Popen | None) -> bool:
    if audio_player_proc is None:
        print("[audio_player][clear_queue_skip] proc=None", flush=True)
        return False

    if audio_player_proc.stdin is None:
        print("[audio_player][clear_queue_skip] stdin=None", flush=True)
        return False

    if audio_player_proc.poll() is not None:
        print("[audio_player][clear_queue_skip] proc_not_running", flush=True)
        return False

    try:
        audio_player_proc.stdin.write(
            json.dumps({"cmd": "clear_queue"}, ensure_ascii=False) + "\n"
        )
        audio_player_proc.stdin.flush()
        print("[audio_player][clear_queue_sent]", flush=True)
        return True
    except Exception as e:
        print(
            f"[audio_player][clear_queue_error] {type(e).__name__}: {e}",
            flush=True,
        )
        return False


def _start_battle_interrupt_cli_thread(
    *,
    q: asyncio.Queue[str],
    loop: asyncio.AbstractEventLoop,
    prefix: str,
) -> Thread:
    """
    ターミナル入力から Battle 割り込み指示を受け取る。
    例:
      speak 今すぐ煽って
    """
    prefix = str(prefix)

    def _target() -> None:
        print(
            f"[battle_interrupt][cli_ready] type: {prefix}<message>",
            flush=True,
        )

        while True:
            try:
                line = input()
            except EOFError:
                break
            except BaseException as e:
                print(
                    f"[battle_interrupt][cli_warn] {type(e).__name__}: {e}",
                    flush=True,
                )
                break

            line = str(line).strip()
            if not line:
                continue

            if prefix and not line.startswith(prefix):
                continue

            text = line[len(prefix):].strip() if prefix else line
            if not text:
                continue

            print(f"[battle_interrupt][queued] text={text}", flush=True)
            loop.call_soon_threadsafe(q.put_nowait, text)

    th = Thread(target=_target, daemon=True)
    th.start()
    return th


async def _battle_interrupt_send_loop(
    *,
    session: Any,
    q: asyncio.Queue[str],
    stop_event: asyncio.Event,
    cut_in_event: asyncio.Event | None = None,
) -> None:
    """
    admin CLI queue から受け取った割り込み指示を、
    現在の Live API session へ即時 text send する。
    """
    while not stop_event.is_set():
        try:
            raw_text = await asyncio.wait_for(q.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue

        prompt = _build_battle_interrupt_prompt(raw_text)

        try:
            send_t0 = time.perf_counter()

            print(
                "[battle_interrupt][send_begin]",
                f"text={raw_text}",
                flush=True,
            )

            if cut_in_event is not None:
                cut_in_event.set()
                print(
                    "[battle_talkover][cut_in_requested]",
                    f"text={raw_text}",
                    flush=True,
                )

            await session.send_realtime_input(text=prompt)

            print(
                "[battle_interrupt][sent]",
                f"text={raw_text}",
                f"send_sec={time.perf_counter() - send_t0:.3f}",
                flush=True,
            )
        except BaseException as e:
            print(
                f"[battle_interrupt][send_error] {type(e).__name__}: {e}",
                flush=True,
            )


# --- [ADD] async task cleanup helper ---
async def _cancel_tasks_safely(
    tasks: list[asyncio.Task | None],
    *,
    tag: str,
) -> None:
    alive: list[asyncio.Task] = []

    for task in tasks:
        if task is None:
            continue
        if task.done():
            continue

        task.cancel()
        alive.append(task)

    if not alive:
        print(
            f"[task_cleanup][SKIP] tag={tag} alive_n=0",
            flush=True,
        )
        return

    results = await asyncio.gather(
        *alive,
        return_exceptions=True,
    )

    cancelled_n = 0
    error_n = 0

    for result in results:
        if isinstance(result, asyncio.CancelledError):
            cancelled_n += 1
        elif isinstance(result, BaseException):
            error_n += 1

    print(
        f"[task_cleanup][DONE] "
        f"tag={tag} "
        f"alive_n={len(alive)} "
        f"cancelled_n={cancelled_n} "
        f"error_n={error_n}",
        flush=True,
    )


# --- [FIX] Battle Runtime: enqueue preloaded queue after first_audio_detected ---
async def _enqueue_battle_interrupt_lines_after_first_audio(
    *,
    q: asyncio.Queue[str],
    lines: list[str],
    interval_s: float,
    turn: int,
) -> None:
    if not lines:
        return

    for idx, text in enumerate(lines):
        print(
            f"[battle_interrupt][after_first_audio_queued] turn={turn} idx={idx} text={text}",
            flush=True,
        )
        await q.put(text)

        if idx < len(lines) - 1:
            await asyncio.sleep(float(interval_s))


# --- [ADD] interrupt mode mapping ---
INTERRUPT_MODE_TO_TEXT = {
    "aggressive": "短く強めに返して",
    "tsukkomi": "短くツッコんで",
    "ignore": "冷たく短く返して",
    "laugh": "少し笑う感じで返して",
    "panic": "少し焦った感じで返して",
}


# --- [ADD] localhost TCP battle interrupt server ---
async def _battle_interrupt_socket_server(
    *,
    host: str,
    port: int,
    q: asyncio.Queue[str],
    stop_event: asyncio.Event,
    abort_event: asyncio.Event | None = None,
) -> None:
    async def _handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        addr = writer.get_extra_info("peername")

        print(
            f"[battle_socket][client_connected] addr={addr}",
            flush=True,
        )

        try:
            while not stop_event.is_set():
                line = await reader.readline()

                if not line:
                    break

                text = line.decode("utf-8").strip()

                if not text:
                    continue

                try:
                    payload = json.loads(text)
                except BaseException as e:
                    print(
                        f"[battle_socket][json_error] {type(e).__name__}: {e}",
                        flush=True,
                    )
                    continue

                if payload.get("type") != "interrupt":
                    continue

                interrupt_text = ""

                mode = str(payload.get("mode", "")).strip()

                if mode:
                    interrupt_text = INTERRUPT_MODE_TO_TEXT.get(mode, "")

                    print(
                        f"[battle_socket][mode] mode={mode} mapped={interrupt_text}",
                        flush=True,
                    )

                if not interrupt_text:
                    interrupt_text = str(payload.get("text", "")).strip()

                if not interrupt_text:
                    continue

                if bool(payload.get("abort", False)) and abort_event is not None:
                    abort_event.set()
                    print(
                        f"[battle_socket][abort_requested] text={interrupt_text}",
                        flush=True,
                    )
                    break

                await q.put(interrupt_text)

                print(
                    f"[battle_socket][queued] text={interrupt_text}",
                    flush=True,
                )

                break

        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except BaseException:
                pass

            print(
                f"[battle_socket][client_closed] addr={addr}",
                flush=True,
            )

    server = await asyncio.start_server(
        _handle_client,
        host=host,
        port=port,
    )

    print(
        f"[battle_socket][LISTEN] {host}:{port}",
        flush=True,
    )

    try:
        async with server:
            while not stop_event.is_set():
                await asyncio.sleep(0.2)
    finally:
        server.close()
        await server.wait_closed()

        print(
            "[battle_socket][STOP]",
            flush=True,
        )


BATTLE_INTERRUPT_PRIORITY_WEIGHT = {
    "normal": 10,
    "battle": 20,
    "critical": 30,
}


def _normalize_battle_interrupt_priority(priority: str | None) -> str:
    priority = str(priority or "normal").strip().lower()

    if priority not in BATTLE_INTERRUPT_PRIORITY_WEIGHT:
        return "normal"

    return priority


def _parse_battle_interrupt_file_control(raw: str) -> dict[str, Any] | None:
    raw = str(raw or "").strip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            if str(obj.get("type", "interrupt")).strip() != "interrupt":
                return None

            text = str(obj.get("text", "")).strip()
            if not text:
                return None

            priority = _normalize_battle_interrupt_priority(
                obj.get("priority")
            )

            expire_sec = float(obj.get("expire_sec", 30.0))

            style = obj.get("style", [])

            if not isinstance(style, list):
                style = []

            style = [str(x).strip() for x in style if str(x).strip()]

            return {
                "type": "interrupt",
                "text": text,
                "priority": priority,
                "priority_weight": int(BATTLE_INTERRUPT_PRIORITY_WEIGHT[priority]),
                "created_at": time.time(),
                "expire_sec": expire_sec,
                "style": style,
            }
    except Exception:
        pass

    return {
        "type": "interrupt",
        "text": raw,
        "priority": "normal",
        "priority_weight": int(BATTLE_INTERRUPT_PRIORITY_WEIGHT["normal"]),
        "created_at": time.time(),
        "expire_sec": 30.0,
    }


# --- [ADD] Battle Runtime: file based interrupt watcher ---
def _start_battle_interrupt_file_thread(
    *,
    path: Path,
    pending_lines: list[str],
    loop: asyncio.AbstractEventLoop,
    poll_s: float,
    immediate_q: asyncio.Queue[str] | None = None,
    immediate_send: bool = False,
) -> Thread:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    last_mtime = 0.0

    def _target() -> None:
        nonlocal last_mtime

        print(
            f"[battle_interrupt][file_ready] path={path} poll_s={float(poll_s):.3f}",
            flush=True,
        )

        while True:
            try:
                stat = path.stat()
                mtime = float(stat.st_mtime)

                if mtime <= last_mtime:
                    time.sleep(float(poll_s))
                    continue

                last_mtime = mtime

                raw = ""
                for enc in ("utf-8-sig", "utf-8", "cp932", "utf-16"):
                    try:
                        raw = path.read_text(encoding=enc).strip()
                        break
                    except UnicodeDecodeError:
                        continue

                control = _parse_battle_interrupt_file_control(raw)

                if control:
                    print(
                        "[battle_interrupt][file_pending_overwrite]",
                        f"priority={control.get('priority')}",
                        f"text={control.get('text')}",
                        flush=True,
                    )

                    def _overwrite_pending_control(control_snapshot: dict[str, Any] | None = control) -> None:
                        if not isinstance(control_snapshot, dict):
                            return

                        if pending_lines:
                            current = pending_lines[-1]

                            if isinstance(current, dict):
                                current_weight = int(current.get("priority_weight", 10))
                            else:
                                current_weight = 10

                            next_weight = int(control_snapshot.get("priority_weight", 10))

                            # 高priorityは上書き。
                            # 同priorityも最新を採用。
                            # 低priorityは既存pendingを維持。
                            if next_weight < current_weight:
                                print(
                                    "[battle_interrupt][file_pending_keep_higher_priority]",
                                    f"current_priority={current.get('priority') if isinstance(current, dict) else 'normal'}",
                                    f"new_priority={control_snapshot.get('priority')}",
                                    f"text={control_snapshot.get('text')}",
                                    flush=True,
                                )
                                return

                        pending_lines.clear()
                        pending_lines.append(control_snapshot)

                    loop.call_soon_threadsafe(_overwrite_pending_control)

                    if bool(immediate_send) and immediate_q is not None:
                        immediate_text = str(control.get("text", "")).strip()

                        if immediate_text:
                            loop.call_soon_threadsafe(
                                immediate_q.put_nowait,
                                immediate_text,
                            )

                            print(
                                "[battle_interrupt][file_immediate_queued]",
                                f"priority={control.get('priority')}",
                                f"text={immediate_text}",
                                flush=True,
                            )

                    # 同じ文面を連続投入できるように、読んだら空にする
                    path.write_text("", encoding="utf-8")

            except BaseException as e:
                print(
                    f"[battle_interrupt][file_warn] {type(e).__name__}: {e}",
                    flush=True,
                )

            time.sleep(float(poll_s))

    th = Thread(target=_target, daemon=True)
    th.start()
    return th


def _load_event_catalog_json(path: Path | str | None) -> dict[str, Any]:
    if not path:
        return {}

    p = Path(path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"event_catalog_json not found: {p}")

    with p.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"event_catalog_json must be object: {p}")

    return obj


def _resolve_event_from_catalog(
    *,
    event_id: str,
    catalog: dict[str, Any],
    fallback_duration_s: float = 0.0,
) -> dict[str, Any]:
    event_id = str(event_id).strip()

    item = catalog.get(event_id, {})

    if item is None:
        item = {}

    if not isinstance(item, dict):
        raise ValueError(f"invalid event catalog item: event_id={event_id} item={item}")

    duration_s = float(
        item.get("duration_s", fallback_duration_s) or fallback_duration_s or 0.0
    )

    return {
        "event_id": event_id,
        "event_mode": str(item.get("event_mode", "bg_only")),
        "bg_video": str(item.get("bg_video", "")),
        "pose_json": str(item.get("pose_json", "")),
        "duration_s": duration_s,
        "audio": bool(item.get("audio", False)),
        "audio_file": str(item.get("audio_file", "")),
        "audio_source": str(item.get("audio_source", "")).strip().lower(),
        "post_control": str(item.get("post_control", "")).strip(),
    }


def _parse_event_runtime_file_text(raw: str) -> dict[str, Any] | None:
    raw = str(raw or "").strip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    if str(obj.get("type", "")).strip() != "event":
        return None

    event_id = str(obj.get("event_id", "")).strip()
    if not event_id:
        return None

    duration_s = float(obj.get("duration_s", 0.0) or 0.0)

    return {
        "type": "event",
        "event_id": event_id,
        "duration_s": duration_s,
        "created_at": time.time(),
    }


def _resolve_m35_path(m35_repo: Path, value: str) -> Path:
    p = Path(str(value or "").strip())
    if p.is_absolute():
        return p.resolve()
    return (Path(m35_repo) / p).resolve()


def _ffprobe_duration_s(path: Path) -> float:
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"missing media for duration probe: {path}")

    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    raw = str(proc.stdout or "").strip()
    return float(raw)


def _extract_event_mp4_audio_to_wav(
    *,
    src_mp4: Path,
    out_wav: Path,
) -> Path:
    src_mp4 = Path(src_mp4).resolve()
    out_wav = Path(out_wav).resolve()

    if not src_mp4.exists():
        raise FileNotFoundError(f"missing mp4 for audio extract: {src_mp4}")

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src_mp4),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-sample_fmt",
            "s16",
            str(out_wav),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return out_wav


def _play_event_wav(
    *,
    audio_player_proc: subprocess.Popen | None,
    m35_repo: Path,
    audio_file: str,
    event_id: str,
) -> bool:
    if audio_player_proc is None:
        print("[event_runtime][play_wav_skip] proc=None", flush=True)
        return False

    if audio_player_proc.stdin is None:
        print("[event_runtime][play_wav_skip] stdin=None", flush=True)
        return False

    if audio_player_proc.poll() is not None:
        print("[event_runtime][play_wav_skip] proc_not_running", flush=True)
        return False

    audio_file = str(audio_file or "").strip()
    if not audio_file:
        print(
            "[event_runtime][play_wav_skip]",
            f"event_id={event_id}",
            "audio_file_empty",
            flush=True,
        )
        return False

    audio_path = Path(audio_file)
    if not audio_path.is_absolute():
        audio_path = (Path(m35_repo) / audio_path).resolve()
    else:
        audio_path = audio_path.resolve()

    if not audio_path.exists():
        print(
            "[event_runtime][play_wav_error]",
            f"event_id={event_id}",
            f"missing={audio_path}",
            flush=True,
        )
        return False

    try:
        payload = {
            "cmd": "play_wav",
            "path": str(audio_path),
            "chunk_id": 9001,
        }
        audio_player_proc.stdin.write(
            json.dumps(payload, ensure_ascii=False) + "\n"
        )
        audio_player_proc.stdin.flush()

        print(
            "[event_runtime][play_wav_sent]",
            f"event_id={event_id}",
            f"path={audio_path}",
            flush=True,
        )
        return True

    except Exception as e:
        print(
            f"[event_runtime][play_wav_error] {type(e).__name__}: {e}",
            flush=True,
        )
        return False


def _write_post_control_file(
    *,
    control_file: Path | None,
    text: str,
    event_id: str,
) -> bool:
    if control_file is None:
        print("[event_runtime][post_control_skip] control_file=None", flush=True)
        return False

    text = str(text or "").strip()
    if not text:
        print(
            "[event_runtime][post_control_skip]",
            f"event_id={event_id}",
            "post_control_empty",
            flush=True,
        )
        return False

    payload = {
        "type": "control",
        "text": text,
    }

    control_file = Path(control_file).resolve()
    control_file.parent.mkdir(parents=True, exist_ok=True)
    control_file.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "[event_runtime][post_control_written]",
        f"event_id={event_id}",
        f"path={control_file}",
        f"text={text}",
        flush=True,
    )

    return True


def _write_bg_override_file(
    *,
    path: Path | None,
    m35_repo: Path,
    bg_video: str,
    duration_s: float,
    event_id: str,
) -> bool:
    if path is None:
        print("[event_runtime][bg_override_skip] path=None", flush=True)
        return False

    bg_video = str(bg_video or "").strip()
    if not bg_video:
        print(
            "[event_runtime][bg_override_skip]",
            f"event_id={event_id}",
            "bg_video_empty",
            flush=True,
        )
        return False

    bg_path = Path(bg_video)

    if not bg_path.is_absolute():
        bg_path = (Path(m35_repo) / bg_path).resolve()
    else:
        bg_path = bg_path.resolve()

    payload = {
        "type": "bg_override",
        "event_id": str(event_id),
        "bg_video": str(bg_path),
        "duration_s": float(duration_s),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "[event_runtime][bg_override_written]",
        f"event_id={event_id}",
        f"bg_video={bg_path}",
        f"duration_s={float(duration_s):.3f}",
        f"path={path}",
        flush=True,
    )

    return True


def _start_event_runtime_file_thread(
    *,
    path: Path,
    poll_s: float,
    battle_mic_gate_ref: dict[str, str],
    audio_player_proc: subprocess.Popen | None,
    event_catalog: dict[str, Any] | None = None,
    bg_override_file: Path | None = None,
    m35_repo: Path | None = None,
    m1_repo: Path | None = None,
    battle_control_file: Path | None = None,
) -> Thread:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    last_mtime = 0.0

    def _open_later(event_id: str, duration_s: float, post_control: str = "") -> None:
        time.sleep(max(0.0, float(duration_s)))
        battle_mic_gate_ref["value"] = "open"
        print(
            "[event_runtime][mic_gate_open]",
            f"event_id={event_id}",
            f"duration_s={duration_s:.3f}",
            flush=True,
        )

        _write_post_control_file(
            control_file=battle_control_file,
            text=post_control,
            event_id=event_id,
        )

    def _target() -> None:
        nonlocal last_mtime

        print(
            f"[event_runtime][file_ready] path={path} poll_s={float(poll_s):.3f}",
            flush=True,
        )

        while True:
            try:
                stat = path.stat()
                mtime = float(stat.st_mtime)

                if mtime <= last_mtime:
                    time.sleep(float(poll_s))
                    continue

                last_mtime = mtime
                raw = path.read_text(encoding="utf-8-sig").strip()
                ev = _parse_event_runtime_file_text(raw)

                if not ev:
                    time.sleep(float(poll_s))
                    continue

                event_id = str(ev["event_id"])
                fallback_duration_s = float(ev.get("duration_s", 0.0) or 0.0)

                resolved_event = _resolve_event_from_catalog(
                    event_id=event_id,
                    catalog=event_catalog or {},
                    fallback_duration_s=fallback_duration_s,
                )

                duration_s = float(resolved_event.get("duration_s", 0.0) or 0.0)

                bg_video_for_event = str(resolved_event.get("bg_video", ""))
                bg_video_path_for_event: Path | None = None

                if m35_repo is not None and bg_video_for_event:
                    bg_video_path_for_event = _resolve_m35_path(
                        Path(m35_repo).resolve(),
                        bg_video_for_event,
                    )

                    if duration_s <= 0:
                        try:
                            duration_s = _ffprobe_duration_s(bg_video_path_for_event)
                            resolved_event["duration_s"] = duration_s
                            print(
                                "[event_runtime][duration_from_mp4]",
                                f"event_id={event_id}",
                                f"duration_s={duration_s:.3f}",
                                f"bg_video={bg_video_path_for_event}",
                                flush=True,
                            )
                        except Exception as e:
                            print(
                                f"[event_runtime][duration_probe_warn] {type(e).__name__}: {e}",
                                flush=True,
                            )

                print(
                    "[event_runtime][resolved]",
                    f"event_id={event_id}",
                    f"event_mode={resolved_event.get('event_mode')}",
                    f"bg_video={resolved_event.get('bg_video')}",
                    f"pose_json={resolved_event.get('pose_json')}",
                    f"duration_s={duration_s:.3f}",
                    f"audio={resolved_event.get('audio')}",
                    f"audio_source={resolved_event.get('audio_source')}",
                    f"audio_file={resolved_event.get('audio_file')}",
                    f"post_control={bool(str(resolved_event.get('post_control', '')).strip())}",
                    flush=True,
                )

                battle_mic_gate_ref["value"] = "mute"
                print(
                    "[event_runtime][trigger]",
                    f"event_id={event_id}",
                    f"duration_s={duration_s:.3f}",
                    flush=True,
                )
                print(
                    "[event_runtime][mic_gate_mute]",
                    f"event_id={event_id}",
                    flush=True,
                )

                _clear_audio_player_queue(audio_player_proc)

                if bool(resolved_event.get("audio", False)) and m35_repo is not None:
                    audio_source = str(resolved_event.get("audio_source", "")).strip().lower()
                    audio_file_for_play = str(resolved_event.get("audio_file", ""))

                    if audio_source == "mp4":
                        try:
                            if bg_video_path_for_event is None:
                                raise ValueError("bg_video_path_for_event is None")

                            cache_wav = (
                                Path(m1_repo)
                                / "out"
                                / "event_audio_cache"
                                / f"{event_id}.wav"
                            )

                            _extract_event_mp4_audio_to_wav(
                                src_mp4=bg_video_path_for_event,
                                out_wav=cache_wav,
                            )

                            audio_file_for_play = str(cache_wav)

                            print(
                                "[event_runtime][mp4_audio_extracted]",
                                f"event_id={event_id}",
                                f"src={bg_video_path_for_event}",
                                f"wav={cache_wav}",
                                flush=True,
                            )

                        except Exception as e:
                            audio_file_for_play = ""
                            print(
                                f"[event_runtime][mp4_audio_extract_error] {type(e).__name__}: {e}",
                                flush=True,
                            )

                    _play_event_wav(
                        audio_player_proc=audio_player_proc,
                        m35_repo=Path(m35_repo).resolve(),
                        audio_file=audio_file_for_play,
                        event_id=event_id,
                    )

                if m35_repo is not None:
                    _write_bg_override_file(
                        path=bg_override_file,
                        m35_repo=Path(m35_repo).resolve(),
                        bg_video=str(resolved_event.get("bg_video", "")),
                        duration_s=duration_s,
                        event_id=event_id,
                    )

                if duration_s > 0:
                    Thread(
                        target=_open_later,
                        args=(
                            event_id,
                            duration_s,
                            str(resolved_event.get("post_control", "")),
                        ),
                        daemon=True,
                    ).start()

                path.write_text("", encoding="utf-8")

            except BaseException as e:
                print(
                    f"[event_runtime][file_warn] {type(e).__name__}: {e}",
                    flush=True,
                )

            time.sleep(float(poll_s))

    th = Thread(target=_target, daemon=True)
    th.start()
    return th


def _parse_battle_control_file_text(raw: str) -> dict[str, Any] | None:
    raw = str(raw or "").strip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            if str(obj.get("type", "control")).strip() != "control":
                return None

            action = str(obj.get("action", "")).strip().lower()
            mic_gate = str(obj.get("mic_gate", "")).strip().lower()
            text = str(obj.get("text", "")).strip()

            if action == "clear":
                return {
                    "type": "control",
                    "action": "clear",
                    "text": "",
                    "mic_gate": mic_gate,
                }

            if mic_gate not in ("", "mute", "open"):
                mic_gate = ""

            return {
                "type": "control",
                "action": action,
                "text": text,
                "mic_gate": mic_gate,
            }

    except Exception:
        pass

    return {
        "type": "control",
        "action": "",
        "text": raw,
        "mic_gate": "",
    }


def _start_battle_control_file_thread(
    *,
    path: Path,
    pending_lines: list[dict[str, Any]],
    loop: asyncio.AbstractEventLoop,
    poll_s: float,
    battle_mic_gate_ref: dict[str, str] | None = None,
) -> Thread:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    last_mtime = 0.0

    def _target() -> None:
        nonlocal last_mtime

        print(
            f"[battle_control][file_ready] path={path} poll_s={float(poll_s):.3f}",
            flush=True,
        )

        while True:
            try:
                stat = path.stat()
                mtime = float(stat.st_mtime)

                if mtime <= last_mtime:
                    time.sleep(float(poll_s))
                    continue

                last_mtime = mtime

                raw = ""
                for enc in ("utf-8-sig", "utf-8", "cp932", "utf-16"):
                    try:
                        raw = path.read_text(encoding=enc).strip()
                        break
                    except UnicodeDecodeError:
                        continue

                control = _parse_battle_control_file_text(raw)

                if control is not None:
                    mic_gate = str(control.get("mic_gate", "")).strip().lower()

                    print(
                        "[battle_control][file_pending_overwrite]",
                        f"text={control.get('text', '')}",
                        f"mic_gate={mic_gate}",
                        f"action={control.get('action', '')}",
                        flush=True,
                    )

                    if mic_gate in ("mute", "open") and battle_mic_gate_ref is not None:
                        def _set_mic_gate(mic_gate_snapshot: str = mic_gate) -> None:
                            battle_mic_gate_ref["value"] = mic_gate_snapshot

                            print(
                                "[battle_mic_gate][file_set]",
                                f"state={mic_gate_snapshot}",
                                flush=True,
                            )

                        loop.call_soon_threadsafe(_set_mic_gate)

                    def _overwrite_pending_control(control_snapshot: dict[str, Any] = control) -> None:
                        pending_lines.clear()
                        pending_lines.append(control_snapshot)

                    loop.call_soon_threadsafe(_overwrite_pending_control)

                    path.write_text("", encoding="utf-8")

            except BaseException as e:
                print(
                    f"[battle_control][file_warn] {type(e).__name__}: {e}",
                    flush=True,
                )

            time.sleep(float(poll_s))

    th = Thread(target=_target, daemon=True)
    th.start()
    return th


def _start_dev_battle_file_writer_thread(
    *,
    path: Path,
    text: str,
    delay_s: float,
) -> Thread:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _target() -> None:
        time.sleep(float(delay_s))

        payload = {
            "type": "interrupt",
            "text": str(text),
        }

        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"[dev_battle_file_writer][wrote] delay_s={float(delay_s):.3f} path={path} text={text}",
            flush=True,
        )

    th = Thread(target=_target, daemon=True)
    th.start()
    return th


# --- [ADD] Battle Runtime: preloaded queue file interrupt ---
def _start_battle_interrupt_queue_file_thread(
    *,
    path: Path,
    q: asyncio.Queue[str],
    loop: asyncio.AbstractEventLoop,
    interval_s: float,
    start_delay_s: float,
) -> Thread:
    """
    事前に複数行を書いた queue file を読み込み、
    interval_s ごとに1行ずつ Battle interrupt queue へ投入する。

    例: battle_interrupt_queue.txt
      今すぐ割り込んで短くツッコんで
      もう一回、さらに強く煽って
    """
    path = Path(path).resolve()

    def _read_lines() -> list[str]:
        if not path.exists():
            return []

        for enc in ("utf-8-sig", "cp932", "utf-16"):
            try:
                return [
                    line.strip()
                    for line in path.read_text(encoding=enc).splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
            except UnicodeDecodeError:
                continue

        return []

    def _target() -> None:
        lines = _read_lines()

        print(
            f"[battle_interrupt][queue_file_ready] path={path} lines={len(lines)} "
            f"start_delay_s={float(start_delay_s):.3f} interval_s={float(interval_s):.3f}",
            flush=True,
        )

        if not lines:
            return

        if float(start_delay_s) > 0:
            time.sleep(float(start_delay_s))

        for idx, text in enumerate(lines):
            print(
                f"[battle_interrupt][queue_file_queued] idx={idx} text={text}",
                flush=True,
            )
            loop.call_soon_threadsafe(q.put_nowait, text)

            if idx < len(lines) - 1:
                time.sleep(float(interval_s))

    th = Thread(target=_target, daemon=True)
    th.start()
    return th


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

            try:
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

                        # output_transcription は分割で届くことがあるため、
                        # turn内で累積して複数 [emo:ID] を検出する。
                        accum = str(turn_state.get("transcription_accum") or "")
                        accum += str(transcription_text)
                        turn_state["transcription_accum"] = accum

                        seen_tags = list(turn_state.get("live_emo_seen_tags") or [])
                        emo_ids = _extract_emo_ids_from_transcription(accum)

                        new_emo_ids: list[str] = []

                        for live_emo_id in emo_ids:
                            if live_emo_id in seen_tags:
                                continue

                            seen_tags.append(live_emo_id)
                            new_emo_ids.append(live_emo_id)

                        base_t_ms = 0
                        if turn_state.get("turn_start_perf") is not None:
                            base_t_ms = int(
                                round(
                                    (
                                        time.perf_counter()
                                        - float(turn_state["turn_start_perf"])
                                    )
                                    * 1000.0
                                )
                            )

                        # 同一 transcription chunk 内に複数 [emo:ID] が来た場合、
                        # 全部同じ t_ms になると expression timeline 上で同時刻イベントになる。
                        # そのため暫定的に 400ms 間隔の仮想 offset を付ける。
                        virtual_emo_interval_ms = 400

                        for local_i, live_emo_id in enumerate(new_emo_ids):
                            live_emo_id = _normalize_emo_id(live_emo_id)
                            turn_state["live_emo_id"] = live_emo_id

                            t_ms = int(base_t_ms + local_i * virtual_emo_interval_ms)

                            events = list(turn_state.get("live_emo_events") or [])
                            events.append(
                                {
                                    "t_ms": t_ms,
                                    "emo_id": live_emo_id,
                                    "source": "output_transcription_virtual_offset",
                                    "base_t_ms": int(base_t_ms),
                                    "virtual_offset_ms": int(local_i * virtual_emo_interval_ms),
                                }
                            )
                            turn_state["live_emo_events"] = events

                            print(
                                f"[transcription][emo_id] "
                                f"active_turn={turn_state.get('active_turn')} "
                                f"emo_id={live_emo_id} "
                                f"t_ms={t_ms} "
                                f"base_t_ms={base_t_ms} "
                                f"virtual_offset_ms={local_i * virtual_emo_interval_ms} "
                                f"n={len(events)}",
                                flush=True,
                            )

                        turn_state["live_emo_seen_tags"] = seen_tags

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

            except Exception as e:
                import traceback

                print(
                    "[session_loop][receive_loop][EXCEPTION]",
                    f"type={type(e).__name__}",
                    f"detail={e}",
                    flush=True,
                )

                traceback.print_exc()

            print(
                "[session_loop][receive_loop][END]",
                f"active_turn={turn_state.get('active_turn')}",
                f"audio_chunks={turn_state.get('audio_chunk_idx_log')}",
                f"got_any={got_any}",
                f"stop_event={stop_event.is_set()}",
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
    bg_override_file: Path | None = None,
) -> subprocess.Popen:
    cmd = [
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
    ]

    if bg_override_file is not None:
        cmd.extend(
            [
                "--bg_override_file",
                str(bg_override_file),
            ]
        )

    return subprocess.Popen(
        cmd,
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
    print(
        "[debug][run_entry]",
        f"file={__file__}",
        f"session_id={args.session_id}",
        f"battle_interrupt_cli={args.battle_interrupt_cli}",
        f"battle_interrupt_file={args.battle_interrupt_file}",
        f"battle_interrupt_queue_file={args.battle_interrupt_queue_file}",
        flush=True,
    )
    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    # --- [ADD] Battle Runtime audio routing fallback ---
    if args.mic_input_device is None:
        args.mic_input_device = None  # sounddevice default input を使う

    if args.ai_audio_output_device is None:
        args.ai_audio_output_device = args.audio_device

    print(
        "[battle_audio_routing] "
        f"mic_input_device={args.mic_input_device} "
        f"ai_audio_output_device={args.ai_audio_output_device}",
        flush=True,
    )

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

    bg_override_file = (
        Path(args.bg_override_file).resolve()
        if args.bg_override_file
        else m1_repo / "in" / "bg_override_live.txt"
    )
    bg_override_file.parent.mkdir(parents=True, exist_ok=True)
    bg_override_file.write_text("", encoding="utf-8")
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
            bg_override_file=bg_override_file,
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
            audio_device=str(args.ai_audio_output_device),
            cwd=m1_repo,
            env=env,
        )

        mouth_streamer = None

        m0_result_box: dict[str, Any] = {}
        m0_error_box: list[BaseException] = []
        m0_thread: Thread | None = None

        # --- [ADD] Battle Runtime: admin CLI interrupt state ---
        battle_interrupt_queue: asyncio.Queue[str] | None = None
        battle_cli_thread: Thread | None = None
        battle_file_thread: Thread | None = None
        battle_control_file_thread: Thread | None = None
        event_runtime_file_thread: Thread | None = None
        battle_dev_file_writer_thread: Thread | None = None
        battle_queue_file_thread: Thread | None = None
        battle_interrupt_lines: list[str] = []
        battle_control_lines: list[dict[str, Any]] = []
        battle_control_active_ref: dict[str, str] = {"value": ""}
        battle_mic_gate_ref: dict[str, str] = {"value": "open"}
        battle_file_pending_lines: list[str] = []
        battle_interrupt_queue_enqueued_once = False
        battle_abort_requested = asyncio.Event()
        battle_talkover_cut_in_event = asyncio.Event()
        # turnごとに作るため、ここでは初期化しない
        m0_stream_dir_current: Path | None = None
        mouth_json_current: Path | None = None
        turn_start_perf_current: float | None = None
        frame_offset_current = 0
        next_frame_offset = 0
        inline_emo_id_current: str | None = None

        event_catalog: dict[str, Any] = {}

        if args.event_catalog_json:
            event_catalog = _load_event_catalog_json(args.event_catalog_json)
            print(
                "[event_runtime][catalog_loaded]",
                f"path={Path(args.event_catalog_json).resolve()}",
                f"events={len(event_catalog)}",
                flush=True,
            )

        if args.event_runtime_file:
            event_runtime_file_thread = _start_event_runtime_file_thread(
                path=Path(args.event_runtime_file).resolve(),
                poll_s=float(args.event_runtime_file_poll_s),
                battle_mic_gate_ref=battle_mic_gate_ref,
                audio_player_proc=audio_player_proc,
                event_catalog=event_catalog,
                bg_override_file=bg_override_file,
                m35_repo=m35_repo,
                m1_repo=m1_repo,
                battle_control_file=(
                    Path(args.battle_control_file).resolve()
                    if args.battle_control_file
                    else None
                ),
            )

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
                        str(inline_emo_id_current)
                        if bool(args.inline_emo_tag_mode)
                        else None
                    ),
                    live_emo_id_getter=(
                        (lambda: turn_state.get("live_emo_id"))
                        if bool(args.inline_emo_tag_mode)
                        else None
                    ),
                    live_emo_events_getter=(
                        (lambda: list(turn_state.get("live_emo_events") or []))
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

        prompt_dir = (
            Path(args.prompt_dir).resolve()
            if args.prompt_dir
            else (Path(args.m1_repo_root).resolve() / "configs" / "prompts")
        )

        system_instruction = _build_system_instruction_from_prompt_dir(
            prompt_dir=prompt_dir,
            inline_emo_tag_mode=bool(args.inline_emo_tag_mode),
            audio_priority_mode=bool(args.audio_priority_mode),
        )

        config = _build_live_config(
            system_instruction,
            enable_tools=not bool(args.inline_emo_tag_mode),
            output_audio_transcription=bool(args.output_audio_transcription),
        )
        _aad = getattr(
            getattr(config, "realtime_input_config", None),
            "automatic_activity_detection",
            None,
        )
        print(
            "[session_loop][live_config]",
            f"automatic_activity_detection.disabled={getattr(_aad, 'disabled', None)}",
            f"mic_vad_end_enabled={bool(args.mic_vad_end_enabled)}",
            f"skip_response_trigger={bool(args.skip_response_trigger)}",
            f"drop_initial_audio_ms_forced=0",
            flush=True,
        )

        turn_state: dict[str, Any] = {}
        recv_stop = asyncio.Event()

        async def _bootstrap_probe_audio_session(
            *,
            session: Any,
            attempt: int,
        ) -> bool:
            """
            Live API session が audio chunk を返す状態かを検査する。
            audio が1個でも来れば合格。
            失敗した session は async with を抜けて破棄する。
            """
            print(
                f"[session_loop][bootstrap] start attempt={attempt}",
                flush=True,
            )

            turn_state.clear()
            turn_state["active_turn"] = f"bootstrap_{attempt}"
            turn_state["turn_start_perf"] = time.perf_counter()
            turn_state["first_audio_sec"] = None
            turn_state["last_audio_perf"] = None
            turn_state["audio_chunk_idx_log"] = 0
            turn_state["transcription_accum"] = ""
            turn_state["live_emo_seen_tags"] = []
            turn_state["live_emo_events"] = []
            turn_state["debug_receive_raw_count"] = 0

            bootstrap_dir = out_root / "_bootstrap" / f"attempt_{attempt:03d}"
            pcm_dir = bootstrap_dir / "pcm_stream_chunks_discard"
            bootstrap_dir.mkdir(parents=True, exist_ok=True)
            pcm_dir.mkdir(parents=True, exist_ok=True)

            dummy_streamer_json = bootstrap_dir / "mouth_streamer.json"
            dummy_raw_json = bootstrap_dir / "mouth_timeline.formant.raw.json"
            dummy_mouth_json = bootstrap_dir / "mouth.json"
            dummy_audio_pcm = bootstrap_dir / "audio_response.pcm"

            dummy_streamer = MouthStreamerOC(
                out_json=str(dummy_streamer_json),
                session_id=f"{args.session_id}_bootstrap{attempt:03d}",
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

            bootstrap_stop = asyncio.Event()
            bootstrap_recv_task = asyncio.create_task(
                _receive_loop(
                    session=session,
                    stop_event=bootstrap_stop,
                    pcm_stream_chunks_dir=pcm_dir,
                    audio_response_pcm=dummy_audio_pcm,
                    mouth_streamer=dummy_streamer,
                    mouth_streamer_json=dummy_streamer_json,
                    mouth_raw_json=dummy_raw_json,
                    mouth_json=dummy_mouth_json,
                    knn_script=knn_script,
                    gt_glob=str(m3_repo / "data" / "knn_db" / "*.f1f2.json"),
                    step_ms=int(args.step_ms),
                    turn_state=turn_state,
                    mouth_updated_event=Event(),
                    drop_initial_audio_ms=0,
                    debug_receive=bool(args.debug_receive),
                    debug_receive_raw=bool(args.debug_receive_raw),
                )
            )

            try:
                # bootstrap は短時間固定長 + activity_start/end（テキストトリガー廃止）
                sent_bytes = await _send_mic_once(
                    session=session,
                    duration_s=float(args.mic_send_max_s),
                    input_sr=int(args.input_sr),
                    chunk_ms=int(args.step_ms),
                    device=args.mic_input_device,
                    mic_vad_end_enabled=False,
                    send_activity_signals=True,
                )

                print(
                    f"[session_loop][bootstrap][activity_end_path] "
                    f"attempt={attempt} (no text response_trigger)",
                    flush=True,
                )

                input_audio_ms = int(round((sent_bytes // 2) * 1000.0 / int(args.input_sr)))
                print(
                    f"[session_loop][bootstrap][turn_sent] "
                    f"attempt={attempt} input_audio_ms={input_audio_ms}",
                    flush=True,
                )

                wait_t0 = time.perf_counter()

                while True:
                    if turn_state.get("first_audio_sec") is not None:
                        print(
                            f"[session_loop][bootstrap][audio_ready] "
                            f"attempt={attempt} first_audio_sec={turn_state.get('first_audio_sec')}",
                            flush=True,
                        )
                        return True

                    wait_sec = time.perf_counter() - wait_t0

                    if wait_sec >= float(args.bootstrap_timeout_s):
                        print(
                            f"[session_loop][bootstrap][WARN] audio timeout "
                            f"attempt={attempt} timeout_s={float(args.bootstrap_timeout_s)}",
                            flush=True,
                        )
                        return False

                    await asyncio.sleep(0.02)

            finally:
                bootstrap_stop.set()
                await _cancel_tasks_safely(
                    [bootstrap_recv_task],
                    tag=f"bootstrap_attempt_{attempt}",
                )

        async def _run_warmup_turns(*, session: Any) -> None:
            n = max(0, int(args.warmup_turns))
            if n <= 0:
                return

            for wi in range(n):
                turn_no = wi + 1
                print(
                    f"[session_loop][warmup] start {turn_no}/{n}",
                    flush=True,
                )

                turn_state.clear()
                turn_state["active_turn"] = f"warmup_{turn_no}"
                turn_state["turn_start_perf"] = time.perf_counter()
                turn_state["first_audio_sec"] = None
                turn_state["last_audio_perf"] = None
                turn_state["audio_chunk_idx_log"] = 0
                turn_state["transcription_accum"] = ""
                turn_state["live_emo_seen_tags"] = []
                turn_state["live_emo_events"] = []
                turn_state["debug_receive_raw_count"] = 0

                warmup_dir = out_root / "_warmup" / f"warmup_{turn_no:03d}"
                pcm_dir = warmup_dir / "pcm_stream_chunks_discard"
                warmup_dir.mkdir(parents=True, exist_ok=True)
                pcm_dir.mkdir(parents=True, exist_ok=True)

                dummy_streamer_json = warmup_dir / "mouth_streamer.json"
                dummy_raw_json = warmup_dir / "mouth_timeline.formant.raw.json"
                dummy_mouth_json = warmup_dir / "mouth.json"
                dummy_audio_pcm = warmup_dir / "audio_response.pcm"

                dummy_streamer = MouthStreamerOC(
                    out_json=str(dummy_streamer_json),
                    session_id=f"{args.session_id}_warmup{turn_no:03d}",
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

                warmup_stop = asyncio.Event()
                warmup_recv_task = asyncio.create_task(
                    _receive_loop(
                        session=session,
                        stop_event=warmup_stop,
                        pcm_stream_chunks_dir=pcm_dir,
                        audio_response_pcm=dummy_audio_pcm,
                        mouth_streamer=dummy_streamer,
                        mouth_streamer_json=dummy_streamer_json,
                        mouth_raw_json=dummy_raw_json,
                        mouth_json=dummy_mouth_json,
                        knn_script=knn_script,
                        gt_glob=str(m3_repo / "data" / "knn_db" / "*.f1f2.json"),
                        step_ms=int(args.step_ms),
                        turn_state=turn_state,
                        mouth_updated_event=Event(),
                        drop_initial_audio_ms=0,
                        debug_receive=bool(args.debug_receive),
                        debug_receive_raw=bool(args.debug_receive_raw),
                    )
                )

                # receive 起動後に mic + activity_start/end（テキストトリガー廃止）
                sent_bytes = await _send_mic_once(
                    session=session,
                    duration_s=float(args.mic_send_max_s),
                    input_sr=int(args.input_sr),
                    chunk_ms=int(args.step_ms),
                    device=args.mic_input_device,
                    mic_vad_end_enabled=bool(args.mic_vad_end_enabled),
                    mic_vad_rms_threshold=float(args.mic_vad_rms_threshold),
                    mic_vad_end_rms_threshold=(
                        float(args.mic_vad_end_rms_threshold)
                        if args.mic_vad_end_rms_threshold is not None
                        else None
                    ),
                    mic_vad_min_voice_ms=int(args.mic_vad_min_voice_ms),
                    mic_vad_silence_ms=int(args.mic_vad_silence_ms),
                    mic_vad_min_listen_ms=int(args.mic_vad_min_listen_ms),
                    mic_vad_debug=bool(args.mic_vad_debug),
                    send_activity_signals=True,
                )

                print(
                    f"[session_loop][warmup][activity_end_path] turn={turn_no} "
                    f"(no text response_trigger)",
                    flush=True,
                )

                input_audio_ms = int(round((sent_bytes // 2) * 1000.0 / int(args.input_sr)))
                print(
                    f"[session_loop][warmup][turn_sent] turn={turn_no} input_audio_ms={input_audio_ms}",
                    flush=True,
                )

                wait_t0 = time.perf_counter()
                warmup_audio_ok = False

                while True:
                    if turn_state.get("first_audio_sec") is not None:
                        warmup_audio_ok = True
                        print(
                            f"[session_loop][warmup][first_audio_detected] turn={turn_no}",
                            flush=True,
                        )
                        break

                    if time.perf_counter() - wait_t0 >= float(args.turn_first_audio_timeout_s):
                        print(
                            f"[session_loop][warmup][WARN] first_audio timeout turn={turn_no}",
                            flush=True,
                        )
                        break

                    await asyncio.sleep(0.02)

                drain_t0 = time.perf_counter()
                while True:
                    latest_audio_perf = turn_state.get("last_audio_perf")
                    if latest_audio_perf is None:
                        if time.perf_counter() - drain_t0 >= float(args.turn_idle_wait_s):
                            break
                        await asyncio.sleep(0.05)
                        continue

                    if time.perf_counter() - float(latest_audio_perf) >= float(args.turn_idle_wait_s):
                        break

                    if time.perf_counter() - drain_t0 >= 3.0:
                        break

                    await asyncio.sleep(0.05)

                warmup_stop.set()
                await _cancel_tasks_safely(
                    [warmup_recv_task],
                    tag=f"warmup_turn_{turn_no}",
                )

                print(
                    f"[session_loop][warmup] done {turn_no}/{n} ok={warmup_audio_ok}",
                    flush=True,
                )

        async def _run_one_turn(
            *,
            session: Any,
            i: int,
            battle_interrupt_queue: asyncio.Queue[str] | None = None,
            battle_interrupt_lines: list[str] | None = None,
            battle_control_lines: list[str] | None = None,
            battle_control_active_ref: dict[str, str] | None = None,
            battle_mic_gate_ref: dict[str, str] | None = None,
            battle_file_pending_lines: list[str] | None = None,
        ) -> bool:
            nonlocal next_frame_offset
            nonlocal m0_thread
            nonlocal audio_thread
            nonlocal m0_stop_event
            nonlocal producer_done_event
            nonlocal mouth_updated_event
            nonlocal m0_stream_dir_current
            nonlocal mouth_json_current
            nonlocal turn_start_perf_current
            nonlocal frame_offset_current
            nonlocal inline_emo_id_current
            nonlocal battle_interrupt_queue_enqueued_once

            turn_no = i + 1
            battle_abort_requested.clear()
            battle_talkover_cut_in_event.clear()

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

            inline_emo_id_current = str(inline_emo_id_for_turn)

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
            turn_state["live_emo_seen_tags"] = []
            turn_state["live_emo_events"] = []
            turn_state["transcription_accum"] = ""
            turn_state["debug_receive_raw_count"] = 0
            turn_state["audio_player_proc"] = audio_player_proc
            turn_state["ai_audio_output_device"] = str(args.ai_audio_output_device)

            if bool(args.inline_emo_tag_mode):
                turn_state["live_emo_id_getter"] = (
                    lambda: turn_state.get("live_emo_id")
                )
                turn_state["live_emo_events_getter"] = (
                    lambda: list(turn_state.get("live_emo_events") or [])
                )
            else:
                turn_state.pop("live_emo_id_getter", None)
                turn_state.pop("live_emo_events_getter", None)

            dev_live_emo_events = _parse_dev_live_emo_events_csv(
                getattr(args, "dev_live_emo_events_csv", None)
            )

            if dev_live_emo_events:
                turn_state["live_emo_events"] = list(dev_live_emo_events)
                turn_state["live_emo_id"] = str(dev_live_emo_events[-1].get("emo_id"))

                # dev注入テスト時は、transcription由来の同一emoを重複appendしない。
                turn_state["live_emo_seen_tags"] = [
                    str(ev.get("emo_id"))
                    for ev in dev_live_emo_events
                    if ev.get("emo_id") is not None
                ]

                print(
                    f"[session_loop][dev_live_emo_events] "
                    f"turn={turn_no} events={dev_live_emo_events}",
                    flush=True,
                )

            # --- Phase 1:
            # client VAD + activity_end では、mic 完了直後にモデルが応答を開始する。
            # 取りこぼし防止のため、通常ターンも mic 送信前に receive/M0/audio を起動する。
            turn_audio_stop_event = Event()
            recv_stop = asyncio.Event()
            recv_task: asyncio.Task | None = None

            turn_state["active_turn"] = turn_no
            turn_state["turn_start_perf"] = time.perf_counter()
            turn_state["first_audio_sec"] = None

            m0_stop_event = Event()
            producer_done_event = Event()
            mouth_updated_event = Event()
            m0_stream_dir_current = m0_stream_dir
            mouth_json_current = mouth_json
            turn_start_perf_current = float(turn_state["turn_start_perf"])
            frame_offset_current = int(next_frame_offset)

            m0_thread = Thread(target=_m0_target, daemon=True)
            m0_thread.start()

            audio_thread = _watch_stream_pcm_chunks(
                audio_player_proc=audio_player_proc,
                pcm_stream_chunks_dir=pcm_stream_chunks_dir,
                audio_device=str(args.ai_audio_output_device),
                stop_event=turn_audio_stop_event,
            )

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
                    drop_initial_audio_ms=0,
                    debug_receive=bool(args.debug_receive),
                    debug_receive_raw=bool(args.debug_receive_raw),
                )
            )

            print(
                "[session_loop][receive_before_mic_ready]",
                f"turn={turn_no}",
                flush=True,
            )

            sent_bytes = await _send_mic_once(
                session=session,
                duration_s=float(args.mic_send_max_s),
                input_sr=int(args.input_sr),
                chunk_ms=int(args.step_ms),
                device=args.mic_input_device,
                stop_event=(
                    battle_talkover_cut_in_event
                    if bool(args.battle_talkover_cut_in_on_interrupt)
                    else None
                ),
                mic_gate_ref=battle_mic_gate_ref,
                mic_vad_end_enabled=bool(args.mic_vad_end_enabled),
                mic_vad_rms_threshold=float(args.mic_vad_rms_threshold),
                mic_vad_end_rms_threshold=(
                    float(args.mic_vad_end_rms_threshold)
                    if args.mic_vad_end_rms_threshold is not None
                    else None
                ),
                mic_vad_min_voice_ms=int(args.mic_vad_min_voice_ms),
                mic_vad_silence_ms=int(args.mic_vad_silence_ms),
                mic_vad_min_listen_ms=int(args.mic_vad_min_listen_ms),
                mic_vad_debug=bool(args.mic_vad_debug),
                send_activity_signals=True,
            )

            if (
                bool(args.battle_talkover_cut_in_on_interrupt)
                and battle_talkover_cut_in_event.is_set()
            ):
                # activity_end は _send_mic_once 内で送信済み（audio_stream_end は使わない）
                print(
                    "[battle_talkover][activity_end_path]",
                    f"turn={turn_no}",
                    flush=True,
                )
            elif bool(args.audio_stream_end_per_turn):
                print(
                    f"[session_loop][WARN] --audio_stream_end_per_turn is deprecated; "
                    f"activity_end already sent by client VAD path turn={turn_no}",
                    flush=True,
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

            consumed_file_control = False

            has_pending_interrupt = bool(battle_file_pending_lines)
            has_pending_control = bool(battle_control_lines)

            if has_pending_interrupt or has_pending_control:
                if has_pending_interrupt:
                    arbitration_winner = "interrupt"
                elif has_pending_control:
                    arbitration_winner = "control"
                else:
                    arbitration_winner = "none"

                print(
                    "[battle_arbitration]",
                    f"turn={turn_no}",
                    f"interrupt={has_pending_interrupt}",
                    f"control={has_pending_control}",
                    f"winner={arbitration_winner}",
                    flush=True,
                )

            if battle_control_lines:
                latest_control = battle_control_lines.pop(-1)

                if not isinstance(latest_control, dict):
                    latest_control = {
                        "type": "control",
                        "action": "",
                        "text": str(latest_control).strip(),
                        "mic_gate": "",
                    }

                latest_action = str(latest_control.get("action", "")).strip().lower()
                latest_text = str(latest_control.get("text", "")).strip()
                latest_mic_gate = str(latest_control.get("mic_gate", "")).strip().lower()

                if latest_action == "clear":
                    if battle_control_active_ref is not None:
                        battle_control_active_ref["value"] = ""

                    print(
                        "[battle_control][clear]",
                        f"turn={turn_no}",
                        flush=True,
                    )
                elif latest_text:
                    if battle_control_active_ref is not None:
                        battle_control_active_ref["value"] = latest_text

                if latest_mic_gate in ("mute", "open"):
                    if battle_mic_gate_ref is not None:
                        battle_mic_gate_ref["value"] = latest_mic_gate

                    print(
                        "[battle_mic_gate][set]",
                        f"turn={turn_no}",
                        f"state={latest_mic_gate}",
                        flush=True,
                    )

                print(
                    "[battle_control][activate]",
                    f"turn={turn_no}",
                    f"remaining={len(battle_control_lines)}",
                    f"active={battle_control_active_ref.get('value', '') if battle_control_active_ref is not None else ''}",
                    f"mic_gate={battle_mic_gate_ref.get('value', 'open') if battle_mic_gate_ref is not None else 'open'}",
                    flush=True,
                )

            active_control = str(
                battle_control_active_ref.get("value", "")
                if battle_control_active_ref is not None
                else ""
            ).strip()

            if active_control:
                print(
                    "[battle_control][apply_active]",
                    f"turn={turn_no}",
                    f"active={active_control}",
                    flush=True,
                )

                response_trigger = (
                    f"{response_trigger}\n"
                    f"【管理者制御】{active_control}"
                )

            if battle_file_pending_lines:
                latest_control = battle_file_pending_lines[-1]

                if isinstance(latest_control, dict):
                    created_at = float(latest_control.get("created_at", 0.0) or 0.0)
                    expire_sec = float(latest_control.get("expire_sec", 30.0) or 30.0)
                    age_sec = time.time() - created_at if created_at > 0 else 0.0

                    if age_sec > expire_sec:
                        print(
                            "[battle_interrupt][file_expired]",
                            f"turn={turn_no}",
                            f"age_sec={age_sec:.3f}",
                            f"expire_sec={expire_sec:.3f}",
                            f"text={latest_control.get('text')}",
                            flush=True,
                        )
                        battle_file_pending_lines.clear()
                    else:
                        latest = str(latest_control.get("text", "")).strip()
                        latest_priority = _normalize_battle_interrupt_priority(
                            latest_control.get("priority")
                        )

                        print(
                            "[battle_interrupt][file_apply_as_control]",
                            f"turn={turn_no}",
                            f"priority={latest_priority}",
                            f"age_sec={age_sec:.3f}",
                            f"expire_sec={expire_sec:.3f}",
                            f"latest={latest}",
                            flush=True,
                        )

                        response_trigger = (
                            f"{response_trigger}\n"
                            f"【管理者割り込み予約】{latest}"
                        )

                else:
                    latest = str(latest_control).strip()
                    latest_priority = "normal"

                    print(
                        "[battle_interrupt][file_apply_as_control]",
                        f"turn={turn_no}",
                        f"priority={latest_priority}",
                        f"latest={latest}",
                        flush=True,
                    )

                    response_trigger = (
                        f"{response_trigger}\n"
                        f"【管理者割り込み予約】{latest}"
                    )

            if bool(args.skip_response_trigger):
                print(
                    f"[session_loop][response_trigger][SKIP] turn={turn_no} "
                    f"(client VAD / activity_end path)",
                    flush=True,
                )
            else:
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
            turn_audio_ok = False

            while True:
                if turn_state.get("first_audio_sec") is not None:
                    turn_audio_ok = True
                    print(
                        f"[session_loop][turn_first_audio_detected] turn={turn_no}",
                        flush=True,
                    )

                    # --- [FIX] Battle Runtime: enqueue queue-file interrupts only once after first audio ---
                    if (
                        battle_interrupt_queue is not None
                        and battle_interrupt_lines
                        and not battle_interrupt_queue_enqueued_once
                    ):
                        battle_interrupt_queue_enqueued_once = True

                        asyncio.create_task(
                            _enqueue_battle_interrupt_lines_after_first_audio(
                                q=battle_interrupt_queue,
                                lines=battle_interrupt_lines,
                                interval_s=float(args.battle_interrupt_queue_interval_s),
                                turn=turn_no,
                            )
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

            # --- [ADD] Battle Runtime: retry first turn once if no audio chunk returned ---
            if (
                not turn_audio_ok
                and turn_no == 1
                and bool(args.first_turn_audio_retry)
                and not bool(args.skip_response_trigger)
                and response_trigger
            ):
                wait_s = float(args.first_turn_audio_retry_s)
                if wait_s > 0:
                    await asyncio.sleep(wait_s)

                print(
                    f"[session_loop][first_turn_audio_retry] "
                    f"turn={turn_no} wait_s={wait_s} text={response_trigger}",
                    flush=True,
                )

                turn_state["first_audio_sec"] = None
                turn_state["last_audio_perf"] = None

                await session.send_realtime_input(text=response_trigger)

                retry_t0 = time.perf_counter()

                while True:
                    if turn_state.get("first_audio_sec") is not None:
                        turn_audio_ok = True
                        print(
                            f"[session_loop][first_turn_audio_retry_ok] turn={turn_no}",
                            flush=True,
                        )
                        break

                    retry_wait_sec = time.perf_counter() - retry_t0

                    if retry_wait_sec >= float(args.turn_first_audio_timeout_s):
                        print(
                            f"[session_loop][WARN] first_turn_audio_retry_timeout turn={turn_no}",
                            flush=True,
                        )
                        break

                    await asyncio.sleep(0.02)

            # このturnの応答音声が止まるまで待つ
            drain_t0 = time.perf_counter()

            while True:
                if battle_abort_requested.is_set():
                    print(
                        f"[battle_interrupt][abort_turn] turn={turn_no}",
                        flush=True,
                    )
                    break

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
            await _cancel_tasks_safely(
                [recv_task],
                tag=f"turn_{turn_no}_recv",
            )

            # このturn用 audio watcher を停止
            turn_audio_stop_event.set()
            if audio_thread is not None:
                audio_thread.join(timeout=2.0)
                audio_thread = None

            # このturn用 M0 watcher を停止
            if producer_done_event is not None:
                producer_done_event.set()

            if m0_thread is not None:
                m0_thread.join(timeout=8.0)

                if m0_thread.is_alive():
                    print(
                        "[session_loop][WARN] m0 watcher did not stop after producer_done; force stop",
                        flush=True,
                    )

                    if m0_stop_event is not None:
                        m0_stop_event.set()

                    m0_thread.join(timeout=3.0)

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
            if turn_audio_ok and battle_file_pending_lines:
                print(
                    "[battle_interrupt][file_consume_confirmed]",
                    f"turn={turn_no}",
                    f"n={len(battle_file_pending_lines)}",
                    flush=True,
                )
                battle_file_pending_lines.clear()

            m0_result_box.clear()

            return bool(turn_audio_ok)

        if (
            args.battle_interrupt_file
            and battle_file_thread is None
            and not bool(args.battle_interrupt_file_immediate_send)
        ):
            loop = asyncio.get_running_loop()

            battle_file_thread = _start_battle_interrupt_file_thread(
                path=Path(args.battle_interrupt_file),
                pending_lines=battle_file_pending_lines,
                loop=loop,
                poll_s=float(args.battle_interrupt_file_poll_s),
                immediate_q=None,
                immediate_send=False,
            )

            if args.dev_battle_file_writer_text:
                battle_dev_file_writer_thread = _start_dev_battle_file_writer_thread(
                    path=Path(args.battle_interrupt_file),
                    text=str(args.dev_battle_file_writer_text),
                    delay_s=float(args.dev_battle_file_writer_delay_s),
                )

        if args.battle_control_file and battle_control_file_thread is None:
            loop = asyncio.get_running_loop()

            battle_control_file_thread = _start_battle_control_file_thread(
                path=Path(args.battle_control_file),
                pending_lines=battle_control_lines,
                loop=loop,
                poll_s=float(args.battle_control_file_poll_s),
                battle_mic_gate_ref=battle_mic_gate_ref,
            )

        if bool(args.reconnect_per_turn):
            for i in range(int(args.turns)):
                max_attempts = (
                    max(1, int(args.bootstrap_retry_n))
                    if bool(args.bootstrap_audio_required)
                    else max(1, int(args.turn_audio_retry_n) + 1)
                )
                ok = False

                for attempt in range(max_attempts):
                    print(
                        f"[session_loop] connect model={args.model} "
                        f"reconnect_turn={i + 1} attempt={attempt + 1}/{max_attempts}",
                        flush=True,
                    )

                    async with client.aio.live.connect(model=args.model, config=config) as session:
                        ok = await _run_one_turn(
                            session=session,
                            i=i,
                            battle_interrupt_queue=battle_interrupt_queue,
                            battle_interrupt_lines=battle_interrupt_lines,
                            battle_control_lines=battle_control_lines,
                            battle_control_active_ref=battle_control_active_ref,
                            battle_mic_gate_ref=battle_mic_gate_ref,
                            battle_file_pending_lines=battle_file_pending_lines,
                        )

                    if ok:
                        break

                    print(
                        f"[session_loop][WARN] retry turn={i + 1} "
                        f"because first_audio timeout",
                        flush=True,
                    )

                    await asyncio.sleep(1.0)
        else:
            reconnect_turn_total = int(args.turns) if bool(args.reconnect_per_turn) else 1

            for current_turn_i in range(reconnect_turn_total):
                bootstrap_max_attempts = (
                    max(1, int(args.bootstrap_retry_n))
                    if bool(args.bootstrap_audio_required)
                    else 1
                )

                bootstrap_ready = False

                for bootstrap_attempt in range(1, bootstrap_max_attempts + 1):
                    print(
                        f"[session_loop] connect model={args.model} "
                        f"bootstrap_attempt={bootstrap_attempt}/{bootstrap_max_attempts}",
                        flush=True,
                    )

                    async with client.aio.live.connect(model=args.model, config=config) as session:
                        if bool(args.bootstrap_audio_required):
                            bootstrap_ready = await _bootstrap_probe_audio_session(
                                session=session,
                                attempt=bootstrap_attempt,
                            )

                            if not bootstrap_ready:
                                print(
                                    f"[session_loop][bootstrap][retry] "
                                    f"attempt={bootstrap_attempt}/{bootstrap_max_attempts}",
                                    flush=True,
                                )
                                await asyncio.sleep(1.0)
                                continue
                        else:
                            bootstrap_ready = True

                        # bootstrap probe自体がwarmupを兼ねる。
                        # bootstrap_audio_required有効時は、旧warmupは二重実行しない。
                        if not bool(args.bootstrap_audio_required):
                            await _run_warmup_turns(session=session)

                        print(
                            "[battle_interrupt][post_bootstrap_enter]",
                            f"attempt={bootstrap_attempt}",
                            f"bootstrap_ready={bootstrap_ready}",
                            flush=True,
                        )

                        print(
                            "[battle_interrupt][args]",
                            f"cli={bool(args.battle_interrupt_cli)}",
                            f"file={args.battle_interrupt_file}",
                            f"queue_file={args.battle_interrupt_queue_file}",
                            flush=True,
                        )

                        battle_interrupt_stop = asyncio.Event()
                        battle_interrupt_task: asyncio.Task | None = None
                        battle_socket_task: asyncio.Task | None = None

                        if (
                            bool(args.battle_socket)
                            or bool(args.battle_interrupt_cli)
                            or args.battle_interrupt_file
                            or args.battle_interrupt_queue_file
                        ):
                            battle_interrupt_queue = asyncio.Queue()
                            loop = asyncio.get_running_loop()

                            if bool(args.battle_interrupt_cli):
                                battle_cli_thread = _start_battle_interrupt_cli_thread(
                                    q=battle_interrupt_queue,
                                    loop=loop,
                                    prefix=str(args.battle_interrupt_prefix),
                                )

                            if args.battle_interrupt_file:
                                print(
                                    "[battle_interrupt][session_file_watcher_start]",
                                    f"immediate_send={bool(args.battle_interrupt_file_immediate_send)}",
                                    flush=True,
                                )

                                battle_file_thread = _start_battle_interrupt_file_thread(
                                    path=Path(args.battle_interrupt_file),
                                    pending_lines=battle_file_pending_lines,
                                    loop=loop,
                                    poll_s=float(args.battle_interrupt_file_poll_s),
                                    immediate_q=(
                                        battle_interrupt_queue
                                        if bool(args.battle_interrupt_file_immediate_send)
                                        else None
                                    ),
                                    immediate_send=bool(args.battle_interrupt_file_immediate_send),
                                )

                                if args.dev_battle_file_writer_text:
                                    battle_dev_file_writer_thread = _start_dev_battle_file_writer_thread(
                                        path=Path(args.battle_interrupt_file),
                                        text=str(args.dev_battle_file_writer_text),
                                        delay_s=float(args.dev_battle_file_writer_delay_s),
                                    )

                            if args.battle_interrupt_queue_file:
                                battle_interrupt_lines = _read_battle_interrupt_queue_file(
                                    Path(args.battle_interrupt_queue_file)
                                )
                                print(
                                    f"[battle_interrupt][queue_file_loaded] "
                                    f"path={Path(args.battle_interrupt_queue_file).resolve()} "
                                    f"lines={len(battle_interrupt_lines)}",
                                    flush=True,
                                )

                            if args.battle_control_queue_file:
                                control_path = Path(args.battle_control_queue_file).resolve()

                                if control_path.exists():
                                    battle_control_lines = [
                                        x.strip()
                                        for x in control_path.read_text(
                                            encoding="utf-8"
                                        ).splitlines()
                                        if x.strip()
                                    ]

                                    print(
                                        "[battle_control][queue_file_loaded]",
                                        f"path={control_path}",
                                        f"lines={len(battle_control_lines)}",
                                        flush=True,
                                    )

                            if (
                                bool(args.battle_interrupt_cli)
                                or args.battle_interrupt_queue_file
                                or bool(args.battle_socket)
                                or bool(args.battle_interrupt_file_immediate_send)
                            ):
                                battle_interrupt_task = asyncio.create_task(
                                    _battle_interrupt_send_loop(
                                        session=session,
                                        q=battle_interrupt_queue,
                                        stop_event=battle_interrupt_stop,
                                        cut_in_event=(
                                            battle_talkover_cut_in_event
                                            if bool(args.battle_talkover_cut_in_on_interrupt)
                                            else None
                                        ),
                                    )
                                )

                            if bool(args.battle_socket):
                                battle_socket_task = asyncio.create_task(
                                    _battle_interrupt_socket_server(
                                        host=str(args.battle_socket_host),
                                        port=int(args.battle_socket_port),
                                        q=battle_interrupt_queue,
                                        stop_event=battle_interrupt_stop,
                                        abort_event=None,
                                    )
                                )

                        try:
                            if bool(args.reconnect_per_turn):
                                # reconnect_per_turn では、このsessionでは1turnだけ実行する。
                                # 外側の turn loop 側で次turn用に新しい Live API session を張り直す。
                                await _run_one_turn(
                                    session=session,
                                    i=current_turn_i,
                                    battle_interrupt_queue=battle_interrupt_queue,
                                    battle_interrupt_lines=battle_interrupt_lines,
                                    battle_control_lines=battle_control_lines,
                                    battle_control_active_ref=battle_control_active_ref,
                                    battle_mic_gate_ref=battle_mic_gate_ref,
                                    battle_file_pending_lines=battle_file_pending_lines,
                                )
                            else:
                                for i in range(int(args.turns)):
                                    await _run_one_turn(
                                        session=session,
                                        i=i,
                                        battle_interrupt_queue=battle_interrupt_queue,
                                        battle_interrupt_lines=battle_interrupt_lines,
                                        battle_control_lines=battle_control_lines,
                                        battle_control_active_ref=battle_control_active_ref,
                                        battle_mic_gate_ref=battle_mic_gate_ref,
                                        battle_file_pending_lines=battle_file_pending_lines,
                                    )
                        finally:
                            battle_interrupt_stop.set()

                            await _cancel_tasks_safely(
                                [
                                    battle_interrupt_task,
                                    battle_socket_task,
                                ],
                                tag=f"battle_interrupt_bootstrap_attempt_{bootstrap_attempt}",
                            )

                        # 通常turnが完了したので bootstrap retry loop を抜ける
                        break

                if bool(args.bootstrap_audio_required) and not bootstrap_ready:
                    raise RuntimeError(
                        f"bootstrap audio probe failed after {bootstrap_max_attempts} attempts "
                        f"for turn={current_turn_i + 1}"
                    )

                if bool(args.reconnect_per_turn) and current_turn_i < reconnect_turn_total - 1:
                    await asyncio.sleep(float(args.gap_s))

        if m0_error_box:
            raise m0_error_box[0]

        # --- [FIX] Summarize all rendered realtime M0 chunks across all turns ---
        rendered_chunk_dirs = sorted(
            out_root.glob(
                "turn_*/03_stream_mouth_m0/m0_stream_work/chunk_*/realtime_step1_chunk"
            )
        )

        chunks_n = len(rendered_chunk_dirs)
        total_frames = 0

        for chunk_dir in rendered_chunk_dirs:
            total_frames += len(list(chunk_dir.glob("*.png")))

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
    ap.add_argument("--reconnect_per_turn", action="store_true")
    ap.add_argument("--turn_audio_retry_n", type=int, default=0)
    # client VAD の最大聴取窓（無音で早期終了）。旧固定長 0.6s では VAD 不能。
    ap.add_argument("--mic_send_max_s", type=float, default=10.0)

    # --- Phase 1: client / local RMS mic VAD（mouth_vad_* とは別）---
    ap.add_argument(
        "--mic_vad_end_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable client RMS VAD end detection (default: True).",
    )
    ap.add_argument(
        "--mic_vad_rms_threshold",
        type=float,
        default=0.015,
        help="RMS threshold to start user speech (client VAD).",
    )
    ap.add_argument(
        "--mic_vad_end_rms_threshold",
        type=float,
        default=None,
        help="RMS threshold to keep speech active; default=mic_vad_rms_threshold.",
    )
    ap.add_argument(
        "--mic_vad_min_voice_ms",
        type=int,
        default=200,
        help="Min voiced ms before activity_start.",
    )
    ap.add_argument(
        "--mic_vad_silence_ms",
        type=int,
        default=700,
        help="Silence ms after speech before activity_end.",
    )
    ap.add_argument(
        "--mic_vad_min_listen_ms",
        type=int,
        default=800,
        help="Min listen ms before allowing VAD end.",
    )
    ap.add_argument(
        "--mic_vad_debug",
        action="store_true",
        help="Verbose client mic VAD logs.",
    )

    # --- [ADD] Live API warmup before production turns ---
    ap.add_argument(
        "--warmup_turns",
        type=int,
        default=0,
        help="Run N Live API warmup turns before normal turns. Warmup output is discarded.",
    )
    ap.add_argument(
        "--warmup_response_trigger",
        default="短く返答してください。",
        help="Text trigger used only for Live API warmup turns.",
    )

    # --- [ADD] Bootstrap: require audio-ready Live API session ---
    ap.add_argument(
        "--bootstrap_audio_required",
        action="store_true",
        help="Reconnect until bootstrap probe receives at least one audio chunk.",
    )
    ap.add_argument(
        "--bootstrap_retry_n",
        type=int,
        default=5,
        help="Max bootstrap reconnect attempts when --bootstrap_audio_required is set.",
    )
    ap.add_argument(
        "--bootstrap_timeout_s",
        type=float,
        default=2.0,
        help="Seconds to wait for first audio chunk during bootstrap probe.",
    )
    ap.add_argument(
        "--bootstrap_response_trigger",
        default="短く返答してください。",
        help="Text trigger used for bootstrap audio probe.",
    )
    ap.add_argument(
        "--audio_priority_mode",
        action="store_true",
        help="Strengthen system instruction to prioritize AUDIO response and short spoken output.",
    )

    ap.add_argument("--audio_device", default="15")
    ap.add_argument("--input_sr", type=int, default=16000)

    # --- [ADD] Battle Runtime audio routing ---
    ap.add_argument(
        "--mic_input_device",
        default=None,
        help="Input device used for Live API microphone send.",
    )
    ap.add_argument(
        "--ai_audio_output_device",
        default=None,
        help="Output device used for AI cat audio playback.",
    )

    ap.add_argument("--model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--api_version", default="v1alpha")
    ap.add_argument("--api_key_env", default="GEMINI_API_KEY")

    ap.add_argument("--response_trigger", default="短く返答してください。返答前にset_emotionを1回呼んでください。")
    ap.add_argument(
        "--prompt_dir",
        default=None,
        help="Directory for external prompt txt files. Default: <m1_repo_root>/configs/prompts",
    )
    ap.add_argument(
        "--skip_response_trigger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip text response_trigger for normal turns (default True). "
            "Client VAD + activity_end drives the model response. "
            "Use --no-skip_response_trigger only for debug."
        ),
    )
    ap.add_argument(
        "--battle_talkover_receive_during_mic",
        action="store_true",
        help=(
            "Start receive/M0/audio watchers before mic send, so immediate battle "
            "interrupt audio can be observed during user speech. Experimental; no abort."
        ),
    )
    ap.add_argument(
        "--battle_talkover_cut_in_on_interrupt",
        action="store_true",
        help=(
            "Soft cut-in mode: when battle interrupt is sent, stop mic send early "
            "and complete the turn via activity_end (not audio_stream_end). "
            "This is not abort."
        ),
    )
    ap.add_argument(
        "--response_triggers",
        default=None,
        help=(
            "Per-turn response triggers separated by '|'. "
            "Example: 'sleepy|normal|very sleepy'"
        ),
    )

    # --- [ADD] Battle Runtime: admin CLI interrupt ---
    ap.add_argument(
        "--battle_socket",
        action="store_true",
        help="Enable localhost TCP battle interrupt socket without txt queue/control files.",
    )
    ap.add_argument(
        "--battle_socket_host",
        default="127.0.0.1",
        help="Host for Battle Runtime socket server.",
    )
    ap.add_argument(
        "--battle_socket_port",
        type=int,
        default=39395,
        help="Port for Battle Runtime socket server.",
    )

    ap.add_argument(
        "--battle_interrupt_cli",
        action="store_true",
        help="Enable admin CLI interrupt. Type 'speak ...' during runtime.",
    )
    ap.add_argument(
        "--battle_interrupt_prefix",
        default="speak ",
        help="CLI prefix for Battle interrupt commands.",
    )
    ap.add_argument(
        "--battle_interrupt_file",
        default=None,
        help="Path to text file used as Battle interrupt trigger.",
    )
    ap.add_argument(
        "--battle_interrupt_file_poll_s",
        type=float,
        default=0.05,
        help="Polling interval for --battle_interrupt_file.",
    )
    ap.add_argument(
        "--battle_interrupt_file_immediate_send",
        action="store_true",
        help=(
            "Also send --battle_interrupt_file content immediately to the active "
            "Live API session via send_realtime_input(text=...). "
            "This does not abort current audio."
        ),
    )
    ap.add_argument(
        "--dev_battle_file_writer_text",
        default=None,
        help="DEV only: write this interrupt text into --battle_interrupt_file after delay.",
    )
    ap.add_argument(
        "--dev_battle_file_writer_delay_s",
        type=float,
        default=3.0,
        help="DEV only: delay seconds before writing dev battle interrupt file.",
    )
    ap.add_argument(
        "--battle_interrupt_queue_file",
        default=None,
        help="Path to line-based Battle interrupt queue file.",
    )
    ap.add_argument(
        "--battle_control_queue_file",
        default="",
    )
    ap.add_argument(
        "--battle_control_file",
        default=None,
        help="Path to text file used as Battle control trigger.",
    )
    ap.add_argument(
        "--battle_control_file_poll_s",
        type=float,
        default=0.05,
        help="Polling interval for --battle_control_file.",
    )
    ap.add_argument(
        "--event_runtime_file",
        default=None,
        help="Path to event_runtime_live.txt used as M3.5 event trigger.",
    )
    ap.add_argument(
        "--event_runtime_file_poll_s",
        type=float,
        default=0.05,
        help="Polling interval for --event_runtime_file.",
    )
    ap.add_argument(
        "--event_catalog_json",
        default=None,
        help="Path to event_catalog.json for resolving event_id to bg_video/pose_json/duration/audio.",
    )
    ap.add_argument(
        "--bg_override_file",
        default=None,
        help="Path to bg_override_live.txt passed to virtualcam for temporary BG video override.",
    )
    ap.add_argument(
        "--battle_interrupt_queue_interval_s",
        type=float,
        default=1.2,
        help="Interval seconds between queue file interrupt sends.",
    )
    ap.add_argument(
        "--battle_interrupt_queue_start_delay_s",
        type=float,
        default=0.5,
        help="Delay seconds before sending first queue file interrupt.",
    )

    # --- [ADD] Battle Runtime: first-turn audio retry ---
    ap.add_argument(
        "--first_turn_audio_retry",
        action="store_true",
        help="Retry response_trigger once when turn=1 receives no audio chunk.",
    )
    ap.add_argument(
        "--first_turn_audio_retry_s",
        type=float,
        default=1.0,
        help="Seconds to wait before retrying response_trigger for first turn.",
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
    ap.add_argument(
        "--dev_live_emo_events_csv",
        default=None,
        help="Dev only. Example: 1_1@600,2_0@1000,9_1@1400",
    )
    ap.add_argument(
        "--drop_initial_audio_ms",
        type=int,
        default=0,
        help=(
            "DEPRECATED/disabled in Phase 1 client-VAD mode. "
            "Always treated as 0 (no initial playback drop)."
        ),
    )
    ap.add_argument(
        "--audio_stream_end_per_turn",
        action="store_true",
        help=(
            "DEPRECATED: ignored. Normal turns complete via client VAD "
            "activity_end (not audio_stream_end)."
        ),
    )

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

    if int(args.stream_mouth_m0_chunk_len_ms) not in (80, 120, 200, 400):
        raise ValueError("stream_mouth_m0_chunk_len_ms must be 80, 120, 200, or 400")

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())