#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import glob
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
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
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
    _M0_PIPELINE_HANG_TIMEOUT_MS,
    _create_m0_pipeline_ref,
    _make_audio_playback_state_ref,
    _m0_pipeline_advance_sync,
    _m0_pipeline_enqueue_timeline_end_ms,
    _m0_pipeline_rendered_end_ms,
    _m0_pipeline_verify_pngs_exist,
    _m0_pool_reset_tcp,
    _m0_timing_add,
    _m0_timing_empty,
    _mouth_obj_with_hold_extend,
    _start_audio_player,
    _stop_audio_player,
    _send_audio_chunk,
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


def _project_streamer_obj_to_raw(streamer: MouthStreamerOC) -> dict[str, Any]:
    """Phase 5b: project in-memory streamer frames without flush/disk read."""
    frames_in = list(getattr(streamer, "_frames", []) or [])
    frames_out: list[dict[str, Any]] = []
    for fr in frames_in:
        if not isinstance(fr, dict):
            continue
        meta = fr.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        frames_out.append(
            {
                "t_ms": fr.get("t_ms"),
                "vad_active": int(meta.get("vad_active", 0) or 0),
                "f1_hz": meta.get("f1_hz"),
                "f2_hz": meta.get("f2_hz"),
                "src": "session_loop_stream_mouth",
            }
        )

    step_ms = int(getattr(getattr(streamer, "cfg", None), "step_ms", 40) or 40)
    meta_obj = getattr(streamer, "_meta", {}) or {}
    return {
        "version": "m3p.mouth.timeline.v1",
        "step_ms": int(step_ms),
        "frames": frames_out,
        "meta": meta_obj if isinstance(meta_obj, dict) else {},
    }


_KNN_MODULE_CACHE: dict[str, Any] = {}
_KNN_FUNC_CACHE: dict[str, Any] = {}
_KNN_GT_CACHE: dict[str, dict[str, Any]] = {}


def _load_knn_runtime_module(*, knn_script: Path) -> Any:
    key = str(knn_script.resolve())
    if key not in _KNN_MODULE_CACHE:
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

        _KNN_MODULE_CACHE[key] = mod
        _KNN_FUNC_CACHE[key] = fn
    return _KNN_MODULE_CACHE[key]


def _load_knn_from_raw_obj_fn(*, knn_script: Path) -> Any:
    _load_knn_runtime_module(knn_script=knn_script)
    return _KNN_FUNC_CACHE[str(knn_script.resolve())]


def _ensure_knn_gt_runtime(*, knn_script: Path, gt_glob: str) -> dict[str, Any]:
    """Cache GT DB + z-score once per (knn_script, gt_glob). Phase 7 O(N²) fix."""
    cache_key = f"{knn_script.resolve()}|{gt_glob}"
    if cache_key not in _KNN_GT_CACHE:
        mod = _load_knn_runtime_module(knn_script=knn_script)
        gt_paths = sorted(glob.glob(str(gt_glob)))
        if not gt_paths:
            raise RuntimeError(f"[knn] no gt files matched: {gt_glob}")

        db = mod._load_knn_db(gt_paths)
        z = mod._compute_z(db)
        db_z = [(mod._z_point(f1, f2, z), vid) for (f1, f2, vid) in db]
        _KNN_GT_CACHE[cache_key] = {
            "mod": mod,
            "db_z": db_z,
            "z": z,
        }
    return _KNN_GT_CACHE[cache_key]


def _knn_mouth_frames_from_raw_frames(
    *,
    frames_in: list[dict[str, Any]],
    gt_runtime: dict[str, Any],
    k: int = 5,
    fallback_id_active: int = 2,
    min_conf_ratio: float = 1.0,
) -> list[dict[str, Any]]:
    mod = gt_runtime["mod"]
    db_z = gt_runtime["db_z"]
    z = gt_runtime["z"]
    out_frames: list[dict[str, Any]] = []

    for fr in frames_in:
        t_ms = fr.get("t_ms")
        vad = fr.get("vad_active")
        if t_ms is None:
            continue

        t_ms = int(t_ms)
        vad = int(vad) if vad is not None else 0
        if vad == 0:
            out_frames.append({"t_ms": t_ms, "mouth_id": 0})
            continue

        f1 = fr.get("f1_hz")
        f2 = fr.get("f2_hz")
        if not (
            isinstance(f1, (int, float))
            and isinstance(f2, (int, float))
            and f1 == f1
            and f2 == f2
        ):
            mid = int(fallback_id_active)
            if mid == 0:
                mid = 2
            out_frames.append({"t_ms": t_ms, "mouth_id": mid})
            continue

        qz = mod._z_point(float(f1), float(f2), z)
        pred, top, top2 = mod._predict_knn(qz, db_z, int(k))
        ratio = (top / top2) if top2 > 0 else 999.0
        if ratio < float(min_conf_ratio):
            pred = int(fallback_id_active) if int(fallback_id_active) != 0 else 2
        out_frames.append({"t_ms": t_ms, "mouth_id": int(pred)})

    return out_frames


def _run_knn_from_raw_obj(
    *,
    knn_script: Path,
    raw_obj: dict[str, Any],
    gt_glob: str,
    step_ms: int,
) -> tuple[dict[str, Any], float]:
    """Run KNN from raw_obj with cached GT DB (no per-call GT reload)."""
    t0 = time.perf_counter()
    gt_runtime = _ensure_knn_gt_runtime(knn_script=knn_script, gt_glob=gt_glob)
    frames_in = list(raw_obj.get("frames") or [])
    out_frames = _knn_mouth_frames_from_raw_frames(
        frames_in=frames_in,
        gt_runtime=gt_runtime,
    )
    out_obj = {
        "audio": "",
        "step_ms": int(step_ms),
        "frames": out_frames,
    }
    return out_obj, time.perf_counter() - t0


def _run_knn_incremental_from_raw_obj(
    *,
    knn_script: Path,
    raw_obj: dict[str, Any],
    gt_glob: str,
    step_ms: int,
    mouth_obj_ref: dict[str, Any],
) -> tuple[dict[str, Any], float, int, int]:
    """Delta-only KNN: process new raw frames and append to mouth_obj_ref.

    Safe under 図A because push+KNN is serialized by pipeline_seq.
    """
    t0 = time.perf_counter()
    raw_frames = list(raw_obj.get("frames") or [])
    prev_n = int(mouth_obj_ref.get("knn_raw_frames_done", 0) or 0)

    if len(raw_frames) < prev_n:
        prev_n = 0
        mouth_obj_ref["obj"] = None

    if len(raw_frames) == prev_n:
        existing = mouth_obj_ref.get("obj")
        if isinstance(existing, dict):
            total_n = len(existing.get("frames") or existing.get("timeline") or [])
            return existing, 0.0, 0, int(total_n)
        prev_n = 0

    delta_frames = raw_frames[prev_n:]
    if not delta_frames:
        existing = mouth_obj_ref.get("obj")
        if isinstance(existing, dict):
            total_n = len(existing.get("frames") or existing.get("timeline") or [])
            return existing, 0.0, 0, int(total_n)
        out_obj = {
            "audio": "",
            "step_ms": int(step_ms),
            "frames": [],
        }
        mouth_obj_ref["knn_raw_frames_done"] = len(raw_frames)
        mouth_obj_ref["obj"] = out_obj
        return out_obj, time.perf_counter() - t0, 0, 0

    gt_runtime = _ensure_knn_gt_runtime(knn_script=knn_script, gt_glob=gt_glob)
    delta_out = _knn_mouth_frames_from_raw_frames(
        frames_in=delta_frames,
        gt_runtime=gt_runtime,
    )

    prev_obj = mouth_obj_ref.get("obj")
    if isinstance(prev_obj, dict) and prev_n > 0:
        prev_mouth_frames = list(
            prev_obj.get("frames") or prev_obj.get("timeline") or []
        )
    else:
        prev_mouth_frames = []

    merged_frames = prev_mouth_frames + delta_out
    out_obj = {
        "audio": "",
        "step_ms": int(step_ms),
        "frames": merged_frames,
    }
    mouth_obj_ref["knn_raw_frames_done"] = len(raw_frames)
    mouth_obj_ref["obj"] = out_obj
    return (
        out_obj,
        time.perf_counter() - t0,
        len(delta_frames),
        len(merged_frames),
    )


def _run_knn_in_process(
    *,
    knn_script: Path,
    raw_json: Path,
    out_json: Path,
    gt_glob: str,
    step_ms: int,
) -> float:
    raw_obj = json.loads(raw_json.read_text(encoding="utf-8"))
    out_obj, elapsed = _run_knn_from_raw_obj(
        knn_script=knn_script,
        raw_obj=raw_obj,
        gt_glob=gt_glob,
        step_ms=step_ms,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(out_obj, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return elapsed


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
    mic_vad_silence_ms: int = 350,
    mic_vad_silence_ms_ref: dict[str, int] | None = None,
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
    min_listen_blocks = max(
        1,
        int(round(float(mic_vad_min_listen_ms) / actual_chunk_ms)),
    )

    def _silence_ms_now() -> int:
        if mic_vad_silence_ms_ref is not None:
            try:
                return int(mic_vad_silence_ms_ref["value"])
            except Exception:
                pass
        return int(mic_vad_silence_ms)

    def _silence_blocks_limit_now() -> int:
        return max(
            1,
            int(round(float(_silence_ms_now()) / actual_chunk_ms)),
        )

    silence_blocks_limit = _silence_blocks_limit_now()

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
            f"silence_ms={_silence_ms_now()}",
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
        _ae_perf_ms = time.perf_counter() * 1000.0
        print(
            "[mic_vad][ACTIVITY_END]",
            f"reason={reason}",
            f"elapsed_s={time.perf_counter() - t0:.3f}",
            flush=True,
        )
        # Phase R1: response-timing SSOT (same clock as speech_end / first_audio)
        print(
            "[resp_timing] activity_end",
            f"perf_ms={_ae_perf_ms:.3f}",
            f"reason={reason}",
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
                    # Phase R2: re-read silence limit each chunk (file may switch 350↔250)
                    silence_blocks_limit = _silence_blocks_limit_now()
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
                            if silence_blocks == 1:
                                # Phase R1: speech_end = silence accumulation start
                                print(
                                    "[resp_timing] speech_end",
                                    f"perf_ms={time.perf_counter() * 1000.0:.3f}",
                                    flush=True,
                                )
                            if mic_vad_debug and (
                                silence_blocks == 1
                                or silence_blocks % 5 == 0
                                or silence_blocks >= silence_blocks_limit
                            ):
                                print(
                                    "[mic_vad][SILENCE]",
                                    f"i={i}",
                                    f"rms={rms:.5f}",
                                    f"silence_ms={_silence_ms_now()}",
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
                                    f"silence_ms={_silence_ms_now()}",
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


def _clear_audio_player_queue(
    audio_player_proc: subprocess.Popen | None,
    *,
    reason: str = "unspecified",
) -> bool:
    """Send clear_queue to player.

    Phase 4: normal turn / generation_complete / tail drain must NOT call this.
    Allowed: user interrupt (talkover cut-in) and event_runtime overlay only.
    """
    if audio_player_proc is None:
        print(
            "[audio_player][clear_queue_skip] proc=None",
            f"reason={reason}",
            flush=True,
        )
        return False

    if audio_player_proc.stdin is None:
        print(
            "[audio_player][clear_queue_skip] stdin=None",
            f"reason={reason}",
            flush=True,
        )
        return False

    if audio_player_proc.poll() is not None:
        print(
            "[audio_player][clear_queue_skip] proc_not_running",
            f"reason={reason}",
            flush=True,
        )
        return False

    try:
        audio_player_proc.stdin.write(
            json.dumps(
                {"cmd": "clear_queue", "reason": str(reason)},
                ensure_ascii=False,
            )
            + "\n"
        )
        audio_player_proc.stdin.flush()
        print(
            "[audio_player][clear_queue_sent]",
            f"reason={reason}",
            flush=True,
        )
        return True
    except Exception as e:
        print(
            f"[audio_player][clear_queue_error] {type(e).__name__}: {e}",
            f"reason={reason}",
            flush=True,
        )
        return False


@dataclass(frozen=True)
class _PipelineInterruptClear:
    """Dispatcher marker: drop stale pending and jump expected_seq after interrupt."""

    min_seq: int


def _clear_draw_queue_state(turn_state: dict[str, Any]) -> None:
    """Minimal draw-queue clear for interrupt exception (mouth in-memory only)."""
    mouth_ref = turn_state.get("mouth_obj_ref")
    cleared = False
    if isinstance(mouth_ref, dict):
        obj = mouth_ref.get("obj")
        if isinstance(obj, dict):
            if isinstance(obj.get("frames"), list):
                obj["frames"] = []
                cleared = True
            if isinstance(obj.get("timeline"), list):
                obj["timeline"] = []
                cleared = True
        mouth_ref["obj"] = {"frames": [], "timeline": []} if not isinstance(obj, dict) else obj
        # Phase 7: reset incremental cursor so next KNN rebuilds from raw[0:].
        mouth_ref["knn_raw_frames_done"] = 0
        cleared = True
    print(
        "[battle_talkover][draw_queue_cleared]",
        f"cleared={bool(cleared)}",
        flush=True,
    )


async def _apply_user_interrupt_clear(
    *,
    audio_player_proc: subprocess.Popen | None,
    turn_state: dict[str, Any],
    reason: str = "user_interrupt",
) -> None:
    """唯一の clear_queue 例外: ユーザー割り込み（talkover cut-in）。

    順序: player clear → 描画キュー clear → 進行中 chunk タスク cancel
    （mic 停止 / activity_end は呼び出し側で先行済み、または cut_in_event 経由）
    """
    if bool(turn_state.get("interrupt_clear_in_progress")):
        print(
            "[battle_talkover][interrupt_clear_skip]",
            "reason=already_in_progress",
            flush=True,
        )
        return

    turn_state["interrupt_clear_in_progress"] = True
    try:
        idle_stop = turn_state.get("idle_silent_stop_event")
        if isinstance(idle_stop, asyncio.Event) and not idle_stop.is_set():
            idle_stop.set()
            print(
                "[idle_silent_pcm][STOP_ON_INTERRUPT]",
                f"reason={reason}",
                flush=True,
            )

        min_seq = int(turn_state.get("pipeline_next_seq", 0) or 0)
        turn_state["pipeline_interrupt_min_seq"] = min_seq

        print(
            "[battle_talkover][interrupt_clear_begin]",
            f"reason={reason}",
            f"min_seq={min_seq}",
            flush=True,
        )

        _clear_audio_player_queue(audio_player_proc, reason=reason)
        _clear_draw_queue_state(turn_state)

        order = turn_state.get("pipeline_enqueue_order")
        if isinstance(order, dict):
            cond = order.get("cond")
            if isinstance(cond, asyncio.Condition):
                async with cond:
                    order["expected_push_seq"] = min_seq
                    order["expected_seq"] = min_seq
                    cond.notify_all()
            else:
                order["expected_push_seq"] = min_seq
                order["expected_seq"] = min_seq

        eq = turn_state.get("pipeline_enqueue_queue_ref")
        if eq is not None:
            try:
                eq.put_nowait(_PipelineInterruptClear(min_seq=min_seq))
            except Exception as e:
                print(
                    "[battle_talkover][interrupt_clear_queue_marker_error]",
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

        tasks_ref = turn_state.get("pipeline_active_tasks_ref")
        cancelled_n = 0
        if isinstance(tasks_ref, list):
            pending = [t for t in list(tasks_ref) if t is not None and not t.done()]
            for t in pending:
                t.cancel()
                cancelled_n += 1
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            tasks_ref[:] = [t for t in tasks_ref if t is not None and not t.done()]

        print(
            "[battle_talkover][interrupt_clear_done]",
            f"reason={reason}",
            f"chunk_tasks_cancelled={cancelled_n}",
            f"min_seq={min_seq}",
            flush=True,
        )
    finally:
        turn_state["interrupt_clear_in_progress"] = False


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
    audio_player_proc: subprocess.Popen | None = None,
    turn_state: dict[str, Any] | None = None,
) -> None:
    """
    admin CLI queue から受け取った割り込み指示を、
    現在の Live API session へ即時 text send する。

    talkover cut-in 時（cut_in_event あり）:
      mic 停止シグナル →（mic 側で activity_end）→ 割り込み clear_queue
      → chunk cancel → text send（新応答開始）
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
                # mic 停止は cut_in_event。activity_end は _send_mic_once finally。
                # 割り込み例外 clear はここで即時（再生中 AI 音声を止める）。
                if turn_state is not None:
                    await _apply_user_interrupt_clear(
                        audio_player_proc=audio_player_proc,
                        turn_state=turn_state,
                        reason="talkover_cut_in",
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

                _clear_audio_player_queue(
                    audio_player_proc,
                    reason="event_runtime",
                )

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


# Phase R2: runtime VAD silence profile (350=usual / 250=aggressive). File watch only.
_VAD_PROFILE_ALLOWED_SILENCE_MS = frozenset({250, 350})
_VAD_PROFILE_DEFAULT_SILENCE_MS = 350


def _parse_vad_profile_file_text(raw: str) -> int | None:
    """Parse allowed mic_vad_silence_ms from profile file text. Invalid → None."""
    raw = str(raw or "").strip()
    if not raw:
        return None

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for key in ("mic_vad_silence_ms", "silence_ms", "vad_silence_ms"):
                if key not in obj:
                    continue
                try:
                    v = int(obj[key])
                except (TypeError, ValueError):
                    return None
                if v in _VAD_PROFILE_ALLOWED_SILENCE_MS:
                    return v
                return None
            return None
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            v = int(obj)
            if v in _VAD_PROFILE_ALLOWED_SILENCE_MS:
                return v
            return None
    except Exception:
        pass

    try:
        v = int(raw.split()[0])
    except (TypeError, ValueError):
        return None
    if v in _VAD_PROFILE_ALLOWED_SILENCE_MS:
        return v
    return None


def _read_vad_profile_file(path: Path) -> int | None:
    path = Path(path)
    if not path.is_file():
        return None
    raw = ""
    for enc in ("utf-8-sig", "utf-8", "cp932", "utf-16"):
        try:
            raw = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return _parse_vad_profile_file_text(raw)


def _resolve_mic_vad_silence_ms(
    *,
    cli_value: int | None,
    profile_path: Path | None,
) -> tuple[int, str]:
    """Priority: CLI explicit > persistent file > default 350."""
    if cli_value is not None:
        return int(cli_value), "cli"
    if profile_path is not None:
        file_v = _read_vad_profile_file(profile_path)
        if file_v is not None:
            return int(file_v), "file"
    return int(_VAD_PROFILE_DEFAULT_SILENCE_MS), "default"


def _start_vad_profile_file_thread(
    *,
    path: Path,
    silence_ms_ref: dict[str, int],
    loop: asyncio.AbstractEventLoop,
    poll_s: float,
) -> Thread:
    """Watch vad_profile file; apply 250/350 only. Does not clear file (persistence)."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    last_raw = ""
    try:
        last_mtime = float(path.stat().st_mtime)
        for enc in ("utf-8-sig", "utf-8", "cp932", "utf-16"):
            try:
                last_raw = path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
    except OSError:
        last_mtime = 0.0

    def _target() -> None:
        nonlocal last_mtime, last_raw

        print(
            "[vad_profile][file_ready]",
            f"path={path}",
            f"poll_s={float(poll_s):.3f}",
            f"current={int(silence_ms_ref.get('value', _VAD_PROFILE_DEFAULT_SILENCE_MS))}",
            flush=True,
        )

        while True:
            try:
                stat = path.stat()
                mtime = float(stat.st_mtime)

                raw = ""
                for enc in ("utf-8-sig", "utf-8", "cp932", "utf-16"):
                    try:
                        raw = path.read_text(encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue

                # Content-aware: Windows mtime granularity can hide same-second writes.
                if mtime <= last_mtime and raw == last_raw:
                    time.sleep(float(poll_s))
                    continue

                last_mtime = mtime
                last_raw = raw

                parsed = _parse_vad_profile_file_text(raw)
                if parsed is None:
                    if str(raw or "").strip():
                        print(
                            "[vad_profile][reject]",
                            f"raw={str(raw).strip()!r}",
                            f"keep={int(silence_ms_ref.get('value', _VAD_PROFILE_DEFAULT_SILENCE_MS))}",
                            flush=True,
                        )
                    time.sleep(float(poll_s))
                    continue

                prev = int(
                    silence_ms_ref.get("value", _VAD_PROFILE_DEFAULT_SILENCE_MS)
                )
                if parsed == prev:
                    print(
                        "[vad_profile][noop]",
                        f"silence_ms={parsed}",
                        flush=True,
                    )
                else:

                    def _apply(
                        v: int = parsed,
                        p: int = prev,
                    ) -> None:
                        silence_ms_ref["value"] = int(v)
                        print(
                            "[vad_profile][set]",
                            f"from={p}",
                            f"to={int(v)}",
                            flush=True,
                        )

                    loop.call_soon_threadsafe(_apply)

            except BaseException as e:
                print(
                    f"[vad_profile][file_warn] {type(e).__name__}: {e}",
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


_PIPELINE_MAX_INFLIGHT = 6
_AUDIO_PLAYBACK_EPOCH_GAP = -1


@dataclass
class _AudioPipelineJob:
    ctx_id: str | None
    playback_chunk_idx: int
    playback_epoch: int
    audio: bytes
    playback_audio: bytes
    pcm_stream_chunks_dir: Path
    audio_response_pcm: Path
    mouth_streamer: MouthStreamerOC
    mouth_streamer_json: Path
    mouth_raw_json: Path
    mouth_json: Path
    knn_script: Path
    gt_glob: str
    step_ms: int
    knn_inmemory: bool
    m0_inmemory: bool
    mouth_obj_ref: dict[str, Any] | None
    fast_inmemory: bool
    skip_archive_pcm: bool
    is_idle_silent: bool = False


@dataclass
class _PipelineEnqueueItem:
    pipeline_seq: int
    job: _AudioPipelineJob
    effective_playback: bytes
    enqueue_blocked: bool
    enqueue_timeline_end_ms: int
    emitted: int
    knn_ms: float
    m0_ms: float
    m0_chunks: int
    hang_used: bool
    png_verified: bool
    frames_n: int
    m0_last_cid: int
    m0_last_global: int
    t_pipeline0: float
    stage_knn_done_ms: float
    stage_m0_done_ms: float
    m0_breakdown: dict[str, float] | None = None
    push_wait_ms: float = 0.0
    t_enqueued: float = 0.0


def _tail_hold_extend_live_mouth(
    mouth_obj_ref: dict[str, Any] | None,
    *,
    until_t1_ms: int,
    step_ms: int,
) -> int:
    """Turn-end only: mutate live mouth so the last PCM until can clear mouth_ready.

    Phase22 hold-extend is render-copy only and runs after claim; at generation end
    mouth often stops ~1 step short of until, so claim never starts. Extending the
    live ref with the last mouth_id (not mouth_closed) lets M0 cover then enqueue
    without audio-before.
    """
    if mouth_obj_ref is None or int(until_t1_ms) <= 0:
        return 0
    obj = mouth_obj_ref.get("obj")
    if not isinstance(obj, dict):
        return 0
    step = max(1, int(step_ms) or 40)
    # hold_extend upper bound is exclusive (t1//step); +step ensures cov crosses until.
    target_t1 = int(until_t1_ms) + int(step)
    new_obj, added = _mouth_obj_with_hold_extend(
        obj, t1_ms=int(target_t1), step_ms=int(step)
    )
    if int(added) <= 0:
        return 0
    mouth_obj_ref["obj"] = new_obj
    return int(added)


def _tail_hold_gate_ready(
    turn_state: dict[str, Any],
    stop_event: asyncio.Event,
    no_progress_rounds: int,
) -> bool:
    """Phase30: allow turn-end hold-extend as soon as generation_complete is seen.

    Before: gate required recv_stop (turn teardown), so last chunks spun ~1–2s on
    wait_mouth while player depleted → turn-end REB / lip freeze. generation_complete
    is input-end only (no clear_queue); using it here only unlocks mouth hold-extend.
    """
    gen_done = bool(turn_state.get("generation_complete_seen"))
    if not gen_done and not stop_event.is_set():
        return False
    # After input end, one no-progress round (~50ms) is enough; recv_stop path keeps 4.
    need = 1 if gen_done else 4
    return int(no_progress_rounds) >= int(need)


def _mouth_shortfall_frames(
    mouth_obj_ref: dict[str, Any] | None,
    *,
    until_t1_ms: int,
    step_ms: int,
) -> int:
    """How many step frames mouth last is short of until (0 if caught up / unknown)."""
    if mouth_obj_ref is None or int(until_t1_ms) <= 0:
        return 0
    obj = mouth_obj_ref.get("obj")
    if not isinstance(obj, dict):
        return 0
    frames = obj.get("frames")
    if not isinstance(frames, list) or not frames:
        return 0
    step = max(1, int(step_ms) or 40)
    last_t = int(frames[-1].get("t_ms", 0) or 0)
    # mouth_ready needs last_t + step >= until; shortfall in frames beyond that.
    need_t = int(until_t1_ms)
    cov = int(last_t) + int(step)
    if cov >= need_t:
        return 0
    return max(0, (int(need_t) - int(cov) + int(step) - 1) // int(step))


# Phase30: gen_complete may unlock hold while streamer still far behind; only auto-hold
# tiny shortfalls (Before post_gen was always 1 frame). Larger gaps wait for mouth/stop.
_TAIL_HOLD_GEN_COMPLETE_MAX_SHORTFALL_FRAMES = 3


def _try_tail_hold_extend(
    mouth_obj_ref: dict[str, Any] | None,
    *,
    until_t1_ms: int,
    step_ms: int,
    turn_state: dict[str, Any],
    stop_event: asyncio.Event,
) -> int:
    """Apply live tail hold-extend with Phase30 gen_complete shortfall guard."""
    gen_done = bool(turn_state.get("generation_complete_seen"))
    if gen_done and not stop_event.is_set():
        shortfall = _mouth_shortfall_frames(
            mouth_obj_ref,
            until_t1_ms=int(until_t1_ms),
            step_ms=int(step_ms),
        )
        if int(shortfall) > int(_TAIL_HOLD_GEN_COMPLETE_MAX_SHORTFALL_FRAMES):
            return 0
    return _tail_hold_extend_live_mouth(
        mouth_obj_ref,
        until_t1_ms=int(until_t1_ms),
        step_ms=int(step_ms),
    )


def _pipeline_inflight_inc(turn_state: dict[str, Any]) -> int:
    n = int(turn_state.get("pipeline_inflight", 0)) + 1
    turn_state["pipeline_inflight"] = n
    return n


def _pipeline_inflight_dec(turn_state: dict[str, Any]) -> int:
    n = max(0, int(turn_state.get("pipeline_inflight", 0)) - 1)
    turn_state["pipeline_inflight"] = n
    return n


def _ensure_pipeline_enqueue_order(turn_state: dict[str, Any]) -> dict[str, Any]:
    order = turn_state.get("pipeline_enqueue_order")
    if order is None:
        order = {
            "expected_push_seq": 0,
            "expected_seq": 0,
            "cond": asyncio.Condition(),
        }
        turn_state["pipeline_enqueue_order"] = order
    else:
        order.setdefault("expected_push_seq", 0)
        order.setdefault("expected_seq", 0)
        if "cond" not in order or order.get("cond") is None:
            order["cond"] = asyncio.Condition()
    return order


async def _await_pipeline_push_turn(
    turn_state: dict[str, Any],
    pipeline_seq: int,
) -> None:
    order = _ensure_pipeline_enqueue_order(turn_state)
    cond: asyncio.Condition = order["cond"]
    async with cond:
        while int(order["expected_push_seq"]) != int(pipeline_seq):
            await cond.wait()


async def _advance_pipeline_push_turn(
    turn_state: dict[str, Any],
    pipeline_seq: int,
) -> None:
    order = _ensure_pipeline_enqueue_order(turn_state)
    cond: asyncio.Condition = order["cond"]
    async with cond:
        if int(order["expected_push_seq"]) == int(pipeline_seq):
            order["expected_push_seq"] = int(pipeline_seq) + 1
            cond.notify_all()


def _ensure_pipeline_audio_file(
    *,
    job: _AudioPipelineJob,
    pipeline_io_ref: dict[str, Any],
) -> Any | None:
    if bool(job.skip_archive_pcm):
        return None

    if job.ctx_id != pipeline_io_ref.get("ctx_id"):
        audio_f = pipeline_io_ref.get("audio_f")
        if audio_f is not None:
            try:
                audio_f.close()
            except Exception:
                pass
            pipeline_io_ref["audio_f"] = None

        pipeline_io_ref["ctx_id"] = job.ctx_id
        job.pcm_stream_chunks_dir.mkdir(parents=True, exist_ok=True)
        job.audio_response_pcm.parent.mkdir(parents=True, exist_ok=True)
        pipeline_io_ref["audio_f"] = job.audio_response_pcm.open("ab")

    audio_f = pipeline_io_ref.get("audio_f")
    if audio_f is None and not bool(job.skip_archive_pcm):
        job.audio_response_pcm.parent.mkdir(parents=True, exist_ok=True)
        pipeline_io_ref["audio_f"] = job.audio_response_pcm.open("ab")
        audio_f = pipeline_io_ref["audio_f"]

    return audio_f


def _process_audio_chunk_knn_sync(
    *,
    raw_obj: dict[str, Any],
    job: _AudioPipelineJob,
    turn_state: dict[str, Any],
) -> int:
    if not bool(job.knn_inmemory):
        # Phase 7: compact JSON (indent=2 grew with N and inflated knn_ms on OFF path).
        job.mouth_raw_json.write_text(
            json.dumps(raw_obj, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    if not turn_state.get("first_mouth_json_logged", False):
        active_turn = turn_state.get("active_turn")
        if active_turn is not None and turn_state.get("turn_start_perf") is not None:
            dt = time.perf_counter() - float(turn_state["turn_start_perf"])
            print(
                "[perf][session_first_stream_mouth_json_written_from_turn_start_sec] "
                f"{dt:.3f}",
                flush=True,
            )
        turn_state["first_mouth_json_logged"] = True

    frames_n = 0
    knn_incremental = bool(turn_state.get("knn_incremental", True))

    # Phase 7: delta-only KNN (+ GT cache). Push+KNN is arrival-ordered, so merge is safe.
    if job.mouth_obj_ref is not None and knn_incremental:
        out_obj, knn_sec, delta_n, frames_n = _run_knn_incremental_from_raw_obj(
            knn_script=job.knn_script,
            raw_obj=raw_obj,
            gt_glob=job.gt_glob,
            step_ms=job.step_ms,
            mouth_obj_ref=job.mouth_obj_ref,
        )
        if not bool(job.knn_inmemory):
            job.mouth_json.parent.mkdir(parents=True, exist_ok=True)
            job.mouth_json.write_text(
                json.dumps(out_obj, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
        tag = (
            "[knn_inmemory][mouth_obj_updated]"
            if bool(job.knn_inmemory)
            else "[knn][mouth_obj_updated]"
        )
        print(
            tag,
            f"frames={int(frames_n)}",
            f"delta={int(delta_n)}",
            f"knn_sec={float(knn_sec):.3f}",
            "mode=incremental",
            flush=True,
        )
    elif bool(job.knn_inmemory):
        if job.mouth_obj_ref is None:
            raise RuntimeError("knn_inmemory requires mouth_obj_ref")
        mouth_obj, knn_sec = _run_knn_from_raw_obj(
            knn_script=job.knn_script,
            raw_obj=raw_obj,
            gt_glob=job.gt_glob,
            step_ms=job.step_ms,
        )
        job.mouth_obj_ref["obj"] = mouth_obj
        job.mouth_obj_ref["knn_raw_frames_done"] = len(raw_obj.get("frames") or [])
        frames_n = len(mouth_obj.get("frames") or mouth_obj.get("timeline") or [])
        print(
            "[knn_inmemory][mouth_obj_updated]",
            f"frames={int(frames_n)}",
            f"knn_sec={float(knn_sec):.3f}",
            "mode=full",
            flush=True,
        )
    else:
        _run_knn_in_process(
            knn_script=job.knn_script,
            raw_json=job.mouth_raw_json,
            out_json=job.mouth_json,
            gt_glob=job.gt_glob,
            step_ms=job.step_ms,
        )
        try:
            mouth_obj = json.loads(job.mouth_json.read_text(encoding="utf-8"))
            frames_n = len(mouth_obj.get("frames") or mouth_obj.get("timeline") or [])
        except Exception:
            frames_n = 0

        if job.mouth_obj_ref is not None:
            try:
                job.mouth_obj_ref["obj"] = json.loads(
                    job.mouth_json.read_text(encoding="utf-8")
                )
                job.mouth_obj_ref["knn_raw_frames_done"] = len(
                    raw_obj.get("frames") or []
                )
            except Exception:
                pass

    return int(frames_n)


def _process_audio_chunk_push_knn_sync(
    *,
    job: _AudioPipelineJob,
    turn_state: dict[str, Any],
    pipeline_io_ref: dict[str, Any],
) -> dict[str, Any]:
    t_knn0 = time.perf_counter()
    frames_n = 0
    emitted = 0
    knn_ms = 0.0
    effective_playback = job.playback_audio
    enqueue_timeline_end_ms = 0

    io_lock = pipeline_io_ref.get("file_io_lock")
    if io_lock is not None:
        with io_lock:
            audio_f = _ensure_pipeline_audio_file(job=job, pipeline_io_ref=pipeline_io_ref)
            if audio_f is not None and not bool(job.skip_archive_pcm):
                audio_f.write(job.audio)
                audio_f.flush()
            if effective_playback and not bool(job.skip_archive_pcm):
                chunk_path = (
                    job.pcm_stream_chunks_dir
                    / f"chunk_{job.playback_chunk_idx:06d}.pcm"
                )
                chunk_path.write_bytes(effective_playback)
    else:
        audio_f = _ensure_pipeline_audio_file(job=job, pipeline_io_ref=pipeline_io_ref)
        if audio_f is not None and not bool(job.skip_archive_pcm):
            audio_f.write(job.audio)
            audio_f.flush()
        if effective_playback and not bool(job.skip_archive_pcm):
            chunk_path = (
                job.pcm_stream_chunks_dir
                / f"chunk_{job.playback_chunk_idx:06d}.pcm"
            )
            chunk_path.write_bytes(effective_playback)

    emitted = job.mouth_streamer.push_pcm16_mono(job.audio, input_sr=24000)
    # Phase24: KNN always projects from in-memory streamer frames (avoid
    # flush→full JSON read spikes). Disk archive flush remains for
    # --no-fast_inmemory (hot path omits debug_frames; finalize keeps them).
    if not bool(job.fast_inmemory):
        job.mouth_streamer.flush(include_debug=False)

    if int(emitted) > 0:
        raw_obj = _project_streamer_obj_to_raw(job.mouth_streamer)

        frames_n = _process_audio_chunk_knn_sync(
            raw_obj=raw_obj,
            job=job,
            turn_state=turn_state,
        )
        knn_ms = (time.perf_counter() - t_knn0) * 1000.0

        mouth_updated_event = turn_state.get("mouth_updated_event")
        if mouth_updated_event is not None:
            try:
                mouth_updated_event.set()
            except Exception:
                pass
        mouth_frames_async_event = turn_state.get("mouth_frames_async_event")
        if mouth_frames_async_event is not None:
            try:
                mouth_frames_async_event.set()
            except Exception:
                pass

    playback_ref = turn_state.get("audio_playback_state_ref")
    if effective_playback:
        enqueue_timeline_end_ms = int(
            _m0_pipeline_enqueue_timeline_end_ms(playback_ref, effective_playback)
        )

    print(
        "[sync][pipeline_chunk][knn_done]",
        f"chunk_idx={int(job.playback_chunk_idx)}",
        f"emitted={int(emitted)}",
        f"knn_ms={knn_ms:.1f}",
        f"frames_n={int(frames_n)}",
        flush=True,
    )

    return {
        "emitted": int(emitted),
        "frames_n": int(frames_n),
        "knn_ms": float(knn_ms),
        "effective_playback": effective_playback,
        "enqueue_timeline_end_ms": int(enqueue_timeline_end_ms),
    }


def _process_audio_chunk_m0_sync(
    *,
    job: _AudioPipelineJob,
    turn_state: dict[str, Any],
    m0_pipeline_ref: dict[str, Any],
    effective_playback: bytes,
    m0_max_chunks: int = 8,
    until_t1_ms: int | None = None,
) -> dict[str, Any]:
    playback_ref = turn_state.get("audio_playback_state_ref")
    enqueue_timeline_end_ms = int(until_t1_ms or 0)
    if effective_playback and enqueue_timeline_end_ms <= 0:
        enqueue_timeline_end_ms = int(
            _m0_pipeline_enqueue_timeline_end_ms(playback_ref, effective_playback)
        )

    m0_result = _m0_pipeline_advance_sync(
        m0_pipeline_ref=m0_pipeline_ref,
        mouth_obj_ref=job.mouth_obj_ref,
        audio_playback_state_ref=playback_ref,
        live_emo_id_getter=turn_state.get("live_emo_id_getter"),
        live_emo_events_getter=turn_state.get("live_emo_events_getter"),
        hang_timeout_ms=int(_M0_PIPELINE_HANG_TIMEOUT_MS),
        max_chunks=int(m0_max_chunks),
        until_t1_ms=int(enqueue_timeline_end_ms) if enqueue_timeline_end_ms > 0 else None,
    )

    m0_chunks_rendered = int(m0_result.get("chunks_rendered", 0))
    png_verified = bool(m0_result.get("png_verified", True))
    rendered_end = _m0_pipeline_rendered_end_ms(m0_pipeline_ref, playback_ref)
    covered = int(rendered_end) >= int(enqueue_timeline_end_ms) if enqueue_timeline_end_ms > 0 else True
    enqueue_blocked = bool(effective_playback and (not png_verified or not covered))

    breakdown = m0_result.get("m0_breakdown")
    if not isinstance(breakdown, dict):
        breakdown = _m0_timing_empty()

    return {
        "m0_ms": float(m0_result.get("m0_ms_total", 0.0)),
        "hang_used": bool(m0_result.get("hang_used", False)),
        "m0_chunks": int(m0_chunks_rendered),
        "m0_last_cid": int(m0_result.get("last_cid", -1)),
        "m0_last_global": int(m0_result.get("last_global_frame1", -1)),
        "png_verified": bool(png_verified),
        "enqueue_blocked": bool(enqueue_blocked),
        "enqueue_timeline_end_ms": int(enqueue_timeline_end_ms),
        "covered": bool(covered),
        "rendered_end_ms": int(rendered_end),
        "m0_breakdown": breakdown,
    }


def _enqueue_playback_audio_sync(
    *,
    job: _AudioPipelineJob,
    turn_state: dict[str, Any],
    effective_playback: bytes,
) -> None:
    if not effective_playback:
        return

    audio_player_proc = turn_state.get("audio_player_proc")
    audio_device = turn_state.get("ai_audio_output_device")
    if audio_player_proc is None or audio_device is None:
        return

    # Phase29: publish sync_meta immediately before the first player enqueue.
    _commit_pending_virtualcam_sync_meta(turn_state)

    playback_ref = turn_state.get("audio_playback_state_ref")
    chunk_path = job.pcm_stream_chunks_dir / f"chunk_{job.playback_chunk_idx:06d}.pcm"
    if not chunk_path.exists():
        chunk_path.write_bytes(effective_playback)

    _send_audio_chunk(
        audio_player_proc=audio_player_proc,
        pcm=chunk_path,
        chunk_id=int(job.playback_chunk_idx),
        audio_device=str(audio_device),
        single_chunk=True,
        audio_playback_state_ref=playback_ref,
    )
    # Phase29hf: refresh last-good clock after player ack updates playback_ref.
    _refresh_playback_clock_guard(turn_state)

    print(
        "[sync][pipeline_chunk][enqueue_done]",
        f"chunk_idx={int(job.playback_chunk_idx)}",
        f"bytes={len(effective_playback)}",
        f"delivered_epoch_ms={time.time() * 1000.0:.1f}",
        flush=True,
    )


def _finalize_m0_breakdown(
    *,
    m0_ms: float,
    breakdown: dict[str, float] | None,
) -> dict[str, float]:
    out = _m0_timing_empty()
    if isinstance(breakdown, dict):
        for k in out:
            out[k] = float(breakdown.get(k, 0.0) or 0.0)
    parts = (
        out["m0_wait_mouth_ms"]
        + out["m0_lock_ms"]
        + out["m0_slice_ms"]
        + out["m0_disk_ms"]
        + out["m0_req_send_ms"]
        + out["m0_png_wait_ms"]
        + out["m0_verify_ms"]
    )
    out["m0_other_ms"] = max(0.0, float(m0_ms) - float(parts))
    return out


def _log_pipeline_chunk_result(
    *,
    job: _AudioPipelineJob,
    pipeline_seq: int,
    emitted: int,
    knn_ms: float,
    m0_ms: float,
    m0_chunks: int,
    enqueue_ms: float,
    total_ms: float,
    hang_used: bool,
    png_verified: bool,
    frames_n: int,
    enqueue_timeline_end_ms: int,
    m0_last_cid: int,
    m0_last_global: int,
    turn_state: dict[str, Any],
    queue_wait_ms: float = 0.0,
    push_wait_ms: float = 0.0,
    order_wait_ms: float = 0.0,
    stage_knn_done_ms: float = 0.0,
    stage_m0_done_ms: float = 0.0,
    m0_breakdown: dict[str, float] | None = None,
) -> None:
    bd = _finalize_m0_breakdown(m0_ms=float(m0_ms), breakdown=m0_breakdown)
    print(
        "[sync][pipeline_chunk]",
        f"pipeline_seq={int(pipeline_seq)}",
        f"chunk_idx={int(job.playback_chunk_idx)}",
        f"emitted={int(emitted)}",
        f"knn_ms={knn_ms:.1f}",
        f"m0_ms={m0_ms:.1f}",
        f"m0_chunks={int(m0_chunks)}",
        f"enqueue_ms={enqueue_ms:.1f}",
        f"queue_wait_ms={queue_wait_ms:.1f}",
        f"push_wait_ms={push_wait_ms:.1f}",
        f"order_wait_ms={order_wait_ms:.1f}",
        f"total_ms={total_ms:.1f}",
        f"stage_knn_done_ms={stage_knn_done_ms:.1f}",
        f"stage_m0_done_ms={stage_m0_done_ms:.1f}",
        f"hang_used={bool(hang_used)}",
        f"png_verified={bool(png_verified)}",
        f"mouth_frames_n={int(frames_n)}",
        f"enqueue_timeline_end_ms={int(enqueue_timeline_end_ms)}",
        f"m0_last_cid={int(m0_last_cid)}",
        f"m0_last_global_frame1={int(m0_last_global)}",
        f"m0_wait_mouth_ms={bd['m0_wait_mouth_ms']:.1f}",
        f"m0_lock_ms={bd['m0_lock_ms']:.1f}",
        f"m0_slice_ms={bd['m0_slice_ms']:.1f}",
        f"m0_disk_ms={bd['m0_disk_ms']:.1f}",
        f"m0_req_send_ms={bd['m0_req_send_ms']:.1f}",
        f"m0_png_wait_ms={bd['m0_png_wait_ms']:.1f}",
        f"m0_verify_ms={bd['m0_verify_ms']:.1f}",
        f"m0_other_ms={bd['m0_other_ms']:.1f}",
        flush=True,
    )
    print(
        "[sync][pipeline_chunk][m0_breakdown]",
        f"chunk_idx={int(job.playback_chunk_idx)}",
        f"m0_ms={m0_ms:.1f}",
        f"wait_mouth={bd['m0_wait_mouth_ms']:.1f}",
        f"lock={bd['m0_lock_ms']:.1f}",
        f"slice={bd['m0_slice_ms']:.1f}",
        f"disk={bd['m0_disk_ms']:.1f}",
        f"req_send={bd['m0_req_send_ms']:.1f}",
        f"png_wait={bd['m0_png_wait_ms']:.1f}",
        f"verify={bd['m0_verify_ms']:.1f}",
        f"other={bd['m0_other_ms']:.1f}",
        flush=True,
    )


async def _process_pipeline_chunk_task(
    *,
    job: _AudioPipelineJob,
    pipeline_seq: int,
    m0_pipeline_ref: dict[str, Any] | None,
    enqueue_queue: asyncio.Queue,
    turn_state: dict[str, Any],
    pipeline_io_ref: dict[str, Any],
    inflight_sem: asyncio.Semaphore,
    stop_event: asyncio.Event,
) -> None:
    """図A: チャンク内は KNN → M0完了 → enqueue 一本道。チャンク間は並列開始可。"""
    await inflight_sem.acquire()
    _pipeline_inflight_inc(turn_state)
    try:
        t_pipeline0 = time.perf_counter()
        print(
            "[sync][pipeline_chunk][start]",
            f"pipeline_seq={int(pipeline_seq)}",
            f"chunk_idx={int(job.playback_chunk_idx)}",
            flush=True,
        )

        t_push0 = time.perf_counter()
        await _await_pipeline_push_turn(turn_state, int(pipeline_seq))
        push_wait_ms = (time.perf_counter() - t_push0) * 1000.0
        try:
            push_knn_result = await asyncio.to_thread(
                _process_audio_chunk_push_knn_sync,
                job=job,
                turn_state=turn_state,
                pipeline_io_ref=pipeline_io_ref,
            )
        finally:
            await _advance_pipeline_push_turn(turn_state, int(pipeline_seq))

        stage_knn_done_ms = (time.perf_counter() - t_pipeline0) * 1000.0
        emitted = int(push_knn_result.get("emitted", 0))
        frames_n = int(push_knn_result.get("frames_n", 0))
        knn_ms = float(push_knn_result.get("knn_ms", 0.0))
        effective_playback = push_knn_result.get("effective_playback") or b""
        enqueue_timeline_end_ms = int(push_knn_result.get("enqueue_timeline_end_ms", 0))

        m0_ms = 0.0
        hang_used = False
        m0_chunks = 0
        m0_last_cid = -1
        m0_last_global = -1
        png_verified = True
        enqueue_blocked = False
        covered = False
        stage_m0_done_ms = stage_knn_done_ms
        m0_breakdown = _m0_timing_empty()

        # Always resolve live refs (turn_state SSOT); do not trust a stale closure.
        m0_ref = turn_state.get("m0_pipeline_ref")
        if m0_ref is None:
            m0_ref = m0_pipeline_ref
        mouth_ref = job.mouth_obj_ref
        if mouth_ref is None:
            mouth_ref = turn_state.get("mouth_obj_ref")

        if effective_playback and m0_ref is not None and mouth_ref is not None:
            if job.mouth_obj_ref is None:
                job.mouth_obj_ref = mouth_ref
            t_m0 = time.perf_counter()
            covered = False
            no_progress_rounds = 0
            tail_hold_attempted = False
            # Coverage wait must not abort immediately on recv_stop: turn-end stop
            # fires while tail chunks still need claim/watermark. Interrupt cancels
            # the task (CancelledError). After stop, allow a short no-progress budget.
            while True:
                t_adv0 = time.perf_counter()
                rendered_before = _m0_pipeline_rendered_end_ms(
                    m0_ref, turn_state.get("audio_playback_state_ref")
                )
                m0_result = await asyncio.to_thread(
                    _process_audio_chunk_m0_sync,
                    job=job,
                    turn_state=turn_state,
                    m0_pipeline_ref=m0_ref,
                    effective_playback=effective_playback,
                    m0_max_chunks=8,
                    until_t1_ms=int(enqueue_timeline_end_ms),
                )
                adv_wall_ms = (time.perf_counter() - t_adv0) * 1000.0
                part = m0_result.get("m0_breakdown")
                part_sum = 0.0
                if isinstance(part, dict):
                    for k in m0_breakdown:
                        if k == "m0_wait_mouth_ms" or k == "m0_lock_ms":
                            continue
                        v = float(part.get(k, 0.0) or 0.0)
                        _m0_timing_add(m0_breakdown, k, v)
                        part_sum += v
                # Wall time in advance not explained by instrumented work ≈ lock contention.
                _m0_timing_add(
                    m0_breakdown,
                    "m0_lock_ms",
                    max(0.0, float(adv_wall_ms) - float(part_sum)),
                )
                hang_used = hang_used or bool(m0_result.get("hang_used", False))
                chunk_n = int(m0_result.get("m0_chunks", 0))
                m0_chunks += chunk_n
                if int(m0_result.get("m0_last_cid", -1)) >= 0:
                    m0_last_cid = int(m0_result.get("m0_last_cid", -1))
                if int(m0_result.get("m0_last_global", -1)) >= 0:
                    m0_last_global = int(m0_result.get("m0_last_global", -1))
                png_verified = bool(m0_result.get("png_verified", True))
                covered = bool(m0_result.get("covered", False))
                enqueue_blocked = bool(m0_result.get("enqueue_blocked", False))
                rendered_after = int(m0_result.get("rendered_end_ms", 0) or 0)
                if rendered_after <= 0:
                    rendered_after = _m0_pipeline_rendered_end_ms(
                        m0_ref, turn_state.get("audio_playback_state_ref")
                    )
                progressed = chunk_n > 0 or int(rendered_after) > int(rendered_before)
                if progressed:
                    no_progress_rounds = 0
                else:
                    no_progress_rounds += 1

                if covered and png_verified:
                    enqueue_blocked = False
                    m0_ms = (time.perf_counter() - t_m0) * 1000.0
                    print(
                        "[sync][pipeline_chunk][m0_done]",
                        f"chunk_idx={int(job.playback_chunk_idx)}",
                        f"m0_ms={m0_ms:.1f}",
                        f"m0_chunks={int(m0_chunks)}",
                        f"png_verified={bool(png_verified)}",
                        flush=True,
                    )
                    break

                if hang_used and not covered:
                    enqueue_blocked = True
                    m0_ms = (time.perf_counter() - t_m0) * 1000.0
                    print(
                        "[sync][pipeline_chunk][m0_hang_uncovered]",
                        f"chunk_idx={int(job.playback_chunk_idx)}",
                        f"until_t1_ms={int(enqueue_timeline_end_ms)}",
                        flush=True,
                    )
                    break

                # Turn-end mouth/claim stall: hold-extend once (mouth often ends ~1
                # step short of until → mouth_ready never clears). Phase30: also gate
                # on generation_complete_seen so we do not wait for recv_stop teardown.
                if _tail_hold_gate_ready(turn_state, stop_event, no_progress_rounds):
                    if (
                        not tail_hold_attempted
                        and int(enqueue_timeline_end_ms) > 0
                        and int(rendered_after) < int(enqueue_timeline_end_ms)
                    ):
                        added = _try_tail_hold_extend(
                            mouth_ref,
                            until_t1_ms=int(enqueue_timeline_end_ms),
                            step_ms=int(job.step_ms),
                            turn_state=turn_state,
                            stop_event=stop_event,
                        )
                        if added > 0:
                            tail_hold_attempted = True
                            print(
                                "[sync][pipeline_chunk][mouth_tail_hold_extend]",
                                f"chunk_idx={int(job.playback_chunk_idx)}",
                                f"until_t1_ms={int(enqueue_timeline_end_ms)}",
                                f"rendered_end_ms={int(rendered_after)}",
                                f"hold_frames={int(added)}",
                                f"via={'gen_complete' if turn_state.get('generation_complete_seen') else 'recv_stop'}",
                                flush=True,
                            )
                            ev = turn_state.get("mouth_frames_async_event")
                            if isinstance(ev, asyncio.Event):
                                ev.set()
                            no_progress_rounds = 0
                            continue
                        if stop_event.is_set():
                            # stop path already allows unbounded hold; do not retry.
                            tail_hold_attempted = True
                    if no_progress_rounds >= 40:
                        enqueue_blocked = True
                        m0_ms = (time.perf_counter() - t_m0) * 1000.0
                        print(
                            "[sync][pipeline_chunk][m0_tail_uncovered]",
                            f"chunk_idx={int(job.playback_chunk_idx)}",
                            f"until_t1_ms={int(enqueue_timeline_end_ms)}",
                            f"rendered_end_ms={int(rendered_after)}",
                            f"no_progress_rounds={int(no_progress_rounds)}",
                            flush=True,
                        )
                        break

                t_wait0 = time.perf_counter()
                # Release inflight slot while waiting for mouth so later chunks can
                # KNN and grow frames (otherwise all slots can stall on coverage).
                mouth_wait_released = False
                if chunk_n <= 0 and not covered:
                    inflight_sem.release()
                    mouth_wait_released = True
                try:
                    ev = turn_state.get("mouth_frames_async_event")
                    if isinstance(ev, asyncio.Event):
                        ev.clear()
                        try:
                            await asyncio.wait_for(ev.wait(), timeout=0.05)
                        except asyncio.TimeoutError:
                            pass
                    else:
                        await asyncio.sleep(0.02)
                finally:
                    if mouth_wait_released:
                        await inflight_sem.acquire()
                _m0_timing_add(
                    m0_breakdown,
                    "m0_wait_mouth_ms",
                    (time.perf_counter() - t_wait0) * 1000.0,
                )
            m0_ms = (time.perf_counter() - t_m0) * 1000.0
            stage_m0_done_ms = (time.perf_counter() - t_pipeline0) * 1000.0
            if effective_playback and not covered:
                enqueue_blocked = True
        elif effective_playback and m0_ref is None:
            enqueue_blocked = True
            png_verified = False
            print(
                "[sync][pipeline_chunk][m0_skip]",
                f"chunk_idx={int(job.playback_chunk_idx)}",
                "reason=m0_pipeline_ref_none",
                flush=True,
            )
        elif effective_playback and mouth_ref is None:
            enqueue_blocked = True
            png_verified = False
            print(
                "[sync][pipeline_chunk][m0_skip]",
                f"chunk_idx={int(job.playback_chunk_idx)}",
                "reason=mouth_obj_ref_none",
                flush=True,
            )

        # Drop stale work cancelled/superseded by user interrupt clear.
        min_keep = int(turn_state.get("pipeline_interrupt_min_seq", 0) or 0)
        if int(pipeline_seq) < min_keep:
            print(
                "[sync][pipeline_chunk][interrupt_drop]",
                f"pipeline_seq={int(pipeline_seq)}",
                f"min_seq={min_keep}",
                flush=True,
            )
            return

        turn_state["pipeline_continuous_errors"] = 0
        item = _PipelineEnqueueItem(
            pipeline_seq=int(pipeline_seq),
            job=job,
            effective_playback=effective_playback,
            enqueue_blocked=bool(enqueue_blocked),
            enqueue_timeline_end_ms=int(enqueue_timeline_end_ms),
            emitted=int(emitted),
            knn_ms=float(knn_ms),
            m0_ms=float(m0_ms),
            m0_chunks=int(m0_chunks),
            hang_used=bool(hang_used),
            png_verified=bool(png_verified),
            frames_n=int(frames_n),
            m0_last_cid=int(m0_last_cid),
            m0_last_global=int(m0_last_global),
            t_pipeline0=float(t_pipeline0),
            stage_knn_done_ms=float(stage_knn_done_ms),
            stage_m0_done_ms=float(stage_m0_done_ms),
            m0_breakdown=dict(m0_breakdown),
            push_wait_ms=float(push_wait_ms),
            t_enqueued=float(time.perf_counter()),
        )
        enqueue_queue.put_nowait(item)
    except asyncio.CancelledError:
        # Interrupt cancel: advance push turn if held; never enqueue leftover audio.
        try:
            await _advance_pipeline_push_turn(turn_state, int(pipeline_seq))
        except BaseException:
            pass
        print(
            "[sync][pipeline_chunk][cancelled]",
            f"pipeline_seq={int(pipeline_seq)}",
            f"chunk_idx={int(job.playback_chunk_idx)}",
            flush=True,
        )
        raise
    except BaseException as e:
        err_cnt = int(turn_state.get("pipeline_continuous_errors", 0)) + 1
        turn_state["pipeline_continuous_errors"] = err_cnt
        print(
            "[sync][pipeline_worker][ERROR]",
            f"type={type(e).__name__}",
            f"detail={e}",
            f"pipeline_seq={int(pipeline_seq)}",
            f"continuous_errors={err_cnt}",
            flush=True,
        )
        order = turn_state.get("pipeline_enqueue_order")
        if isinstance(order, dict):
            try:
                if int(order.get("expected_push_seq", 0)) == int(pipeline_seq):
                    await _advance_pipeline_push_turn(turn_state, int(pipeline_seq))
            except BaseException:
                pass
        min_keep = int(turn_state.get("pipeline_interrupt_min_seq", 0) or 0)
        if int(pipeline_seq) >= min_keep:
            enqueue_queue.put_nowait(
                _PipelineEnqueueItem(
                    pipeline_seq=int(pipeline_seq),
                    job=job,
                    effective_playback=b"",
                    enqueue_blocked=True,
                    enqueue_timeline_end_ms=0,
                    emitted=0,
                    knn_ms=0.0,
                    m0_ms=0.0,
                    m0_chunks=0,
                    hang_used=False,
                    png_verified=False,
                    frames_n=0,
                    m0_last_cid=-1,
                    m0_last_global=-1,
                    t_pipeline0=time.perf_counter(),
                    stage_knn_done_ms=0.0,
                    stage_m0_done_ms=0.0,
                )
            )
    finally:
        _pipeline_inflight_dec(turn_state)
        inflight_sem.release()


async def _pipeline_enqueue_dispatcher_loop(
    *,
    enqueue_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    turn_state: dict[str, Any],
) -> None:
    """pipeline_seq / chunk_idx 到着順で player enqueue。

    後続 chunk が先に M0 完了しても pending に保持し、
    expected_seq の順でのみ enqueue する（順序保証・デッドロック回避）。
    """
    pending: dict[int, _PipelineEnqueueItem] = {}
    saw_sentinel = False

    async def _enqueue_one(item: _PipelineEnqueueItem) -> None:
        order = _ensure_pipeline_enqueue_order(turn_state)
        t_enqueue0 = time.perf_counter()
        order_wait_ms = 0.0
        if float(item.t_enqueued) > 0.0:
            order_wait_ms = max(
                0.0, (t_enqueue0 - float(item.t_enqueued)) * 1000.0
            )
        try:
            if item.effective_playback:
                # Final guard + late catch-up: M0 coverage must reach until_t1
                # before player enqueue (図A). Runs even when chunk task left
                # enqueue_blocked after a stalled claim/watermark.
                m0_ref = turn_state.get("m0_pipeline_ref")
                playback_ref = turn_state.get("audio_playback_state_ref")
                if m0_ref is not None and int(item.enqueue_timeline_end_ms) > 0:
                    rendered_end = _m0_pipeline_rendered_end_ms(m0_ref, playback_ref)
                    catch_rounds = 0
                    tail_hold_tried = False
                    while (
                        int(rendered_end) < int(item.enqueue_timeline_end_ms)
                        and catch_rounds < 8
                    ):
                        catch_rounds += 1
                        mouth_ref = getattr(item.job, "mouth_obj_ref", None)
                        if mouth_ref is None:
                            mouth_ref = turn_state.get("mouth_obj_ref")
                        if mouth_ref is None:
                            break
                        try:
                            catch = await asyncio.to_thread(
                                _m0_pipeline_advance_sync,
                                m0_pipeline_ref=m0_ref,
                                mouth_obj_ref=mouth_ref,
                                audio_playback_state_ref=playback_ref,
                                live_emo_id_getter=turn_state.get(
                                    "live_emo_id_getter"
                                ),
                                live_emo_events_getter=turn_state.get(
                                    "live_emo_events_getter"
                                ),
                                hang_timeout_ms=int(_M0_PIPELINE_HANG_TIMEOUT_MS),
                                max_chunks=8,
                                until_t1_ms=int(item.enqueue_timeline_end_ms),
                            )
                        except BaseException as e:
                            print(
                                "[sync][pipeline_chunk][m0_catchup_error]",
                                f"chunk_idx={int(item.job.playback_chunk_idx)}",
                                f"type={type(e).__name__}",
                                f"detail={e}",
                                flush=True,
                            )
                            break
                        rendered_end = _m0_pipeline_rendered_end_ms(
                            m0_ref, playback_ref
                        )
                        n_catch = int(catch.get("chunks_rendered", 0) or 0)
                        if n_catch > 0:
                            print(
                                "[sync][pipeline_chunk][m0_catchup]",
                                f"chunk_idx={int(item.job.playback_chunk_idx)}",
                                f"m0_chunks={n_catch}",
                                f"rendered_end_ms={int(rendered_end)}",
                                f"until_t1_ms={int(item.enqueue_timeline_end_ms)}",
                                flush=True,
                            )
                        if int(rendered_end) >= int(item.enqueue_timeline_end_ms):
                            item.enqueue_blocked = False
                            item.png_verified = True
                            break
                        if n_catch <= 0:
                            # Turn-end safety net: same hold-extend as chunk task.
                            # Phase30: generation_complete_seen unlocks without recv_stop.
                            if (
                                (
                                    stop_event.is_set()
                                    or bool(
                                        turn_state.get("generation_complete_seen")
                                    )
                                )
                                and not tail_hold_tried
                                and mouth_ref is not None
                            ):
                                added = _try_tail_hold_extend(
                                    mouth_ref,
                                    until_t1_ms=int(item.enqueue_timeline_end_ms),
                                    step_ms=int(item.job.step_ms),
                                    turn_state=turn_state,
                                    stop_event=stop_event,
                                )
                                if added > 0:
                                    tail_hold_tried = True
                                    print(
                                        "[sync][pipeline_chunk][mouth_tail_hold_extend]",
                                        f"chunk_idx={int(item.job.playback_chunk_idx)}",
                                        f"until_t1_ms={int(item.enqueue_timeline_end_ms)}",
                                        f"rendered_end_ms={int(rendered_end)}",
                                        f"hold_frames={int(added)}",
                                        f"via=enqueue_dispatcher",
                                        f"gen_complete={1 if turn_state.get('generation_complete_seen') else 0}",
                                        flush=True,
                                    )
                                    continue
                                if stop_event.is_set():
                                    tail_hold_tried = True
                            # Mouth not ready yet for next cid; stop spinning here.
                            break

                    if int(rendered_end) < int(item.enqueue_timeline_end_ms):
                        print(
                            "[sync][pipeline_chunk][AUDIO_BEFORE_M0]",
                            f"chunk_idx={int(item.job.playback_chunk_idx)}",
                            f"rendered_end_ms={int(rendered_end)}",
                            f"until_t1_ms={int(item.enqueue_timeline_end_ms)}",
                            flush=True,
                        )
                        item.enqueue_blocked = True
                        item.png_verified = False

                if not item.enqueue_blocked:
                    await asyncio.to_thread(
                        _enqueue_playback_audio_sync,
                        job=item.job,
                        turn_state=turn_state,
                        effective_playback=item.effective_playback,
                    )
                    if bool(getattr(item.job, "is_idle_silent", False)):
                        n = int(turn_state.get("idle_silent_enqueue_n", 0) or 0) + 1
                        turn_state["idle_silent_enqueue_n"] = n
                        if n == 1 or (n % 25) == 0:
                            print(
                                "[idle_silent_pcm][enqueue]",
                                f"n={n}",
                                f"chunk_idx={int(item.job.playback_chunk_idx)}",
                                f"pipeline_seq={int(item.pipeline_seq)}",
                                f"bytes={len(item.effective_playback)}",
                                flush=True,
                            )
                else:
                    print(
                        "[sync][pipeline_chunk][ENQUEUE_BLOCKED]",
                        f"chunk_idx={int(item.job.playback_chunk_idx)}",
                        f"pipeline_seq={int(item.pipeline_seq)}",
                        f"reason=m0_png_missing",
                        f"enqueue_timeline_end_ms={int(item.enqueue_timeline_end_ms)}",
                        flush=True,
                    )
        finally:
            cond: asyncio.Condition = order["cond"]
            async with cond:
                if int(order["expected_seq"]) == int(item.pipeline_seq):
                    order["expected_seq"] = int(item.pipeline_seq) + 1
                    cond.notify_all()
            enqueue_ms = (time.perf_counter() - t_enqueue0) * 1000.0
            total_ms = (time.perf_counter() - item.t_pipeline0) * 1000.0
            queue_wait_ms = max(
                0.0,
                total_ms - float(item.knn_ms) - float(item.m0_ms) - enqueue_ms,
            )
            _log_pipeline_chunk_result(
                job=item.job,
                pipeline_seq=int(item.pipeline_seq),
                emitted=int(item.emitted),
                knn_ms=float(item.knn_ms),
                m0_ms=float(item.m0_ms),
                m0_chunks=int(item.m0_chunks),
                enqueue_ms=float(enqueue_ms),
                total_ms=float(total_ms),
                hang_used=bool(item.hang_used),
                png_verified=bool(item.png_verified),
                frames_n=int(item.frames_n),
                enqueue_timeline_end_ms=int(item.enqueue_timeline_end_ms),
                m0_last_cid=int(item.m0_last_cid),
                m0_last_global=int(item.m0_last_global),
                turn_state=turn_state,
                queue_wait_ms=float(queue_wait_ms),
                push_wait_ms=float(item.push_wait_ms),
                order_wait_ms=float(order_wait_ms),
                stage_knn_done_ms=float(item.stage_knn_done_ms),
                stage_m0_done_ms=float(item.stage_m0_done_ms),
                m0_breakdown=item.m0_breakdown,
            )

    while True:
        got = None
        timed_out = False
        try:
            got = await asyncio.wait_for(enqueue_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            timed_out = True

        if not timed_out:
            enqueue_queue.task_done()
            if got is None:
                saw_sentinel = True
            elif isinstance(got, _PipelineInterruptClear):
                drop_before = int(got.min_seq)
                dropped = [s for s in list(pending.keys()) if int(s) < drop_before]
                for s in dropped:
                    pending.pop(s, None)
                order = _ensure_pipeline_enqueue_order(turn_state)
                cond: asyncio.Condition = order["cond"]
                async with cond:
                    order["expected_seq"] = max(int(order["expected_seq"]), drop_before)
                    order["expected_push_seq"] = max(
                        int(order["expected_push_seq"]), drop_before
                    )
                    cond.notify_all()
                print(
                    "[battle_talkover][enqueue_pending_dropped]",
                    f"min_seq={drop_before}",
                    f"dropped_n={len(dropped)}",
                    f"expected_seq={int(order['expected_seq'])}",
                    flush=True,
                )
            else:
                assert isinstance(got, _PipelineEnqueueItem)
                seq = int(got.pipeline_seq)
                min_keep = int(turn_state.get("pipeline_interrupt_min_seq", 0) or 0)
                if seq < min_keep:
                    print(
                        "[sync][enqueue_order][interrupt_drop]",
                        f"pipeline_seq={seq}",
                        f"min_seq={min_keep}",
                        flush=True,
                    )
                else:
                    pending[seq] = got
                    order = _ensure_pipeline_enqueue_order(turn_state)
                    if seq != int(order["expected_seq"]):
                        print(
                            "[sync][enqueue_order][pending]",
                            f"pipeline_seq={seq}",
                            f"chunk_idx={int(got.job.playback_chunk_idx)}",
                            f"expected_seq={int(order['expected_seq'])}",
                            f"hol_gap={int(seq) - int(order['expected_seq'])}",
                            f"pending_n={len(pending)}",
                            f"push_wait_ms={float(got.push_wait_ms):.1f}",
                            flush=True,
                        )

        order = _ensure_pipeline_enqueue_order(turn_state)
        while int(order["expected_seq"]) in pending:
            nxt = pending.pop(int(order["expected_seq"]))
            await _enqueue_one(nxt)
            order = _ensure_pipeline_enqueue_order(turn_state)

        if saw_sentinel and not pending:
            break
        if stop_event.is_set() and saw_sentinel and not pending:
            break


def _read_player_buffer_snapshot(
    turn_state: dict[str, Any],
) -> tuple[float, str, int]:
    """Return (pending_ms, state, initial_buffer_ms)."""
    pending_ms = 0.0
    state = "UNKNOWN"
    initial_buffer_ms = 0
    path = turn_state.get("playback_state_file")
    if path is not None:
        try:
            p = Path(path)
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8-sig"))
                if isinstance(obj, dict):
                    state = str(obj.get("state", "UNKNOWN") or "UNKNOWN")
                    pending_ms = float(obj.get("pending_ms", 0.0) or 0.0)
                    initial_buffer_ms = int(obj.get("initial_buffer_ms", 0) or 0)
                    return float(pending_ms), state, int(initial_buffer_ms)
        except Exception:
            pass
    ref = turn_state.get("audio_playback_state_ref")
    if isinstance(ref, dict):
        try:
            with ref["lock"]:
                pending_ms = float(ref.get("pending_ms", 0.0) or 0.0)
        except Exception:
            pending_ms = 0.0
    return float(pending_ms), state, int(initial_buffer_ms)


async def _alloc_pipeline_slot(
    turn_state: dict[str, Any],
    *,
    advance_playback_idx: bool,
) -> tuple[int, int]:
    """Allocate (pipeline_seq, playback_chunk_idx) under shared lock."""
    lock = turn_state.get("pipeline_alloc_lock")
    if not isinstance(lock, asyncio.Lock):
        lock = asyncio.Lock()
        turn_state["pipeline_alloc_lock"] = lock
    async with lock:
        seq = int(turn_state.get("pipeline_next_seq", 0) or 0)
        turn_state["pipeline_next_seq"] = int(seq) + 1
        pb_idx = int(turn_state.get("pipeline_playback_chunk_idx", 0) or 0)
        if advance_playback_idx:
            turn_state["pipeline_playback_chunk_idx"] = int(pb_idx) + 1
        return int(seq), int(pb_idx)


async def _idle_silent_pcm_loop(
    *,
    stop_event: asyncio.Event,
    ai_audio_started_event: asyncio.Event | None,
    turn_state: dict[str, Any],
    pcm_stream_chunks_dir: Path,
    audio_response_pcm: Path,
    mouth_streamer: MouthStreamerOC,
    mouth_streamer_json: Path,
    mouth_raw_json: Path,
    mouth_json: Path,
    knn_script: Path,
    gt_glob: str,
    step_ms: int,
    mouth_obj_ref: dict[str, Any] | None,
    knn_inmemory: bool,
    m0_inmemory: bool,
    fast_inmemory: bool,
    interval_ms: int,
    pending_target_ms: int,
) -> None:
    """Phase12: feed silent PCM through 図A (KNN→M0→enqueue) while waiting.

    Keeps player PLAYING / audio_ms advancing so VirtualCam BGV + idle mouth move.
    Just-in-time: only inject when pending_ms < pending_target_ms to limit residual
    silence ahead of real AI audio (no normal-turn clear_queue).
    """
    input_sr = 24000
    samples_per_chunk = max(1, int(round(input_sr * int(interval_ms) / 1000.0)))
    silent_pcm = bytes(samples_per_chunk * 2)
    target_pending = max(int(interval_ms), int(pending_target_ms))

    i = 0
    injected = 0
    t0 = time.perf_counter()
    print(
        "[idle_silent_pcm][START]",
        f"input_sr={input_sr}",
        f"interval_ms={int(interval_ms)}",
        f"samples={samples_per_chunk}",
        f"bytes={len(silent_pcm)}",
        f"pending_target_ms={int(target_pending)}",
        flush=True,
    )

    try:
        # Wait until receive_loop enables 図A pipeline.
        while not stop_event.is_set():
            if ai_audio_started_event is not None and ai_audio_started_event.is_set():
                print(
                    "[idle_silent_pcm][STOP_ON_REAL_AUDIO]",
                    "phase=wait_pipeline",
                    flush=True,
                )
                return
            if turn_state.get("pipeline_enqueue_queue_ref") is not None:
                break
            await asyncio.sleep(0.02)

        while not stop_event.is_set():
            if ai_audio_started_event is not None and ai_audio_started_event.is_set():
                print(
                    "[idle_silent_pcm][STOP_ON_REAL_AUDIO]",
                    f"i={i}",
                    f"injected={injected}",
                    f"elapsed_s={time.perf_counter() - t0:.3f}",
                    flush=True,
                )
                break

            pending_ms, player_state, initial_buffer_ms = _read_player_buffer_snapshot(
                turn_state
            )
            effective_target = max(int(target_pending), int(initial_buffer_ms))
            # Maintain a shallow buffer so handoff residual stays small.
            # While BUFFERING, fill at least initial_buffer so PLAYING can start.
            if float(pending_ms) >= float(effective_target) and str(player_state) in (
                "PLAYING",
                "BUFFERING",
            ):
                i += 1
                await asyncio.sleep(max(0.001, float(interval_ms) / 1000.0))
                continue

            enqueue_queue = turn_state.get("pipeline_enqueue_queue_ref")
            inflight_sem = turn_state.get("pipeline_inflight_sem")
            pipeline_io_ref = turn_state.get("pipeline_io_ref")
            m0_pipeline_ref = turn_state.get("m0_pipeline_ref")
            active_tasks = turn_state.get("pipeline_active_tasks_ref")
            if (
                enqueue_queue is None
                or inflight_sem is None
                or not isinstance(pipeline_io_ref, dict)
                or not isinstance(active_tasks, list)
            ):
                await asyncio.sleep(0.02)
                continue

            # Avoid unbounded tasks waiting on inflight_sem (interrupt cancel spam).
            alive_n = len([t for t in active_tasks if t is not None and not t.done()])
            if int(alive_n) >= int(_PIPELINE_MAX_INFLIGHT):
                i += 1
                await asyncio.sleep(max(0.001, float(interval_ms) / 1000.0))
                continue

            lock = turn_state.get("pipeline_alloc_lock")
            if not isinstance(lock, asyncio.Lock):
                lock = asyncio.Lock()
                turn_state["pipeline_alloc_lock"] = lock

            async with lock:
                if (
                    ai_audio_started_event is not None
                    and ai_audio_started_event.is_set()
                ):
                    print(
                        "[idle_silent_pcm][STOP_ON_REAL_AUDIO]",
                        f"i={i}",
                        f"injected={injected}",
                        f"elapsed_s={time.perf_counter() - t0:.3f}",
                        flush=True,
                    )
                    break
                if stop_event.is_set():
                    break
                seq = int(turn_state.get("pipeline_next_seq", 0) or 0)
                turn_state["pipeline_next_seq"] = int(seq) + 1
                pb_idx = int(turn_state.get("pipeline_playback_chunk_idx", 0) or 0)
                turn_state["pipeline_playback_chunk_idx"] = int(pb_idx) + 1

            job = _AudioPipelineJob(
                ctx_id="turn",
                playback_chunk_idx=int(pb_idx),
                playback_epoch=int(
                    turn_state.get(
                        "audio_playback_accept_epoch",
                        _AUDIO_PLAYBACK_EPOCH_GAP,
                    )
                ),
                audio=silent_pcm,
                playback_audio=silent_pcm,
                pcm_stream_chunks_dir=pcm_stream_chunks_dir,
                audio_response_pcm=audio_response_pcm,
                mouth_streamer=mouth_streamer,
                mouth_streamer_json=mouth_streamer_json,
                mouth_raw_json=mouth_raw_json,
                mouth_json=mouth_json,
                knn_script=knn_script,
                gt_glob=gt_glob,
                step_ms=int(step_ms),
                knn_inmemory=bool(knn_inmemory),
                m0_inmemory=bool(m0_inmemory),
                mouth_obj_ref=mouth_obj_ref,
                fast_inmemory=bool(fast_inmemory),
                # Idle silence: skip archive I/O; still enqueues to player for SSOT.
                skip_archive_pcm=True,
                is_idle_silent=True,
            )
            task = asyncio.create_task(
                _process_pipeline_chunk_task(
                    job=job,
                    pipeline_seq=int(seq),
                    m0_pipeline_ref=m0_pipeline_ref,
                    enqueue_queue=enqueue_queue,
                    turn_state=turn_state,
                    pipeline_io_ref=pipeline_io_ref,
                    inflight_sem=inflight_sem,
                    stop_event=stop_event,
                ),
                name=f"idle_silent_chunk_{seq}",
            )
            active_tasks.append(task)
            active_tasks[:] = [t for t in active_tasks if not t.done()]
            injected += 1
            if injected == 1 or (injected % 25) == 0:
                print(
                    "[idle_silent_pcm][inject]",
                    f"i={i}",
                    f"injected={injected}",
                    f"pipeline_seq={int(seq)}",
                    f"chunk_idx={int(pb_idx)}",
                    f"pending_ms={float(pending_ms):.1f}",
                    f"state={player_state}",
                    f"elapsed_s={time.perf_counter() - t0:.3f}",
                    flush=True,
                )
            i += 1
            await asyncio.sleep(max(0.001, float(interval_ms) / 1000.0))
    finally:
        print(
            "[idle_silent_pcm][STOP]",
            f"chunks_loop={i}",
            f"injected={injected}",
            f"elapsed_s={time.perf_counter() - t0:.3f}",
            flush=True,
        )


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
    pipeline_sync: bool = True,
    mouth_obj_ref: dict[str, Any] | None = None,
    knn_inmemory: bool = False,
    m0_inmemory: bool = False,
    fast_inmemory: bool = False,
    skip_archive_pcm: bool = False,
    ai_audio_started_event: asyncio.Event | None = None,
) -> None:
    audio_response_pcm.parent.mkdir(parents=True, exist_ok=True)
    pcm_stream_chunks_dir.mkdir(parents=True, exist_ok=True)

    audio_chunk_idx = 0
    drop_initial_audio_bytes_remaining = max(
        0,
        int(round(24000 * 2 * int(drop_initial_audio_ms) / 1000.0)),
    )

    pipeline_io_ref: dict[str, Any] = {
        "ctx_id": "turn",
        "audio_f": None,
        "file_io_lock": Lock(),
    }
    pipeline_enqueue_queue: asyncio.Queue | None = None
    pipeline_enqueue_dispatcher_task: asyncio.Task | None = None
    pipeline_inflight_sem: asyncio.Semaphore | None = None
    pipeline_active_tasks: list[asyncio.Task[None]] = []
    m0_pipeline_ref = turn_state.get("m0_pipeline_ref")
    turn_state["pipeline_active_tasks_ref"] = pipeline_active_tasks
    turn_state["pipeline_next_seq"] = 0
    turn_state["pipeline_playback_chunk_idx"] = 0
    turn_state["pipeline_interrupt_min_seq"] = 0
    turn_state["pipeline_alloc_lock"] = asyncio.Lock()
    turn_state["pipeline_io_ref"] = pipeline_io_ref
    if mouth_obj_ref is not None:
        turn_state["mouth_obj_ref"] = mouth_obj_ref

    if bool(pipeline_sync):
        turn_state["pipeline_inflight"] = 0
        turn_state["mouth_updated_event"] = mouth_updated_event
        turn_state["mouth_frames_async_event"] = asyncio.Event()
        _ensure_pipeline_enqueue_order(turn_state)
        pipeline_enqueue_queue = asyncio.Queue()
        turn_state["pipeline_enqueue_queue_ref"] = pipeline_enqueue_queue
        pipeline_inflight_sem = asyncio.Semaphore(_PIPELINE_MAX_INFLIGHT)
        turn_state["pipeline_inflight_sem"] = pipeline_inflight_sem
        pipeline_enqueue_dispatcher_task = asyncio.create_task(
            _pipeline_enqueue_dispatcher_loop(
                enqueue_queue=pipeline_enqueue_queue,
                stop_event=stop_event,
                turn_state=turn_state,
            ),
            name="pipeline_enqueue_dispatcher",
        )
        print(
            "[sync][pipeline][ENABLED]",
            f"m0_hang_timeout_ms={int(_M0_PIPELINE_HANG_TIMEOUT_MS)}",
            f"fast_inmemory={bool(fast_inmemory)}",
            f"knn_inmemory={bool(knn_inmemory)}",
            f"knn_incremental={bool(turn_state.get('knn_incremental', True))}",
            f"skip_archive_pcm={bool(skip_archive_pcm)}",
            flush=True,
        )
        if bool(fast_inmemory):
            print(
                "[fast_inmemory][ENABLED]",
                f"knn_inmemory={bool(knn_inmemory)}",
                f"skip_archive_pcm={bool(skip_archive_pcm)}",
                "streamer_flush=skipped",
                flush=True,
            )

    audio_f_legacy = None
    try:
        if not bool(skip_archive_pcm):
            audio_f_legacy = audio_response_pcm.open("ab")
            if bool(pipeline_sync):
                pipeline_io_ref["audio_f"] = audio_f_legacy

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
                            # generation_complete = 入力終了マーカーのみ。
                            # clear_queue / ループ停止 / tail 打ち切りは禁止（Phase 4）。
                            if bool(
                                getattr(server_content, "generation_complete", False)
                            ):
                                # Phase30: unlock turn-end mouth hold-extend early.
                                # Still marker_only for player/clear (Phase 4).
                                turn_state["generation_complete_seen"] = True
                                ev_gc = turn_state.get("mouth_frames_async_event")
                                if isinstance(ev_gc, asyncio.Event):
                                    ev_gc.set()
                                print(
                                    "[session_loop][generation_complete]",
                                    "marker_only",
                                    "no_clear_queue",
                                    f"active_turn={turn_state.get('active_turn')}",
                                    "tail_hold_gate=1",
                                    flush=True,
                                )
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
                                transcription_text = getattr(
                                    output_transcription, "text", None
                                )
                    except Exception:
                        pass

                    if transcription_text:
                        print(
                            f"[transcription][output] {str(transcription_text)[:160]}",
                            flush=True,
                        )

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
                                    "virtual_offset_ms": int(
                                        local_i * virtual_emo_interval_ms
                                    ),
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
                        if (
                            ai_audio_started_event is not None
                            and not ai_audio_started_event.is_set()
                        ):
                            ai_audio_started_event.set()
                            print(
                                "[idle_silent_pcm][real_audio_started]",
                                f"active_turn={turn_state.get('active_turn')}",
                                flush=True,
                            )

                        now = time.perf_counter()
                        turn_state["last_audio_perf"] = now

                        active_turn = turn_state.get("active_turn")
                        audio_chunk_idx_log = int(
                            turn_state.get("audio_chunk_idx_log", 0)
                        )

                        sec_from_turn = None
                        if turn_state.get("turn_start_perf") is not None:
                            sec_from_turn = now - float(turn_state["turn_start_perf"])

                        sec_s = (
                            f"{sec_from_turn:.3f}"
                            if sec_from_turn is not None
                            else "None"
                        )
                        print(
                            "[perf][session_audio_chunk] "
                            f"idx={audio_chunk_idx_log} "
                            f"active_turn={active_turn} "
                            f"sec_from_turn={sec_s}",
                            flush=True,
                        )

                        turn_state["audio_chunk_idx_log"] = audio_chunk_idx_log + 1

                        if (
                            active_turn is not None
                            and turn_state.get("first_audio_sec") is None
                        ):
                            first_audio = now - float(turn_state["turn_start_perf"])
                            turn_state["first_audio_sec"] = first_audio
                            print(
                                f"[perf][turn{active_turn}_first_response_audio_chunk_sec] "
                                f"{first_audio:.3f}",
                                flush=True,
                            )
                            # Phase R1: response-timing SSOT (same clock as speech_end / activity_end)
                            print(
                                "[resp_timing] first_audio",
                                f"perf_ms={now * 1000.0:.3f}",
                                f"turn={active_turn}",
                                flush=True,
                            )

                        playback_audio = audio
                        if drop_initial_audio_bytes_remaining > 0:
                            if len(playback_audio) <= drop_initial_audio_bytes_remaining:
                                drop_initial_audio_bytes_remaining -= len(
                                    playback_audio
                                )
                                playback_audio = b""
                            else:
                                playback_audio = playback_audio[
                                    drop_initial_audio_bytes_remaining:
                                ]
                                drop_initial_audio_bytes_remaining = 0

                        if (
                            bool(pipeline_sync)
                            and pipeline_enqueue_queue is not None
                            and pipeline_inflight_sem is not None
                        ):
                            # 図A: PCM到着で非同期フォーク。KNN+M0はタスク内、enqueueはdispatcher。
                            # Shared alloc with idle_silent so seq/chunk_idx stay ordered.
                            pipeline_seq, playback_chunk_idx = await _alloc_pipeline_slot(
                                turn_state,
                                advance_playback_idx=bool(playback_audio),
                            )
                            job = _AudioPipelineJob(
                                ctx_id="turn",
                                playback_chunk_idx=int(playback_chunk_idx),
                                playback_epoch=int(
                                    turn_state.get(
                                        "audio_playback_accept_epoch",
                                        _AUDIO_PLAYBACK_EPOCH_GAP,
                                    )
                                ),
                                audio=audio,
                                playback_audio=playback_audio,
                                pcm_stream_chunks_dir=pcm_stream_chunks_dir,
                                audio_response_pcm=audio_response_pcm,
                                mouth_streamer=mouth_streamer,
                                mouth_streamer_json=mouth_streamer_json,
                                mouth_raw_json=mouth_raw_json,
                                mouth_json=mouth_json,
                                knn_script=knn_script,
                                gt_glob=gt_glob,
                                step_ms=int(step_ms),
                                knn_inmemory=bool(knn_inmemory),
                                m0_inmemory=bool(m0_inmemory),
                                mouth_obj_ref=mouth_obj_ref,
                                fast_inmemory=bool(fast_inmemory),
                                skip_archive_pcm=bool(skip_archive_pcm),
                                is_idle_silent=False,
                            )
                            audio_chunk_idx += 1
                            task = asyncio.create_task(
                                _process_pipeline_chunk_task(
                                    job=job,
                                    pipeline_seq=int(pipeline_seq),
                                    m0_pipeline_ref=m0_pipeline_ref,
                                    enqueue_queue=pipeline_enqueue_queue,
                                    turn_state=turn_state,
                                    pipeline_io_ref=pipeline_io_ref,
                                    inflight_sem=pipeline_inflight_sem,
                                    stop_event=stop_event,
                                ),
                                name=f"pipeline_chunk_{pipeline_seq}",
                            )
                            pipeline_active_tasks.append(task)
                            pipeline_active_tasks[:] = [
                                t for t in pipeline_active_tasks if not t.done()
                            ]
                        else:
                            # Legacy Phase10 file-watch path (pipeline_sync=False).
                            playback_chunk_idx = int(
                                turn_state.get("pipeline_playback_chunk_idx", 0) or 0
                            )
                            if audio_f_legacy is not None:
                                audio_f_legacy.write(audio)
                                audio_f_legacy.flush()

                            if playback_audio:
                                chunk_path = (
                                    pcm_stream_chunks_dir
                                    / f"chunk_{playback_chunk_idx:06d}.pcm"
                                )
                                chunk_path.write_bytes(playback_audio)
                                turn_state["pipeline_playback_chunk_idx"] = (
                                    int(playback_chunk_idx) + 1
                                )

                            audio_chunk_idx += 1

                            emitted = mouth_streamer.push_pcm16_mono(
                                audio, input_sr=24000
                            )
                            mouth_streamer.flush(include_debug=False)

                            raw_obj = _project_streamer_obj_to_raw(mouth_streamer)
                            mouth_raw_json.write_text(
                                json.dumps(
                                    raw_obj,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                                encoding="utf-8",
                            )

                            if emitted > 0:
                                if not turn_state.get(
                                    "first_mouth_json_logged", False
                                ):
                                    active_turn = turn_state.get("active_turn")
                                    if active_turn is not None:
                                        dt = time.perf_counter() - float(
                                            turn_state["turn_start_perf"]
                                        )
                                        print(
                                            "[perf][session_first_stream_mouth_json_written_from_turn_start_sec] "
                                            f"{dt:.3f}",
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

            print(
                "[session_loop][receive_loop] receive() ended; restart",
                flush=True,
            )
            await asyncio.sleep(0.05)

            if not got_any:
                await asyncio.sleep(0.1)
    finally:
        if bool(pipeline_sync):
            pending = [t for t in pipeline_active_tasks if not t.done()]
            if pending:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    for t in pending:
                        t.cancel()

            if pipeline_enqueue_queue is not None:
                pipeline_enqueue_queue.put_nowait(None)
            if pipeline_enqueue_dispatcher_task is not None:
                try:
                    await asyncio.wait_for(
                        pipeline_enqueue_dispatcher_task, timeout=10.0
                    )
                except asyncio.TimeoutError:
                    pipeline_enqueue_dispatcher_task.cancel()

            turn_state.pop("pipeline_enqueue_order", None)
            turn_state.pop("pipeline_enqueue_queue_ref", None)
            turn_state.pop("pipeline_inflight_sem", None)
            turn_state.pop("pipeline_io_ref", None)
            turn_state.pop("mouth_frames_async_event", None)

        # Close archive PCM only after pipeline tasks + dispatcher drain.
        try:
            af = pipeline_io_ref.get("audio_f")
            if af is not None:
                af.close()
                pipeline_io_ref["audio_f"] = None
        except Exception:
            pass
        try:
            if (
                audio_f_legacy is not None
                and pipeline_io_ref.get("audio_f") is not audio_f_legacy
            ):
                audio_f_legacy.close()
        except Exception:
            pass


def _read_played_samples_from_state_file(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(obj, dict):
            return int(obj.get("played_samples", 0) or 0)
    except Exception:
        return 0
    return 0


def _read_playback_clock_from_state_file(
    path: Path | None,
) -> tuple[int, int, int]:
    """Return (played_samples, pending_samples, sample_rate)."""
    played = 0
    pending_samples = 0
    sample_rate = 24000
    if path is None or not Path(path).exists():
        return played, pending_samples, sample_rate
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(obj, dict):
            played = int(obj.get("played_samples", 0) or 0)
            pending_samples = int(obj.get("pending_samples", 0) or 0)
            if pending_samples <= 0:
                # Fallback when only pending_ms is published.
                pending_ms = float(obj.get("pending_ms", 0.0) or 0.0)
                sr = int(obj.get("sample_rate", 24000) or 24000)
                if sr > 0 and pending_ms > 0.0:
                    pending_samples = int(round(pending_ms * float(sr) / 1000.0))
            sample_rate = int(obj.get("sample_rate", 24000) or 24000)
            if sample_rate <= 0:
                sample_rate = 24000
    except Exception:
        return 0, 0, 24000
    return int(played), int(pending_samples), int(sample_rate)


def _read_playback_clock_from_ref(
    turn_state: dict[str, Any],
) -> tuple[int, int, int]:
    """Return (played_samples, pending_samples, sample_rate) from in-proc ref."""
    ref = turn_state.get("audio_playback_state_ref")
    if not isinstance(ref, dict):
        return 0, 0, 24000
    try:
        with ref["lock"]:
            played = int(ref.get("played_samples", 0) or 0)
            pending_ms = float(ref.get("pending_ms", 0.0) or 0.0)
            sr = int(ref.get("sample_rate", 24000) or 24000)
        if sr <= 0:
            sr = 24000
        pending_samples = 0
        if pending_ms > 0.0:
            pending_samples = int(round(pending_ms * float(sr) / 1000.0))
        return int(played), int(pending_samples), int(sr)
    except Exception:
        return 0, 0, 24000


def _playback_clock_guard(turn_state: dict[str, Any]) -> dict[str, int]:
    guard = turn_state.get("playback_clock_guard")
    if not isinstance(guard, dict):
        guard = {"played_samples": 0, "pending_samples": 0}
        turn_state["playback_clock_guard"] = guard
    return guard


def _refresh_playback_clock_guard(turn_state: dict[str, Any]) -> None:
    """Keep session-scoped last-good clock (survives turn_state.clear via reattach)."""
    guard = _playback_clock_guard(turn_state)
    state_file = turn_state.get("playback_state_file")
    f_played, f_pending, _sr = _read_playback_clock_from_state_file(
        Path(state_file) if state_file is not None else None
    )
    r_played, r_pending, _ = _read_playback_clock_from_ref(turn_state)
    if int(f_played) + int(f_pending) >= int(r_played) + int(r_pending):
        played, pending_samples = int(f_played), int(f_pending)
    else:
        played, pending_samples = int(r_played), int(r_pending)
    if played + pending_samples <= 0:
        return
    guard["played_samples"] = max(int(guard.get("played_samples", 0) or 0), played)
    # pending is instantaneous; keep latest non-negative snapshot
    guard["pending_samples"] = max(0, pending_samples)


def _pick_playback_clock_for_commit(
    turn_state: dict[str, Any],
) -> tuple[int, int, int, str]:
    """Pick best (played, pending, sr, source) for sync_meta base.

    Prefers the higher played+pending snapshot among state_file and playback_ref.
    Retries briefly when prior progress exists but both sources read as zero
    (Windows atomic-replace / empty-file race → T8-type false base=0).
    """
    state_file = turn_state.get("playback_state_file")
    path = Path(state_file) if state_file is not None else None
    guard = _playback_clock_guard(turn_state)
    last_played = int(guard.get("played_samples", 0) or 0)
    last_pending = int(guard.get("pending_samples", 0) or 0)
    last_sum = max(0, last_played + last_pending)
    pending_meta = turn_state.get("pending_virtualcam_sync_meta")
    turn_no = 0
    if isinstance(pending_meta, dict) and pending_meta.get("turn_no") is not None:
        try:
            turn_no = int(pending_meta.get("turn_no") or 0)
        except Exception:
            turn_no = 0

    played = 0
    pending_samples = 0
    sample_rate = 24000
    source = "empty"
    max_attempts = 6 if last_sum > 0 or turn_no > 1 else 1
    for attempt in range(max_attempts):
        f_played, f_pending, f_sr = _read_playback_clock_from_state_file(path)
        r_played, r_pending, r_sr = _read_playback_clock_from_ref(turn_state)
        if int(f_played) + int(f_pending) >= int(r_played) + int(r_pending):
            played, pending_samples, sample_rate = int(f_played), int(f_pending), int(f_sr)
            source = "state_file"
        else:
            played, pending_samples, sample_rate = int(r_played), int(r_pending), int(r_sr)
            source = "playback_ref"
        if played > 0 or pending_samples > 0:
            return played, pending_samples, sample_rate, source
        if last_sum <= 0 and turn_no <= 1:
            return 0, 0, sample_rate, source
        print(
            "[sync][virtualcam_meta][COMMIT_CLOCK_RETRY]",
            f"attempt={attempt + 1}/{max_attempts}",
            f"turn_no={turn_no}",
            f"last_played={last_played}",
            f"last_pending={last_pending}",
            f"file=({f_played},{f_pending})",
            f"ref=({r_played},{r_pending})",
            flush=True,
        )
        time.sleep(0.005)
    # Still zero after retries while prior progress exists → refuse (caller retries).
    return 0, 0, sample_rate, "false_zero"


def _commit_pending_virtualcam_sync_meta(turn_state: dict[str, Any]) -> bool:
    """Phase29: publish new-turn sync_meta on first player enqueue.

    Writing frame_offset/base at turn start remaps leftover previous-turn PCM
    onto the new offset (sticky CATCHUP). Commit when new audio is about to be
    queued, with base at the sample position where that audio will start
    (played + pending), so residual old PCM keeps audio_ms≈0 under the new meta.

    Phase29hf: refuse commit on transient false-zero clock reads (retry next enqueue).
    """
    pending = turn_state.get("pending_virtualcam_sync_meta")
    if not isinstance(pending, dict) or bool(pending.get("committed")):
        return False

    meta_file = turn_state.get("virtualcam_sync_meta_file")
    played, pending_samples, _sr, clock_source = _pick_playback_clock_for_commit(
        turn_state
    )
    guard = _playback_clock_guard(turn_state)
    last_played = int(guard.get("played_samples", 0) or 0)
    last_pending = int(guard.get("pending_samples", 0) or 0)
    last_sum = max(0, last_played + last_pending)
    turn_no = pending.get("turn_no")
    if clock_source == "false_zero" or (
        played <= 0
        and pending_samples <= 0
        and (last_sum > 0 or (turn_no is not None and int(turn_no) > 1))
    ):
        defer_n = int(pending.get("false_zero_defer_n", 0) or 0) + 1
        pending["false_zero_defer_n"] = defer_n
        print(
            "[sync][virtualcam_meta][COMMIT_DEFER_FALSE_ZERO]",
            f"frame_offset={int(pending.get('frame_offset', 0) or 0)}",
            f"turn_no={turn_no}",
            f"defer_n={defer_n}",
            f"last_played={last_played}",
            f"last_pending={last_pending}",
            f"clock_source={clock_source}",
            flush=True,
        )
        return False

    base = max(0, int(played) + int(pending_samples))
    # Never regress below last-good total when a partial race under-reports.
    if last_sum > 0 and base < last_sum:
        print(
            "[sync][virtualcam_meta][COMMIT_FLOOR_LAST_GOOD]",
            f"raw_base={base}",
            f"last_sum={last_sum}",
            f"played={played}",
            f"pending_samples={pending_samples}",
            f"turn_no={turn_no}",
            flush=True,
        )
        base = int(last_sum)
        played = int(last_played)
        pending_samples = int(last_pending)

    _write_virtualcam_sync_meta(
        Path(meta_file) if meta_file is not None else None,
        frame_offset=int(pending.get("frame_offset", 0) or 0),
        step_ms=int(pending.get("step_ms", 40) or 40),
        base_played_samples=int(base),
        playback_origin_ms=0,
        turn_no=(
            int(pending["turn_no"]) if pending.get("turn_no") is not None else None
        ),
    )
    playback_ref = turn_state.get("audio_playback_state_ref")
    if isinstance(playback_ref, dict):
        with playback_ref["lock"]:
            playback_ref["response_playback_base_samples"] = int(base)
            playback_ref["playback_origin_ms"] = 0
            # Keep ref clock coherent for subsequent commits / fallbacks.
            if int(played) > int(playback_ref.get("played_samples", 0) or 0):
                playback_ref["played_samples"] = int(played)
    if base > 0:
        guard["played_samples"] = max(int(guard.get("played_samples", 0) or 0), int(played))
        guard["pending_samples"] = max(0, int(pending_samples))
    pending["committed"] = True
    print(
        "[sync][virtualcam_meta][COMMIT_ON_FIRST_ENQUEUE]",
        f"frame_offset={int(pending.get('frame_offset', 0) or 0)}",
        f"base_played_samples={int(base)}",
        f"played_samples={int(played)}",
        f"pending_samples={int(pending_samples)}",
        f"turn_no={pending.get('turn_no')}",
        f"clock_source={clock_source}",
        flush=True,
    )
    return True


def _write_virtualcam_sync_meta(
    path: Path | None,
    *,
    frame_offset: int,
    step_ms: int,
    base_played_samples: int,
    playback_origin_ms: int = 0,
    turn_no: int | None = None,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "virtualcam_sync_meta",
        "frame_offset": int(frame_offset),
        "step_ms": int(step_ms),
        "base_played_samples": int(base_played_samples),
        "playback_origin_ms": int(playback_origin_ms),
        "turn_no": int(turn_no) if turn_no is not None else None,
        "updated_mono_s": float(time.monotonic()),
    }
    text = json.dumps(payload, ensure_ascii=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    replaced = False
    for _ in range(8):
        try:
            tmp.replace(path)
            replaced = True
            break
        except PermissionError:
            # Windows: reader (virtualcam) may briefly lock the target.
            time.sleep(0.01)
    if not replaced:
        path.write_text(text, encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    print(
        "[sync][virtualcam_meta]",
        f"frame_offset={int(frame_offset)}",
        f"step_ms={int(step_ms)}",
        f"base_played_samples={int(base_played_samples)}",
        f"playback_origin_ms={int(playback_origin_ms)}",
        f"turn_no={turn_no}",
        flush=True,
    )


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
    playback_state_file: Path | None = None,
    sync_meta_file: Path | None = None,
    step_ms: int = 40,
    frame_offset: int = 0,
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
        "--step_ms",
        str(int(step_ms)),
        "--frame_offset",
        str(int(frame_offset)),
    ]

    if bg_override_file is not None:
        cmd.extend(
            [
                "--bg_override_file",
                str(bg_override_file),
            ]
        )
    if playback_state_file is not None:
        cmd.extend(
            [
                "--playback_state_file",
                str(Path(playback_state_file).resolve()),
            ]
        )
    if sync_meta_file is not None:
        cmd.extend(
            [
                "--sync_meta_file",
                str(Path(sync_meta_file).resolve()),
            ]
        )

    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
    )


def _clamp_m0_worker_n(n: int) -> int:
    """Phase 13: default 2; hard cap 4 (cores-2 on 6-core). Never 6-all."""
    try:
        v = int(n)
    except Exception:
        v = 2
    if v < 1:
        v = 1
    if v > 4:
        print(
            f"[m0_pool][clamp] requested_n={int(n)} -> 4 (max; leave headroom for OBS/player/Live/OS)",
            flush=True,
        )
        v = 4
    return int(v)


def _m0_worker_ports_for_n(*, base_port: int, n: int) -> list[int]:
    n = _clamp_m0_worker_n(n)
    base = int(base_port)
    return [base + i for i in range(int(n))]


def _start_m0_worker_tcp(
    *,
    py: Path,
    m0_repo: Path,
    host: str,
    port: int,
    env: dict[str, str],
    parent_pid: int | None = None,
) -> subprocess.Popen:
    worker_script = m0_repo / "src" / "m0_persistent_worker.py"
    if not worker_script.exists():
        raise FileNotFoundError(f"missing m0 worker: {worker_script}")

    cmd = [
        str(py),
        str(worker_script),
        "--tcp",
        "--host",
        str(host),
        "--port",
        str(int(port)),
    ]
    if parent_pid is not None and int(parent_pid) > 0:
        cmd.extend(["--parent_pid", str(int(parent_pid))])

    return subprocess.Popen(
        cmd,
        cwd=str(m0_repo),
        env=env,
    )


def _start_m0_worker_pool_tcp(
    *,
    py: Path,
    m0_repo: Path,
    host: str,
    ports: list[int],
    env: dict[str, str],
    parent_pid: int | None = None,
) -> list[subprocess.Popen]:
    procs: list[subprocess.Popen] = []
    for port in ports:
        proc = _start_m0_worker_tcp(
            py=py,
            m0_repo=m0_repo,
            host=str(host),
            port=int(port),
            env=env,
            parent_pid=parent_pid,
        )
        procs.append(proc)
        print(
            f"[m0_pool][spawn] port={int(port)} pid={proc.pid} parent_pid={parent_pid}",
            flush=True,
        )
    return procs


def _stop_m0_worker_tcp(proc: subprocess.Popen | None, host: str, port: int) -> None:
    if proc is None or proc.poll() is not None:
        return

    try:
        with socket.create_connection((host, int(port)), timeout=3.0) as sock:
            sock.sendall((json.dumps({"cmd": "quit"}) + "\n").encode("utf-8"))
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def _stop_m0_worker_pool_tcp(
    procs: list[subprocess.Popen] | None,
    host: str,
    ports: list[int],
) -> None:
    if not procs:
        return
    for proc, port in zip(procs, ports):
        _stop_m0_worker_tcp(proc, str(host), int(port))


def _rss_mb_for_pids(pids: list[int]) -> dict[str, float]:
    """Best-effort RSS (MB) for worker PIDs; used in Phase 13 Before/After.

    On Windows, `.venv\\Scripts\\python.exe` is often a launcher: the real
    worker RSS lives in a child process. Include children so M0 pool totals
    are not stuck near ~4MB.
    """
    out: dict[str, float] = {"sum_mb": 0.0, "n": 0.0}
    if not pids:
        return out
    try:
        import psutil  # type: ignore

        total = 0.0
        alive = 0
        seen: set[int] = set()
        for pid in pids:
            try:
                p = psutil.Process(int(pid))
            except Exception:
                continue
            procs = [p]
            try:
                procs.extend(p.children(recursive=True))
            except Exception:
                pass
            for proc in procs:
                try:
                    cpid = int(proc.pid)
                    if cpid in seen:
                        continue
                    seen.add(cpid)
                    rss = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                    total += rss
                    alive += 1
                    out[f"pid_{cpid}_mb"] = rss
                except Exception:
                    continue
        out["sum_mb"] = total
        out["n"] = float(alive)
        return out
    except Exception:
        pass
    # Windows fallback without psutil.
    try:
        import ctypes
        from ctypes import wintypes

        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
        OpenProcess = kernel32.OpenProcess
        CloseHandle = kernel32.CloseHandle
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010

        total = 0.0
        alive = 0
        for pid in pids:
            h = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, int(pid))
            if not h:
                continue
            try:
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                if GetProcessMemoryInfo(h, ctypes.byref(counters), counters.cb):
                    rss = float(counters.WorkingSetSize) / (1024.0 * 1024.0)
                    total += rss
                    alive += 1
                    out[f"pid_{int(pid)}_mb"] = rss
            finally:
                CloseHandle(h)
        out["sum_mb"] = total
        out["n"] = float(alive)
    except Exception:
        pass
    return out


async def _phase24_rss_tick_loop(
    *,
    stop_event: asyncio.Event,
    m0_pids: list[int],
    turn_state: dict[str, Any],
    csv_path: Path | None,
    interval_s: float = 2.0,
) -> None:
    """Phase24: parent + M0 RSS timeseries (and light qsize / streamer len)."""
    parent_pid = int(os.getpid())
    t0 = time.perf_counter()
    write_header = bool(csv_path is not None and not csv_path.exists())
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    while not stop_event.is_set():
        try:
            parent = _rss_mb_for_pids([parent_pid])
            m0 = _rss_mb_for_pids([int(p) for p in m0_pids if int(p) > 0])
            parent_mb = float(parent.get(f"pid_{parent_pid}_mb", 0.0) or 0.0)
            m0_sum = float(m0.get("sum_mb", 0.0) or 0.0)
            streamer = turn_state.get("mouth_streamer")
            frames_n = 0
            try:
                frames_n = len(getattr(streamer, "_frames", []) or [])
            except Exception:
                frames_n = 0
            inflight = int(turn_state.get("pipeline_inflight", 0) or 0)
            eq = turn_state.get("pipeline_enqueue_queue_ref")
            try:
                qsize = int(eq.qsize()) if eq is not None else -1
            except Exception:
                qsize = -1
            elapsed = time.perf_counter() - t0
            print(
                "[phase24][rss]",
                f"elapsed_s={elapsed:.1f}",
                f"parent_mb={parent_mb:.1f}",
                f"m0_sum_mb={m0_sum:.1f}",
                f"m0_detail={ {k: round(v, 1) for k, v in m0.items() if k.startswith('pid_')} }",
                f"streamer_frames={frames_n}",
                f"pipeline_inflight={inflight}",
                f"enqueue_qsize={qsize}",
                flush=True,
            )
            if csv_path is not None:
                import csv as _csv

                with csv_path.open("a", encoding="utf-8", newline="") as f:
                    w = _csv.DictWriter(
                        f,
                        fieldnames=[
                            "ts",
                            "elapsed_s",
                            "pid",
                            "name",
                            "rss_mb",
                            "pipeline_inflight",
                            "enqueue_qsize",
                            "worker_qsize",
                            "streamer_frames",
                        ],
                    )
                    if write_header:
                        w.writeheader()
                        write_header = False
                    now = time.strftime("%Y-%m-%dT%H:%M:%S")
                    w.writerow(
                        {
                            "ts": now,
                            "elapsed_s": f"{elapsed:.1f}",
                            "pid": str(parent_pid),
                            "name": "session_loop",
                            "rss_mb": f"{parent_mb:.1f}",
                            "pipeline_inflight": str(inflight),
                            "enqueue_qsize": str(qsize),
                            "worker_qsize": "",
                            "streamer_frames": str(frames_n),
                        }
                    )
                    for k, v in m0.items():
                        if not k.startswith("pid_"):
                            continue
                        pid_s = k[len("pid_") : -len("_mb")]
                        w.writerow(
                            {
                                "ts": now,
                                "elapsed_s": f"{elapsed:.1f}",
                                "pid": pid_s,
                                "name": "m0_worker",
                                "rss_mb": f"{float(v):.1f}",
                                "pipeline_inflight": str(inflight),
                                "enqueue_qsize": str(qsize),
                                "worker_qsize": "",
                                "streamer_frames": str(frames_n),
                            }
                        )
        except Exception as e:
            print(
                f"[phase24][rss][WARN] {type(e).__name__}: {e}",
                flush=True,
            )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=float(interval_s))
        except asyncio.TimeoutError:
            pass


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

    # Phase R2: mic_vad_silence_ms resolve (CLI > file > 350) + runtime file override.
    if args.vad_profile_file == "":
        vad_profile_path: Path | None = None
    elif args.vad_profile_file is None:
        vad_profile_path = m1_repo / "in" / "vad_profile_live.txt"
    else:
        vad_profile_path = Path(args.vad_profile_file).resolve()

    cli_silence_raw = getattr(args, "mic_vad_silence_ms", None)
    resolved_silence_ms, silence_source = _resolve_mic_vad_silence_ms(
        cli_value=(int(cli_silence_raw) if cli_silence_raw is not None else None),
        profile_path=vad_profile_path,
    )
    args.mic_vad_silence_ms = int(resolved_silence_ms)
    mic_vad_silence_ms_ref: dict[str, int] = {"value": int(resolved_silence_ms)}
    print(
        "[vad_profile][init]",
        f"silence_ms={int(resolved_silence_ms)}",
        f"source={silence_source}",
        f"path={vad_profile_path}",
        f"allowed={sorted(_VAD_PROFILE_ALLOWED_SILENCE_MS)}",
        flush=True,
    )
    if vad_profile_path is not None:
        vad_profile_path.parent.mkdir(parents=True, exist_ok=True)
        if int(resolved_silence_ms) in _VAD_PROFILE_ALLOWED_SILENCE_MS:
            # Persist starting profile (do not persist non-allowed CLI values).
            vad_profile_path.write_text(
                f"{int(resolved_silence_ms)}\n",
                encoding="utf-8",
            )
        elif not vad_profile_path.exists():
            vad_profile_path.touch()

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

    # Phase 5: player clock + turn timeline meta for VirtualCam audio_ms SSOT.
    sync_dir = out_root / "sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    playback_state_file = sync_dir / "playback_state.json"
    virtualcam_sync_meta_file = sync_dir / "virtualcam_sync_meta.json"
    _write_virtualcam_sync_meta(
        virtualcam_sync_meta_file,
        frame_offset=0,
        step_ms=int(args.step_ms),
        base_played_samples=0,
        playback_origin_ms=0,
        turn_no=0,
    )

    knn_script = m3_repo / "tools" / "knn_from_formant_raw_to_mouth_timeline.py"

    base_cfg_path = (
        Path(args.m0_base_config).resolve()
        if args.m0_base_config
        else m0_repo / "configs" / "smoke_pose_improved.yaml"
    )
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    env = os.environ.copy()

    cam_proc = None
    m0_procs: list[subprocess.Popen] = []
    m0_worker_ports: list[int] = []
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
            playback_state_file=playback_state_file,
            sync_meta_file=virtualcam_sync_meta_file,
            step_ms=int(args.step_ms),
            frame_offset=0,
        )

        m0_n = _clamp_m0_worker_n(int(getattr(args, "m0_worker_n", 2)))
        m0_worker_ports = _m0_worker_ports_for_n(
            base_port=int(args.m0_worker_port),
            n=int(m0_n),
        )
        print(
            "[session_loop] start m0 tcp worker pool",
            f"n={int(m0_n)}",
            f"ports={m0_worker_ports}",
            f"host={args.m0_worker_host}",
            flush=True,
        )
        m0_procs = _start_m0_worker_pool_tcp(
            py=py,
            m0_repo=m0_repo,
            host=str(args.m0_worker_host),
            ports=m0_worker_ports,
            env=env,
            parent_pid=int(os.getpid()),
        )
        time.sleep(1.0)
        rss0 = _rss_mb_for_pids([int(p.pid) for p in m0_procs if p.pid])
        print(
            "[m0_pool][rss_after_spawn]",
            f"n={len(m0_procs)}",
            f"sum_mb={float(rss0.get('sum_mb', 0.0)):.1f}",
            f"detail={ {k: round(v, 1) for k, v in rss0.items() if k.startswith('pid_')} }",
            flush=True,
        )

        print("[session_loop] start audio player", flush=True)
        print(
            "[audio_player][config]",
            f"initial_buffer_ms={int(args.audio_player_initial_buffer_ms)}",
            f"start_fallback_ms={int(args.audio_player_start_fallback_ms)}",
            f"rebuffer_target_ms={int(args.audio_player_rebuffer_target_ms)}",
            f"min_start_pcm_ms={int(args.audio_player_min_start_pcm_ms)}",
            f"m0_hang_timeout_ms={int(_M0_PIPELINE_HANG_TIMEOUT_MS)}",
            "(jitter≠m0_hang; min_start≠hang)",
            flush=True,
        )
        audio_player_proc = _start_audio_player(
            py=py,
            audio_script=audio_script,
            audio_device=str(args.ai_audio_output_device),
            cwd=m1_repo,
            env=env,
            initial_buffer_ms=int(args.audio_player_initial_buffer_ms),
            start_fallback_ms=int(args.audio_player_start_fallback_ms),
            rebuffer_target_ms=int(args.audio_player_rebuffer_target_ms),
            min_start_pcm_ms=int(args.audio_player_min_start_pcm_ms),
            playback_state_file=playback_state_file,
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
        vad_profile_file_thread: Thread | None = None
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
                    m0_worker_port=int(m0_worker_ports[0]) if m0_worker_ports else int(args.m0_worker_port),
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
        # Phase29hf: session-scoped last-good player clock (reattached after clear).
        playback_clock_guard: dict[str, int] = {
            "played_samples": 0,
            "pending_samples": 0,
        }
        turn_state["playback_clock_guard"] = playback_clock_guard
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
                    mic_vad_silence_ms=int(mic_vad_silence_ms_ref["value"]),
                    mic_vad_silence_ms_ref=mic_vad_silence_ms_ref,
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
            turn_state["playback_clock_guard"] = playback_clock_guard
            turn_state["mouth_streamer"] = mouth_streamer
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
            turn_state["generation_complete_seen"] = False
            turn_state["audio_player_proc"] = audio_player_proc
            turn_state["ai_audio_output_device"] = str(args.ai_audio_output_device)
            turn_state["audio_playback_state_ref"] = _make_audio_playback_state_ref()
            turn_state["playback_state_file"] = playback_state_file
            turn_state["virtualcam_sync_meta_file"] = virtualcam_sync_meta_file
            turn_state["fast_inmemory"] = bool(getattr(args, "fast_inmemory", False))
            turn_state["skip_archive_pcm"] = bool(
                getattr(args, "fast_inmemory", False)
                and getattr(args, "skip_archive_pcm", False)
            )

            # Phase29: defer VirtualCam sync_meta (frame_offset/base) until the
            # first player enqueue. Early write remaps leftover previous-turn PCM
            # onto the new offset → sticky CATCHUP (see Phase28 root cause).
            turn_state["pending_virtualcam_sync_meta"] = {
                "frame_offset": int(next_frame_offset),
                "step_ms": int(args.step_ms),
                "turn_no": int(turn_no),
                "committed": False,
            }
            playback_ref_init = turn_state["audio_playback_state_ref"]
            with playback_ref_init["lock"]:
                playback_ref_init["response_playback_base_samples"] = 0
                playback_ref_init["playback_origin_ms"] = 0
                # Phase29hf: seed ref from last-good so commit fallback is not false-zero.
                seed_played = int(playback_clock_guard.get("played_samples", 0) or 0)
                seed_pending = int(playback_clock_guard.get("pending_samples", 0) or 0)
                if seed_played > 0:
                    playback_ref_init["played_samples"] = int(seed_played)
                if seed_pending > 0:
                    sr0 = int(playback_ref_init.get("sample_rate", 24000) or 24000)
                    if sr0 <= 0:
                        sr0 = 24000
                    playback_ref_init["pending_ms"] = (
                        float(seed_pending) * 1000.0 / float(sr0)
                    )
            print(
                "[sync][virtualcam_meta][DEFER_UNTIL_ENQUEUE]",
                f"frame_offset={int(next_frame_offset)}",
                f"step_ms={int(args.step_ms)}",
                f"turn_no={int(turn_no)}",
                f"seed_played={int(playback_clock_guard.get('played_samples', 0) or 0)}",
                flush=True,
            )

            mouth_obj_ref: dict[str, Any] = {"obj": None, "knn_raw_frames_done": 0}
            pipeline_sync = bool(getattr(args, "pipeline_sync", True))
            knn_inmemory = bool(getattr(args, "knn_inmemory", False))
            m0_inmemory = bool(getattr(args, "m0_inmemory", False))
            fast_inmemory = bool(getattr(args, "fast_inmemory", False))
            knn_incremental = bool(getattr(args, "knn_incremental", True))
            skip_archive_pcm = bool(turn_state.get("skip_archive_pcm", False))
            turn_state["knn_incremental"] = bool(knn_incremental)

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
            rss_tick_task: asyncio.Task | None = None
            idle_silent_stop: asyncio.Event | None = None
            ai_audio_started_event: asyncio.Event | None = None
            idle_silent_task: asyncio.Task | None = None

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

            m0_pipeline_ref = None
            if pipeline_sync:
                # Phase 13: turn-boundary Worker flush/reset (local VAD turn).
                _m0_pool_reset_tcp(
                    host=str(args.m0_worker_host),
                    ports=list(m0_worker_ports),
                )
                m0_pipeline_ref = _create_m0_pipeline_ref(
                    py=py,
                    m0_repo=m0_repo,
                    m1_repo=m1_repo,
                    m3_repo=m3_repo,
                    base_cfg=base_cfg,
                    pose_json=Path(args.pose_json).resolve(),
                    session_id=str(args.session_id),
                    work_dir=m0_stream_dir_current,
                    watch_fg_dir=watch_fg_dir,
                    env=env,
                    frame_offset=int(frame_offset_current),
                    step_ms=int(args.step_ms),
                    chunk_len_ms=int(args.stream_mouth_m0_chunk_len_ms),
                    fps=int(args.fps),
                    m0_worker_proc=None,
                    m0_worker_host=str(args.m0_worker_host),
                    m0_worker_port=int(m0_worker_ports[0]) if m0_worker_ports else int(args.m0_worker_port),
                    m0_worker_ports=list(m0_worker_ports),
                    inline_emo_id=(
                        str(inline_emo_id_current)
                        if bool(args.inline_emo_tag_mode)
                        else None
                    ),
                    close_mouth_id=int(args.mouth_close_id),
                )
                turn_state["m0_pipeline_ref"] = m0_pipeline_ref
                print(
                    "[m0_pool][turn_ready]",
                    f"turn={int(turn_no)}",
                    f"n={int(m0_pipeline_ref.get('m0_worker_n', 1))}",
                    f"ports={m0_pipeline_ref.get('m0_worker_ports')}",
                    flush=True,
                )
            else:
                turn_state["m0_pipeline_ref"] = None

            m0_thread = None
            if not pipeline_sync:
                m0_thread = Thread(target=_m0_target, daemon=True)
                m0_thread.start()

            audio_thread = None
            if not pipeline_sync:
                audio_thread = _watch_stream_pcm_chunks(
                    audio_player_proc=audio_player_proc,
                    pcm_stream_chunks_dir=pcm_stream_chunks_dir,
                    audio_device=str(args.ai_audio_output_device),
                    stop_event=turn_audio_stop_event,
                )

            if bool(getattr(args, "idle_silent_pcm_enabled", False)) and bool(
                pipeline_sync
            ):
                ai_audio_started_event = asyncio.Event()
                idle_silent_stop = asyncio.Event()
                turn_state["ai_audio_started_event"] = ai_audio_started_event
                turn_state["idle_silent_stop_event"] = idle_silent_stop
            else:
                turn_state.pop("ai_audio_started_event", None)
                turn_state.pop("idle_silent_stop_event", None)

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
                    pipeline_sync=bool(pipeline_sync),
                    mouth_obj_ref=mouth_obj_ref,
                    knn_inmemory=bool(knn_inmemory),
                    m0_inmemory=bool(m0_inmemory),
                    fast_inmemory=bool(fast_inmemory),
                    skip_archive_pcm=bool(skip_archive_pcm),
                    ai_audio_started_event=ai_audio_started_event,
                )
            )

            try:
                rss_csv = (
                    Path(__file__).resolve().parents[2]
                    / "logs"
                    / f"{args.session_id}_rss.csv"
                )
                rss_tick_task = asyncio.create_task(
                    _phase24_rss_tick_loop(
                        stop_event=recv_stop,
                        m0_pids=[int(p.pid) for p in m0_procs if p.pid],
                        turn_state=turn_state,
                        csv_path=rss_csv,
                        interval_s=2.0,
                    ),
                    name=f"phase24_rss_turn_{turn_no}",
                )
            except Exception as e:
                print(
                    f"[phase24][rss][WARN] tick start failed: {type(e).__name__}: {e}",
                    flush=True,
                )

            if idle_silent_stop is not None and ai_audio_started_event is not None:
                idle_silent_task = asyncio.create_task(
                    _idle_silent_pcm_loop(
                        stop_event=idle_silent_stop,
                        ai_audio_started_event=ai_audio_started_event,
                        turn_state=turn_state,
                        pcm_stream_chunks_dir=pcm_stream_chunks_dir,
                        audio_response_pcm=audio_response_pcm,
                        mouth_streamer=mouth_streamer,
                        mouth_streamer_json=mouth_streamer_json,
                        mouth_raw_json=mouth_raw_json,
                        mouth_json=mouth_json,
                        knn_script=knn_script,
                        gt_glob=str(m3_repo / "data" / "knn_db" / "*.f1f2.json"),
                        step_ms=int(args.step_ms),
                        mouth_obj_ref=mouth_obj_ref,
                        knn_inmemory=bool(knn_inmemory),
                        m0_inmemory=bool(m0_inmemory),
                        fast_inmemory=bool(fast_inmemory),
                        interval_ms=int(args.idle_silent_pcm_interval_ms),
                        pending_target_ms=int(args.idle_silent_pcm_pending_target_ms),
                    ),
                    name=f"idle_silent_pcm_turn_{turn_no}",
                )
                print(
                    "[idle_silent_pcm][ready]",
                    f"turn={turn_no}",
                    f"interval_ms={int(args.idle_silent_pcm_interval_ms)}",
                    f"pending_target_ms={int(args.idle_silent_pcm_pending_target_ms)}",
                    flush=True,
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
                mic_vad_silence_ms=int(mic_vad_silence_ms_ref["value"]),
                mic_vad_silence_ms_ref=mic_vad_silence_ms_ref,
                mic_vad_min_listen_ms=int(args.mic_vad_min_listen_ms),
                mic_vad_debug=bool(args.mic_vad_debug),
                send_activity_signals=True,
            )

            if (
                bool(args.battle_talkover_cut_in_on_interrupt)
                and battle_talkover_cut_in_event.is_set()
            ):
                # activity_end は _send_mic_once 内で送信済み（audio_stream_end は使わない）
                # interrupt clear は _battle_interrupt_send_loop 側で実施済み
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

            control_text_to_send = ""
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
                    control_text_to_send = latest_text

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

            interrupt_text_to_send = ""
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
                            "[battle_interrupt][file_apply_activity_path]",
                            f"turn={turn_no}",
                            f"priority={latest_priority}",
                            f"age_sec={age_sec:.3f}",
                            f"expire_sec={expire_sec:.3f}",
                            f"latest={latest}",
                            flush=True,
                        )
                        interrupt_text_to_send = latest

                else:
                    latest = str(latest_control).strip()
                    latest_priority = "normal"

                    print(
                        "[battle_interrupt][file_apply_activity_path]",
                        f"turn={turn_no}",
                        f"priority={latest_priority}",
                        f"latest={latest}",
                        flush=True,
                    )
                    interrupt_text_to_send = latest

            if bool(args.skip_response_trigger):
                print(
                    f"[session_loop][response_trigger][SKIP] turn={turn_no} "
                    f"(client VAD / activity_end path)",
                    flush=True,
                )
                # Phase 4: battle は方式2整合（activity_end 後の直接 text）。
                # 通常ターンの response_trigger は復活させない。
                # active_control の継続保持は emo 用。text 再送は新規 activate 時のみ。
                # arbitration: interrupt > control（同時時は interrupt のみ送る）
                if active_control:
                    print(
                        "[battle_control][emo_active]",
                        f"turn={turn_no}",
                        f"active={active_control}",
                        flush=True,
                    )
                if interrupt_text_to_send:
                    if bool(args.battle_interrupt_file_immediate_send):
                        # immediate_send 済み。二重送信しない（pending は consume 用に残す）
                        print(
                            "[battle_interrupt][activity_path_skip_already_immediate]",
                            f"turn={turn_no}",
                            f"text={interrupt_text_to_send}",
                            flush=True,
                        )
                    else:
                        prompt = _build_battle_interrupt_prompt(interrupt_text_to_send)
                        print(
                            "[battle_interrupt][activity_path_sent]",
                            f"turn={turn_no}",
                            f"text={interrupt_text_to_send}",
                            flush=True,
                        )
                        await session.send_realtime_input(text=prompt)
                elif control_text_to_send:
                    print(
                        "[battle_control][activity_path_sent]",
                        f"turn={turn_no}",
                        f"text={control_text_to_send}",
                        flush=True,
                    )
                    await session.send_realtime_input(
                        text=f"【管理者制御】{control_text_to_send}"
                    )
            else:
                # debug only: legacy response_trigger 合成（通常運用では skip=True）
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
                if interrupt_text_to_send:
                    response_trigger = (
                        f"{response_trigger}\n"
                        f"【管理者割り込み予約】{interrupt_text_to_send}"
                    )
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

            if idle_silent_stop is not None:
                idle_silent_stop.set()
            if idle_silent_task is not None:
                await _cancel_tasks_safely(
                    [idle_silent_task],
                    tag=f"idle_silent_pcm_turn_{turn_no}",
                )
                idle_silent_task = None
            turn_state.pop("idle_silent_stop_event", None)
            turn_state.pop("ai_audio_started_event", None)

            recv_stop.set()
            if rss_tick_task is not None:
                await _cancel_tasks_safely(
                    [rss_tick_task],
                    tag=f"phase24_rss_turn_{turn_no}",
                )
                rss_tick_task = None
            await _cancel_tasks_safely(
                [recv_task],
                tag=f"turn_{turn_no}_recv",
            )

            # Phase24: archive final mouth_streamer.json with debug_frames for
            # self-gate / analysis (hot-path flushes omit debug).
            try:
                mouth_streamer.finalize()
                print(
                    "[phase24][streamer_finalize]",
                    f"turn={turn_no}",
                    f"frames={len(getattr(mouth_streamer, '_frames', []) or [])}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[phase24][streamer_finalize][WARN] {type(e).__name__}: {e}",
                    flush=True,
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
            elif pipeline_sync and isinstance(turn_state.get("m0_pipeline_ref"), dict):
                next_frame_offset += int(
                    turn_state["m0_pipeline_ref"].get("total_frames", 0) or 0
                )
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

        if vad_profile_path is not None and vad_profile_file_thread is None:
            loop = asyncio.get_running_loop()
            vad_profile_file_thread = _start_vad_profile_file_thread(
                path=vad_profile_path,
                silence_ms_ref=mic_vad_silence_ms_ref,
                loop=loop,
                poll_s=float(args.vad_profile_file_poll_s),
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
                                        audio_player_proc=audio_player_proc,
                                        turn_state=turn_state,
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

        _stop_m0_worker_pool_tcp(
            m0_procs,
            str(args.m0_worker_host),
            list(m0_worker_ports) if m0_worker_ports else [int(args.m0_worker_port)],
        )

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
        default=None,
        help=(
            "Silence ms after speech before activity_end. "
            "Priority: CLI explicit > --vad_profile_file > 350. "
            "CLI sets initial only; runtime file may override (250/350)."
        ),
    )
    ap.add_argument(
        "--vad_profile_file",
        default=None,
        help=(
            "Path to vad_profile_live.txt (plain 250|350 or JSON). "
            "Default: <m1>/in/vad_profile_live.txt. "
            "Empty string disables file watch/persistence."
        ),
    )
    ap.add_argument(
        "--vad_profile_file_poll_s",
        type=float,
        default=0.05,
        help="Polling interval for --vad_profile_file.",
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

    # --- Phase 12: idle silent PCM (waiting motion via 図A + Sync SSOT) ---
    ap.add_argument(
        "--idle_silent_pcm_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Inject silent PCM through KNN→M0→player while waiting for AI audio "
            "(default True). Keeps audio_ms / BGV / idle mouth advancing. "
            "Use --no-idle_silent_pcm_enabled to disable."
        ),
    )
    ap.add_argument(
        "--idle_silent_pcm_interval_ms",
        type=int,
        default=40,
        help="Silent PCM chunk interval for idle waiting motion (default 40).",
    )
    ap.add_argument(
        "--idle_silent_pcm_pending_target_ms",
        type=int,
        default=200,
        help=(
            "Soft cap for player pending_ms while idle-injecting. "
            "Effective target is max(this, player initial_buffer_ms) so PLAYING can start; "
            "keeps residual silence before real AI audio small."
        ),
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

    # --- Phase 3: player data-amount jitter (≠ M0 hang timeout) ---
    ap.add_argument(
        "--audio_player_initial_buffer_ms",
        type=int,
        default=300,
        help=(
            "Initial jitter: queued audio ms before playout starts "
            "(data-amount trigger; typical 300-500). Distinct from M0 hang."
        ),
    )
    ap.add_argument(
        "--audio_player_start_fallback_ms",
        type=int,
        default=1000,
        help=(
            "Force playout if initial buffer not reached after this many ms "
            "from first enqueued sample (prevents infinite wait)."
        ),
    )
    ap.add_argument(
        "--audio_player_rebuffer_target_ms",
        type=int,
        default=240,
        help="Rebuffer data-amount target after an active-playback underrun.",
    )
    ap.add_argument(
        "--audio_player_min_start_pcm_ms",
        type=int,
        default=20,
        help=(
            "Drop PCM shorter than this from player queue/clock/fallback "
            "(Hotfix for tiny leading slices; default 20ms)."
        ),
    )

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
            "Talkover cut-in: on battle interrupt, stop mic early, send activity_end, "
            "then interrupt-exception clear_queue + cancel in-flight chunk tasks "
            "(not audio_stream_end). Normal turns still never clear before tail drain."
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

    ap.add_argument(
        "--pipeline_sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Phase2 arrival-order parallel pipeline (KNN→M0→enqueue). Default ON.",
    )
    ap.add_argument(
        "--fast_inmemory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Phase5b fast in-memory path (keep branch; default OFF).",
    )
    ap.add_argument(
        "--knn_inmemory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="KNN in-memory branch (default OFF for Phase2 disk path).",
    )
    ap.add_argument(
        "--knn_incremental",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Phase7 delta-only KNN with cached GT DB (default ON).",
    )
    ap.add_argument(
        "--m0_inmemory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="M0 in-memory branch flag (pipeline always uses m0_pipeline_ref).",
    )
    ap.add_argument(
        "--skip_archive_pcm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip PCM archive when combined with --fast_inmemory.",
    )

    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--watch_fg_dir", default=None)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--clean_fg", action="store_true")

    ap.add_argument("--m0_worker_host", default="127.0.0.1")
    ap.add_argument("--m0_worker_port", type=int, default=39390)
    ap.add_argument(
        "--m0_worker_n",
        type=int,
        default=2,
        help=(
            "Phase 13: resident M0 worker pool size (default 2). "
            "Clamped to 1..4 (never 6-all on 6-core). N=1 restores serial."
        ),
    )
    ap.add_argument("--m0_base_config", default=None)

    ap.add_argument("--mouth_window_ms", type=int, default=240)
    ap.add_argument("--mouth_analysis_sr", type=int, default=16000)
    ap.add_argument("--mouth_rms_thr", type=float, default=0.015)
    ap.add_argument("--mouth_vad_energy_thr", type=float, default=0.0004)
    ap.add_argument("--mouth_vad_min_speech_ms", type=int, default=80)
    ap.add_argument("--mouth_vad_min_silence_ms", type=int, default=120)
    ap.add_argument("--mouth_open_id", type=int, default=1)
    ap.add_argument("--mouth_close_id", type=int, default=0)
    ap.add_argument(
        "--mouth_flush_every_frames",
        type=int,
        default=25,
        help="Archive flush cadence for MouthStreamerOC (default 25; was 1).",
    )
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