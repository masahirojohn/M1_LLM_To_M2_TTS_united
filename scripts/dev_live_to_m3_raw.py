#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from m3p.live.mouth_streamer_oc import MouthStreamerOC, MouthOCConfig


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None or v == "" else int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None or v == "" else float(v)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _project_streamer_to_raw(streamer_json_path: Path) -> dict[str, Any]:
    data = json.loads(streamer_json_path.read_text(encoding="utf-8"))
    frames_in = data.get("frames", [])
    frames_out = []
    for fr in frames_in:
        frames_out.append(
            {
                "t_ms": fr.get("t_ms"),
                "vad_active": int(fr.get("vad_active", 0) or 0),
                "f1_hz": fr.get("f1_hz", None),
                "f2_hz": fr.get("f2_hz", None),
                "src": "mouth_streamer_oc_live",
            }
        )
    return {
        "version": "m3p.mouth.timeline.v1",
        "step_ms": int(data.get("step_ms", 40)),
        "frames": frames_out,
        "meta": data.get("meta", {}),
    }


def _build_live_connect_config(system_instruction: str) -> types.LiveConnectConfig:
    set_emotion = types.FunctionDeclaration(
        name="set_emotion",
        description="Set current emotion id for avatar expression control.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "emo_id": types.Schema(type=types.Type.STRING),
            },
            required=["emo_id"],
        ),
    )

    return types.LiveConnectConfig(
        system_instruction=system_instruction,
        tools=[types.Tool(function_declarations=[set_emotion])],
        response_modalities=["AUDIO"],
    )


def _current_audio_ms(total_audio_bytes: int, sample_rate: int) -> int:
    total_samples = total_audio_bytes // 2
    return int(round(total_samples * 1000.0 / int(sample_rate)))


async def _run(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var: {args.api_key_env}")

    out_dir = Path(args.out_dir).resolve()
    _ensure_dir(out_dir)

    session_id = str(args.session_id)
    prompt_text = str(args.prompt)

    streamer_json = out_dir / args.out_streamer_json
    raw_json = out_dir / args.out_raw_json
    audio_meta_json = out_dir / args.out_audio_meta_json
    tool_calls_json = out_dir / args.out_tool_calls_json
    text_chunks_json = out_dir / args.out_text_chunks_json
    events_jsonl = out_dir / args.out_events_jsonl
    profile_json = out_dir / args.out_profile_json

    t0_mono_ms = _now_ms()
    t0_wall = time.time()

    profile: dict[str, Any] = {
        "format": "live_to_m3_raw_profile.v1",
        "session_id": session_id,
        "model": args.model,
        "step_ms": int(args.step_ms),
        "analysis_sr": int(args.analysis_sr),
        "live_input_sr": int(args.live_input_sr),
        "dev_stop_s": float(args.dev_stop_s),
        "target_audio_ms": int(args.target_audio_ms),
        "receive_timeout_s": float(args.receive_timeout_s),
        "t_start_ms": int(t0_mono_ms),
        "prompt": prompt_text,
    }

    tool_calls: list[dict[str, Any]] = []
    text_chunks: list[dict[str, Any]] = []

    total_audio_bytes = 0
    total_audio_chunks = 0
    got_audio = False
    stop_reason: str | None = None

    system_instruction = (
        "あなたは感情豊かな猫キャラです。"
        "返答は必ず音声で行ってください。"
        "感情が変わったら必要に応じて set_emotion(emo_id) を呼んでください。"
        "emo_id は文字列IDです。"
    )
    if args.system_instruction:
        system_instruction = str(args.system_instruction)

    streamer = MouthStreamerOC(
        out_json=str(streamer_json),
        session_id=session_id,
        cfg=MouthOCConfig(
            step_ms=int(args.step_ms),
            window_ms=int(args.window_ms),
            analysis_sr=int(args.analysis_sr),
            input_sr_default=int(args.live_input_sr),
            rms_thr=float(args.rms_thr),
            vad_energy_thr=float(args.vad_energy_thr),
            vad_min_speech_ms=int(args.vad_min_speech_ms),
            vad_min_silence_ms=int(args.vad_min_silence_ms),
            open_id=int(args.open_id),
            close_id=int(args.close_id),
            flush_every_frames=int(args.flush_every_frames),
            max_buffer_s=float(args.max_buffer_s),
            vowel_mode=str(args.vowel_mode),
            formant_window_ms=int(args.formant_window_ms),
            formant_max_hz=int(args.formant_max_hz),
        ),
    )

    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": str(args.api_version)},
    )

    _append_jsonl(
        events_jsonl,
        {
            "t_ms": 0,
            "type": "session_start",
            "session_id": session_id,
            "model": args.model,
            "prompt": prompt_text,
        },
    )

    try:
        config = _build_live_connect_config(system_instruction=system_instruction)

        print(f"[dev_live_to_m3_raw] connect model={args.model} session_id={session_id}")
        print(f"[dev_live_to_m3_raw] out_dir={out_dir}")

        async with client.aio.live.connect(model=args.model, config=config) as session:
            profile["t_connected_ms"] = _now_ms()
            _append_jsonl(
                events_jsonl,
                {
                    "t_ms": profile["t_connected_ms"] - t0_mono_ms,
                    "type": "connected",
                },
            )

            await session.send_client_content(
                turns=[types.Content(parts=[types.Part(text=prompt_text)])],
                turn_complete=True,
            )

            _append_jsonl(
                events_jsonl,
                {
                    "t_ms": _now_ms() - t0_mono_ms,
                    "type": "prompt_sent",
                },
            )

            recv_iter = session.receive().__aiter__()
            recv_task = None

            while True:
                elapsed_s = time.time() - t0_wall
                audio_ms_now = _current_audio_ms(total_audio_bytes, int(args.live_input_sr))

                if got_audio and int(args.target_audio_ms) > 0 and audio_ms_now >= int(args.target_audio_ms):
                    stop_reason = "target_audio_ms"
                    print(
                        f"[dev_live_to_m3_raw] stop by target_audio_ms "
                        f"{audio_ms_now} >= {int(args.target_audio_ms)}"
                    )
                    _append_jsonl(
                        events_jsonl,
                        {
                            "t_ms": _now_ms() - t0_mono_ms,
                            "type": "target_audio_ms_stop",
                            "payload": {
                                "audio_ms": int(audio_ms_now),
                                "target_audio_ms": int(args.target_audio_ms),
                            },
                        },
                    )
                    break

                if got_audio and elapsed_s >= float(args.dev_stop_s):
                    stop_reason = "dev_stop_s"
                    print(
                        f"[dev_live_to_m3_raw] dev stop after {float(args.dev_stop_s):.1f}s "
                        "(timer exit before receive)"
                    )
                    _append_jsonl(
                        events_jsonl,
                        {
                            "t_ms": _now_ms() - t0_mono_ms,
                            "type": "dev_stop",
                            "payload": {"dev_stop_s": float(args.dev_stop_s)},
                        },
                    )
                    break

                if "recv_task" not in locals() or recv_task is None:
                    recv_task = asyncio.create_task(recv_iter.__anext__())

                done, pending = await asyncio.wait(
                   {recv_task},
                    timeout=float(args.receive_timeout_s),
                )

                if not done:
                    continue

                try:
                    msg = recv_task.result()
                    recv_task = None
                except StopAsyncIteration:
                    stop_reason = "stop_async_iteration"
                    break

                now_rel_ms = _now_ms() - t0_mono_ms

                tool_call = _safe_getattr(msg, "tool_call", None)
                if tool_call and _safe_getattr(tool_call, "function_calls", None):
                    for fc in tool_call.function_calls:
                        item = {
                            "t_ms": now_rel_ms,
                            "name": _safe_getattr(fc, "name", None),
                            "args": _safe_getattr(fc, "args", None),
                            "id": _safe_getattr(fc, "id", None),
                        }
                        tool_calls.append(item)
                        _append_jsonl(
                            events_jsonl,
                            {
                                "t_ms": now_rel_ms,
                                "type": "tool_call",
                                "payload": item,
                            },
                        )
                        print(f"[TOOL] {item['name']} args={item['args']}")

                server_content = _safe_getattr(msg, "server_content", None)
                if server_content:
                    model_turn = _safe_getattr(server_content, "model_turn", None)
                    parts = _safe_getattr(model_turn, "parts", None) or []

                    for part in parts:
                        text_val = _safe_getattr(part, "text", None)
                        if isinstance(text_val, str) and text_val != "":
                            text_item = {"t_ms": now_rel_ms, "text": text_val}
                            text_chunks.append(text_item)
                            _append_jsonl(
                                events_jsonl,
                                {
                                    "t_ms": now_rel_ms,
                                    "type": "text_chunk",
                                    "payload": text_item,
                                },
                            )
                            print(f"[TEXT] {text_val}")

                        inline_data = _safe_getattr(part, "inline_data", None)
                        data = _safe_getattr(inline_data, "data", None)

                        if data:
                            b = data
                            if not isinstance(b, (bytes, bytearray)):
                                continue

                            if len(b) % 2 != 0:
                                b = b[: len(b) - 1]

                            if not b:
                                continue

                            got_audio = True
                            if "t_first_audio_ms" not in profile:
                                profile["t_first_audio_ms"] = _now_ms()

                            total_audio_chunks += 1
                            total_audio_bytes += len(b)

                            new_frames = streamer.push_pcm16_mono(
                                bytes(b),
                                input_sr=int(args.live_input_sr),
                            )

                            audio_ms_now = _current_audio_ms(
                                total_audio_bytes,
                                int(args.live_input_sr),
                            )

                            _append_jsonl(
                                events_jsonl,
                                {
                                    "t_ms": now_rel_ms,
                                    "type": "audio_chunk",
                                    "payload": {
                                        "bytes": len(b),
                                        "new_frames": int(new_frames),
                                        "audio_ms": int(audio_ms_now),
                                    },
                                },
                            )

                            if int(new_frames) > 0:
                                print(
                                    f"[AUDIO] bytes={len(b)} new_frames={int(new_frames)} "
                                    f"chunks={total_audio_chunks} audio_ms={audio_ms_now}"
                                )

    except Exception as e:
        stop_reason = stop_reason or "exception"
        warn_item = {
            "t_ms": _now_ms() - t0_mono_ms,
            "type": "warning",
            "payload": {
                "error_type": type(e).__name__,
                "message": str(e),
            },
        }
        _append_jsonl(events_jsonl, warn_item)
        print(f"[dev_live_to_m3_raw][WARN] {type(e).__name__}: {e}")

    finally:
        streamer.finalize()

        raw_obj = _project_streamer_to_raw(streamer_json)
        _write_json(raw_json, raw_obj)

        total_samples = total_audio_bytes // 2
        audio_ms = int(round(total_samples * 1000.0 / int(args.live_input_sr)))

        audio_meta = {
            "schema": "audio_meta.live.v0.1",
            "session_id": session_id,
            "sample_rate": int(args.live_input_sr),
            "channels": 1,
            "sample_width_bytes": 2,
            "audio_bytes": int(total_audio_bytes),
            "audio_samples": int(total_samples),
            "audio_ms": int(audio_ms),
            "step_ms": int(args.step_ms),
            "audio_chunks": int(total_audio_chunks),
            "source": "gemini_live_api_inline_data",
        }

        _write_json(audio_meta_json, audio_meta)
        _write_json(tool_calls_json, tool_calls)
        _write_json(text_chunks_json, text_chunks)

        profile["t_finalize_done_ms"] = _now_ms()
        profile["stop_reason"] = stop_reason
        profile["audio_chunks"] = int(total_audio_chunks)
        profile["audio_bytes"] = int(total_audio_bytes)
        profile["audio_ms"] = int(audio_ms)
        profile["tool_calls_n"] = int(len(tool_calls))
        profile["text_chunks_n"] = int(len(text_chunks))
        profile["streamer_json"] = str(streamer_json)
        profile["raw_json"] = str(raw_json)
        profile["audio_meta_json"] = str(audio_meta_json)
        profile["tool_calls_json"] = str(tool_calls_json)
        profile["text_chunks_json"] = str(text_chunks_json)
        _write_json(profile_json, profile)

        _append_jsonl(
            events_jsonl,
            {
                "t_ms": _now_ms() - t0_mono_ms,
                "type": "finalized",
                "payload": {
                    "audio_ms": int(audio_ms),
                    "audio_chunks": int(total_audio_chunks),
                    "tool_calls_n": int(len(tool_calls)),
                    "text_chunks_n": int(len(text_chunks)),
                    "stop_reason": stop_reason,
                },
            },
        )

        print("[dev_live_to_m3_raw][OK]")
        print(f"  streamer_json   : {streamer_json}")
        print(f"  raw_json        : {raw_json}")
        print(f"  audio_meta_json : {audio_meta_json}")
        print(f"  tool_calls_json : {tool_calls_json}")
        print(f"  text_chunks_json: {text_chunks_json}")
        print(f"  events_jsonl    : {events_jsonl}")
        print(f"  profile_json    : {profile_json}")
        print(f"  stop_reason     : {stop_reason}")
        print(f"  audio_ms        : {audio_ms}")
        print(f"  audio_chunks    : {total_audio_chunks}")

    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Gemini Live API -> M3 MouthStreamerOC -> raw mouth/audio_meta dump"
    )

    ap.add_argument("--session_id", default=_env_str("LIVE_SESSION_ID", "sess_live_dev_001"))
    ap.add_argument("--out_dir", default=_env_str("OUT_DIR", "out/live/sess_live_dev_001"))

    ap.add_argument("--model", default=_env_str("GEMINI_LIVE_MODEL", "gemini-2.0-flash-exp"))
    ap.add_argument("--api_key_env", default="GEMINI_API_KEY")
    ap.add_argument("--api_version", default=_env_str("GEMINI_API_VERSION", "v1alpha"))
    ap.add_argument("--prompt", default=_env_str("PROMPT", "元気よく自己紹介して！短めで。"))
    ap.add_argument("--system_instruction", default="")

    ap.add_argument("--step_ms", type=int, default=_env_int("STEP_MS", 40))
    ap.add_argument("--dev_stop_s", type=float, default=_env_float("DEV_STOP_S", 8.0))
    ap.add_argument("--target_audio_ms", type=int, default=_env_int("TARGET_AUDIO_MS", 0))
    ap.add_argument("--receive_timeout_s", type=float, default=_env_float("RECEIVE_TIMEOUT_S", 0.5))

    ap.add_argument("--live_input_sr", type=int, default=_env_int("LIVE_INPUT_SR", 24000))

    ap.add_argument("--analysis_sr", type=int, default=_env_int("ANALYSIS_SR", 16000))
    ap.add_argument("--window_ms", type=int, default=_env_int("WINDOW_MS", 240))
    ap.add_argument("--rms_thr", type=float, default=_env_float("RMS_THR", 0.015))
    ap.add_argument("--vad_energy_thr", type=float, default=_env_float("VAD_ENERGY_THR", 0.0004))
    ap.add_argument("--vad_min_speech_ms", type=int, default=_env_int("VAD_MIN_SPEECH_MS", 80))
    ap.add_argument("--vad_min_silence_ms", type=int, default=_env_int("VAD_MIN_SILENCE_MS", 120))
    ap.add_argument("--open_id", type=int, default=_env_int("OPEN_ID", 1))
    ap.add_argument("--close_id", type=int, default=_env_int("CLOSE_ID", 0))
    ap.add_argument("--flush_every_frames", type=int, default=_env_int("FLUSH_EVERY_FRAMES", 25))
    ap.add_argument("--max_buffer_s", type=float, default=_env_float("MAX_BUFFER_S", 10.0))
    ap.add_argument("--vowel_mode", choices=["formant", "simple"], default=_env_str("VOWEL_MODE", "formant"))
    ap.add_argument("--formant_window_ms", type=int, default=_env_int("FORMANT_WINDOW_MS", 200))
    ap.add_argument("--formant_max_hz", type=int, default=_env_int("FORMANT_MAX_HZ", 5500))

    ap.add_argument("--out_streamer_json", default="mouth_timeline.live.json")
    ap.add_argument("--out_raw_json", default="mouth_timeline.formant.raw.json")
    ap.add_argument("--out_audio_meta_json", default="audio_meta.json")
    ap.add_argument("--out_tool_calls_json", default="tool_calls.live.json")
    ap.add_argument("--out_text_chunks_json", default="text_chunks.live.json")
    ap.add_argument("--out_events_jsonl", default="events.live.jsonl")
    ap.add_argument("--out_profile_json", default="profile.live.json")

    return ap


def main() -> int:
    args = build_argparser().parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())