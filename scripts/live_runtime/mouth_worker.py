#!/usr/bin/env python3
from __future__ import annotations

"""
Live Runtime mouth_worker v0

目的:
- 既存 dev_live_to_m3_raw.py の成功済み Live API 受信ロジックを壊さず、
  完全リアルタイム用の mouth worker として切り出す。
- Live API audio を MouthStreamerOC に push し、
  40ms mouth frame を ring buffer に保持する。
- v0 は単体スモーク用。M0 / M3.5 にはまだ接続しない。

固定:
- step_ms = 40
- chunk_len_ms = 400 は orchestrator 側
- receive() を直接 await しない
- asyncio.wait + non-cancel方式
"""

import argparse
import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque

from google import genai
from google.genai import types

from m3p.live.mouth_streamer_oc import MouthStreamerOC, MouthOCConfig


# ============================================================
# Helpers
# ============================================================

def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _current_audio_ms(total_audio_bytes: int, sample_rate: int) -> int:
    total_samples = int(total_audio_bytes) // 2
    return int(round(total_samples * 1000.0 / int(sample_rate)))


def _project_streamer_to_raw(streamer_json_path: Path) -> dict[str, Any]:
    data = json.loads(streamer_json_path.read_text(encoding="utf-8"))
    frames_in = data.get("frames", []) or []

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
                "src": "mouth_worker_live",
            }
        )

    return {
        "version": "m3p.mouth.timeline.v1",
        "step_ms": int(data.get("step_ms", 40)),
        "frames": frames_out,
        "meta": data.get("meta", {}),
    }


def _extract_audio_bytes(msg: Any) -> bytes | None:
    """
    dev_live_to_m3_raw.py 互換の安全抽出。
    Gemini Live API SDK のレスポンス形状差分を吸収する。
    """
    server_content = _safe_getattr(msg, "server_content", None)
    model_turn = _safe_getattr(server_content, "model_turn", None)

    parts = _safe_getattr(model_turn, "parts", None)
    if not parts:
        return None

    for part in parts:
        inline_data = _safe_getattr(part, "inline_data", None)
        if inline_data is None:
            continue

        data = _safe_getattr(inline_data, "data", None)
        if data:
            return bytes(data)

    return None


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


# ============================================================
# Ring buffer state
# ============================================================

@dataclass
class MouthRingBuffer:
    max_frames: int = 300
    frames: Deque[dict[str, Any]] = field(default_factory=deque)

    def push_many(self, items: list[dict[str, Any]]) -> None:
        for fr in items:
            self.frames.append(fr)
            while len(self.frames) > self.max_frames:
                self.frames.popleft()

    def latest(self, n: int = 10) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        return list(self.frames)[-n:]

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.frames)


def _read_streamer_frames(streamer_json: Path) -> list[dict[str, Any]]:
    if not streamer_json.exists():
        return []

    try:
        data = json.loads(streamer_json.read_text(encoding="utf-8"))
    except Exception:
        return []

    frames = data.get("frames", []) if isinstance(data, dict) else []
    if not isinstance(frames, list):
        return []

    out: list[dict[str, Any]] = []
    for fr in frames:
        if not isinstance(fr, dict):
            continue
        out.append(
            {
                "t_ms": int(fr.get("t_ms", 0) or 0),
                "vad_active": int(fr.get("vad_active", 0) or 0),
                "f1_hz": fr.get("f1_hz"),
                "f2_hz": fr.get("f2_hz"),
                "mouth_id": fr.get("mouth_id"),
                "src": "mouth_streamer_oc_live",
            }
        )
    return out


# ============================================================
# Worker core
# ============================================================

async def run_mouth_worker(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var: {args.api_key_env}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session_id = str(args.session_id)
    prompt_text = str(args.prompt)

    streamer_json = out_dir / "mouth_worker.streamer.json"
    raw_json = out_dir / "mouth_worker.formant.raw.json"
    ring_json = out_dir / "mouth_worker.ring.json"
    audio_meta_json = out_dir / "audio_meta.json"
    events_jsonl = out_dir / "mouth_worker.events.jsonl"
    profile_json = out_dir / "mouth_worker.profile.json"

    t0_mono_ms = _now_ms()
    t0_wall = time.time()

    ring = MouthRingBuffer(max_frames=int(args.ring_frames))

    profile: dict[str, Any] = {
        "format": "mouth_worker_profile.v0",
        "session_id": session_id,
        "model": args.model,
        "step_ms": int(args.step_ms),
        "analysis_sr": int(args.analysis_sr),
        "live_input_sr": int(args.live_input_sr),
        "target_audio_ms": int(args.target_audio_ms),
        "dev_stop_s": float(args.dev_stop_s),
        "receive_timeout_s": float(args.receive_timeout_s),
        "ring_frames": int(args.ring_frames),
        "t_start_ms": int(t0_mono_ms),
        "prompt": prompt_text,
    }

    system_instruction = (
        "あなたは感情豊かな猫キャラです。"
        "返答は必ず音声で行ってください。"
        "短く自然な日本語で話してください。"
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

    total_audio_bytes = 0
    total_audio_chunks = 0
    got_audio = False
    stop_reason: str | None = None
    last_ring_count = 0

    try:
        config = _build_live_connect_config(system_instruction=system_instruction)

        print(f"[mouth_worker] connect model={args.model} session_id={session_id}")
        print(f"[mouth_worker] out_dir={out_dir}")

        async with client.aio.live.connect(model=args.model, config=config) as session:
            profile["t_connected_ms"] = _now_ms()

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
            recv_task: asyncio.Task[Any] | None = None

            while True:
                elapsed_s = time.time() - t0_wall
                audio_ms_now = _current_audio_ms(
                    total_audio_bytes,
                    int(args.live_input_sr),
                )

                if got_audio and int(args.target_audio_ms) > 0 and audio_ms_now >= int(args.target_audio_ms):
                    stop_reason = "target_audio_ms"
                    print(
                        f"[mouth_worker] stop by target_audio_ms "
                        f"{audio_ms_now} >= {int(args.target_audio_ms)}"
                    )
                    break

                if got_audio and elapsed_s >= float(args.dev_stop_s):
                    stop_reason = "dev_stop_s"
                    print(f"[mouth_worker] dev stop after {float(args.dev_stop_s):.1f}s")
                    break

                if recv_task is None:
                    recv_task = asyncio.create_task(recv_iter.__anext__())

                done, _pending = await asyncio.wait(
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

                audio_bytes = _extract_audio_bytes(msg)
                if audio_bytes:
                    got_audio = True
                    total_audio_chunks += 1
                    total_audio_bytes += len(audio_bytes)

                    streamer.push_pcm16_mono(
                        audio_bytes,
                        input_sr=int(args.live_input_sr),
                    )

                    all_frames = _read_streamer_frames(streamer_json)
                    new_frames = all_frames[last_ring_count:]
                    if new_frames:
                        ring.push_many(new_frames)
                        last_ring_count = len(all_frames)

                    if total_audio_chunks % int(args.log_every_audio_chunks) == 0:
                        print(
                            "[mouth_worker] "
                            f"audio_ms={audio_ms_now} "
                            f"audio_chunks={total_audio_chunks} "
                            f"ring_frames={len(ring.frames)}"
                        )

                    _append_jsonl(
                        events_jsonl,
                        {
                            "t_ms": now_rel_ms,
                            "type": "audio_chunk",
                            "payload": {
                                "bytes": len(audio_bytes),
                                "audio_ms": audio_ms_now,
                                "ring_frames": len(ring.frames),
                            },
                        },
                    )

                tool_call = _safe_getattr(msg, "tool_call", None)
                if tool_call and _safe_getattr(tool_call, "function_calls", None):
                    for fc in tool_call.function_calls:
                        _append_jsonl(
                            events_jsonl,
                            {
                                "t_ms": now_rel_ms,
                                "type": "tool_call_ignored_by_mouth_worker",
                                "payload": {
                                    "name": _safe_getattr(fc, "name", None),
                                    "args": _safe_getattr(fc, "args", None),
                                    "id": _safe_getattr(fc, "id", None),
                                },
                            },
                        )

    finally:
        try:
            streamer.close()
        except Exception:
            pass

    audio_ms_final = _current_audio_ms(total_audio_bytes, int(args.live_input_sr))

    if streamer_json.exists():
        raw_obj = _project_streamer_to_raw(streamer_json)
        _write_json(raw_json, raw_obj)

        all_frames = _read_streamer_frames(streamer_json)
        if len(all_frames) > last_ring_count:
            ring.push_many(all_frames[last_ring_count:])

    _write_json(
        ring_json,
        {
            "schema_version": "mouth_ring_buffer.v0",
            "session_id": session_id,
            "step_ms": int(args.step_ms),
            "frames": ring.to_list(),
            "meta": {
                "ring_frames": int(args.ring_frames),
                "audio_ms": int(audio_ms_final),
                "total_audio_chunks": int(total_audio_chunks),
                "total_audio_bytes": int(total_audio_bytes),
                "stop_reason": stop_reason,
            },
        },
    )

    _write_json(
        audio_meta_json,
        {
            "format": "audio_meta.v1",
            "session_id": session_id,
            "utt_id": session_id,
            "step_ms": int(args.step_ms),
            "sample_rate": int(args.live_input_sr),
            "audio_ms": int(audio_ms_final),
        },
    )

    profile.update(
        {
            "stop_reason": stop_reason,
            "audio_ms": int(audio_ms_final),
            "total_audio_chunks": int(total_audio_chunks),
            "total_audio_bytes": int(total_audio_bytes),
            "ring_frames_final": len(ring.frames),
            "t_finished_ms": _now_ms(),
        }
    )
    _write_json(profile_json, profile)

    print("[mouth_worker][OK]")
    print(f"  audio_ms     : {audio_ms_final}")
    print(f"  ring_frames  : {len(ring.frames)}")
    print(f"  streamer_json: {streamer_json}")
    print(f"  raw_json     : {raw_json}")
    print(f"  ring_json    : {ring_json}")
    print(f"  audio_meta   : {audio_meta_json}")

    return 0


# ============================================================
# CLI
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Live Runtime mouth_worker v0"
    )

    ap.add_argument("--session_id", default="sess_mouth_worker_001")
    ap.add_argument("--out_dir", default="out/live_runtime_realtime/mouth_worker_001")

    ap.add_argument("--api_key_env", default="GEMINI_API_KEY")
    ap.add_argument("--api_version", default="v1beta")
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash-native-audio-preview-12-2025",
    )
    ap.add_argument(
        "--prompt",
        default="元気よく短く自己紹介して！",
    )
    ap.add_argument("--system_instruction", default=None)

    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--target_audio_ms", type=int, default=3000)
    ap.add_argument("--dev_stop_s", type=float, default=20.0)
    ap.add_argument("--receive_timeout_s", type=float, default=0.25)

    ap.add_argument("--live_input_sr", type=int, default=24000)
    ap.add_argument("--analysis_sr", type=int, default=16000)
    ap.add_argument("--window_ms", type=int, default=120)

    ap.add_argument("--rms_thr", type=float, default=0.01)
    ap.add_argument("--vad_energy_thr", type=float, default=0.0005)
    ap.add_argument("--vad_min_speech_ms", type=int, default=40)
    ap.add_argument("--vad_min_silence_ms", type=int, default=120)

    ap.add_argument("--open_id", type=int, default=2)
    ap.add_argument("--close_id", type=int, default=1)
    ap.add_argument("--flush_every_frames", type=int, default=1)
    ap.add_argument("--max_buffer_s", type=float, default=10.0)

    ap.add_argument("--vowel_mode", default="formant")
    ap.add_argument("--formant_window_ms", type=int, default=80)
    ap.add_argument("--formant_max_hz", type=int, default=5500)

    ap.add_argument("--ring_frames", type=int, default=300)
    ap.add_argument("--log_every_audio_chunks", type=int, default=5)

    return ap


def main() -> int:
    args = build_argparser().parse_args()

    if int(args.step_ms) != 40:
        raise ValueError("step_ms must be 40 for current realtime contract")

    return asyncio.run(run_mouth_worker(args))


if __name__ == "__main__":
    raise SystemExit(main())