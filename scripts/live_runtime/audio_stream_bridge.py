#!/usr/bin/env python3
from __future__ import annotations

"""
audio_stream_bridge.py

目的:
- Live API 音声入力・音声対話の最小ブリッジ
- Codespacesでは --input_wav / --input_pcm で疑似マイク入力
- ローカルWindowsでは --mic で sounddevice マイク入力

出力:
- audio_input.pcm
- audio_response.pcm
- live_text_chunks.json
- live_tool_calls.json
- audio_stream_bridge.events.jsonl
- audio_stream_bridge.summary.json
"""

import argparse
import asyncio
import base64
import json
import os
import time
import wave
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


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


# --- 修正点1: 追加 ---
def _is_normal_live_close_error(e: BaseException) -> bool:
    s = str(e)
    return ("1000" in s and "None" in s)


def _ensure_pcm16_mono_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()
        data = wf.readframes(frames)

    if channels != 1:
        raise RuntimeError(f"wav must be mono: channels={channels}")
    if sampwidth != 2:
        raise RuntimeError(f"wav must be PCM16: sampwidth={sampwidth}")

    return data, int(sr)


def _read_pcm(path: Path, sample_rate: int) -> tuple[bytes, int]:
    return path.read_bytes(), int(sample_rate)


def _chunk_bytes(data: bytes, *, sample_rate: int, chunk_ms: int) -> list[bytes]:
    bytes_per_sample = 2
    chunk_size = int(sample_rate * chunk_ms / 1000.0) * bytes_per_sample
    chunk_size = max(2, chunk_size)
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def _decode_audio_bytes(data: bytes) -> bytes:
    """
    SDK/モデル差分対策:
    - raw PCM bytes の場合はそのまま返す
    - base64文字列bytesの場合はdecodeして返す
    """
    if not data:
        return b""

    try:
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


def _extract_text(msg: Any) -> str | None:
    server_content = _safe_getattr(msg, "server_content", None)
    model_turn = _safe_getattr(server_content, "model_turn", None)
    parts = _safe_getattr(model_turn, "parts", None)

    if parts:
        texts: list[str] = []
        for part in parts:
            text = _safe_getattr(part, "text", None)
            if text:
                texts.append(str(text))
        if texts:
            return "".join(texts)

    text = _safe_getattr(msg, "text", None)
    if text:
        return str(text)

    return None


def _extract_tool_calls(msg: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    tool_call = _safe_getattr(msg, "tool_call", None)
    function_calls = _safe_getattr(tool_call, "function_calls", None)

    if function_calls:
        for fc in function_calls:
            name = _safe_getattr(fc, "name", None)
            args = _safe_getattr(fc, "args", None)
            call_id = _safe_getattr(fc, "id", None)
            out.append(
                {
                    "id": call_id,
                    "name": name,
                    "args": dict(args) if isinstance(args, dict) else args,
                }
            )

    return out


def _build_live_config(system_instruction: str) -> types.LiveConnectConfig:
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


async def _send_audio_file(
    *,
    session: Any,
    input_pcm: bytes,
    input_sr: int,
    chunk_ms: int,
    realtime_sleep: bool,
    audio_input_path: Path,
    events_jsonl: Path,
    t0_ms: int,
) -> int:
    audio_input_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = _chunk_bytes(input_pcm, sample_rate=input_sr, chunk_ms=chunk_ms)
    sent_bytes = 0

    with audio_input_path.open("wb") as f:
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue

            await session.send_realtime_input(
                audio=types.Blob(
                    data=chunk,
                    mime_type=f"audio/pcm;rate={int(input_sr)}",
                )
            )

            f.write(chunk)
            sent_bytes += len(chunk)

            _append_jsonl(
                events_jsonl,
                {
                    "t_ms": _now_ms() - t0_ms,
                    "type": "input_audio_chunk_sent",
                    "chunk_index": i,
                    "bytes": len(chunk),
                    "input_sr": int(input_sr),
                },
            )

            if realtime_sleep:
                await asyncio.sleep(chunk_ms / 1000.0)

    return sent_bytes


async def _send_audio_mic(
    *,
    session: Any,
    duration_s: float,
    input_sr: int,
    chunk_ms: int,
    audio_input_path: Path,
    events_jsonl: Path,
    t0_ms: int,
) -> int:
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError("sounddevice is required for --mic. Install: pip install sounddevice") from e

    import numpy as np

    audio_input_path.parent.mkdir(parents=True, exist_ok=True)

    block_samples = int(input_sr * chunk_ms / 1000.0)
    total_blocks = int((duration_s * 1000.0) / chunk_ms)
    sent_bytes = 0

    q: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            pass

        x = indata[:, 0]
        x = np.clip(x, -1.0, 1.0)
        pcm = (x * 32767.0).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(q.put_nowait, pcm)

    with audio_input_path.open("wb") as f:
        with sd.InputStream(
            samplerate=int(input_sr),
            channels=1,
            dtype="float32",
            blocksize=block_samples,
            callback=callback,
        ):
            for i in range(total_blocks):
                chunk = await q.get()

                await session.send_realtime_input(
                    audio=types.Blob(
                        data=chunk,
                        mime_type=f"audio/pcm;rate={int(input_sr)}",
                    )
                )

                f.write(chunk)
                sent_bytes += len(chunk)

                _append_jsonl(
                    events_jsonl,
                    {
                        "t_ms": _now_ms() - t0_ms,
                        "type": "mic_audio_chunk_sent",
                        "chunk_index": i,
                        "bytes": len(chunk),
                        "input_sr": int(input_sr),
                    },
                )

    return sent_bytes


# --- 修正点3: 追加 ---
async def _send_response_trigger(
    *,
    session: Any,
    prompt: str,
    events_jsonl: Path,
    t0_ms: int,
) -> None:
    await session.send_realtime_input(text=prompt)

    _append_jsonl(
        events_jsonl,
        {
            "t_ms": _now_ms() - t0_ms,
            "type": "response_trigger_sent",
            "prompt": prompt,
        },
    )

async def _send_audio_stream_end(
    *,
    session: Any,
    events_jsonl: Path,
    t0_ms: int,
) -> None:
    await session.send_realtime_input(audio_stream_end=True)

    _append_jsonl(
        events_jsonl,
        {
            "t_ms": _now_ms() - t0_ms,
            "type": "audio_stream_end_sent",
        },
    )

async def _receive_loop(
    *,
    session: Any,
    duration_s: float,
    receive_timeout_s: float,
    audio_response_path: Path,
    text_chunks_json: Path,
    tool_calls_json: Path,
    events_jsonl: Path,
    t0_ms: int,
) -> dict[str, Any]:
    audio_response_path.parent.mkdir(parents=True, exist_ok=True)

    text_chunks: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    total_audio_bytes = 0
    total_audio_chunks = 0
    total_text_chunks = 0
    total_tool_calls = 0

    recv_iter = session.receive().__aiter__()
    recv_task: asyncio.Task | None = None
    t_start = time.time()

    with audio_response_path.open("wb") as audio_f:
        while True:
            elapsed = time.time() - t_start
            if elapsed >= duration_s + receive_timeout_s:
                break

            if recv_task is None:
                recv_task = asyncio.create_task(recv_iter.__anext__())

            done, _pending = await asyncio.wait(
                {recv_task},
                timeout=0.2,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                continue

            # --- 修正点2: 修正フル版 ---
            try:
                msg = recv_task.result()
            except StopAsyncIteration:
                break
            except Exception as e:
                if _is_normal_live_close_error(e):
                    _append_jsonl(
                        events_jsonl,
                        {
                            "t_ms": _now_ms() - t0_ms,
                            "type": "normal_live_close",
                            "message": str(e),
                        },
                    )
                    break
                raise
            finally:
                recv_task = None
            # --- 修正点2 ここまで ---

            audio = _extract_audio_bytes(msg)
            if audio:
                audio_f.write(audio)
                total_audio_bytes += len(audio)
                total_audio_chunks += 1
                _append_jsonl(
                    events_jsonl,
                    {
                        "t_ms": _now_ms() - t0_ms,
                        "type": "response_audio_chunk",
                        "bytes": len(audio),
                    },
                )

            text = _extract_text(msg)
            if text:
                ev = {
                    "t_ms": _now_ms() - t0_ms,
                    "text": text,
                }
                text_chunks.append(ev)
                total_text_chunks += 1
                _append_jsonl(
                    events_jsonl,
                    {
                        "t_ms": ev["t_ms"],
                        "type": "text_chunk",
                        "text": text,
                    },
                )

            calls = _extract_tool_calls(msg)
            for call in calls:
                ev = {
                    "t_ms": _now_ms() - t0_ms,
                    **call,
                }
                tool_calls.append(ev)
                total_tool_calls += 1
                _append_jsonl(
                    events_jsonl,
                    {
                        "t_ms": ev["t_ms"],
                        "type": "tool_call",
                        "payload": call,
                    },
                )

    _write_json(text_chunks_json, text_chunks)
    _write_json(tool_calls_json, tool_calls)

    if recv_task is not None and not recv_task.done():
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
        except BaseException as e:
            if not _is_normal_live_close_error(e):
                raise    

    return {
        "response_audio_bytes": total_audio_bytes,
        "response_audio_chunks": total_audio_chunks,
        "text_chunks": total_text_chunks,
        "tool_calls": total_tool_calls,
    }


async def _run(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var: {args.api_key_env}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session_id = str(args.session_id)

    audio_input_path = out_dir / "audio_input.pcm"
    audio_response_path = out_dir / "audio_response.pcm"
    text_chunks_json = out_dir / "live_text_chunks.json"
    tool_calls_json = out_dir / "live_tool_calls.json"
    events_jsonl = out_dir / "audio_stream_bridge.events.jsonl"
    summary_json = out_dir / "audio_stream_bridge.summary.json"

    t0_ms = _now_ms()

    system_instruction = str(args.system_instruction) if args.system_instruction else (
        "あなたは感情豊かな猫キャラです。"
        "ユーザーの音声に短く自然な日本語で返答してください。"
        "必ず音声で返答してください。"
        "発話の直前に必要に応じて set_emotion を呼んでください。"
        "emo_id は 1_0, 1_1, 1_2, 2_0, 9_1, 9_2 のいずれかです。"
        "通常は 1_1 を使ってください。"
        "出力中に説明文・作業メモを書いてはいけません。"
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
            "mode": args.mode,
        },
    )

    input_pcm: bytes | None = None
    input_sr = int(args.input_sr)

    if args.input_wav:
        input_pcm, input_sr = _ensure_pcm16_mono_wav(Path(args.input_wav).resolve())
    elif args.input_pcm:
        input_pcm, input_sr = _read_pcm(Path(args.input_pcm).resolve(), int(args.input_sr))

    config = _build_live_config(system_instruction)

    print(f"[audio_stream_bridge] connect model={args.model}")
    print(f"[audio_stream_bridge] out_dir={out_dir}")
    print(f"[audio_stream_bridge] mode={args.mode}")

    async with client.aio.live.connect(model=args.model, config=config) as session:
        _append_jsonl(
            events_jsonl,
            {
                "t_ms": _now_ms() - t0_ms,
                "type": "connected",
            },
        )

        recv_task = asyncio.create_task(
            _receive_loop(
                session=session,
                duration_s=float(args.duration_s),
                receive_timeout_s=float(args.receive_timeout_s),
                audio_response_path=audio_response_path,
                text_chunks_json=text_chunks_json,
                tool_calls_json=tool_calls_json,
                events_jsonl=events_jsonl,
                t0_ms=t0_ms,
            )
        )

        if args.mode == "mic":
            sent_bytes = await _send_audio_mic(
                session=session,
                duration_s=float(args.duration_s),
                input_sr=input_sr,
                chunk_ms=int(args.chunk_ms),
                audio_input_path=audio_input_path,
                events_jsonl=events_jsonl,
                t0_ms=t0_ms,
            )
        else:
            if input_pcm is None:
                raise RuntimeError("--input_wav or --input_pcm is required unless --mode mic")
            sent_bytes = await _send_audio_file(
                session=session,
                input_pcm=input_pcm,
                input_sr=input_sr,
                chunk_ms=int(args.chunk_ms),
                realtime_sleep=not bool(args.no_realtime_sleep),
                audio_input_path=audio_input_path,
                events_jsonl=events_jsonl,
                t0_ms=t0_ms,
            )

        _append_jsonl(
            events_jsonl,
            {
                "t_ms": _now_ms() - t0_ms,
                "type": "input_audio_done",
                "sent_bytes": sent_bytes,
            },
        )

        await _send_audio_stream_end(
            session=session,
            events_jsonl=events_jsonl,
            t0_ms=t0_ms,
        )

        # --- 修正点4: 追加 ---
        if args.response_trigger:
            await _send_response_trigger(
                session=session,
                prompt=str(args.response_trigger),
                events_jsonl=events_jsonl,
                t0_ms=t0_ms,
            )

        recv_summary = await recv_task

    input_audio_ms = int(round((sent_bytes // 2) * 1000.0 / input_sr))

    summary = {
        "format": "audio_stream_bridge.summary.v0",
        "session_id": session_id,
        "model": args.model,
        "api_version": args.api_version,
        "mode": args.mode,
        "input_sr": input_sr,
        "response_sr": 24000,
        "chunk_ms": int(args.chunk_ms),
        "duration_s": float(args.duration_s),
        "input_audio_bytes": int(sent_bytes),
        "input_audio_ms": int(input_audio_ms),
        **recv_summary,
        "outputs": {
            "audio_input_pcm": str(audio_input_path),
            "audio_response_pcm": str(audio_response_path),
            "live_text_chunks_json": str(text_chunks_json),
            "live_tool_calls_json": str(tool_calls_json),
            "events_jsonl": str(events_jsonl),
        },
    }

    _write_json(summary_json, summary)

    print("[audio_stream_bridge][OK]")
    print(f"  input_audio_ms        : {summary['input_audio_ms']}")
    print(f"  response_audio_bytes : {summary['response_audio_bytes']}")
    print(f"  response_audio_chunks: {summary['response_audio_chunks']}")
    print(f"  text_chunks           : {summary['text_chunks']}")
    print(f"  tool_calls            : {summary['tool_calls']}")
    print(f"  summary_json          : {summary_json}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--session_id", default="sess_audio_bridge_001")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--api_version", default="v1alpha")
    ap.add_argument("--api_key_env", default="GEMINI_API_KEY")

    ap.add_argument("--mode", choices=["file", "mic"], default="file")
    ap.add_argument("--input_wav", default=None)
    ap.add_argument("--input_pcm", default=None)
    ap.add_argument("--input_sr", type=int, default=16000)

    ap.add_argument("--duration_s", type=float, default=10.0)
    ap.add_argument("--receive_timeout_s", type=float, default=5.0)
    ap.add_argument("--chunk_ms", type=int, default=40)
    ap.add_argument("--no_realtime_sleep", action="store_true")

    ap.add_argument("--system_instruction", default=None)
    
    # --- 修正点5: 追加 ---
    ap.add_argument(
        "--response_trigger",
        default="今の音声に短く日本語で返答してください。返答の直前に set_emotion を1回呼んでください。",
    )

    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())