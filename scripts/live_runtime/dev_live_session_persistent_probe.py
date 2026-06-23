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


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


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
                return bytes(data)

    data = _safe_getattr(msg, "data", None)
    if data:
        return bytes(data)

    return None


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


async def _receive_audio_loop(
    *,
    session: Any,
    audio_queue: asyncio.Queue[bytes],
    stop_event: asyncio.Event,
) -> None:
    try:
        while not stop_event.is_set():
            got_any = False

            async for msg in session.receive():
                if stop_event.is_set():
                    break

                got_any = True

                audio = _extract_audio_bytes(msg)
                if audio:
                    await audio_queue.put(audio)

            print("[persistent_probe][receive_loop] receive() ended; restart", flush=True)

            # turn完了で receive() が自然終了した場合、次turn用に再度 receive() へ入る
            await asyncio.sleep(0.05)

            if not got_any:
                await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        pass
    except BaseException as e:
        if not stop_event.is_set():
            print(f"[persistent_probe][receive_loop][WARN] {type(e).__name__}: {e}", flush=True)


async def _send_mic_once(
    *,
    session: Any,
    duration_s: float,
    input_sr: int,
    chunk_ms: int,
) -> int:
    try:
        import sounddevice as sd
        import numpy as np
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


async def _measure_one_turn(
    *,
    session: Any,
    audio_queue: asyncio.Queue[bytes],
    turn_index: int,
    duration_s: float,
    input_sr: int,
    chunk_ms: int,
    response_trigger: str,
    receive_timeout_s: float,
    audio_stream_end_per_turn: bool,
) -> dict[str, Any]:
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    t0 = time.perf_counter()

    sent_bytes = await _send_mic_once(
        session=session,
        duration_s=duration_s,
        input_sr=input_sr,
        chunk_ms=chunk_ms,
    )

    if audio_stream_end_per_turn:
        await session.send_realtime_input(audio_stream_end=True)

    await session.send_realtime_input(text=response_trigger)

    first_audio_sec = None
    audio_chunks = 0
    audio_bytes = 0
    last_audio_perf = None

    while True:
        elapsed = time.perf_counter() - t0
        if elapsed > duration_s + receive_timeout_s:
            break

        try:
            audio = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue

        if audio:
            if first_audio_sec is None:
                first_audio_sec = time.perf_counter() - t0
                print(
                    f"[perf][turn{turn_index}_first_response_audio_chunk_sec] "
                    f"{first_audio_sec:.3f}",
                    flush=True,
                )

            audio_chunks += 1
            audio_bytes += len(audio)
            last_audio_perf = time.perf_counter()

        if last_audio_perf is not None:
            if time.perf_counter() - last_audio_perf > 0.8:
                break

    return {
        "turn": int(turn_index),
        "sent_bytes": int(sent_bytes),
        "input_audio_ms": int(round((sent_bytes // 2) * 1000.0 / input_sr)),
        "first_response_audio_chunk_sec": first_audio_sec,
        "response_audio_chunks": int(audio_chunks),
        "response_audio_bytes": int(audio_bytes),
    }


async def _run(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var: {args.api_key_env}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    system_instruction = (
        "あなたは感情豊かな猫キャラです。"
        "ユーザーの音声に短く自然な日本語で返答してください。"
        "必ず音声で返答してください。"
        "返答前に set_emotion を1回呼んでください。"
    )

    client = genai.Client(
        api_key=api_key,
        http_options={"api_version": str(args.api_version)},
    )

    config = _build_live_config(system_instruction)

    print(f"[persistent_probe] connect model={args.model}", flush=True)

    results: list[dict[str, Any]] = []

    async with client.aio.live.connect(model=args.model, config=config) as session:
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        stop_event = asyncio.Event()
        receive_task = asyncio.create_task(
            _receive_audio_loop(
                session=session,
                audio_queue=audio_queue,
                stop_event=stop_event,
            )
        )

        try:
            for i in range(int(args.turns)):
                print(f"[persistent_probe] turn={i+1}", flush=True)

                res = await _measure_one_turn(
                    session=session,
                    audio_queue=audio_queue,
                    turn_index=i + 1,
                    duration_s=float(args.duration_s),
                    input_sr=int(args.input_sr),
                    chunk_ms=int(args.chunk_ms),
                    response_trigger=str(args.response_trigger),
                    receive_timeout_s=float(args.receive_timeout_s),
                    audio_stream_end_per_turn=bool(args.audio_stream_end_per_turn),
                )

                results.append(res)

                if i < int(args.turns) - 1:
                    await asyncio.sleep(float(args.gap_s))
        finally:
            stop_event.set()
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass
            except BaseException:
                pass

    summary = {
        "model": args.model,
        "api_version": args.api_version,
        "turns": results,
    }

    summary_json = out_dir / "dev_live_session_persistent_probe.summary.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[persistent_probe][OK]", flush=True)
    print(f"  summary_json: {summary_json}", flush=True)

    for r in results:
        print(
            f"  turn{r['turn']}: first_audio={r['first_response_audio_chunk_sec']} "
            f"chunks={r['response_audio_chunks']} input_ms={r['input_audio_ms']}",
            flush=True,
        )

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", default="out/dev_live_session_persistent_probe")
    ap.add_argument("--model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--api_version", default="v1alpha")
    ap.add_argument("--api_key_env", default="GEMINI_API_KEY")

    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--duration_s", type=float, default=0.6)
    ap.add_argument("--gap_s", type=float, default=0.8)

    ap.add_argument("--input_sr", type=int, default=16000)
    ap.add_argument("--chunk_ms", type=int, default=40)
    ap.add_argument("--receive_timeout_s", type=float, default=5.0)
    ap.add_argument("--audio_stream_end_per_turn", action="store_true")

    ap.add_argument(
        "--response_trigger",
        default="短く返答してください。返答前にset_emotionを1回呼んでください。",
    )

    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
