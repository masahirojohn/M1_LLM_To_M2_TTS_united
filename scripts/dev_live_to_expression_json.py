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

from m3p.live.expression_streamer import ExpressionStreamer, ExpressionStreamerConfig


def _now_ms() -> int:
    return int(time.monotonic() * 1000)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return float(v)


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


def _default_emo_map() -> dict[str, str]:
    """
    contracts / current ExpressionStreamer minimal set に寄せる。
    - 1_0 -> normal
    - 1_1 -> smile
    - 1_2 -> sad
    - 2_0 -> angry
    未定義は normal に落とす。
    """
    return {
        "1_0": "normal",
        "1_1": "smile",
        "1_2": "sad",
        "2_0": "angry",
    }


def _parse_emo_id(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    try:
        return str(v)
    except Exception:
        return None


async def _run(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var: {args.api_key_env}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session_id = str(args.session_id)
    prompt_text = str(args.prompt)

    expr_json = out_dir / args.out_expr_json
    tool_calls_json = out_dir / args.out_tool_calls_json
    text_chunks_json = out_dir / args.out_text_chunks_json
    events_jsonl = out_dir / args.out_events_jsonl
    profile_json = out_dir / args.out_profile_json

    t0_mono_ms = _now_ms()
    t0_wall = time.time()

    profile: dict[str, Any] = {
        "format": "live_to_expression_profile.v1",
        "session_id": session_id,
        "model": args.model,
        "step_ms": int(args.step_ms),
        "dev_stop_s": float(args.dev_stop_s),
        "stop_after_first_tool_s": float(args.stop_after_first_tool_s),
        "t_start_ms": int(t0_mono_ms),
        "prompt": prompt_text,
    }

    system_instruction = (
        "あなたは感情豊かな猫キャラです。"
        "返答は短く自然な日本語にしてください。"
        "必ず音声で返答してください。"
        "テキストで考え方や作業メモを説明してはいけません。"
        "感情を変えるたびに、発話の直前に必ず set_emotion を呼んでください。"
        "今回の返答では、感情を3回必ず切り替えてください。"
        "順番は 1_1 → 1_2 → 1_0 です。"
        "各感情について、必ず1回ずつ set_emotion(emo_id) を呼んでから、その感情で短く話してください。"
        "set_emotion を呼ばずに次の発話へ進んではいけません。"
        "emo_id は文字列IDです。使用可能なのは 1_0, 1_1, 1_2, 2_0 のみです。"
        "出力中に説明文・要約・英語のメモを書いてはいけません。"
    )

    system_instruction = (
        "あなたは感情豊かな猫キャラです。"
        "返答は短く自然な日本語にしてください。"
        "必ず音声で返答してください。"
        "発話の直前に必ず set_emotion を1回だけ呼んでください。"
        "emo_id は文字列IDです。"
        "使用可能なのは 1_0, 1_1, 1_2, 2_0 のみです。"
        "通常は 1_1 を使ってください。"
        "出力中に説明文・要約・作業メモを書いてはいけません。"
    )




    if args.system_instruction:
        system_instruction = str(args.system_instruction)

    emo_map = _default_emo_map()
    if args.emo_map_json:
        emo_map_path = Path(args.emo_map_json).resolve()
        emo_map = json.loads(emo_map_path.read_text(encoding="utf-8"))

    streamer = ExpressionStreamer(
        ExpressionStreamerConfig(
            step_ms=int(args.step_ms),
            default_expression=str(args.default_expression),
            emo_map=emo_map,
            unknown_emo_to=str(args.unknown_emo_to),
        )
    )
    streamer.init_at(0, str(args.default_expression))

    tool_calls: list[dict[str, Any]] = []
    text_chunks: list[dict[str, Any]] = []
    total_audio_chunks = 0
    total_audio_bytes = 0
    first_tool_wall: float | None = None
    stop_requested = False
    last_tool_t_ms_for_streamer: int | None = None

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

        print(f"[dev_live_to_expression_json] connect model={args.model} session_id={session_id}")
        print(f"[dev_live_to_expression_json] out_dir={out_dir}")

        async with client.aio.live.connect(model=args.model, config=config) as session:
            profile["t_connected_ms"] = _now_ms()
            _append_jsonl(
                events_jsonl,
                {
                    "t_ms": profile["t_connected_ms"] - t0_mono_ms,
                    "type": "connected",
                },
            )

            await session.send_realtime_input(
                text=prompt_text
            )

            _append_jsonl(
                events_jsonl,
                {
                    "t_ms": _now_ms() - t0_mono_ms,
                    "type": "prompt_sent",
                },
            )

            async for msg in session.receive():
                now_rel_ms = _now_ms() - t0_mono_ms

                # ---- tool_call ----
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

                        if item["name"] != "set_emotion":
                            print(f"[TOOL] skip unknown tool name={item['name']}")
                            continue

                        args_dict = item["args"] if isinstance(item["args"], dict) else {}
                        emo_id = _parse_emo_id(args_dict.get("emo_id"))
                        if emo_id is None:
                            print("[TOOL] set_emotion without valid emo_id")
                            continue

                        # 同一40msグリッド衝突回避:
                        # ExpressionStreamer 側は step_ms=40 で量子化し、
                        # 同一 t_ms は後勝ちで潰れるため、
                        # ここで streamer投入用の時刻だけ単調増加にしておく。
                        t_ms_for_streamer = int(now_rel_ms)
                        step_ms = int(args.step_ms)
                        if last_tool_t_ms_for_streamer is not None:
                            prev_q = int(round(last_tool_t_ms_for_streamer / step_ms) * step_ms)
                            curr_q = int(round(t_ms_for_streamer / step_ms) * step_ms)
                            if curr_q <= prev_q:
                                t_ms_for_streamer = prev_q + step_ms

                        ev = streamer.on_emo_id(
                            emo_id=emo_id,
                            t_ms=t_ms_for_streamer,
                            source="tool_call",
                        )
                        last_tool_t_ms_for_streamer = int(t_ms_for_streamer)
                        streamer.write_json(expr_json)

                        if ev is not None:
                            print(
                                f"[TOOL] set_emotion emo_id={emo_id} "
                                f"-> {ev.get('expression')} @t_ms={ev.get('t_ms')}"
                            )
                        else:
                            print(f"[TOOL] set_emotion emo_id={emo_id} (no change)")

                        if first_tool_wall is None:
                            first_tool_wall = time.time()
                            profile["t_first_tool_ms"] = _now_ms()

                            # --- stop after first tool: immediate local timer ---
                            # 次の受信イベントを待たず、この場で少し待って終了する。
                            wait_s = float(args.stop_after_first_tool_s)
                            if wait_s > 0:
                                await asyncio.sleep(wait_s)
                            print(
                                "[dev_live_to_expression_json] "
                                f"dev stop after first tool (+{float(args.stop_after_first_tool_s):.1f}s)"
                            )
                            _append_jsonl(
                                events_jsonl,
                                {
                                    "t_ms": _now_ms() - t0_mono_ms,
                                    "type": "dev_stop_after_first_tool",
                                    "payload": {"stop_after_first_tool_s": float(args.stop_after_first_tool_s)},
                                },
                            )
                            stop_requested = True
                            break

                # ---- text / audio logs ----
                server_content = _safe_getattr(msg, "server_content", None)
                if server_content:
                    model_turn = _safe_getattr(server_content, "model_turn", None)
                    parts = _safe_getattr(model_turn, "parts", None) or []

                    for part in parts:
                        text_val = _safe_getattr(part, "text", None)
                        if isinstance(text_val, str) and text_val != "":
                            text_item = {
                                "t_ms": now_rel_ms,
                                "text": text_val,
                            }
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
                        if data and isinstance(data, (bytes, bytearray)):
                            total_audio_chunks += 1
                            total_audio_bytes += len(data)

                if stop_requested:
                    break            

                # ---- stop conditions ----
                now_wall = time.time()

                if (now_wall - t0_wall) >= float(args.dev_stop_s):
                    print(
                        f"[dev_live_to_expression_json] dev stop by timeout {float(args.dev_stop_s):.1f}s"
                    )
                    _append_jsonl(
                        events_jsonl,
                        {
                            "t_ms": _now_ms() - t0_mono_ms,
                            "type": "dev_stop_timeout",
                            "payload": {"dev_stop_s": float(args.dev_stop_s)},
                        },
                    )
                    break

    except Exception as e:
        warn_item = {
            "t_ms": _now_ms() - t0_mono_ms,
            "type": "warning",
            "payload": {
                "error_type": type(e).__name__,
                "message": str(e),
            },
        }
        _append_jsonl(events_jsonl, warn_item)
        print(f"[dev_live_to_expression_json][WARN] {type(e).__name__}: {e}")

    finally:
        streamer.write_json(expr_json)
        _write_json(tool_calls_json, tool_calls)
        _write_json(text_chunks_json, text_chunks)

        final_events = streamer.to_json()
        profile["t_finalize_done_ms"] = _now_ms()
        profile["tool_calls_n"] = int(len(tool_calls))
        profile["text_chunks_n"] = int(len(text_chunks))
        profile["expression_events_n"] = int(len(final_events))
        profile["audio_chunks_seen"] = int(total_audio_chunks)
        profile["audio_bytes_seen"] = int(total_audio_bytes)
        profile["expr_json"] = str(expr_json)
        profile["tool_calls_json"] = str(tool_calls_json)
        profile["text_chunks_json"] = str(text_chunks_json)
        _write_json(profile_json, profile)

        _append_jsonl(
            events_jsonl,
            {
                "t_ms": _now_ms() - t0_mono_ms,
                "type": "finalized",
                "payload": {
                    "expression_events_n": int(len(final_events)),
                    "tool_calls_n": int(len(tool_calls)),
                    "text_chunks_n": int(len(text_chunks)),
                    "audio_chunks_seen": int(total_audio_chunks),
                },
            },
        )

        print("[dev_live_to_expression_json][OK]")
        print(f"  expr_json        : {expr_json}")
        print(f"  tool_calls_json  : {tool_calls_json}")
        print(f"  text_chunks_json : {text_chunks_json}")
        print(f"  events_jsonl     : {events_jsonl}")
        print(f"  profile_json     : {profile_json}")
        print(f"  expression_events: {len(final_events)}")
        print(f"  tool_calls       : {len(tool_calls)}")
        print(f"  text_chunks      : {len(text_chunks)}")

    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Gemini Live API -> ExpressionStreamer -> expression timeline json"
    )

    ap.add_argument("--session_id", default=_env_str("LIVE_SESSION_ID", "sess_live_expr_001"))
    ap.add_argument("--out_dir", default=_env_str("OUT_DIR", "out/live/sess_live_expr_001"))

    ap.add_argument("--model", default=_env_str("GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"))
    ap.add_argument("--api_key_env", default="GEMINI_API_KEY")
    ap.add_argument("--api_version", default=_env_str("GEMINI_API_VERSION", "v1alpha"))
    ap.add_argument("--prompt", default=_env_str("PROMPT", "感情を少し変えながら、短く自己紹介して。"))
    ap.add_argument("--system_instruction", default="")

    ap.add_argument("--step_ms", type=int, default=_env_int("STEP_MS", 40))
    ap.add_argument("--dev_stop_s", type=float, default=_env_float("DEV_STOP_S", 10.0))
    ap.add_argument("--stop_after_first_tool_s", type=float, default=_env_float("STOP_AFTER_FIRST_TOOL_S", 2.0))

    ap.add_argument("--default_expression", default=_env_str("DEFAULT_EXPRESSION", "normal"))
    ap.add_argument("--unknown_emo_to", default=_env_str("UNKNOWN_EMO_TO", "normal"))
    ap.add_argument("--emo_map_json", default="")

    ap.add_argument("--out_expr_json", default="expression_timeline.live.json")
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