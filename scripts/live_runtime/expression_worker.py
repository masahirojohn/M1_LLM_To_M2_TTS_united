#!/usr/bin/env python3
from __future__ import annotations

"""
Live Runtime expression_worker v0

目的:
- Gemini Live API の tool_call set_emotion を受け取る
- 最新 expression だけを保持する
- v0 は単体スモーク用。M0 / orchestrator にはまだ接続しない

固定:
- step_ms = 40
- latest state は上書き方式
- receive() を直接 await しない
- asyncio.wait + non-cancel方式
"""

import argparse
import asyncio
import json
import os
import time
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from m3p.live.expression_streamer import ExpressionStreamer, ExpressionStreamerConfig


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

def _is_normal_live_close_error(e: BaseException) -> bool:
    s = str(e)
    return "1000" in s and "None" in s

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


def _default_emo_map() -> dict[str, str]:
    return {
        "1_0": "normal",
        "1_1": "smile",
        "1_2": "sad",
        "2_0": "angry",
        "9_1": "sad",
        "9_2": "sad",
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


@dataclass
class LatestExpression:
    expression: str = "normal"
    emo_id: str = "1_0"
    updated_at_ms: int = 0
    source: str = "init"

    def update(self, *, expression: str, emo_id: str, updated_at_ms: int, source: str) -> None:
        self.expression = str(expression)
        self.emo_id = str(emo_id)
        self.updated_at_ms = int(updated_at_ms)
        self.source = str(source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "emo_id": self.emo_id,
            "updated_at_ms": self.updated_at_ms,
            "source": self.source,
        }


async def run_expression_worker(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var: {args.api_key_env}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    session_id = str(args.session_id)
    prompt_text = str(args.prompt)

    latest_json = out_dir / "expression_worker.latest.json"
    events_json = out_dir / "expression_worker.events.json"
    tool_calls_json = out_dir / "expression_worker.tool_calls.json"
    events_jsonl = out_dir / "expression_worker.events.jsonl"
    profile_json = out_dir / "expression_worker.profile.json"

    t0_mono_ms = _now_ms()
    t0_wall = time.time()

    emo_map = _default_emo_map()
    if args.emo_map_json:
        emo_map = json.loads(Path(args.emo_map_json).read_text(encoding="utf-8"))

    latest = LatestExpression(
        expression=str(args.default_expression),
        emo_id="1_0",
        updated_at_ms=0,
        source="init",
    )

    streamer = ExpressionStreamer(
        ExpressionStreamerConfig(
            step_ms=int(args.step_ms),
            default_expression=str(args.default_expression),
            emo_map=emo_map,
            unknown_emo_to=str(args.unknown_emo_to),
        )
    )
    streamer.init_at(0, str(args.default_expression))

    system_instruction = (
        "あなたは感情豊かな猫キャラです。"
        "返答は短く自然な日本語にしてください。"
        "必ず音声で返答してください。"
        "発話の直前に必ず set_emotion を1回だけ呼んでください。"
        "emo_id は文字列IDです。"
        "使用可能なのは 1_0, 1_1, 1_2, 2_0, 9_1, 9_2 のみです。"
        "通常は 1_1 を使ってください。"
        "出力中に説明文・要約・作業メモを書いてはいけません。"
    )
    if args.system_instruction:
        system_instruction = str(args.system_instruction)

    profile: dict[str, Any] = {
        "format": "expression_worker_profile.v0",
        "session_id": session_id,
        "model": args.model,
        "step_ms": int(args.step_ms),
        "dev_stop_s": float(args.dev_stop_s),
        "stop_after_first_tool_s": float(args.stop_after_first_tool_s),
        "receive_timeout_s": float(args.receive_timeout_s),
        "t_start_ms": int(t0_mono_ms),
        "prompt": prompt_text,
    }

    tool_calls: list[dict[str, Any]] = []
    expression_events: list[dict[str, Any]] = []

    first_tool_wall: float | None = None
    stop_reason: str | None = None

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

        print(f"[expression_worker] connect model={args.model} session_id={session_id}")
        print(f"[expression_worker] out_dir={out_dir}")

        async with client.aio.live.connect(model=args.model, config=config) as session:
            profile["t_connected_ms"] = _now_ms()

            await session.send_realtime_input(text=prompt_text)

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

                if first_tool_wall is not None:
                    after_first_tool = time.time() - first_tool_wall
                    if after_first_tool >= float(args.stop_after_first_tool_s):
                        stop_reason = "stop_after_first_tool_s"
                        print(
                            f"[expression_worker] stop after first tool "
                            f"{float(args.stop_after_first_tool_s):.1f}s"
                        )
                        break

                if elapsed_s >= float(args.dev_stop_s):
                    stop_reason = "dev_stop_s"
                    print(f"[expression_worker] dev stop after {float(args.dev_stop_s):.1f}s")
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

                except Exception as e:
                    if _is_normal_live_close_error(e):
                        stop_reason = "live_normal_close_1000"
                        break
                    raise

                now_rel_ms = _now_ms() - t0_mono_ms

                tool_call = _safe_getattr(msg, "tool_call", None)
                if not (tool_call and _safe_getattr(tool_call, "function_calls", None)):
                    continue

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

                    if first_tool_wall is None:
                        first_tool_wall = time.time()

                    if item["name"] != "set_emotion":
                        print(f"[expression_worker] skip unknown tool name={item['name']}")
                        continue

                    args_dict = item["args"] if isinstance(item["args"], dict) else {}
                    emo_id = _parse_emo_id(args_dict.get("emo_id"))
                    if emo_id is None:
                        print("[expression_worker] set_emotion without valid emo_id")
                        continue

                    expression = str(emo_map.get(emo_id, args.unknown_emo_to))

                    ev = streamer.on_emo_id(
                        emo_id=str(emo_id),
                        t_ms=int(now_rel_ms),
                        source="live_tool_call",
                    )

                    if ev is not None:
                        expression = str(ev.get("expression", expression))
                        event_t_ms = int(ev.get("t_ms", now_rel_ms))
                    else:
                        event_t_ms = int(now_rel_ms)

                    event = {
                        "t_ms": event_t_ms,
                        "expression": expression,
                        "emo_id": str(emo_id),
                        "source": "live_tool_call",
                    }
                    expression_events.append(event)

                    latest.update(
                        expression=expression,
                        emo_id=str(emo_id),
                        updated_at_ms=int(now_rel_ms),
                        source="live_tool_call",
                    )

                    _write_json(
                        latest_json,
                        {
                            "schema_version": "latest_expression.v0",
                            "session_id": session_id,
                            "step_ms": int(args.step_ms),
                            "latest": latest.to_dict(),
                        },
                    )

                    print(
                        "[expression_worker] "
                        f"set_emotion emo_id={emo_id} expression={expression} "
                        f"t_ms={now_rel_ms}"
                    )
            
            if recv_task is not None:
                if recv_task.done():
                    try:
                        recv_task.result()
                    except Exception as e:
                        if not _is_normal_live_close_error(e):
                            raise
                else:
                    recv_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await recv_task

    finally:
        pass

    if stop_reason is None:
        stop_reason = "finished"

    if not expression_events:
        expression_events.append(
            {
                "t_ms": 0,
                "expression": str(args.default_expression),
                "emo_id": "1_0",
                "source": "fallback_init",
            }
        )
        latest.update(
            expression=str(args.default_expression),
            emo_id="1_0",
            updated_at_ms=0,
            source="fallback_init",
        )

    _write_json(
        latest_json,
        {
            "schema_version": "latest_expression.v0",
            "session_id": session_id,
            "step_ms": int(args.step_ms),
            "latest": latest.to_dict(),
        },
    )

    _write_json(
        events_json,
        {
            "schema_version": "expression_events.v0",
            "session_id": session_id,
            "step_ms": int(args.step_ms),
            "timeline": expression_events,
            "meta": {
                "source": "expression_worker",
                "events_n": len(expression_events),
                "stop_reason": stop_reason,
            },
        },
    )

    _write_json(tool_calls_json, tool_calls)

    profile.update(
        {
            "stop_reason": stop_reason,
            "tool_calls": len(tool_calls),
            "expression_events": len(expression_events),
            "latest": latest.to_dict(),
            "t_finished_ms": _now_ms(),
        }
    )
    _write_json(profile_json, profile)

    print("[expression_worker][OK]")
    print(f"  latest_expression : {latest.expression}")
    print(f"  latest_emo_id     : {latest.emo_id}")
    print(f"  tool_calls        : {len(tool_calls)}")
    print(f"  events            : {len(expression_events)}")
    print(f"  latest_json       : {latest_json}")
    print(f"  events_json       : {events_json}")
    print(f"  tool_calls_json   : {tool_calls_json}")

    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Live Runtime expression_worker v0"
    )

    ap.add_argument("--session_id", default="sess_expression_worker_001")
    ap.add_argument("--out_dir", default="out/live_runtime_realtime/expression_worker_001")

    ap.add_argument("--api_key_env", default="GEMINI_API_KEY")
    ap.add_argument("--api_version", default="v1beta")
    ap.add_argument("--model", default="gemini-3.1-flash-live-preview")

    ap.add_argument(
        "--prompt",
        default="短く自己紹介して。最初に必ず set_emotion を呼び、感情は smile にしてください。",
    )
    ap.add_argument("--system_instruction", default=None)

    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--dev_stop_s", type=float, default=12.0)
    ap.add_argument("--stop_after_first_tool_s", type=float, default=3.0)
    ap.add_argument("--receive_timeout_s", type=float, default=0.25)

    ap.add_argument("--default_expression", default="normal")
    ap.add_argument("--unknown_emo_to", default="normal")
    ap.add_argument("--emo_map_json", default=None)

    return ap


def main() -> int:
    args = build_argparser().parse_args()

    if int(args.step_ms) != 40:
        raise ValueError("step_ms must be 40 for current realtime contract")

    return asyncio.run(run_expression_worker(args))


if __name__ == "__main__":
    raise SystemExit(main())