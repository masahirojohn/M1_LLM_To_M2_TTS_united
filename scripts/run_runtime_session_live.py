#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], *, cwd: Path | None = None, env: Dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_runtime_env(m1_repo: Path, m3_repo: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{m1_repo / 'src'}:{m3_repo / 'src'}:" + env.get("PYTHONPATH", "")
    return env


def _read_audio_meta(audio_meta_json: Path) -> Dict[str, Any]:
    obj = _load_json(audio_meta_json)
    if not isinstance(obj, dict):
        raise ValueError(f"audio_meta must be object: {audio_meta_json}")
    return obj


def _read_expr(expr_json: Path) -> List[Dict[str, Any]]:
    obj = _load_json(expr_json)
    if not isinstance(obj, list):
        raise ValueError(f"expression json must be list: {expr_json}")
    out: List[Dict[str, Any]] = []
    for item in obj:
        if isinstance(item, dict):
            out.append(item)
    return out


def _read_tool_calls(tool_calls_json: Path) -> List[Dict[str, Any]]:
    obj = _load_json(tool_calls_json)
    if not isinstance(obj, list):
        raise ValueError(f"tool_calls json must be list: {tool_calls_json}")
    out: List[Dict[str, Any]] = []
    for item in obj:
        if isinstance(item, dict):
            out.append(item)
    return out


def run_live_mouth(
    *,
    python_exe: str,
    m1_repo: Path,
    m3_repo: Path,
    out_dir: Path,
    session_id: str,
    model: str,
    prompt: str,
    dev_stop_s: float,
    api_version: str,
) -> Dict[str, Path]:
    env = _build_runtime_env(m1_repo, m3_repo)

    cmd = [
        python_exe,
        str((m1_repo / "scripts" / "dev_live_to_m3_raw.py").resolve()),
        "--session_id", str(session_id),
        "--out_dir", str(out_dir),
        "--model", str(model),
        "--prompt", str(prompt),
        "--dev_stop_s", str(dev_stop_s),
        "--api_version", str(api_version),
    ]
    _run(cmd, cwd=m1_repo, env=env)

    paths = {
        "streamer_json": out_dir / "mouth_timeline.live.json",
        "raw_json": out_dir / "mouth_timeline.formant.raw.json",
        "audio_meta_json": out_dir / "audio_meta.json",
        "tool_calls_json": out_dir / "tool_calls.live.json",
        "text_chunks_json": out_dir / "text_chunks.live.json",
        "events_jsonl": out_dir / "events.live.jsonl",
        "profile_json": out_dir / "profile.live.json",
    }

    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"missing {name}: {p}")

    return paths


def run_live_expression(
    *,
    python_exe: str,
    m1_repo: Path,
    m3_repo: Path,
    out_dir: Path,
    session_id: str,
    model: str,
    prompt: str,
    dev_stop_s: float,
    stop_after_first_tool_s: float,
    api_version: str,
) -> Dict[str, Path]:
    env = _build_runtime_env(m1_repo, m3_repo)

    cmd = [
        python_exe,
        str((m1_repo / "scripts" / "dev_live_to_expression_json.py").resolve()),
        "--session_id", str(session_id),
        "--out_dir", str(out_dir),
        "--model", str(model),
        "--prompt", str(prompt),
        "--dev_stop_s", str(dev_stop_s),
        "--stop_after_first_tool_s", str(stop_after_first_tool_s),
        "--api_version", str(api_version),
    ]
    _run(cmd, cwd=m1_repo, env=env)

    paths = {
        "expr_json": out_dir / "expression_timeline.live.json",
        "tool_calls_json": out_dir / "tool_calls.live.json",
        "text_chunks_json": out_dir / "text_chunks.live.json",
        "events_jsonl": out_dir / "events.live.jsonl",
        "profile_json": out_dir / "profile.live.json",
    }

    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"missing {name}: {p}")

    return paths


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Minimal runtime entry for Gemini Live mouth + expression collection"
    )

    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")

    ap.add_argument("--session_id", default="sess_live_runtime_001")
    ap.add_argument("--out_root", default="out/live_runtime")

    # mouth side
    ap.add_argument(
        "--mouth_model",
        default="gemini-2.5-flash-native-audio-preview-12-2025",
        help="Model for live audio -> mouth pipeline",
    )
    ap.add_argument(
        "--mouth_prompt",
        default="元気よく自己紹介して！短めで。",
        help="Prompt for live audio -> mouth pipeline",
    )
    ap.add_argument("--mouth_dev_stop_s", type=float, default=8.0)

    # expression side
    ap.add_argument(
        "--expr_model",
        default="gemini-3.1-flash-live-preview",
        help="Model for live tool_call -> expression pipeline",
    )
    ap.add_argument(
        "--expr_prompt",
        default="短く自己紹介して。最初に必ず set_emotion を呼び、感情は smile にしてください。",
        help="Prompt for live expression pipeline",
    )
    ap.add_argument("--expr_dev_stop_s", type=float, default=12.0)
    ap.add_argument("--stop_after_first_tool_s", type=float, default=3.0)

    ap.add_argument("--api_version", default="v1alpha")
    ap.add_argument(
        "--skip_expression",
        action="store_true",
        help="If set, run mouth side only",
    )

    args = ap.parse_args()

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    out_root = Path(args.out_root).resolve()
    session_id = str(args.session_id)

    if not m1_repo.exists():
        raise FileNotFoundError(f"missing m1_repo_root: {m1_repo}")
    if not m3_repo.exists():
        raise FileNotFoundError(f"missing m3_repo_root: {m3_repo}")

    session_out = out_root / session_id
    mouth_out = session_out / "mouth"
    expr_out = session_out / "expression"
    _ensure_dir(mouth_out)
    _ensure_dir(expr_out)

    python_exe = sys.executable

    mouth_paths = run_live_mouth(
        python_exe=python_exe,
        m1_repo=m1_repo,
        m3_repo=m3_repo,
        out_dir=mouth_out,
        session_id=session_id,
        model=str(args.mouth_model),
        prompt=str(args.mouth_prompt),
        dev_stop_s=float(args.mouth_dev_stop_s),
        api_version=str(args.api_version),
    )

    expr_paths: Dict[str, Path] | None = None
    if not args.skip_expression:
        expr_paths = run_live_expression(
            python_exe=python_exe,
            m1_repo=m1_repo,
            m3_repo=m3_repo,
            out_dir=expr_out,
            session_id=session_id,
            model=str(args.expr_model),
            prompt=str(args.expr_prompt),
            dev_stop_s=float(args.expr_dev_stop_s),
            stop_after_first_tool_s=float(args.stop_after_first_tool_s),
            api_version=str(args.api_version),
        )

    audio_meta = _read_audio_meta(mouth_paths["audio_meta_json"])
    audio_ms = int(audio_meta.get("audio_ms", 0))

    expr_events: List[Dict[str, Any]] = []
    expr_tool_calls: List[Dict[str, Any]] = []
    if expr_paths is not None:
        expr_events = _read_expr(expr_paths["expr_json"])
        expr_tool_calls = _read_tool_calls(expr_paths["tool_calls_json"])

    summary = {
        "format": "runtime_session_live_summary.v0.1",
        "session_id": session_id,
        "audio_ms": audio_ms,
        "mouth": {
            "streamer_json": str(mouth_paths["streamer_json"]),
            "raw_json": str(mouth_paths["raw_json"]),
            "audio_meta_json": str(mouth_paths["audio_meta_json"]),
            "tool_calls_json": str(mouth_paths["tool_calls_json"]),
            "text_chunks_json": str(mouth_paths["text_chunks_json"]),
            "events_jsonl": str(mouth_paths["events_jsonl"]),
            "profile_json": str(mouth_paths["profile_json"]),
        },
        "expression": (
            {
                "expr_json": str(expr_paths["expr_json"]),
                "tool_calls_json": str(expr_paths["tool_calls_json"]),
                "text_chunks_json": str(expr_paths["text_chunks_json"]),
                "events_jsonl": str(expr_paths["events_jsonl"]),
                "profile_json": str(expr_paths["profile_json"]),
                "expression_events_n": len(expr_events),
                "tool_calls_n": len(expr_tool_calls),
            }
            if expr_paths is not None
            else None
        ),
    }

    summary_json = session_out / "session_live_runtime_summary.json"
    _dump_json(summary_json, summary)

    print("[run_runtime_session_live][OK]")
    print("  session_id       :", session_id)
    print("  session_out      :", session_out)
    print("  mouth_raw_json   :", mouth_paths["raw_json"])
    print("  audio_meta_json  :", mouth_paths["audio_meta_json"])
    if expr_paths is not None:
        print("  expr_json        :", expr_paths["expr_json"])
        print("  expr_tool_calls  :", expr_paths["tool_calls_json"])
        print("  expr_events_n    :", len(expr_events))
        print("  expr_tool_calls_n:", len(expr_tool_calls))
    print("  summary_json     :", summary_json)
    print("  audio_ms         :", audio_ms)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())