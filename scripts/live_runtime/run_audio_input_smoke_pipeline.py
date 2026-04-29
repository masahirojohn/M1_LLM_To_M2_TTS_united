#!/usr/bin/env python3
from __future__ import annotations

"""
run_audio_input_smoke_pipeline.py

目的:
Codespacesで実行できる範囲の音声入力スモークを1コマンド化する。

Pipeline:
1. audio_stream_bridge.py
   WAV/PCM入力 → Gemini Live API → audio_response.pcm / tool_calls.json

2. audio_response_to_mouth_smoke.py
   audio_response.pcm → mouth_timeline.formant.raw.json / audio_meta.json

3. knn_from_formant_raw_to_mouth_timeline.py
   raw formant → mouth.json

4. audio_tool_calls_to_expr_smoke.py
   live_tool_calls.json → expr.json（audio_ms内にclamp）

出力:
out/audio_input_smoke_pipeline/<session_id>/
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_env(m1_repo: Path, m3_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{m1_repo / 'src'}:{m3_repo / 'src'}:"
        + env.get("PYTHONPATH", "")
    )
    return env


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run audio input smoke pipeline"
    )

    ap.add_argument("--session_id", default="sess_audio_input_smoke_001")

    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")

    ap.add_argument("--input_wav", default=None)
    ap.add_argument("--input_pcm", default=None)
    ap.add_argument("--input_sr", type=int, default=16000)

    ap.add_argument("--out_root", default=None)

    ap.add_argument("--model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--api_version", default="v1alpha")
    ap.add_argument("--duration_s", type=float, default=10.0)
    ap.add_argument("--step_ms", type=int, default=40)

    ap.add_argument("--gt_glob", default="data/knn_db/*.f1f2.json")

    args = ap.parse_args()

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    python_exe = sys.executable
    env = _build_env(m1_repo, m3_repo)

    if not args.input_wav and not args.input_pcm:
        raise RuntimeError("--input_wav or --input_pcm is required")

    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else m1_repo / "out" / "audio_input_smoke_pipeline" / args.session_id
    )
    out_root.mkdir(parents=True, exist_ok=True)

    bridge_dir = out_root / "01_audio_stream_bridge"
    mouth_dir = out_root / "02_audio_response_to_mouth"
    expr_dir = out_root / "03_tool_calls_to_expr"

    bridge_script = m1_repo / "scripts" / "live_runtime" / "audio_stream_bridge.py"
    mouth_script = m1_repo / "scripts" / "live_runtime" / "audio_response_to_mouth_smoke.py"
    expr_script = m1_repo / "scripts" / "live_runtime" / "audio_tool_calls_to_expr_smoke.py"
    knn_script = m3_repo / "tools" / "knn_from_formant_raw_to_mouth_timeline.py"

    for p in [bridge_script, mouth_script, expr_script, knn_script]:
        if not p.exists():
            raise FileNotFoundError(f"missing script: {p}")

    # 1. Live API audio input smoke
    cmd = [
        python_exe,
        str(bridge_script),
        "--session_id",
        args.session_id,
        "--out_dir",
        str(bridge_dir),
        "--mode",
        "file",
        "--model",
        str(args.model),
        "--api_version",
        str(args.api_version),
        "--duration_s",
        str(float(args.duration_s)),
        "--response_trigger",
        "",
    ]

    if args.input_wav:
        cmd += ["--input_wav", str(Path(args.input_wav).resolve())]
    else:
        cmd += [
            "--input_pcm",
            str(Path(args.input_pcm).resolve()),
            "--input_sr",
            str(int(args.input_sr)),
        ]

    _run(cmd, cwd=m1_repo, env=env)

    audio_response_pcm = bridge_dir / "audio_response.pcm"
    tool_calls_json = bridge_dir / "live_tool_calls.json"

    if not audio_response_pcm.exists():
        raise FileNotFoundError(f"missing audio_response.pcm: {audio_response_pcm}")
    if not tool_calls_json.exists():
        raise FileNotFoundError(f"missing live_tool_calls.json: {tool_calls_json}")

    # 2. Gemini response audio -> raw mouth formants
    _run(
        [
            python_exe,
            str(mouth_script),
            "--session_id",
            args.session_id,
            "--input_pcm",
            str(audio_response_pcm),
            "--out_dir",
            str(mouth_dir),
            "--input_sr",
            "24000",
            "--step_ms",
            str(int(args.step_ms)),
            "--chunk_ms",
            str(int(args.step_ms)),
        ],
        cwd=m1_repo,
        env=env,
    )

    raw_json = mouth_dir / "mouth_timeline.formant.raw.json"
    audio_meta_json = mouth_dir / "audio_meta.json"
    mouth_json = mouth_dir / "mouth.json"

    if not raw_json.exists():
        raise FileNotFoundError(f"missing raw_json: {raw_json}")
    if not audio_meta_json.exists():
        raise FileNotFoundError(f"missing audio_meta_json: {audio_meta_json}")

    # 3. raw formants -> mouth.json
    _run(
        [
            python_exe,
            str(knn_script),
            "--raw",
            str(raw_json),
            "--gt_glob",
            str(args.gt_glob),
            "--out",
            str(mouth_json),
            "--audio",
            str(audio_response_pcm),
            "--step_ms",
            str(int(args.step_ms)),
        ],
        cwd=m3_repo,
        env=env,
    )

    if not mouth_json.exists():
        raise FileNotFoundError(f"missing mouth_json: {mouth_json}")

    # 4. tool_calls -> expr.json with audio_ms clamp
    _run(
        [
            python_exe,
            str(expr_script),
            "--session_id",
            args.session_id,
            "--tool_calls_json",
            str(tool_calls_json),
            "--audio_meta_json",
            str(audio_meta_json),
            "--out_dir",
            str(expr_dir),
            "--step_ms",
            str(int(args.step_ms)),
        ],
        cwd=m1_repo,
        env=env,
    )

    expr_json = expr_dir / "expr.json"
    if not expr_json.exists():
        raise FileNotFoundError(f"missing expr_json: {expr_json}")

    bridge_summary = _load_json(bridge_dir / "audio_stream_bridge.summary.json")
    mouth_summary = _load_json(mouth_dir / "audio_response_to_mouth_smoke.summary.json")
    expr_summary = _load_json(expr_dir / "audio_tool_calls_to_expr_smoke.summary.json")

    mouth_obj = _load_json(mouth_json)
    expr_obj = _load_json(expr_json)

    frames = mouth_obj.get("frames", [])
    timeline = expr_obj.get("timeline", [])

    summary = {
        "format": "audio_input_smoke_pipeline.summary.v0",
        "session_id": args.session_id,
        "step_ms": int(args.step_ms),
        "bridge": {
            "input_audio_ms": bridge_summary.get("input_audio_ms"),
            "response_audio_bytes": bridge_summary.get("response_audio_bytes"),
            "response_audio_chunks": bridge_summary.get("response_audio_chunks"),
            "tool_calls": bridge_summary.get("tool_calls"),
        },
        "mouth": {
            "audio_ms": mouth_summary.get("audio_ms"),
            "target_frames": mouth_summary.get("target_frames"),
            "frames_n": len(frames),
            "voiced_frames_n": mouth_summary.get("voiced_frames_n"),
        },
        "expr": {
            "tool_calls_n": expr_summary.get("tool_calls_n"),
            "accepted_n": expr_summary.get("accepted_n"),
            "events_n": len(timeline),
            "timeline": timeline,
        },
        "outputs": {
            "audio_response_pcm": str(audio_response_pcm),
            "mouth_raw_json": str(raw_json),
            "mouth_json": str(mouth_json),
            "audio_meta_json": str(audio_meta_json),
            "expr_json": str(expr_json),
        },
    }

    summary_json = out_root / "run_audio_input_smoke_pipeline.summary.json"
    _write_json(summary_json, summary)

    print("[run_audio_input_smoke_pipeline][OK]")
    print(f"  session_id          : {args.session_id}")
    print(f"  response_audio_bytes: {summary['bridge']['response_audio_bytes']}")
    print(f"  response_audio_chunks: {summary['bridge']['response_audio_chunks']}")
    print(f"  tool_calls          : {summary['bridge']['tool_calls']}")
    print(f"  mouth_frames_n      : {summary['mouth']['frames_n']}")
    print(f"  voiced_frames_n     : {summary['mouth']['voiced_frames_n']}")
    print(f"  expr_events_n       : {summary['expr']['events_n']}")
    print(f"  mouth_json          : {mouth_json}")
    print(f"  expr_json           : {expr_json}")
    print(f"  summary_json        : {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())