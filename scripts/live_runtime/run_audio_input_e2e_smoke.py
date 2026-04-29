#!/usr/bin/env python3
from __future__ import annotations

"""
run_audio_input_e2e_smoke.py

目的:
Codespacesで可能な音声入力版E2Eスモークを1コマンド化する。

Flow:
1. run_audio_input_smoke_pipeline.py
   WAV/PCM → Live API → audio_response.pcm / mouth.json / expr.json

2. run_audio_input_to_chunks_smoke.py
   mouth.json / expr.json / pose_json → chunk JSON

3. run_audio_input_m0_all_chunks_smoke.py
   chunk JSON → M0 global FG PNG

4. run_audio_input_m35_compose_smoke.py
   FG PNG + BG動画 → M3.5合成MP4
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
    if not path.exists():
        raise FileNotFoundError(f"missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_env(m1_repo: Path, m3_repo: Path, m35_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{m1_repo / 'src'}:"
        f"{m3_repo / 'src'}:"
        f"{m35_repo}:"
        + env.get("PYTHONPATH", "")
    )
    return env


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run audio input E2E smoke: Live API audio input -> M3.5 MP4"
    )

    ap.add_argument("--session_id", default="sess_audio_input_e2e_001")

    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m0_repo_root", default="/workspaces/M0_session_renderer_final_1")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    ap.add_argument("--input_wav", default=None)
    ap.add_argument("--input_pcm", default=None)
    ap.add_argument("--input_sr", type=int, default=16000)

    ap.add_argument(
        "--pose_json",
        default="/workspaces/M0_session_renderer_final_1/timelines/pose/pose_timeline_final_with1_4.json",
    )
    ap.add_argument("--bg_video", default="/workspaces/M3.5_final/in/with1.mp4")

    ap.add_argument("--model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--api_version", default="v1alpha")

    ap.add_argument("--duration_s", type=float, default=10.0)
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--chunk_len_ms", type=int, default=400)
    ap.add_argument("--fps", type=int, default=25)

    ap.add_argument("--out_root", default=None)
    ap.add_argument("--clean", action="store_true")

    args = ap.parse_args()

    if not args.input_wav and not args.input_pcm:
        raise RuntimeError("--input_wav or --input_pcm is required")

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    python_exe = sys.executable
    env = _build_env(m1_repo, m3_repo, m35_repo)

    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else m1_repo / "out" / "audio_input_e2e_smoke" / args.session_id
    )
    out_root.mkdir(parents=True, exist_ok=True)

    pipeline_dir = out_root / "01_audio_input_smoke_pipeline"
    chunks_dir = out_root / "02_audio_input_to_chunks"
    m0_dir = out_root / "03_m0_all_chunks"
    m35_dir = out_root / "04_m35_compose"

    script_pipeline = m1_repo / "scripts" / "live_runtime" / "run_audio_input_smoke_pipeline.py"
    script_chunks = m1_repo / "scripts" / "live_runtime" / "run_audio_input_to_chunks_smoke.py"
    script_m0 = m1_repo / "scripts" / "live_runtime" / "run_audio_input_m0_all_chunks_smoke.py"
    script_m35 = m1_repo / "scripts" / "live_runtime" / "run_audio_input_m35_compose_smoke.py"

    for p in [script_pipeline, script_chunks, script_m0, script_m35]:
        if not p.exists():
            raise FileNotFoundError(f"missing script: {p}")

    # 1. Audio input smoke pipeline
    cmd1 = [
        python_exe,
        str(script_pipeline),
        "--session_id",
        args.session_id,
        "--m1_repo_root",
        str(m1_repo),
        "--m3_repo_root",
        str(m3_repo),
        "--out_root",
        str(pipeline_dir),
        "--model",
        str(args.model),
        "--api_version",
        str(args.api_version),
        "--duration_s",
        str(float(args.duration_s)),
        "--step_ms",
        str(int(args.step_ms)),
    ]

    if args.input_wav:
        cmd1 += ["--input_wav", str(Path(args.input_wav).resolve())]
    else:
        cmd1 += [
            "--input_pcm",
            str(Path(args.input_pcm).resolve()),
            "--input_sr",
            str(int(args.input_sr)),
        ]

    _run(cmd1, cwd=m1_repo, env=env)

    pipeline_summary = pipeline_dir / "run_audio_input_smoke_pipeline.summary.json"

    # 2. outputs -> chunk JSON
    _run(
        [
            python_exe,
            str(script_chunks),
            "--session_id",
            args.session_id,
            "--pipeline_dir",
            str(pipeline_dir),
            "--pose_json",
            str(Path(args.pose_json).resolve()),
            "--out_dir",
            str(chunks_dir),
            "--step_ms",
            str(int(args.step_ms)),
            "--chunk_len_ms",
            str(int(args.chunk_len_ms)),
            "--fps",
            str(int(args.fps)),
        ],
        cwd=m1_repo,
        env=env,
    )

    chunks_summary = chunks_dir / "run_audio_input_to_chunks_smoke.summary.json"

    # 3. chunk JSON -> M0 global FG
    cmd3 = [
        python_exe,
        str(script_m0),
        "--session_id",
        args.session_id,
        "--m1_repo_root",
        str(m1_repo),
        "--m3_repo_root",
        str(m3_repo),
        "--m0_repo_root",
        str(m0_repo),
        "--chunks_summary_json",
        str(chunks_summary),
        "--out_dir",
        str(m0_dir),
    ]
    if args.clean:
        cmd3.append("--clean")

    _run(cmd3, cwd=m1_repo, env=env)

    m0_summary = m0_dir / "run_audio_input_m0_all_chunks_smoke.summary.json"

    # 4. M3.5 compose
    _run(
        [
            python_exe,
            str(script_m35),
            "--session_id",
            args.session_id,
            "--m1_repo_root",
            str(m1_repo),
            "--m3_repo_root",
            str(m3_repo),
            "--m35_repo_root",
            str(m35_repo),
            "--m0_all_chunks_summary_json",
            str(m0_summary),
            "--pose_json",
            str(Path(args.pose_json).resolve()),
            "--bg_video",
            str(Path(args.bg_video).resolve()),
            "--out_dir",
            str(m35_dir),
        ],
        cwd=m1_repo,
        env=env,
    )

    m35_summary = m35_dir / "run_audio_input_m35_compose_smoke.summary.json"

    pipeline_obj = _load_json(pipeline_summary)
    chunks_obj = _load_json(chunks_summary)
    m0_obj = _load_json(m0_summary)
    m35_obj = _load_json(m35_summary)

    summary = {
        "format": "audio_input_e2e_smoke.summary.v0",
        "session_id": str(args.session_id),
        "step_ms": int(args.step_ms),
        "chunk_len_ms": int(args.chunk_len_ms),
        "fps": int(args.fps),
        "bridge": pipeline_obj.get("bridge", {}),
        "mouth": pipeline_obj.get("mouth", {}),
        "expr": pipeline_obj.get("expr", {}),
        "chunks": {
            "audio_ms": chunks_obj.get("audio_ms"),
            "target_frames": chunks_obj.get("target_frames"),
            "chunks_n": chunks_obj.get("chunks_n"),
        },
        "m0": {
            "global_png_n": m0_obj.get("global_png_n"),
            "target_frames": m0_obj.get("target_frames"),
        },
        "m35": {
            "fg_png_n": m35_obj.get("fg_png_n"),
            "mp4": m35_obj.get("outputs", {}).get("mp4"),
            "log_csv": m35_obj.get("outputs", {}).get("log_csv"),
        },
        "outputs": {
            "out_root": str(out_root),
            "pipeline_summary": str(pipeline_summary),
            "chunks_summary": str(chunks_summary),
            "m0_summary": str(m0_summary),
            "m35_summary": str(m35_summary),
            "final_mp4": m35_obj.get("outputs", {}).get("mp4"),
        },
    }

    summary_json = out_root / "run_audio_input_e2e_smoke.summary.json"
    _write_json(summary_json, summary)

    print("[run_audio_input_e2e_smoke][OK]")
    print(f"  session_id          : {args.session_id}")
    print(f"  response_audio_bytes: {summary['bridge'].get('response_audio_bytes')}")
    print(f"  tool_calls          : {summary['bridge'].get('tool_calls')}")
    print(f"  mouth_frames_n      : {summary['mouth'].get('frames_n')}")
    print(f"  voiced_frames_n     : {summary['mouth'].get('voiced_frames_n')}")
    print(f"  expr_events_n       : {summary['expr'].get('events_n')}")
    print(f"  chunks_n            : {summary['chunks'].get('chunks_n')}")
    print(f"  target_frames       : {summary['chunks'].get('target_frames')}")
    print(f"  global_png_n        : {summary['m0'].get('global_png_n')}")
    print(f"  final_mp4           : {summary['outputs'].get('final_mp4')}")
    print(f"  summary_json        : {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())