#!/usr/bin/env python3
from __future__ import annotations

"""
run_audio_input_m35_compose_smoke.py

目的:
- run_audio_input_m0_all_chunks_smoke.py の global FG PNG を使い、
  M3.5 direct mode で BG 合成 MP4 を生成する。
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _build_env(m1_repo: Path, m3_repo: Path, m35_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{m1_repo / 'src'}:"
        f"{m3_repo / 'src'}:"
        f"{m35_repo}:"
        + env.get("PYTHONPATH", "")
    )
    return env


def _count_pngs(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compose audio-input FG PNGs with BG using M3.5 direct mode"
    )

    ap.add_argument("--session_id", default="sess_audio_input_m35_compose_001")
    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    ap.add_argument("--m0_all_chunks_summary_json", required=True)
    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--bg_video", default="/workspaces/M3.5_final/in/with1.mp4")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--no_color_match", action="store_true", default=True)
    ap.add_argument("--feather_px", type=int, default=0)

    args = ap.parse_args()

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()
    python_exe = sys.executable
    env = _build_env(m1_repo, m3_repo, m35_repo)

    m0_summary_json = Path(args.m0_all_chunks_summary_json).resolve()
    m0_summary = _load_json(m0_summary_json)

    fg_dir = Path(m0_summary["outputs"]["global_fg_dir"]).resolve()
    if not fg_dir.exists():
        raise FileNotFoundError(f"missing fg_dir: {fg_dir}")

    pose_json = Path(args.pose_json).resolve()
    bg_video = Path(args.bg_video).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")
    if not bg_video.exists():
        raise FileNotFoundError(f"missing bg_video: {bg_video}")

    audio_ms = int(m0_summary["audio_ms"])
    target_frames = int(m0_summary["target_frames"])
    png_n = _count_pngs(fg_dir)

    if png_n != target_frames:
        raise RuntimeError(
            f"FG count mismatch: target_frames={target_frames} png_n={png_n} fg_dir={fg_dir}"
        )

    cmd = [
        python_exe,
        "-m",
        "m3_5.cli",
        "--mode",
        "direct",
        "--pose_json",
        str(pose_json),
        "--fg_dir",
        str(fg_dir),
        "--bg_video",
        str(bg_video),
        "--out_dir",
        str(out_dir),
        "--session_audio_ms",
        str(audio_ms),
        "--feather_px",
        str(int(args.feather_px)),
    ]

    if args.no_color_match:
        cmd.append("--no-color_match")

    _run(cmd, cwd=m35_repo, env=env)

    mp4 = out_dir / "m3_5_composite.mp4"
    log_csv = out_dir / "m3_5_composite.log.csv"

    if not mp4.exists():
        raise FileNotFoundError(f"missing output mp4: {mp4}")

    summary = {
        "format": "audio_input_m35_compose_smoke.summary.v0",
        "session_id": str(args.session_id),
        "audio_ms": audio_ms,
        "target_frames": target_frames,
        "fg_png_n": png_n,
        "inputs": {
            "m0_all_chunks_summary_json": str(m0_summary_json),
            "fg_dir": str(fg_dir),
            "pose_json": str(pose_json),
            "bg_video": str(bg_video),
        },
        "outputs": {
            "out_dir": str(out_dir),
            "mp4": str(mp4),
            "log_csv": str(log_csv),
        },
    }

    summary_json = out_dir / "run_audio_input_m35_compose_smoke.summary.json"
    _write_json(summary_json, summary)

    print("[run_audio_input_m35_compose_smoke][OK]")
    print(f"  session_id   : {args.session_id}")
    print(f"  audio_ms     : {audio_ms}")
    print(f"  target_frames: {target_frames}")
    print(f"  fg_png_n     : {png_n}")
    print(f"  mp4          : {mp4}")
    print(f"  log_csv      : {log_csv}")
    print(f"  summary_json : {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())