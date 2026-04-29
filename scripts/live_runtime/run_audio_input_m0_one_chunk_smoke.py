#!/usr/bin/env python3
from __future__ import annotations

"""
run_audio_input_m0_one_chunk_smoke.py

目的:
- audio input chunk JSON 1個を M0 に投入し、FG PNG が生成できるか確認する。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(obj, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.png")))


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _build_env(m1_repo: Path, m3_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{m1_repo / 'src'}:{m3_repo / 'src'}:"
        + env.get("PYTHONPATH", "")
    )
    return env


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one M0 chunk from audio input chunks smoke"
    )

    ap.add_argument("--session_id", default="sess_audio_input_m0_chunk_001")
    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m0_repo_root", default="/workspaces/M0_session_renderer_final_1")

    ap.add_argument("--chunks_dir", required=True)
    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument(
        "--m0_base_config",
        default="/workspaces/M0_session_renderer_final_1/configs/smoke_pose_improved.yaml",
    )
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--chunk_len_ms", type=int, default=400)

    args = ap.parse_args()

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    python_exe = sys.executable
    env = _build_env(m1_repo, m3_repo)

    chunks_dir = Path(args.chunks_dir).resolve()
    chunk_dir = chunks_dir / f"{int(args.chunk_id):06d}"

    pose_json = chunk_dir / "pose.chunk.json"
    mouth_json = chunk_dir / "mouth.chunk.json"
    expr_json = chunk_dir / "expr.chunk.json"

    for p in [pose_json, mouth_json, expr_json]:
        if not p.exists():
            raise FileNotFoundError(f"missing chunk input: {p}")

    out_dir = Path(args.out_dir).resolve()
    run_dir = out_dir / f"chunk_{int(args.chunk_id):06d}"
    fg_dir = run_dir / "fg" / "in" / "fg"
    fg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = Path(args.m0_base_config).resolve()
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"missing m0_base_config: {base_cfg_path}")

    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    cfg = json.loads(json.dumps(base_cfg))

    cfg.setdefault("io", {})
    cfg.setdefault("video", {})
    cfg.setdefault("render", {})
    cfg.setdefault("inputs", {})
    cfg.setdefault("atlas", {})

    cfg["io"]["assets_dir"] = str((m0_repo / "assets" / "sprites").resolve())
    cfg["io"]["out_dir"] = str(run_dir.resolve())
    cfg["io"]["exp_name"] = "audio_input_m0_one_chunk"

    cfg["video"]["fps"] = int(args.fps)
    cfg["video"]["duration_s"] = float(int(args.chunk_len_ms) / 1000.0)

    cfg["render"]["dump_fg_png"] = True
    cfg["render"]["fg_png_dir"] = str(fg_dir.resolve())

    cfg["atlas"]["atlas_json"] = str((m0_repo / "assets" / "atlas.min.json").resolve())
    cfg["atlas"]["affine_points_yaml"] = str((m0_repo / "configs" / "affine_points.yaml").resolve())
    cfg["render"]["affine_points_yaml_rel"] = str((m0_repo / "configs" / "affine_points.yaml").resolve())

    cfg["inputs"]["pose_timeline"] = str(pose_json.resolve())
    cfg["inputs"]["mouth_timeline"] = str(mouth_json.resolve())
    cfg["inputs"]["expression_timeline"] = str(expr_json.resolve())

    cfg_path = run_dir / "m0_chunk_config.yaml"
    _write_yaml(cfg_path, cfg)

    _run(
        [
            python_exe,
            str((m0_repo / "src" / "m0_runner.py").resolve()),
            "--config",
            str(cfg_path),
        ],
        cwd=m0_repo,
        env=env,
    )

    png_n = _count_pngs(fg_dir)
    expected_frames = int(round(int(args.chunk_len_ms) / 1000.0 * int(args.fps)))

    summary = {
        "format": "audio_input_m0_one_chunk_smoke.summary.v0",
        "session_id": str(args.session_id),
        "chunk_id": int(args.chunk_id),
        "expected_frames": expected_frames,
        "png_n": png_n,
        "inputs": {
            "pose_json": str(pose_json),
            "mouth_json": str(mouth_json),
            "expr_json": str(expr_json),
            "m0_base_config": str(base_cfg_path),
        },
        "outputs": {
            "run_dir": str(run_dir),
            "fg_dir": str(fg_dir),
            "config_yaml": str(cfg_path),
        },
    }

    summary_json = run_dir / "run_audio_input_m0_one_chunk_smoke.summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if png_n != expected_frames:
        raise RuntimeError(
            f"FG frame count mismatch: expected={expected_frames} actual={png_n} dir={fg_dir}"
        )

    print("[run_audio_input_m0_one_chunk_smoke][OK]")
    print(f"  chunk_id       : {int(args.chunk_id)}")
    print(f"  expected_frames: {expected_frames}")
    print(f"  png_n          : {png_n}")
    print(f"  fg_dir         : {fg_dir}")
    print(f"  summary_json   : {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())