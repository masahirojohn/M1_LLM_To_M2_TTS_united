#!/usr/bin/env python3
from __future__ import annotations

"""
run_audio_input_m0_all_chunks_smoke.py

目的:
- audio input chunk JSON 全件を M0 に投入
- global FG PNG 連番を生成
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


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _count_pngs(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.exists() else 0


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _build_env(m1_repo: Path, m3_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{m1_repo / 'src'}:{m3_repo / 'src'}:" + env.get("PYTHONPATH", "")
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_id", default="sess_audio_input_m0_all_chunks_001")
    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m0_repo_root", default="/workspaces/M0_session_renderer_final_1")

    ap.add_argument("--chunks_summary_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument(
        "--m0_base_config",
        default="/workspaces/M0_session_renderer_final_1/configs/smoke_pose_improved.yaml",
    )
    ap.add_argument("--clean", action="store_true")

    args = ap.parse_args()

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    env = _build_env(m1_repo, m3_repo)
    python_exe = sys.executable

    chunks_summary_json = Path(args.chunks_summary_json).resolve()
    chunks_summary = _load_json(chunks_summary_json)

    out_dir = Path(args.out_dir).resolve()
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_fg_dir = out_dir / "fg_streaming" / "in" / "fg"
    global_fg_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = Path(args.m0_base_config).resolve()
    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    chunk_results: list[dict[str, Any]] = []
    total_copied = 0

    for ch in chunks_summary["chunks"]:
        cid = int(ch["chunk_id"])
        frame0 = int(ch["frame0"])
        frame1 = int(ch["frame1"])
        expected_frames = frame1 - frame0

        pose_json = Path(ch["pose_chunk_json"]).resolve()
        mouth_json = Path(ch["mouth_chunk_json"]).resolve()
        expr_json = Path(ch["expr_chunk_json"]).resolve()

        run_dir = out_dir / "m0_chunks" / f"chunk_{cid:06d}"
        local_fg_dir = run_dir / "fg_local" / "in" / "fg"
        local_fg_dir.mkdir(parents=True, exist_ok=True)

        cfg = json.loads(json.dumps(base_cfg))
        cfg.setdefault("io", {})
        cfg.setdefault("video", {})
        cfg.setdefault("render", {})
        cfg.setdefault("inputs", {})
        cfg.setdefault("atlas", {})

        cfg["io"]["assets_dir"] = str((m0_repo / "assets" / "sprites").resolve())
        cfg["io"]["out_dir"] = str(run_dir.resolve())
        cfg["io"]["exp_name"] = "audio_input_m0_all_chunks"

        cfg["video"]["fps"] = int(chunks_summary["fps"])
        cfg["video"]["duration_s"] = float(expected_frames / int(chunks_summary["fps"]))

        cfg["render"]["dump_fg_png"] = True
        cfg["render"]["fg_png_dir"] = str(local_fg_dir.resolve())

        cfg["atlas"]["atlas_json"] = str((m0_repo / "assets" / "atlas.min.json").resolve())
        cfg["atlas"]["affine_points_yaml"] = str((m0_repo / "configs" / "affine_points.yaml").resolve())
        cfg["render"]["affine_points_yaml_rel"] = str((m0_repo / "configs" / "affine_points.yaml").resolve())

        cfg["inputs"]["pose_timeline"] = str(pose_json)
        cfg["inputs"]["mouth_timeline"] = str(mouth_json)
        cfg["inputs"]["expression_timeline"] = str(expr_json)

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

        local_n = _count_pngs(local_fg_dir)
        if local_n != expected_frames:
            raise RuntimeError(
                f"chunk {cid:06d} frame mismatch: expected={expected_frames} actual={local_n}"
            )

        copied = 0
        for i in range(expected_frames):
            src = local_fg_dir / f"{i:08d}.png"
            dst = global_fg_dir / f"{frame0 + i:08d}.png"
            if not src.exists():
                raise FileNotFoundError(f"missing local fg: {src}")
            shutil.copy2(src, dst)
            copied += 1

        total_copied += copied
        chunk_results.append(
            {
                "chunk_id": cid,
                "frame0": frame0,
                "frame1": frame1,
                "expected_frames": expected_frames,
                "local_png_n": local_n,
                "copied_frames": copied,
                "run_dir": str(run_dir),
                "local_fg_dir": str(local_fg_dir),
            }
        )

    global_png_n = _count_pngs(global_fg_dir)
    target_frames = int(chunks_summary["target_frames"])

    if global_png_n != target_frames:
        raise RuntimeError(
            f"global FG mismatch: expected={target_frames} actual={global_png_n}"
        )

    summary = {
        "format": "audio_input_m0_all_chunks_smoke.summary.v0",
        "session_id": str(args.session_id),
        "audio_ms": int(chunks_summary["audio_ms"]),
        "step_ms": int(chunks_summary["step_ms"]),
        "chunk_len_ms": int(chunks_summary["chunk_len_ms"]),
        "fps": int(chunks_summary["fps"]),
        "target_frames": target_frames,
        "chunks_n": len(chunk_results),
        "global_png_n": global_png_n,
        "outputs": {
            "global_fg_dir": str(global_fg_dir),
        },
        "chunks": chunk_results,
    }

    summary_json = out_dir / "run_audio_input_m0_all_chunks_smoke.summary.json"
    _write_json(summary_json, summary)

    print("[run_audio_input_m0_all_chunks_smoke][OK]")
    print(f"  session_id    : {args.session_id}")
    print(f"  audio_ms      : {summary['audio_ms']}")
    print(f"  chunks_n      : {summary['chunks_n']}")
    print(f"  target_frames : {target_frames}")
    print(f"  global_png_n  : {global_png_n}")
    print(f"  global_fg_dir : {global_fg_dir}")
    print(f"  summary_json  : {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())