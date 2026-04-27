#!/usr/bin/env python3
from __future__ import annotations

"""
Live Runtime Realtime v0.1

目的:
- mouth_worker
- expression_worker
- chunk_orchestrator
- M0 run_chunk (v0.1.1: individual chunk inputs)

を1コマンドで接続する。

v0.1:
- 完全常時workerではなく、安全な統合スモーク版
- 既存成功コードを壊さない
- まずは 3000ms / 8 chunks / 75 frames を固定確認する

固定:
- step_ms = 40
- chunk_len_ms = 400
- fps = 25
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


DEFAULT_STEP_MS = 40
DEFAULT_CHUNK_LEN_MS = 400
DEFAULT_FPS = 25


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd), env=env)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_env(m1_repo: Path, m3_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{m1_repo / 'src'}:{m3_repo / 'src'}:" + env.get("PYTHONPATH", "")
    return env


def _read_audio_ms(audio_meta_json: Path) -> int:
    obj = _load_json(audio_meta_json)
    return int(obj["audio_ms"])


def _copy_tree_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_m0_session_json(
    *,
    path: Path,
    session_id: str,
    audio_ms: int,
    pose_rel: str,
    mouth_rel: str,
    expr_rel: str,
    step_ms: int,
) -> None:
    _write_json(
        path,
        {
            "schema_version": "session_runtime_v0.2",
            "session_id": session_id,
            "session_audio_ms": int(audio_ms),
            "step_ms": int(step_ms),
            "pose_timeline": pose_rel,
            "mouth_timeline": mouth_rel,
            "expression_timeline": expr_rel,
            "audio_meta": {
                "audio_ms": int(audio_ms),
                "step_ms": int(step_ms),
            },
        },
    )


def _prepare_m0_inputs_first_chunk_reference(
    *,
    m0_repo: Path,
    orchestrator_chunks_root: Path,
    session_id: str,
    audio_ms: int,
    step_ms: int,
) -> Path:
    """
    v0.1:
    M0 run_chunk.py は session.json 内の timeline を読み、
    manifestに従ってsliceする。
    ここでは現行検証済みの方式として 000000 chunk を参照する。
    完全なchunk個別入力は次STEPで bridge/run_chunk方式へ寄せる。
    """
    dst_root = m0_repo / "in" / "live_runtime_realtime" / session_id / "chunk_000000"
    _ensure_dir(dst_root)

    _copy_tree_file(orchestrator_chunks_root / "000000" / "pose.chunk.json", dst_root / "pose.json")
    _copy_tree_file(orchestrator_chunks_root / "000000" / "mouth.chunk.json", dst_root / "mouth.json")
    _copy_tree_file(orchestrator_chunks_root / "000000" / "expr.chunk.json", dst_root / "expr.json")

    session_json = m0_repo / "sessions" / f"{session_id}.session.json"

    pose_rel = str((dst_root / "pose.json").relative_to(m0_repo))
    mouth_rel = str((dst_root / "mouth.json").relative_to(m0_repo))
    expr_rel = str((dst_root / "expr.json").relative_to(m0_repo))

    _write_m0_session_json(
        path=session_json,
        session_id=session_id,
        audio_ms=audio_ms,
        pose_rel=pose_rel,
        mouth_rel=mouth_rel,
        expr_rel=expr_rel,
        step_ms=step_ms,
    )

    return session_json


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def _run_m0_individual_chunks(
    *,
    python_exe: str,
    m0_repo: Path,
    session_id: str,
    orchestrator_chunks_root: Path,
    orchestrator_manifest_json: Path,
    out_fg_dir: Path,
    work_root: Path,
    base_config: str,
    fps: int,
    step_ms: int,
) -> dict[str, Any]:
    """
    chunk_orchestrator が生成した各chunk個別JSONを、
    そのままM0へ渡してFGを連番で生成する。

    入力:
      chunks/000000/pose.chunk.json
      chunks/000000/mouth.chunk.json
      chunks/000000/expr.chunk.json
      ...

    出力:
      out_fg_dir/00000000.png ... 00000074.png
    """
    manifest = _load_json(orchestrator_manifest_json)
    chunks = manifest.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"manifest chunks must be list: {orchestrator_manifest_json}")

    base_cfg_path = (m0_repo / base_config).resolve()
    base_cfg = _load_yaml(base_cfg_path)

    if out_fg_dir.exists():
        shutil.rmtree(out_fg_dir)
    if work_root.exists():
        shutil.rmtree(work_root)

    out_fg_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    rows: list[dict[str, Any]] = []

    for ch in chunks:
        cid = int(ch["chunk_id"])
        t0 = int(ch["t0_ms"])
        t1 = int(ch["t1_ms"])
        frame0 = int(ch["frame0"])
        frame1 = int(ch["frame1"])
        chunk_frames = int(frame1 - frame0)

        src_chunk_dir = orchestrator_chunks_root / f"{cid:06d}"
        pose_json = src_chunk_dir / "pose.chunk.json"
        mouth_json = src_chunk_dir / "mouth.chunk.json"
        expr_json = src_chunk_dir / "expr.chunk.json"

        if not pose_json.exists():
            raise FileNotFoundError(f"missing pose_json: {pose_json}")
        if not mouth_json.exists():
            raise FileNotFoundError(f"missing mouth_json: {mouth_json}")
        if not expr_json.exists():
            raise FileNotFoundError(f"missing expr_json: {expr_json}")

        run_chunk_dir = work_root / "chunks" / f"chunk_{cid:04d}"
        local_fg_dir = run_chunk_dir / "fg_local" / "in" / "fg"
        local_fg_dir.mkdir(parents=True, exist_ok=True)

        cfg = json.loads(json.dumps(base_cfg))
        cfg.setdefault("io", {})
        cfg.setdefault("video", {})
        cfg.setdefault("render", {})
        cfg.setdefault("inputs", {})
        cfg.setdefault("atlas", {})

        cfg["io"]["assets_dir"] = str((m0_repo / "assets" / "sprites").resolve())
        cfg["io"]["out_dir"] = str(run_chunk_dir.resolve())
        cfg["io"]["exp_name"] = "m0_chunk"

        cfg["video"]["fps"] = int(fps)
        cfg["video"]["duration_s"] = float(chunk_frames / int(fps))

        cfg["render"]["dump_fg_png"] = True
        cfg["render"]["fg_png_dir"] = str(local_fg_dir.resolve())

        cfg["atlas"]["atlas_json"] = str((m0_repo / "assets" / "atlas.min.json").resolve())
        cfg["atlas"]["affine_points_yaml"] = str((m0_repo / "configs" / "affine_points.yaml").resolve())
        cfg["render"]["affine_points_yaml_rel"] = str((m0_repo / "configs" / "affine_points.yaml").resolve())

        cfg["inputs"]["pose_timeline"] = str(pose_json.resolve())
        cfg["inputs"]["mouth_timeline"] = str(mouth_json.resolve())
        cfg["inputs"]["expression_timeline"] = str(expr_json.resolve())

        cfg_path = run_chunk_dir / "chunk_render.yaml"
        _write_yaml(cfg_path, cfg)

        cmd = [
            python_exe,
            str((m0_repo / "src" / "m0_runner.py").resolve()),
            "--config",
            str(cfg_path),
        ]

        print(
            f"[m0_individual_chunks] chunk={cid:06d} "
            f"t=[{t0},{t1}) frames=[{frame0},{frame1}) "
            f"chunk_frames={chunk_frames}"
        )
        _run(cmd, cwd=m0_repo)

        local_n = _count_pngs(local_fg_dir)
        if local_n != chunk_frames:
            raise RuntimeError(
                f"chunk {cid:06d} local frame count mismatch: "
                f"expected={chunk_frames} actual={local_n} dir={local_fg_dir}"
            )

        for i in range(chunk_frames):
            src = local_fg_dir / f"{i:08d}.png"
            dst = out_fg_dir / f"{frame0 + i:08d}.png"
            if not src.exists():
                raise FileNotFoundError(f"missing local fg: {src}")
            if dst.exists():
                raise FileExistsError(f"dst already exists: {dst}")
            shutil.copy2(src, dst)

            rows.append(
                {
                    "t_ms": int((frame0 + i) * step_ms),
                    "path": str(dst),
                }
            )
            total_copied += 1

    fg_index_csv = work_root / "fg_index.individual_chunks.csv"
    fg_index_csv.parent.mkdir(parents=True, exist_ok=True)
    with fg_index_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_ms", "path"])
        w.writeheader()
        w.writerows(rows)

    target_frames = int(manifest.get("target_frames", total_copied))

    verify_script = m0_repo / "tools" / "verify_chunk_outputs.py"
    if verify_script.exists():
        _run(
            [
                python_exe,
                str(verify_script),
                "--fg_dir",
                str(out_fg_dir),
                "--target_frames",
                str(target_frames),
            ],
            cwd=m0_repo,
        )

    return {
        "mode": "individual_chunks",
        "chunks": len(chunks),
        "target_frames": target_frames,
        "copied_frames": total_copied,
        "out_fg_dir": str(out_fg_dir),
        "fg_index_csv": str(fg_index_csv),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live Runtime Realtime v0.1 integrated smoke"
    )

    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m0_repo_root", default="/workspaces/M0_session_renderer_final_1")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    ap.add_argument("--session_id", default="sess_live_runtime_realtime_001")

    ap.add_argument(
        "--pose_json",
        default="/workspaces/M0_session_renderer_final_1/timelines/pose/pose_timeline_final_with1_4.json",
    )

    ap.add_argument("--target_audio_ms", type=int, default=3000)

    ap.add_argument(
        "--mouth_model",
        default="gemini-2.5-flash-native-audio-preview-12-2025",
    )
    ap.add_argument(
        "--mouth_prompt",
        default="元気よく短く自己紹介して！",
    )
    ap.add_argument("--mouth_dev_stop_s", type=float, default=20.0)

    ap.add_argument(
        "--expr_model",
        default="gemini-3.1-flash-live-preview",
    )
    ap.add_argument(
        "--expr_prompt",
        default="短く自己紹介して。最初に必ず set_emotion を呼び、感情は smile にしてください。",
    )
    ap.add_argument("--expr_dev_stop_s", type=float, default=12.0)
    ap.add_argument("--stop_after_first_tool_s", type=float, default=3.0)

    ap.add_argument("--api_version", default="v1beta")

    ap.add_argument("--step_ms", type=int, default=DEFAULT_STEP_MS)
    ap.add_argument("--chunk_len_ms", type=int, default=DEFAULT_CHUNK_LEN_MS)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)

    ap.add_argument("--m0_base_config", default="configs/smoke_pose_improved.yaml")
    ap.add_argument("--skip_m0", action="store_true")

    ap.add_argument(
        "--normal_bg_video",
        default="/workspaces/M3.5_final/in/with1.mp4",
    )
    ap.add_argument("--skip_m35", action="store_true")

    args = ap.parse_args()

    if int(args.step_ms) != 40:
        raise ValueError("step_ms must be 40")
    if int(args.chunk_len_ms) != 400:
        raise ValueError("chunk_len_ms must be 400")

    python_exe = sys.executable

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    session_id = str(args.session_id)

    out_root = m1_repo / "out" / "live_runtime_realtime" / session_id
    mouth_out = out_root / "mouth_worker"
    expr_out = out_root / "expression_worker"
    orch_out = out_root / "chunk_orchestrator"

    _ensure_dir(out_root)

    env = _build_env(m1_repo, m3_repo)

    # ------------------------------------------------------------
    # 1) mouth_worker
    # ------------------------------------------------------------
    _run(
        [
            python_exe,
            str(m1_repo / "scripts" / "live_runtime" / "mouth_worker.py"),
            "--session_id", session_id,
            "--out_dir", str(mouth_out),
            "--model", str(args.mouth_model),
            "--prompt", str(args.mouth_prompt),
            "--target_audio_ms", str(args.target_audio_ms),
            "--dev_stop_s", str(args.mouth_dev_stop_s),
            "--api_version", str(args.api_version),
            "--step_ms", str(args.step_ms),
        ],
        cwd=m1_repo,
        env=env,
    )

    audio_meta_json = mouth_out / "audio_meta.json"
    mouth_ring_json = mouth_out / "mouth_worker.ring.json"
    audio_ms = _read_audio_ms(audio_meta_json)

    # ------------------------------------------------------------
    # 2) expression_worker
    # ------------------------------------------------------------
    _run(
        [
            python_exe,
            str(m1_repo / "scripts" / "live_runtime" / "expression_worker.py"),
            "--session_id", session_id,
            "--out_dir", str(expr_out),
            "--model", str(args.expr_model),
            "--prompt", str(args.expr_prompt),
            "--dev_stop_s", str(args.expr_dev_stop_s),
            "--stop_after_first_tool_s", str(args.stop_after_first_tool_s),
            "--api_version", str(args.api_version),
            "--step_ms", str(args.step_ms),
        ],
        cwd=m1_repo,
        env=env,
    )

    latest_expr_json = expr_out / "expression_worker.latest.json"

    # ------------------------------------------------------------
    # 3) chunk_orchestrator
    # ------------------------------------------------------------
    _run(
        [
            python_exe,
            str(m1_repo / "scripts" / "live_runtime" / "chunk_orchestrator.py"),
            "--session_id", session_id,
            "--pose_json", str(Path(args.pose_json).resolve()),
            "--mouth_ring_json", str(mouth_ring_json),
            "--latest_expr_json", str(latest_expr_json),
            "--audio_meta_json", str(audio_meta_json),
            "--out_dir", str(orch_out),
            "--step_ms", str(args.step_ms),
            "--chunk_len_ms", str(args.chunk_len_ms),
            "--fps", str(args.fps),
        ],
        cwd=m1_repo,
        env=env,
    )

    orch_summary = _load_json(orch_out / "chunk_orchestrator.summary.json")
    chunks_n = int(orch_summary["chunks"])

    # ------------------------------------------------------------
    # 4) M0 smoke: individual chunk inputs
    # ------------------------------------------------------------
    m0_result: dict[str, Any] = {
        "skipped": bool(args.skip_m0),
    }

    if not args.skip_m0:
        orchestrator_chunks_root = orch_out / "chunks"
        orchestrator_manifest_json = orch_out / "manifest.realtime_chunks.json"

        out_fg_dir = m0_repo / "out" / "live_runtime_realtime" / session_id / "fg_individual_chunks"
        work_root = m0_repo / "out" / "live_runtime_realtime" / session_id / "m0_individual_chunk_runs"

        m0_result = _run_m0_individual_chunks(
            python_exe=python_exe,
            m0_repo=m0_repo,
            session_id=session_id,
            orchestrator_chunks_root=orchestrator_chunks_root,
            orchestrator_manifest_json=orchestrator_manifest_json,
            out_fg_dir=out_fg_dir,
            work_root=work_root,
            base_config=str(args.m0_base_config),
            fps=int(args.fps),
            step_ms=int(args.step_ms),
        )

    # ------------------------------------------------------------
    # 5) M3.5 composite
    # ------------------------------------------------------------
    m35_result: dict[str, Any] = {
        "skipped": bool(args.skip_m35),
    }

    if (not args.skip_m35) and (not args.skip_m0):
        if not m0_result.get("out_fg_dir"):
            raise RuntimeError("M3.5 requires m0_result.out_fg_dir")

        m35_out_dir = (
            m35_repo
            / "out"
            / "live_runtime_realtime"
            / session_id
            / "stage2"
        )

        _run(
            [
                python_exe,
                "-m",
                "m3_5.cli",
                "--mode",
                "direct",
                "--pose_json",
                str(Path(args.pose_json).resolve()),
                "--fg_dir",
                str(m0_result["out_fg_dir"]),
                "--bg_video",
                str(Path(args.normal_bg_video).resolve()),
                "--out_dir",
                str(m35_out_dir),
                "--session_audio_ms",
                str(audio_ms),
                "--no-color_match",
                "--feather_px",
                "0",
            ],
            cwd=m35_repo,
        )

        m35_result = {
            "skipped": False,
            "mode": "direct",
            "out_dir": str(m35_out_dir),
            "mp4": str(m35_out_dir / "m3_5_composite.mp4"),
            "log_csv": str(m35_out_dir / "m3_5_composite.log.csv"),
            "bg_video": str(Path(args.normal_bg_video).resolve()),
        }

    # ------------------------------------------------------------
    # 6) Summary
    # ------------------------------------------------------------
    summary = {
        "schema_version": "run_live_runtime_realtime_summary.v0.1",
        "session_id": session_id,
        "audio_ms": int(audio_ms),
        "step_ms": int(args.step_ms),
        "chunk_len_ms": int(args.chunk_len_ms),
        "fps": int(args.fps),
        "chunks": int(chunks_n),
        "mouth": {
            "out_dir": str(mouth_out),
            "ring_json": str(mouth_ring_json),
            "audio_meta_json": str(audio_meta_json),
        },
        "expression": {
            "out_dir": str(expr_out),
            "latest_expr_json": str(latest_expr_json),
        },
        "orchestrator": {
            "out_dir": str(orch_out),
            "summary_json": str(orch_out / "chunk_orchestrator.summary.json"),
            "manifest_json": str(orch_out / "manifest.realtime_chunks.json"),
        },
        "m0": m0_result,
        "m35": m35_result,
    }

    summary_json = out_root / "run_live_runtime_realtime.summary.json"
    _write_json(summary_json, summary)

    print("[run_live_runtime_realtime][OK]")
    print(f"  session_id   : {session_id}")
    print(f"  audio_ms     : {audio_ms}")
    print(f"  chunks       : {chunks_n}")
    print(f"  out_root     : {out_root}")
    print(f"  summary_json : {summary_json}")

    if not args.skip_m0:
        print(f"  m0_mode      : {m0_result.get('mode')}")
        print(f"  m0_fg_dir    : {m0_result.get('out_fg_dir')}")
        print(f"  m0_frames    : {m0_result.get('copied_frames')}/{m0_result.get('target_frames')}")

    if not args.skip_m35:
        print(f"  m35_out_dir  : {m35_result.get('out_dir')}")
        print(f"  m35_mp4      : {m35_result.get('mp4')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())