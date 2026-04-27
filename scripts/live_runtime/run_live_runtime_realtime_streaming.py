#!/usr/bin/env python3
from __future__ import annotations

"""
Live Runtime Realtime Streaming v0

目的:
- run_live_runtime_realtime.py で合格した流れをベースに、
  chunk_orchestrator が作った chunks/000000... を
  400ms単位で M0 → M3.5 へ逐次処理する。

注意:
- v0 は「chunk逐次実行」版。
- Live API受信とM0/M3.5を完全並列常時worker化する前段。
- 既存成功コードを壊さないため、まずは安全な段階実装。

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
import time
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


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(obj, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_env(m1_repo: Path, m3_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{m1_repo / 'src'}:{m3_repo / 'src'}:" + env.get("PYTHONPATH", "")
    return env


def _read_audio_ms(audio_meta_json: Path) -> int:
    obj = _load_json(audio_meta_json)
    return int(obj["audio_ms"])


def _count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob("*.png")))


def _run_m0_one_chunk(
    *,
    python_exe: str,
    m0_repo: Path,
    base_cfg: dict[str, Any],
    src_chunk_dir: Path,
    out_fg_dir: Path,
    work_root: Path,
    cid: int,
    t0: int,
    t1: int,
    frame0: int,
    frame1: int,
    fps: int,
) -> dict[str, Any]:
    chunk_frames = int(frame1 - frame0)

    pose_json = src_chunk_dir / "pose.chunk.json"
    mouth_json = src_chunk_dir / "mouth.chunk.json"
    expr_json = src_chunk_dir / "expr.chunk.json"

    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")
    if not mouth_json.exists():
        raise FileNotFoundError(f"missing mouth_json: {mouth_json}")
    if not expr_json.exists():
        raise FileNotFoundError(f"missing expr_json: {expr_json}")

    run_chunk_dir = work_root / "m0_chunks" / f"chunk_{cid:04d}"
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

    print(
        f"[streaming_m0] chunk={cid:06d} "
        f"t=[{t0},{t1}) frames=[{frame0},{frame1}) "
        f"chunk_frames={chunk_frames}"
    )

    _run(
        [
            python_exe,
            str((m0_repo / "src" / "m0_runner.py").resolve()),
            "--config",
            str(cfg_path),
        ],
        cwd=m0_repo,
    )

    local_n = _count_pngs(local_fg_dir)
    if local_n != chunk_frames:
        raise RuntimeError(
            f"chunk {cid:06d} local frame count mismatch: "
            f"expected={chunk_frames} actual={local_n} dir={local_fg_dir}"
        )

    copied = 0
    for i in range(chunk_frames):
        src = local_fg_dir / f"{i:08d}.png"
        dst = out_fg_dir / f"{frame0 + i:08d}.png"
        if not src.exists():
            raise FileNotFoundError(f"missing local fg: {src}")
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        copied += 1

    return {
        "chunk_id": cid,
        "t0_ms": t0,
        "t1_ms": t1,
        "frame0": frame0,
        "frame1": frame1,
        "chunk_frames": chunk_frames,
        "copied_frames": copied,
        "local_fg_dir": str(local_fg_dir),
    }


def _run_m35_one_chunk(
    *,
    python_exe: str,
    m35_repo: Path,
    pose_json: Path,
    fg_dir: Path,
    bg_video: Path,
    out_dir: Path,
    chunk_audio_ms: int,
) -> dict[str, Any]:
    _run(
        [
            python_exe,
            "-m",
            "m3_5.cli",
            "--mode",
            "direct",
            "--pose_json",
            str(pose_json.resolve()),
            "--fg_dir",
            str(fg_dir.resolve()),
            "--bg_video",
            str(bg_video.resolve()),
            "--out_dir",
            str(out_dir.resolve()),
            "--session_audio_ms",
            str(chunk_audio_ms),
            "--no-color_match",
            "--feather_px",
            "0",
        ],
        cwd=m35_repo,
    )

    return {
        "out_dir": str(out_dir),
        "mp4": str(out_dir / "m3_5_composite.mp4"),
        "log_csv": str(out_dir / "m3_5_composite.log.csv"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live Runtime Realtime Streaming v0"
    )

    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m0_repo_root", default="/workspaces/M0_session_renderer_final_1")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    ap.add_argument("--session_id", default="sess_live_runtime_realtime_streaming_001")
    ap.add_argument(
        "--pose_json",
        default="/workspaces/M0_session_renderer_final_1/timelines/pose/pose_timeline_final_with1_4.json",
    )
    ap.add_argument(
        "--normal_bg_video",
        default="/workspaces/M3.5_final/in/with1.mp4",
    )

    ap.add_argument("--target_audio_ms", type=int, default=3000)

    ap.add_argument("--mouth_model", default="gemini-2.5-flash-native-audio-preview-12-2025")
    ap.add_argument("--mouth_prompt", default="元気よく短く自己紹介して！")
    ap.add_argument("--mouth_dev_stop_s", type=float, default=20.0)

    ap.add_argument("--expr_model", default="gemini-3.1-flash-live-preview")
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
    ap.add_argument("--skip_m35", action="store_true")

    # デバッグ用：M0/M3.5を最速で回す。実時間sleepしない。
    ap.add_argument("--no_realtime_sleep", action="store_true")

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

    out_root = m1_repo / "out" / "live_runtime_realtime_streaming" / session_id
    mouth_out = out_root / "mouth_worker"
    expr_out = out_root / "expression_worker"
    orch_out = out_root / "chunk_orchestrator"

    m0_out_fg_dir = m0_repo / "out" / "live_runtime_realtime_streaming" / session_id / "fg_streaming"
    m0_work_root = m0_repo / "out" / "live_runtime_realtime_streaming" / session_id / "chunk_runs"
    m35_out_root = m35_repo / "out" / "live_runtime_realtime_streaming" / session_id

    _ensure_dir(out_root)

    env = _build_env(m1_repo, m3_repo)

    # ------------------------------------------------------------
    # 1) mouth_worker
    # ------------------------------------------------------------
    _run(
        [
            python_exe,
            str(m1_repo / "scripts" / "live_runtime" / "mouth_worker.py"),
            "--session_id",
            session_id,
            "--out_dir",
            str(mouth_out),
            "--model",
            str(args.mouth_model),
            "--prompt",
            str(args.mouth_prompt),
            "--target_audio_ms",
            str(args.target_audio_ms),
            "--dev_stop_s",
            str(args.mouth_dev_stop_s),
            "--api_version",
            str(args.api_version),
            "--step_ms",
            str(args.step_ms),
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
            "--session_id",
            session_id,
            "--out_dir",
            str(expr_out),
            "--model",
            str(args.expr_model),
            "--prompt",
            str(args.expr_prompt),
            "--dev_stop_s",
            str(args.expr_dev_stop_s),
            "--stop_after_first_tool_s",
            str(args.stop_after_first_tool_s),
            "--api_version",
            str(args.api_version),
            "--step_ms",
            str(args.step_ms),
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
            "--session_id",
            session_id,
            "--pose_json",
            str(Path(args.pose_json).resolve()),
            "--mouth_ring_json",
            str(mouth_ring_json),
            "--latest_expr_json",
            str(latest_expr_json),
            "--audio_meta_json",
            str(audio_meta_json),
            "--out_dir",
            str(orch_out),
            "--step_ms",
            str(args.step_ms),
            "--chunk_len_ms",
            str(args.chunk_len_ms),
            "--fps",
            str(args.fps),
        ],
        cwd=m1_repo,
        env=env,
    )

    manifest_json = orch_out / "manifest.realtime_chunks.json"
    manifest = _load_json(manifest_json)
    chunks = manifest.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"manifest chunks must be list: {manifest_json}")

    orchestrator_chunks_root = orch_out / "chunks"

    if m0_out_fg_dir.exists():
        shutil.rmtree(m0_out_fg_dir)
    if m0_work_root.exists():
        shutil.rmtree(m0_work_root)
    m0_out_fg_dir.mkdir(parents=True, exist_ok=True)
    m0_work_root.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml((m0_repo / args.m0_base_config).resolve())

    # ------------------------------------------------------------
    # 4) streaming loop: 400ms chunkごとに M0 -> M3.5
    # ------------------------------------------------------------
    chunk_results: list[dict[str, Any]] = []
    fg_rows: list[dict[str, Any]] = []

    stream_t0 = time.monotonic()

    for ch in chunks:
        cid = int(ch["chunk_id"])
        t0 = int(ch["t0_ms"])
        t1 = int(ch["t1_ms"])
        frame0 = int(ch["frame0"])
        frame1 = int(ch["frame1"])

        if not args.no_realtime_sleep:
            target_s = t0 / 1000.0
            elapsed_s = time.monotonic() - stream_t0
            sleep_s = target_s - elapsed_s
            if sleep_s > 0:
                time.sleep(sleep_s)

        src_chunk_dir = orchestrator_chunks_root / f"{cid:06d}"

        m0_res = _run_m0_one_chunk(
            python_exe=python_exe,
            m0_repo=m0_repo,
            base_cfg=base_cfg,
            src_chunk_dir=src_chunk_dir,
            out_fg_dir=m0_out_fg_dir,
            work_root=m0_work_root,
            cid=cid,
            t0=t0,
            t1=t1,
            frame0=frame0,
            frame1=frame1,
            fps=int(args.fps),
        )

        for i in range(frame0, frame1):
            fg_rows.append(
                {
                    "t_ms": int(i * int(args.step_ms)),
                    "path": str(m0_out_fg_dir / f"{i:08d}.png"),
                }
            )

        m35_res: dict[str, Any] = {"skipped": bool(args.skip_m35)}
        if not args.skip_m35:
            m35_chunk_out = m35_out_root / "chunks" / f"{cid:06d}" / "stage2"

            # M3.5 directはfg_dir全体を見るので、v0では蓄積済みfg_dirを渡す。
            # session_audio_ms は該当chunk末尾までにする。
            m35_res = _run_m35_one_chunk(
                python_exe=python_exe,
                m35_repo=m35_repo,
                pose_json=Path(args.pose_json).resolve(),
                fg_dir=m0_out_fg_dir,
                bg_video=Path(args.normal_bg_video).resolve(),
                out_dir=m35_chunk_out,
                chunk_audio_ms=int(t1),
            )

        chunk_results.append(
            {
                "chunk_id": cid,
                "t0_ms": t0,
                "t1_ms": t1,
                "frame0": frame0,
                "frame1": frame1,
                "m0": m0_res,
                "m35": m35_res,
            }
        )

    # ------------------------------------------------------------
    # 5) verify final FG
    # ------------------------------------------------------------
    target_frames = int(manifest.get("target_frames", 0))

    verify_script = m0_repo / "tools" / "verify_chunk_outputs.py"
    if verify_script.exists():
        _run(
            [
                python_exe,
                str(verify_script),
                "--fg_dir",
                str(m0_out_fg_dir),
                "--target_frames",
                str(target_frames),
            ],
            cwd=m0_repo,
        )

    fg_index_csv = m0_work_root / "fg_index.streaming.csv"
    with fg_index_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t_ms", "path"])
        w.writeheader()
        w.writerows(fg_rows)

    # ------------------------------------------------------------
    # 6) final M3.5 composite: 1本MP4
    # ------------------------------------------------------------
    final_m35_result: dict[str, Any] = {
        "skipped": bool(args.skip_m35),
    }

    if not args.skip_m35:
        final_m35_out_dir = m35_out_root / "final" / "stage2"

        final_m35_result = _run_m35_one_chunk(
            python_exe=python_exe,
            m35_repo=m35_repo,
            pose_json=Path(args.pose_json).resolve(),
            fg_dir=m0_out_fg_dir,
            bg_video=Path(args.normal_bg_video).resolve(),
            out_dir=final_m35_out_dir,
            chunk_audio_ms=int(audio_ms),
        )    

    summary = {
        "schema_version": "run_live_runtime_realtime_streaming_summary.v0",
        "session_id": session_id,
        "audio_ms": int(audio_ms),
        "step_ms": int(args.step_ms),
        "chunk_len_ms": int(args.chunk_len_ms),
        "fps": int(args.fps),
        "chunks": len(chunks),
        "target_frames": target_frames,
        "m0": {
            "out_fg_dir": str(m0_out_fg_dir),
            "fg_index_csv": str(fg_index_csv),
        },
        "m35": {
            "skipped": bool(args.skip_m35),
            "out_root": str(m35_out_root),
        },
        "final_m35": final_m35_result,
        "chunk_results": chunk_results,
    }

    summary_json = out_root / "run_live_runtime_realtime_streaming.summary.json"
    _write_json(summary_json, summary)

    print("[run_live_runtime_realtime_streaming][OK]")
    print(f"  session_id   : {session_id}")
    print(f"  audio_ms     : {audio_ms}")
    print(f"  chunks       : {len(chunks)}")
    print(f"  target_frames: {target_frames}")
    print(f"  m0_fg_dir    : {m0_out_fg_dir}")
    print(f"  fg_index_csv : {fg_index_csv}")
    print(f"  m35_out_root : {m35_out_root}")
    if not args.skip_m35:
        print(f"  final_mp4    : {final_m35_result.get('mp4')}")
    print(f"  summary_json : {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())