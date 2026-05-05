#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(obj, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _count_pngs(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.exists() else 0


def _safe_clean_dir(path: Path, *, required_parent: Path) -> None:
    path = path.resolve()
    required_parent = required_parent.resolve()

    if required_parent not in path.parents and path != required_parent:
        raise ValueError(f"refuse to clean unsafe path: {path}")

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _build_env(m1_repo: Path, m3_repo: Path, m35_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(m1_repo / "src"),
            str(m3_repo / "src"),
            str(m35_repo),
            env.get("PYTHONPATH", ""),
        ]
    )
    return env


def _play_audio_async(
    *,
    py: Path,
    audio_script: Path,
    pcm: Path,
    audio_device: str,
    cwd: Path,
    env: dict[str, str],
) -> threading.Thread:
    def _target() -> None:
        if not pcm.exists():
            raise FileNotFoundError(f"missing pcm: {pcm}")

        _run(
            [
                str(py),
                str(audio_script),
                "--pcm",
                str(pcm),
                "--sr",
                "24000",
                "--device",
                str(audio_device),
            ],
            cwd=cwd,
            env=env,
        )

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    return th


# B. _run_m0_one_chunk() 引数追加
def _run_m0_one_chunk(
    *,
    py: Path,
    m0_repo: Path,
    m1_repo: Path,
    m3_repo: Path,
    base_cfg: dict[str, Any],
    chunk: dict[str, Any],
    chunks_summary: dict[str, Any],
    work_dir: Path,
    watch_fg_dir: Path,
    env: dict[str, str],
    frame_offset: int,
) -> int:
    cid = int(chunk["chunk_id"])
    frame0 = int(chunk["frame0"])
    frame1 = int(chunk["frame1"])
    expected_frames = frame1 - frame0

    pose_json = Path(chunk["pose_chunk_json"]).resolve()
    mouth_json = Path(chunk["mouth_chunk_json"]).resolve()
    expr_json = Path(chunk["expr_chunk_json"]).resolve()

    run_dir = work_dir / f"chunk_{cid:06d}"
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
    cfg["io"]["exp_name"] = "realtime_step1_chunk"

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
            str(py),
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
    watch_fg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(expected_frames):
        src = local_fg_dir / f"{i:08d}.png"
        # C. PNG出力番号を変更
        dst = watch_fg_dir / f"{frame_offset + frame0 + i:08d}.png"
        if not src.exists():
            raise FileNotFoundError(f"missing local fg: {src}")
        shutil.copy2(src, dst)
        copied += 1

    # D. ログも変更
    print(
        f"[realtime_step1][m0_chunk_done] "
        f"idx={cid} frames={copied} "
        f"global_range=[{frame_offset + frame0},{frame_offset + frame1})",
        flush=True,
    )

    return copied


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Realtime step1: mic pipeline -> chunks -> M0 per chunk -> persistent OBS"
    )

    ap.add_argument("--session_id", required=True)

    ap.add_argument("--m1_repo_root", required=True)
    ap.add_argument("--m3_repo_root", required=True)
    ap.add_argument("--m0_repo_root", required=True)
    ap.add_argument("--m35_repo_root", required=True)

    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--bg_video", required=True)

    ap.add_argument("--duration_s", type=float, default=5.0)
    ap.add_argument("--audio_device", default="15")

    ap.add_argument("--model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--api_version", default="v1alpha")
    ap.add_argument("--response_trigger", default="短く返答してください。返答前にset_emotionを1回呼んでください")

    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--chunk_len_ms", type=int, default=400)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--m0_base_config", default=None)
    ap.add_argument("--watch_fg_dir", default=None)
    ap.add_argument("--clean", action="store_true")

    # A. 引数追加
    ap.add_argument("--external_virtualcam", action="store_true")
    ap.add_argument("--frame_offset", type=int, default=0)

    ap.add_argument("--chunk_display_gap_s", type=float, default=0.0)

    args = ap.parse_args()

    if int(args.step_ms) != 40:
        raise ValueError("step_ms must be 40")
    if int(args.chunk_len_ms) != 400:
        raise ValueError("chunk_len_ms must be 400")
    if int(args.fps) != 25:
        raise ValueError("fps must be 25")

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    py = m1_repo / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        raise FileNotFoundError(f"missing python exe: {py}")

    pose_json = Path(args.pose_json).resolve()
    bg_video = Path(args.bg_video).resolve()

    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")
    if not bg_video.exists():
        raise FileNotFoundError(f"missing bg_video: {bg_video}")

    env = _build_env(m1_repo, m3_repo, m35_repo)

    out_root = m1_repo / "out" / "obs_realtime_step1" / args.session_id
    pipeline_dir = out_root / "01_audio_input_smoke_pipeline"
    chunks_dir = out_root / "02_audio_input_to_chunks"
    m0_work_dir = out_root / "03_m0_chunks_work"

    obs_stream_root = m1_repo / "out" / "obs_stream_step1"
    watch_fg_dir = (
        Path(args.watch_fg_dir).resolve()
        if args.watch_fg_dir
        else obs_stream_root / "fg"
    )

    if args.clean:
        _safe_clean_dir(out_root, required_parent=m1_repo / "out")
        _safe_clean_dir(watch_fg_dir, required_parent=m1_repo / "out")

    pipeline_script = m1_repo / "scripts" / "live_runtime" / "run_audio_input_smoke_pipeline.py"
    chunks_script = m1_repo / "scripts" / "live_runtime" / "run_audio_input_to_chunks_smoke.py"
    virtualcam_script = m1_repo / "scripts" / "live_runtime" / "run_virtualcam_persistent.py"
    audio_script = m1_repo / "scripts" / "live_runtime" / "dev_play_pcm_to_vbcable.py"

    for p in [pipeline_script, chunks_script, virtualcam_script, audio_script]:
        if not p.exists():
            raise FileNotFoundError(f"missing script: {p}")

    base_cfg_path = (
        Path(args.m0_base_config).resolve()
        if args.m0_base_config
        else m0_repo / "configs" / "smoke_pose_improved.yaml"
    )
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"missing m0_base_config: {base_cfg_path}")

    base_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))

    # E. virtualcam起動を条件分岐
    cam_proc = None

    if not args.external_virtualcam:
        print("[realtime_step1] start virtualcam persistent", flush=True)
        cam_proc = subprocess.Popen(
            [
                str(py),
                str(virtualcam_script),
                "--fg_dir",
                str(watch_fg_dir),
                "--bg_video",
                str(bg_video),
                "--fps",
                str(int(args.fps)),
                "--width",
                str(int(args.width)),
                "--height",
                str(int(args.height)),
                "--idle_hold",
                "--loop_bg",
            ],
            cwd=str(m1_repo),
            env=env,
        )
    else:
        print("[realtime_step1] external virtualcam mode", flush=True)

    try:
        time.sleep(1.0)

        print("[realtime_step1] run audio pipeline", flush=True)
        _run(
            [
                str(py),
                str(pipeline_script),
                "--session_id",
                str(args.session_id),
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
                "--mode",
                "mic",
                "--response_trigger",
                str(args.response_trigger),
            ],
            cwd=m1_repo,
            env=env,
        )

        pcm = pipeline_dir / "01_audio_stream_bridge" / "audio_response.pcm"

        print("[realtime_step1] run chunks build", flush=True)
        _run(
            [
                str(py),
                str(chunks_script),
                "--session_id",
                str(args.session_id),
                "--pipeline_dir",
                str(pipeline_dir),
                "--pose_json",
                str(pose_json),
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

        chunks_summary_json = chunks_dir / "run_audio_input_to_chunks_smoke.summary.json"
        chunks_summary = _load_json(chunks_summary_json)

        audio_thread = None

        total_frames = 0
        for ch in chunks_summary["chunks"]:
            # F. _run_m0_one_chunk() 呼び出しに追加
            total_frames += _run_m0_one_chunk(
                py=py,
                m0_repo=m0_repo,
                m1_repo=m1_repo,
                m3_repo=m3_repo,
                base_cfg=base_cfg,
                chunk=ch,
                chunks_summary=chunks_summary,
                work_dir=m0_work_dir,
                watch_fg_dir=watch_fg_dir,
                env=env,
                frame_offset=int(args.frame_offset),
            )

            if audio_thread is None:
                print("[realtime_step1] start audio after first chunk", flush=True)
                audio_thread = _play_audio_async(
                    py=py,
                    audio_script=audio_script,
                    pcm=pcm,
                    audio_device=str(args.audio_device),
                    cwd=m1_repo,
                    env=env,
                )

            if float(args.chunk_display_gap_s) > 0:
                print(
                    f"[realtime_step1] chunk display gap {args.chunk_display_gap_s}s",
                    flush=True,
                )
                time.sleep(float(args.chunk_display_gap_s))

        if audio_thread is not None:
            audio_thread.join(timeout=30.0)
            if audio_thread.is_alive():
                raise TimeoutError("audio thread did not finish")

        # G. summary_json 出力追加
        summary_json = out_root / "run_mic_input_obs_realtime_step1.summary.json"
        summary = {
            "session_id": str(args.session_id),
            "chunks_n": len(chunks_summary["chunks"]),
            "total_frames": int(total_frames),
            "frame_offset": int(args.frame_offset),
            "watch_fg_dir": str(watch_fg_dir),
        }
        summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("[run_mic_input_obs_realtime_step1][OK]", flush=True)
        print(f"  session_id   : {args.session_id}", flush=True)
        print(f"  chunks_n     : {len(chunks_summary['chunks'])}", flush=True)
        print(f"  total_frames : {total_frames}", flush=True)
        print(f"  watch_fg_dir : {watch_fg_dir}", flush=True)

        time.sleep(2.0)

    finally:
        # H. finallyを修正
        if cam_proc is not None and cam_proc.poll() is None:
            print("[realtime_step1] terminate virtualcam", flush=True)
            cam_proc.terminate()
            try:
                cam_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cam_proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())