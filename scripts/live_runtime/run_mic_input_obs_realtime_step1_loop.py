#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Loop runner for realtime_step1 with optional persistent virtualcam"
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

    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--gap_s", type=float, default=0.5)

    ap.add_argument("--clean_each", action="store_true")
    ap.add_argument("--persistent_cam", action="store_true")
    ap.add_argument("--watch_fg_dir", default=None)

    args = ap.parse_args()

    m1 = Path(args.m1_repo_root).resolve()
    py = m1 / ".venv" / "Scripts" / "python.exe"

    step1_script = m1 / "scripts" / "live_runtime" / "run_mic_input_obs_realtime_step1.py"
    virtualcam_script = m1 / "scripts" / "live_runtime" / "run_virtualcam_persistent.py"

    if not step1_script.exists():
        raise FileNotFoundError(f"missing script: {step1_script}")
    if not virtualcam_script.exists():
        raise FileNotFoundError(f"missing script: {virtualcam_script}")

    watch_fg_dir = (
        Path(args.watch_fg_dir).resolve()
        if args.watch_fg_dir
        else m1 / "out" / "obs_stream_step1_loop" / "fg"
    )

    cam_proc = None

    if args.persistent_cam:
        print("[realtime_step1_loop] clean persistent watch dir:", watch_fg_dir, flush=True)
        _clean_dir(watch_fg_dir)

        print("[realtime_step1_loop] start persistent virtualcam", flush=True)
        cam_proc = subprocess.Popen(
            [
                str(py),
                str(virtualcam_script),
                "--fg_dir",
                str(watch_fg_dir),
                "--bg_video",
                str(Path(args.bg_video).resolve()),
                "--fps",
                "25",
                "--width",
                "720",
                "--height",
                "720",
                "--idle_hold",
                "--loop_bg",
            ],
            cwd=str(m1),
        )

        time.sleep(1.0)

    ok = 0
    frame_offset = 0

    try:
        for i in range(int(args.turns)):
            turn_id = f"{args.session_id}_turn{i+1:03d}"

            print("\n" + "=" * 64, flush=True)
            print(f"[realtime_step1_loop] turn={i+1} session_id={turn_id}", flush=True)
            print("=" * 64, flush=True)

            cmd = [
                str(py),
                str(step1_script),
                "--session_id",
                str(turn_id),
                "--m1_repo_root",
                str(m1),
                "--m3_repo_root",
                str(Path(args.m3_repo_root).resolve()),
                "--m0_repo_root",
                str(Path(args.m0_repo_root).resolve()),
                "--m35_repo_root",
                str(Path(args.m35_repo_root).resolve()),
                "--pose_json",
                str(Path(args.pose_json).resolve()),
                "--bg_video",
                str(Path(args.bg_video).resolve()),
                "--duration_s",
                str(float(args.duration_s)),
                "--audio_device",
                str(args.audio_device),
                "--frame_offset",
                str(int(frame_offset)),
            ]

            if args.persistent_cam:
                cmd += [
                    "--external_virtualcam",
                    "--watch_fg_dir",
                    str(watch_fg_dir),
                ]

            if args.clean_each and not args.persistent_cam:
                cmd.append("--clean")

            _run(cmd, cwd=m1)

            summary_json = (
                m1
                / "out"
                / "obs_realtime_step1"
                / turn_id
                / "run_mic_input_obs_realtime_step1.summary.json"
            )
            summary = _load_json(summary_json)

            turn_frames = int(summary["total_frames"])
            frame_offset += turn_frames
            ok += 1

            print(
                f"[realtime_step1_loop] turn_frames={turn_frames} next_frame_offset={frame_offset}",
                flush=True,
            )

            if i < int(args.turns) - 1 and float(args.gap_s) > 0:
                print(f"[realtime_step1_loop] gap {args.gap_s}s", flush=True)
                time.sleep(float(args.gap_s))

        time.sleep(2.0)

    finally:
        if cam_proc is not None and cam_proc.poll() is None:
            print("[realtime_step1_loop] terminate persistent virtualcam", flush=True)
            cam_proc.terminate()
            try:
                cam_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cam_proc.kill()

    print("\n[run_mic_input_obs_realtime_step1_loop][DONE]", flush=True)
    print(f"  ok_turns: {ok}/{args.turns}", flush=True)
    print(f"  total_frames: {frame_offset}", flush=True)
    print(f"  watch_fg_dir: {watch_fg_dir}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())