#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import threading
import time
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _wait_and_play_pcm(
    *,
    py: Path,
    audio_script: Path,
    pcm: Path,
    audio_device: str,
    cwd: Path,
    timeout_s: float,
) -> None:
    print("[follow_smoke][audio] waiting pcm:", pcm, flush=True)

    t0 = time.time()
    last_size = -1
    stable_count = 0

    while time.time() - t0 < timeout_s:
        if pcm.exists():
            size = pcm.stat().st_size
            if size > 0 and size == last_size:
                stable_count += 1
            else:
                stable_count = 0
                last_size = size

            # 0.4秒程度サイズが安定したら再生開始
            if stable_count >= 2:
                print(f"[follow_smoke][audio] pcm ready bytes={size}", flush=True)
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
                )
                return

        time.sleep(0.2)

    raise TimeoutError(f"timeout waiting pcm: {pcm}")


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--session_id", required=True)
    ap.add_argument("--m1_repo_root", required=True)
    ap.add_argument("--m3_repo_root", required=True)
    ap.add_argument("--m0_repo_root", required=True)
    ap.add_argument("--m35_repo_root", required=True)

    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--bg_video", required=True)

    ap.add_argument("--duration_s", type=float, default=5.0)
    ap.add_argument("--audio_device", default="15")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hold_after_s", type=float, default=3.0)

    args = ap.parse_args()
    pose_json = Path(args.pose_json).resolve()
    bg_video = Path(args.bg_video).resolve()

    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")
    if not bg_video.exists():
        raise FileNotFoundError(f"missing bg_video: {bg_video}")

    m1 = Path(args.m1_repo_root).resolve()
    py = m1 / ".venv" / "Scripts" / "python.exe"

    e2e_script = m1 / "scripts" / "live_runtime" / "run_mic_input_e2e_smoke.py"
    cam_script = m1 / "scripts" / "live_runtime" / "dev_fg_bg_to_virtualcam.py"
    audio_script = m1 / "scripts" / "live_runtime" / "dev_play_pcm_to_vbcable.py"

    base = m1 / "out" / "audio_input_e2e_smoke" / args.session_id
    fg_dir = base / "03_m0_all_chunks" / "fg_streaming" / "in" / "fg"
    pcm = base / "01_audio_input_smoke_pipeline" / "01_audio_stream_bridge" / "audio_response.pcm"

    print("[follow_smoke] start virtualcam follow", flush=True)

    cam_proc = subprocess.Popen(
        [
            str(py),
            str(cam_script),
            "--fg_dir", str(fg_dir),
            "--bg_video", str(Path(args.bg_video).resolve()),
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--follow",
            "--start_timeout_s", "120",
            "--frame_timeout_s", "10",
            "--hold_last_after_s", str(args.hold_after_s),
        ],
        cwd=str(m1),
    )

    audio_thread = threading.Thread(
        target=_wait_and_play_pcm,
        kwargs={
            "py": py,
            "audio_script": audio_script,
            "pcm": pcm,
            "audio_device": str(args.audio_device),
            "cwd": m1,
            "timeout_s": 180.0,
        },
        daemon=True,
    )
    audio_thread.start()

    try:
        _run([
            str(py),
            str(e2e_script),
            "--session_id", args.session_id,
            "--m1_repo_root", str(Path(args.m1_repo_root).resolve()),
            "--m3_repo_root", str(Path(args.m3_repo_root).resolve()),
            "--m0_repo_root", str(Path(args.m0_repo_root).resolve()),
            "--m35_repo_root", str(Path(args.m35_repo_root).resolve()),
            "--pose_json", str(pose_json),
            "--bg_video", str(bg_video),
            "--duration_s", str(args.duration_s),
            "--clean",
        ], cwd=m1)

        audio_thread.join(timeout=30.0)
        if audio_thread.is_alive():
            raise TimeoutError("audio thread did not finish")

        # 追加：仮想カメラプロセスの終了を、hold_after_sを考慮して待機
        cam_wait_s = float(args.hold_after_s) + 15.0
        print(f"[follow_smoke] wait virtualcam finish timeout={cam_wait_s}s", flush=True)

        try:
            cam_proc.wait(timeout=cam_wait_s)
        except subprocess.TimeoutExpired:
            raise TimeoutError("virtualcam process did not finish")

    finally:
        # 更新：より安全なプロセスの終了処理
        if cam_proc.poll() is None:
            print("[follow_smoke] terminate virtualcam", flush=True)
            cam_proc.terminate()
            try:
                cam_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cam_proc.kill()

    print("[run_mic_input_obs_follow_smoke][DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())