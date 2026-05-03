from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd, cwd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


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

    args = ap.parse_args()

    m1 = Path(args.m1_repo_root)
    py = str(m1 / ".venv" / "Scripts" / "python.exe")

    out_root = m1 / "out" / "mic_input_obs" / args.session_id

    # =========================
    # ① mic → E2E
    # =========================
    e2e_script = m1 / "scripts" / "live_runtime" / "run_mic_input_e2e_smoke.py"

    _run([
        py,
        str(e2e_script),
        "--session_id", args.session_id,
        "--m1_repo_root", args.m1_repo_root,
        "--m3_repo_root", args.m3_repo_root,
        "--m0_repo_root", args.m0_repo_root,
        "--m35_repo_root", args.m35_repo_root,
        "--pose_json", args.pose_json,
        "--bg_video", args.bg_video,
        "--duration_s", str(args.duration_s),
        "--clean",
    ], cwd=m1)

    # =========================
    # パス生成
    # =========================
    base = m1 / "out" / "audio_input_e2e_smoke" / args.session_id

    fg_dir = base / "03_m0_all_chunks" / "fg_streaming" / "in" / "fg"
    pcm = base / "01_audio_input_smoke_pipeline" / "01_audio_stream_bridge" / "audio_response.pcm"

    # =========================
    # ② OBS出力
    # =========================
    av_script = m1 / "scripts" / "live_runtime" / "dev_av_to_obs.py"

    _run([
        py,
        str(av_script),
        "--fg_dir", str(fg_dir),
        "--bg_video", args.bg_video,
        "--pcm", str(pcm),
        "--audio_device", str(args.audio_device),
        "--fps", str(args.fps),
        "--width", str(args.width),
        "--height", str(args.height),
    ], cwd=m1)

    print("[run_mic_input_obs_smoke][DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())