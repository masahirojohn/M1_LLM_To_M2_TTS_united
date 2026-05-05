from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Loop smoke: mic -> AI -> M0/M3.5 -> OBS, repeated by turn"
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

    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Number of mic->OBS turns to run.",
    )
    ap.add_argument(
        "--gap_s",
        type=float,
        default=0.5,
        help="Sleep seconds between turns.",
    )
    ap.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="First turn index used in generated session ids.",
    )

    args = ap.parse_args()

    # --- 修正箇所: APIキーのチェックロジック追加 ---
    import os

    gemini_key = os.environ.get("GEMINI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    if google_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is set. Please clear it to avoid API key precedence issues: "
            "Remove-Item Env:GOOGLE_API_KEY -ErrorAction SilentlyContinue"
        )

    if not gemini_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Set it before running: "
            '$env:GEMINI_API_KEY="YOUR_KEY"'
        )
    # ----------------------------------------------

    if int(args.fps) != 25:
        raise ValueError("fps must be 25")
    if int(args.turns) <= 0:
        raise ValueError("--turns must be > 0")
    if float(args.duration_s) <= 0:
        raise ValueError("--duration_s must be > 0")

    m1 = Path(args.m1_repo_root).resolve()
    py = m1 / ".venv" / "Scripts" / "python.exe"

    if not py.exists():
        raise FileNotFoundError(f"missing python exe: {py}")

    smoke_script = m1 / "scripts" / "live_runtime" / "run_mic_input_obs_smoke.py"
    if not smoke_script.exists():
        raise FileNotFoundError(f"missing smoke script: {smoke_script}")

    print("[run_mic_input_obs_loop_smoke][START]", flush=True)
    print(f"  base_session_id : {args.session_id}", flush=True)
    print(f"  turns           : {args.turns}", flush=True)
    print(f"  duration_s      : {args.duration_s}", flush=True)
    print(f"  gap_s           : {args.gap_s}", flush=True)
    print(f"  audio_device    : {args.audio_device}", flush=True)

    ok_turns = 0

    for n in range(int(args.start_index), int(args.start_index) + int(args.turns)):
        turn_session_id = f"{args.session_id}_turn{n:03d}"

        print("", flush=True)
        print("=" * 72, flush=True)
        print(f"[run_mic_input_obs_loop_smoke] turn={n} session_id={turn_session_id}", flush=True)
        print("=" * 72, flush=True)

        cmd = [
            str(py),
            str(smoke_script),
            "--session_id",
            turn_session_id,
            "--m1_repo_root",
            str(Path(args.m1_repo_root).resolve()),
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
            "--fps",
            str(int(args.fps)),
            "--width",
            str(int(args.width)),
            "--height",
            str(int(args.height)),
        ]

        _run(cmd, cwd=m1)

        ok_turns += 1

        if n < int(args.start_index) + int(args.turns) - 1:
            if float(args.gap_s) > 0:
                print(f"[run_mic_input_obs_loop_smoke] sleep {args.gap_s}s", flush=True)
                time.sleep(float(args.gap_s))

    print("", flush=True)
    print("[run_mic_input_obs_loop_smoke][DONE]", flush=True)
    print(f"  ok_turns: {ok_turns}/{args.turns}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())