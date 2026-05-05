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
        description="Loop wrapper for run_mic_input_obs_follow_smoke.py"
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

    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--gap_s", type=float, default=0.5)
    ap.add_argument("--hold_after_s", type=float, default=3.0)
    ap.add_argument("--start_index", type=int, default=1)

    args = ap.parse_args()

    if int(args.turns) <= 0:
        raise ValueError("--turns must be > 0")
    if int(args.fps) != 25:
        raise ValueError("fps must be 25")

    m1 = Path(args.m1_repo_root).resolve()
    py = m1 / ".venv" / "Scripts" / "python.exe"

    follow_script = m1 / "scripts" / "live_runtime" / "run_mic_input_obs_follow_smoke.py"

    if not py.exists():
        raise FileNotFoundError(f"missing python exe: {py}")
    if not follow_script.exists():
        raise FileNotFoundError(f"missing follow script: {follow_script}")

    pose_json = Path(args.pose_json).resolve()
    bg_video = Path(args.bg_video).resolve()

    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")
    if not bg_video.exists():
        raise FileNotFoundError(f"missing bg_video: {bg_video}")

    print("[run_mic_input_obs_follow_loop_smoke][START]", flush=True)
    print(f"  base_session_id : {args.session_id}", flush=True)
    print(f"  turns           : {args.turns}", flush=True)
    print(f"  duration_s      : {args.duration_s}", flush=True)
    print(f"  gap_s           : {args.gap_s}", flush=True)

    ok_turns = 0

    for n in range(int(args.start_index), int(args.start_index) + int(args.turns)):
        turn_session_id = f"{args.session_id}_turn{n:03d}"

        print("", flush=True)
        print("=" * 72, flush=True)
        print(
            f"[run_mic_input_obs_follow_loop_smoke] "
            f"turn={n} session_id={turn_session_id}",
            flush=True,
        )
        print("=" * 72, flush=True)

        _run(
            [
                str(py),
                str(follow_script),
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
                str(pose_json),
                "--bg_video",
                str(bg_video),
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
                 "--hold_after_s",
                str(float(args.hold_after_s)),
            ],
            cwd=m1,
        )

        ok_turns += 1

        if n < int(args.start_index) + int(args.turns) - 1:
            if float(args.gap_s) > 0:
                print(
                    f"[run_mic_input_obs_follow_loop_smoke] sleep {args.gap_s}s",
                    flush=True,
                )
                time.sleep(float(args.gap_s))

    print("", flush=True)
    print("[run_mic_input_obs_follow_loop_smoke][DONE]", flush=True)
    print(f"  ok_turns: {ok_turns}/{args.turns}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())