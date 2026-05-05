#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
import os
from pathlib import Path


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# A. コピー関数はフォールバック用に残します
def _copy_fg_to_watch_dir(src_fg_dir: Path, watch_fg_dir: Path) -> int:
    pngs = sorted(src_fg_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"no pngs in src_fg_dir: {src_fg_dir}")

    watch_fg_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    start_index = len(list(watch_fg_dir.glob("*.png")))

    for i, src in enumerate(pngs):
        dst = watch_fg_dir / f"{start_index + i:08d}.png"
        shutil.copy2(src, dst)
        copied += 1

    return copied


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Persistent OBS smoke: keep virtualcam alive, copy generated FG into watched dir"
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

    ap.add_argument(
        "--watch_fg_dir",
        default=None,
        help="Persistent virtualcam watch dir. Default: <m1>/out/obs_stream/...",
    )

    args = ap.parse_args()

    if int(args.turns) <= 0:
        raise ValueError("--turns must be > 0")
    if int(args.fps) != 25:
        raise ValueError("fps must be 25")

    m1 = Path(args.m1_repo_root).resolve()
    
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(Path(args.m3_repo_root).resolve())
        + ";"
        + str((Path(args.m3_repo_root).resolve() / "src"))
        + ";"
        + str(Path(args.m35_repo_root).resolve())
        + ";"
        + env.get("PYTHONPATH", "")
    )

    py = m1 / ".venv" / "Scripts" / "python.exe"

    if not py.exists():
        raise FileNotFoundError(f"missing python exe: {py}")

    pose_json = Path(args.pose_json).resolve()
    bg_video = Path(args.bg_video).resolve()

    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")
    if not bg_video.exists():
        raise FileNotFoundError(f"missing bg_video: {bg_video}")

    # B. watch構成を変更
    obs_stream_root = m1 / "out" / "obs_stream"

    watch_fg_dir = (
        Path(args.watch_fg_dir).resolve()
        if args.watch_fg_dir
        else obs_stream_root / "m0_all_chunks" / "fg_streaming" / "in" / "fg"
    )

    watch_m0_dir = obs_stream_root / "m0_all_chunks"

    virtualcam_script = m1 / "scripts" / "live_runtime" / "run_virtualcam_persistent.py"
    e2e_script = m1 / "scripts" / "live_runtime" / "run_mic_input_e2e_smoke.py"
    audio_script = m1 / "scripts" / "live_runtime" / "dev_play_pcm_to_vbcable.py"

    for p in [virtualcam_script, e2e_script, audio_script]:
        if not p.exists():
            raise FileNotFoundError(f"missing script: {p}")

    print("[persistent_smoke] clean watch dir:", watch_fg_dir, flush=True)
    _clean_dir(watch_fg_dir)

    print("[persistent_smoke] start virtualcam", flush=True)
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
        cwd=str(m1),
    )

    ok_turns = 0

    try:
        time.sleep(1.0)

        for turn in range(1, int(args.turns) + 1):
            turn_session_id = f"{args.session_id}_turn{turn:03d}"

            print("", flush=True)
            print("=" * 72, flush=True)
            print(f"[persistent_smoke] turn={turn} session_id={turn_session_id}", flush=True)
            print("=" * 72, flush=True)

            # ① M0を「別プロセスで起動」 (Popenに変更)
            e2e_cmd = [
                str(py),
                str(e2e_script),
                "--session_id",
                turn_session_id,
                "--m1_repo_root",
                str(m1),
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
                "--clean",
                "--m0_out_dir",
                str(watch_m0_dir),
            ]
            print("[RUN (Async)]", " ".join(e2e_cmd), flush=True)
            proc = subprocess.Popen(
                e2e_cmd,
                cwd=str(m1),
                env=env,
            )

            # ② すぐにOBS側へ進む
            time.sleep(0.5)  # 少し待ってFG生成開始させる

            # D. 直接参照ログを出力
            base = m1 / "out" / "audio_input_e2e_smoke" / turn_session_id
            pcm = base / "01_audio_input_smoke_pipeline" / "01_audio_stream_bridge" / "audio_response.pcm"

            print(f"[persistent_smoke] m0_direct_fg_dir={watch_fg_dir}", flush=True)

            # オーディオ再生 (ここは同期待ちでOK)
            _run(
                [
                    str(py),
                    str(audio_script),
                    "--pcm",
                    str(pcm),
                    "--sr",
                    "24000",
                    "--device",
                    str(args.audio_device),
                ],
                cwd=m1,
                env=env,
            )

            # ③ プロセス待ちは各ターンの最後で行い、終了ステータスをチェックする
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"e2e process failed rc={rc}")

            ok_turns += 1

            if turn < int(args.turns) and float(args.gap_s) > 0:
                print(f"[persistent_smoke] sleep {args.gap_s}s", flush=True)
                time.sleep(float(args.gap_s))

        time.sleep(2.0)

    finally:
        if cam_proc.poll() is None:
            print("[persistent_smoke] terminate virtualcam", flush=True)
            cam_proc.terminate()
            try:
                cam_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cam_proc.kill()

    print("[run_mic_input_obs_persistent_smoke][DONE]", flush=True)
    print(f"  ok_turns: {ok_turns}/{args.turns}", flush=True)
    print(f"  watch_fg_dir: {watch_fg_dir}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())