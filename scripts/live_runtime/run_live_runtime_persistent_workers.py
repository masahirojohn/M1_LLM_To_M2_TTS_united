#!/usr/bin/env python3
from __future__ import annotations

"""
Live Runtime Persistent Workers v0.5

目的:
- 既存成功版 run_live_runtime_realtime_streaming.py を壊さず、
  session単位で連続実行する「常時ループ入口」を追加する。

重要:
- これは短期安全版。
- M0 は subprocess のまま。
- M3.5 は chunk MP4 + final MP4 確認のまま。
- 真の常時worker化（Live API常時接続 / M0常駐 / OBS出力）は次フェーズ。

固定:
- step_ms = 40
- chunk_len_ms = 400
- fps = 25
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_STEP_MS = 40
DEFAULT_CHUNK_LEN_MS = 400
DEFAULT_FPS = 25


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_env(m1_repo: Path, m3_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{m1_repo / 'src'}:{m3_repo / 'src'}:"
        + env.get("PYTHONPATH", "")
    )
    return env


def _run_one_session(
    *,
    python_exe: str,
    m1_repo: Path,
    m3_repo: Path,
    session_id: str,
    pose_json: Path,
    normal_bg_video: Path,
    target_audio_ms: int,
    mouth_model: str,
    mouth_prompt: str,
    mouth_dev_stop_s: float,
    expr_model: str,
    expr_prompt: str,
    expr_dev_stop_s: float,
    stop_after_first_tool_s: float,
    api_version: str,
    step_ms: int,
    chunk_len_ms: int,
    fps: int,
    m0_base_config: str,
    no_realtime_sleep: bool,
    skip_m35: bool,
) -> dict[str, Any]:
    script = m1_repo / "scripts" / "live_runtime" / "run_live_runtime_realtime_streaming.py"
    if not script.exists():
        raise FileNotFoundError(f"missing script: {script}")

    cmd = [
        python_exe,
        str(script),
        "--session_id",
        session_id,
        "--pose_json",
        str(pose_json.resolve()),
        "--normal_bg_video",
        str(normal_bg_video.resolve()),
        "--target_audio_ms",
        str(int(target_audio_ms)),
        "--mouth_model",
        str(mouth_model),
        "--mouth_prompt",
        str(mouth_prompt),
        "--mouth_dev_stop_s",
        str(float(mouth_dev_stop_s)),
        "--expr_model",
        str(expr_model),
        "--expr_prompt",
        str(expr_prompt),
        "--expr_dev_stop_s",
        str(float(expr_dev_stop_s)),
        "--stop_after_first_tool_s",
        str(float(stop_after_first_tool_s)),
        "--api_version",
        str(api_version),
        "--step_ms",
        str(int(step_ms)),
        "--chunk_len_ms",
        str(int(chunk_len_ms)),
        "--fps",
        str(int(fps)),
        "--m0_base_config",
        str(m0_base_config),
    ]

    if no_realtime_sleep:
        cmd.append("--no_realtime_sleep")

    if skip_m35:
        cmd.append("--skip_m35")

    env = _build_env(m1_repo, m3_repo)

    print("[persistent][RUN]", " ".join(cmd), flush=True)

    started = time.time()

    proc = subprocess.run(
        cmd,
        cwd=str(m1_repo),
        env=env,
        text=True,
    )

    elapsed_s = round(time.time() - started, 3)

    summary_json = (
        m1_repo
        / "out"
        / "live_runtime_realtime_streaming"
        / session_id
        / "run_live_runtime_realtime_streaming.summary.json"
    )

    summary: dict[str, Any] | None = None
    if summary_json.exists():
        try:
            summary = _load_json(summary_json)
        except Exception as e:
            summary = {
                "summary_read_error": str(e),
                "summary_json": str(summary_json),
            }

    return {
        "session_id": session_id,
        "returncode": int(proc.returncode),
        "elapsed_s": elapsed_s,
        "summary_json": str(summary_json),
        "summary": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live Runtime Persistent Workers v0.5 safe loop"
    )

    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")

    ap.add_argument(
        "--session_prefix",
        default="sess_persistent",
    )

    ap.add_argument(
        "--pose_json",
        default="/workspaces/M0_session_renderer_final_1/timelines/pose/pose_timeline_final_with1_4.json",
    )
    ap.add_argument(
        "--normal_bg_video",
        default="/workspaces/M3.5_final/in/with1.mp4",
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

    ap.add_argument(
        "--max_sessions",
        type=int,
        default=1,
        help="Number of sessions to run. Use 0 for infinite loop.",
    )
    ap.add_argument(
        "--interval_s",
        type=float,
        default=0.0,
        help="Sleep seconds between sessions.",
    )

    ap.add_argument(
        "--no_realtime_sleep",
        action="store_true",
        help="Pass through to realtime streaming smoke for fastest debug.",
    )
    ap.add_argument("--skip_m35", action="store_true")

    ap.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop loop if one session fails.",
    )

    args = ap.parse_args()

    if int(args.step_ms) != 40:
        raise ValueError("step_ms must be 40")
    if int(args.chunk_len_ms) != 400:
        raise ValueError("chunk_len_ms must be 400")
    if int(args.fps) != 25:
        raise ValueError("fps must be 25")

    python_exe = sys.executable
    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()

    pose_json = Path(args.pose_json).resolve()
    normal_bg_video = Path(args.normal_bg_video).resolve()

    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")
    if not normal_bg_video.exists():
        raise FileNotFoundError(f"missing normal_bg_video: {normal_bg_video}")

    run_id = _now_id()
    out_root = m1_repo / "out" / "live_runtime_persistent_workers" / run_id
    _ensure_dir(out_root)

    run_log: dict[str, Any] = {
        "schema_version": "live_runtime_persistent_workers.v0.5",
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "mode": {
            "m0": "subprocess",
            "m35": "chunk_mp4_plus_final_mp4",
            "true_persistent_live_api": False,
        },
        "fixed_contract": {
            "step_ms": int(args.step_ms),
            "chunk_len_ms": int(args.chunk_len_ms),
            "fps": int(args.fps),
        },
        "args": vars(args),
        "sessions": [],
    }

    loop_i = 0

    print("[persistent][START]")
    print(f"  run_id     : {run_id}")
    print(f"  out_root   : {out_root}")
    print(f"  max_sessions: {args.max_sessions}")

    while True:
        if int(args.max_sessions) > 0 and loop_i >= int(args.max_sessions):
            break

        session_id = f"{args.session_prefix}_{run_id}_{loop_i:04d}"

        print("[persistent][SESSION_START]")
        print(f"  index     : {loop_i}")
        print(f"  session_id: {session_id}")

        result = _run_one_session(
            python_exe=python_exe,
            m1_repo=m1_repo,
            m3_repo=m3_repo,
            session_id=session_id,
            pose_json=pose_json,
            normal_bg_video=normal_bg_video,
            target_audio_ms=int(args.target_audio_ms),
            mouth_model=str(args.mouth_model),
            mouth_prompt=str(args.mouth_prompt),
            mouth_dev_stop_s=float(args.mouth_dev_stop_s),
            expr_model=str(args.expr_model),
            expr_prompt=str(args.expr_prompt),
            expr_dev_stop_s=float(args.expr_dev_stop_s),
            stop_after_first_tool_s=float(args.stop_after_first_tool_s),
            api_version=str(args.api_version),
            step_ms=int(args.step_ms),
            chunk_len_ms=int(args.chunk_len_ms),
            fps=int(args.fps),
            m0_base_config=str(args.m0_base_config),
            no_realtime_sleep=bool(args.no_realtime_sleep),
            skip_m35=bool(args.skip_m35),
        )

        run_log["sessions"].append(result)
        _write_json(out_root / "persistent_workers.summary.json", run_log)

        if result["returncode"] != 0:
            print("[persistent][ERROR]")
            print(f"  session_id: {session_id}")
            print(f"  returncode: {result['returncode']}")

            if args.stop_on_error:
                break
        else:
            print("[persistent][SESSION_OK]")
            print(f"  session_id: {session_id}")
            print(f"  elapsed_s : {result['elapsed_s']}")

        loop_i += 1

        if float(args.interval_s) > 0:
            time.sleep(float(args.interval_s))

    run_log["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_log["sessions_n"] = len(run_log["sessions"])
    _write_json(out_root / "persistent_workers.summary.json", run_log)

    print("[persistent][OK]")
    print(f"  sessions_n : {len(run_log['sessions'])}")
    print(f"  summary_json: {out_root / 'persistent_workers.summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())