#!/usr/bin/env python3
from __future__ import annotations

"""
Live Runtime Realtime Streaming + BG Pipe v0

目的:
1. 既存成功版 run_live_runtime_realtime_streaming.py を実行
2. 生成された FG PNG 連番を取得
3. ffmpeg rawvideo pipe で BG と合成
4. pipe_bg.mp4 を生成

重要:
- 既存成功コードは変更しない
- M0 = subprocess のまま
- M3.5 final MP4 も従来どおり生成
- 追加で pipe_bg.mp4 を生成する安全な統合版
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2


DEFAULT_STEP_MS = 40
DEFAULT_CHUNK_LEN_MS = 400
DEFAULT_FPS = 25


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(
        cmd,
        check=True,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_env(m1_repo: Path, m3_repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{m1_repo / 'src'}:{m3_repo / 'src'}:"
        + env.get("PYTHONPATH", "")
    )
    return env


def _run_streaming(
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
) -> Path:
    script = m1_repo / "scripts" / "live_runtime" / "run_live_runtime_realtime_streaming.py"
    if not script.exists():
        raise FileNotFoundError(f"missing streaming script: {script}")

    cmd = [
        python_exe,
        str(script),
        "--m1_repo_root",
        str(m1_repo.resolve()),
        "--m3_repo_root",
        str(m3_repo.resolve()),
        "--m0_repo_root",
        str(Path("C:/dev/M0_session_renderer_final_1").resolve()),
        "--m35_repo_root",
        str(Path("C:/dev/M3.5_final").resolve()),
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

    _run(cmd, cwd=m1_repo, env=_build_env(m1_repo, m3_repo))

    summary_json = (
        m1_repo
        / "out"
        / "live_runtime_realtime_streaming"
        / session_id
        / "run_live_runtime_realtime_streaming.summary.json"
    )

    if not summary_json.exists():
        raise FileNotFoundError(f"missing summary_json: {summary_json}")

    return summary_json


def _start_ffmpeg_bg_pipe(
    *,
    bg_video: Path,
    out_mp4: Path,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
) -> subprocess.Popen:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(bg_video.resolve()),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgra",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "-",
        "-filter_complex",
        (
            f"[0:v]scale={width}:{height},format=rgba[bg];"
            "[1:v]format=rgba[fg];"
            "[bg][fg]overlay=0:0:format=auto:shortest=1,format=yuv420p[v]"
        ),
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-frames:v",
        str(int(num_frames)),
        "-shortest",
        str(out_mp4.resolve()),
    ]

    print("[FFMPEG START]", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _pipe_fg_to_bg_mp4(
    *,
    fg_dir: Path,
    bg_video: Path,
    out_mp4: Path,
    width: int,
    height: int,
    fps: int,
    realtime_sleep: bool,
) -> dict[str, Any]:
    if not fg_dir.exists():
        raise FileNotFoundError(f"missing fg_dir: {fg_dir}")
    if not bg_video.exists():
        raise FileNotFoundError(f"missing bg_video: {bg_video}")

    pngs = sorted(fg_dir.glob("*.png"))
    if not pngs:
        raise RuntimeError(f"no png found in fg_dir: {fg_dir}")

    print(f"[PIPE_BG][INFO] frames: {len(pngs)}", flush=True)

    ff = _start_ffmpeg_bg_pipe(
        bg_video=bg_video,
        out_mp4=out_mp4,
        width=width,
        height=height,
        fps=fps,
        num_frames=len(pngs),
    )

    frame_s = 1.0 / float(fps)
    t0 = time.time()

    try:
        if ff.stdin is None:
            raise RuntimeError("ffmpeg stdin is None")

        for i, p in enumerate(pngs):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"failed to read png: {p}")
            if len(img.shape) != 3 or img.shape[2] != 4:
                raise RuntimeError(f"not BGRA png: {p} shape={img.shape}")

            if img.shape[1] != width or img.shape[0] != height:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

            ff.stdin.write(img.tobytes())

            if i % 10 == 0:
                print(f"[PIPE_BG] {i}", flush=True)

            if realtime_sleep:
                target = t0 + ((i + 1) * frame_s)
                sleep_s = target - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)

    finally:
        if ff.stdin:
            ff.stdin.close()
        rc = ff.wait()

    if rc != 0:
        raise RuntimeError(f"ffmpeg failed rc={rc}")

    return {
        "out_mp4": str(out_mp4),
        "frames": len(pngs),
        "duration_ms": int(round(len(pngs) * 1000 / fps)),
        "fps": int(fps),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live Runtime Realtime Streaming + BG Pipe v0"
    )

    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    ap.add_argument("--session_id", required=True)

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

    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument(
        "--pipe_out_name",
        default="pipe_bg.mp4",
        help="Output filename under M3.5 session out root.",
    )
    ap.add_argument(
        "--realtime_sleep",
        action="store_true",
        help="Sleep per frame while piping. Use for real-time-like behavior.",
    )
    ap.add_argument(
        "--no_realtime_sleep",
        action="store_true",
        help="Pass through to run_live_runtime_realtime_streaming.py.",
    )

    args = ap.parse_args()

    if int(args.step_ms) != 40:
        raise ValueError("step_ms must be 40")
    if int(args.chunk_len_ms) != 400:
        raise ValueError("chunk_len_ms must be 400")
    if int(args.fps) != 25:
        raise ValueError("fps must be 25")

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    pose_json = Path(args.pose_json).resolve()
    bg_video = Path(args.normal_bg_video).resolve()

    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")
    if not bg_video.exists():
        raise FileNotFoundError(f"missing normal_bg_video: {bg_video}")

    t0 = time.time()

    summary_json = _run_streaming(
        python_exe=sys.executable,
        m1_repo=m1_repo,
        m3_repo=m3_repo,
        session_id=str(args.session_id),
        pose_json=pose_json,
        normal_bg_video=bg_video,
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
    )

    summary = _load_json(summary_json)

    m0_info = summary.get("m0", {}) if isinstance(summary, dict) else {}
    fg_dir_raw = m0_info.get("out_fg_dir")
    if not fg_dir_raw:
        raise RuntimeError(f"summary missing m0.out_fg_dir: {summary_json}")

    fg_dir = Path(fg_dir_raw).resolve()

    m35_out_root = (
        m35_repo
        / "out"
        / "live_runtime_realtime_streaming"
        / str(args.session_id)
    )
    pipe_out_mp4 = m35_out_root / str(args.pipe_out_name)

    pipe_result = _pipe_fg_to_bg_mp4(
        fg_dir=fg_dir,
        bg_video=bg_video,
        out_mp4=pipe_out_mp4,
        width=int(args.width),
        height=int(args.height),
        fps=int(args.fps),
        realtime_sleep=bool(args.realtime_sleep),
    )

    result = {
        "schema_version": "run_live_runtime_realtime_streaming_pipe_bg.v0",
        "session_id": str(args.session_id),
        "source_summary_json": str(summary_json),
        "audio_ms": summary.get("audio_ms"),
        "chunks": summary.get("chunks"),
        "target_frames": summary.get("target_frames"),
        "fg_dir": str(fg_dir),
        "m35_final_mp4": (summary.get("final_m35") or {}).get("mp4"),
        "pipe_bg": pipe_result,
        "elapsed_s_total": round(time.time() - t0, 3),
    }

    out_summary = (
        m1_repo
        / "out"
        / "live_runtime_realtime_streaming_pipe_bg"
        / str(args.session_id)
        / "run_live_runtime_realtime_streaming_pipe_bg.summary.json"
    )
    _write_json(out_summary, result)

    print("[run_live_runtime_realtime_streaming_pipe_bg][OK]")
    print(f"  session_id   : {args.session_id}")
    print(f"  audio_ms     : {result['audio_ms']}")
    print(f"  chunks       : {result['chunks']}")
    print(f"  target_frames: {result['target_frames']}")
    print(f"  fg_dir       : {fg_dir}")
    print(f"  pipe_bg_mp4  : {pipe_out_mp4}")
    print(f"  summary_json : {out_summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())