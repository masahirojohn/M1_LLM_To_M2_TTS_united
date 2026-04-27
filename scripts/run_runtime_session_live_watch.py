#!/usr/bin/env python3
from __future__ import annotations

"""
Live Runtime Watcher (v0)

役割:
- inbox に投げられた JSON を順次処理
- run_runtime_session_live_e2e.py を呼ぶ
- done / error に振り分け

前提:
- 既存 watcher を壊さない（完全に別ファイル）
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], cwd: Path):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def process_one(
    *,
    python_exe: str,
    repo_root: Path,
    m3_repo: Path,
    m0_repo: Path,
    m35_repo: Path,
    inbox_json: Path,
    done_dir: Path,
    error_dir: Path,
    pose_json: Path,
    bg_video: Path,
    target_audio_ms: int,
    skip_bridge: bool,
):
    try:
        data = _load_json(inbox_json)

        # 必須: prompt
        prompt = data.get("prompt")
        if not prompt:
            raise ValueError("missing prompt")

        session_id = data.get("session_id", inbox_json.stem)

        cmd = [
            python_exe,
            str((repo_root / "scripts" / "run_live_runtime_streaming.py").resolve()),
            "--m1_repo_root", str(repo_root),
            "--m3_repo_root", str(m3_repo),
            "--m0_repo_root", str(m0_repo),
            "--m35_repo_root", str(m35_repo),
            "--session_id", session_id,
            "--pose_json", str(pose_json),
            "--normal_bg_video", str(bg_video),
            "--mouth_prompt", str(prompt),
            "--target_audio_ms", str(target_audio_ms),
        ]

        #if skip_bridge:
            #cmd.append("--skip_bridge")

        _run(cmd, cwd=repo_root)

        _move(inbox_json, done_dir / inbox_json.name)
        print(f"[DONE] {inbox_json.name}")

    except Exception as e:
        print(f"[ERROR] {inbox_json.name}: {e}")
        _move(inbox_json, error_dir / inbox_json.name)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m0_repo_root", default="/workspaces/M0_session_renderer_final_1")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    ap.add_argument("--inbox_dir", default="out/runtime_watch_inbox")
    ap.add_argument("--done_dir", default="out/runtime_watch_inbox/done")
    ap.add_argument("--error_dir", default="out/runtime_watch_inbox/error")

    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--bg_video", required=True)

    ap.add_argument("--target_audio_ms", type=int, default=3000)
    ap.add_argument("--skip_bridge", action="store_true")

    ap.add_argument("--poll_sec", type=float, default=2.0)
    ap.add_argument("--max_runtime_sec", type=int, default=60)

    args = ap.parse_args()

    python_exe = sys.executable

    repo_root = Path(args.repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    inbox_dir = Path(args.inbox_dir).resolve()
    done_dir = Path(args.done_dir).resolve()
    error_dir = Path(args.error_dir).resolve()

    pose_json = Path(args.pose_json).resolve()
    bg_video = Path(args.bg_video).resolve()

    inbox_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    print("[watcher] start")

    t0 = time.time()

    while True:
        if args.max_runtime_sec > 0 and (time.time() - t0) > args.max_runtime_sec:
            print("[watcher] timeout exit")
            break

        files = sorted(inbox_dir.glob("*.json"))

        if not files:
            time.sleep(args.poll_sec)
            continue

        for f in files:
            process_one(
                python_exe=python_exe,
                repo_root=repo_root,
                m3_repo=m3_repo,
                m0_repo=m0_repo,
                m35_repo=m35_repo,
                inbox_json=f,
                done_dir=done_dir,
                error_dir=error_dir,
                pose_json=pose_json,
                bg_video=bg_video,
                target_audio_ms=args.target_audio_ms,
                skip_bridge=args.skip_bridge,
            )


if __name__ == "__main__":
    main()