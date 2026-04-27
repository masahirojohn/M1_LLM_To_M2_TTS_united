#!/usr/bin/env python3
from __future__ import annotations

"""
Live Runtime Streaming → Single MP4 v0

目的:
- 既存 run_live_runtime_realtime_streaming.py をそのまま利用
- chunk MP4 を逐次連結して「1本のMP4」を生成

重要:
- M0 / M3.5 の既存成功コードは一切変更しない
- ffmpeg concat demuxer を使う
- 安全版（後処理型）→ 次フェーズでリアルタイム化

"""

import argparse
import subprocess
import time
from pathlib import Path
import json


def _run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _append_concat_list(list_path: Path, mp4_path: Path):
    with list_path.open("a", encoding="utf-8") as f:
        f.write(f"file '{mp4_path.resolve()}'\n")


def _build_final_mp4(list_path: Path, out_mp4: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(out_mp4),
    ]
    _run(cmd)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--session_id", required=True)
    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--normal_bg_video", required=True)
    ap.add_argument("--target_audio_ms", type=int, default=3000)

    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    args = ap.parse_args()

    m1_repo = Path(args.m1_repo_root)
    m35_repo = Path(args.m35_repo_root)

    # -----------------------------------
    # 1. 既存 streaming 実行
    # -----------------------------------
    run_script = m1_repo / "scripts/live_runtime/run_live_runtime_realtime_streaming.py"

    cmd = [
        "python",
        str(run_script),
        "--session_id",
        args.session_id,
        "--pose_json",
        args.pose_json,
        "--normal_bg_video",
        args.normal_bg_video,
        "--target_audio_ms",
        str(args.target_audio_ms),
    ]

    _run(cmd)

    # -----------------------------------
    # 2. chunk MP4 を収集
    # -----------------------------------
    chunks_root = (
        m35_repo
        / "out/live_runtime_realtime_streaming"
        / args.session_id
        / "chunks"
    )

    if not chunks_root.exists():
        raise RuntimeError(f"chunks not found: {chunks_root}")

    chunk_dirs = sorted(chunks_root.glob("*"))

    chunk_mp4s = []
    for d in chunk_dirs:
        mp4 = d / "stage2/m3_5_composite.mp4"
        if mp4.exists():
            chunk_mp4s.append(mp4)

    if not chunk_mp4s:
        raise RuntimeError("no chunk mp4 found")

    print("[INFO] chunks:", len(chunk_mp4s))

    # -----------------------------------
    # 3. concat list 作成
    # -----------------------------------
    concat_list = chunks_root / "concat_list.txt"
    if concat_list.exists():
        concat_list.unlink()

    for mp4 in chunk_mp4s:
        _append_concat_list(concat_list, mp4)

    # -----------------------------------
    # 4. 最終MP4生成
    # -----------------------------------
    out_mp4 = (
        m35_repo
        / "out/live_runtime_realtime_streaming"
        / args.session_id
        / "final_single.mp4"
    )

    _build_final_mp4(concat_list, out_mp4)

    print("[OK] single MP4:", out_mp4)


if __name__ == "__main__":
    main()