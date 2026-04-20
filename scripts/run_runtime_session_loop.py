#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_list_file(path: Path) -> List[str]:
    obj = _load_json(path)
    if not isinstance(obj, list):
        raise ValueError(f"json list file must be list: {path}")
    out: List[str] = []
    for i, v in enumerate(obj):
        if not isinstance(v, str):
            raise ValueError(f"json list file item[{i}] must be str: {path}")
        out.append(v)
    return out


def _run(cmd: List[str], *, cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
    except subprocess.CalledProcessError as e:
        print("[ERROR] child command failed")
        print("[ERROR] returncode:", e.returncode)
        print("[ERROR] cmd:", " ".join(e.cmd))
        raise

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sequential runtime loop runner for multiple m1_unified_json inputs"
    )
    ap.add_argument("--m1_unified_json_list", required=True)
    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--normal_bg_video", required=True)

    ap.add_argument("--runtime_entry", default="/workspaces/M1_LLM_To_M2_TTS_united/scripts/run_runtime_session.py")
    ap.add_argument("--session_id_prefix", default="sess_runtime_loop")
    ap.add_argument("--utt_id_prefix", default="utt_runtime_loop")

    ap.add_argument("--skip_bridge", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_clean", action="store_true")
    ap.add_argument("--no_verify", action="store_true")

    args = ap.parse_args()

    runtime_entry = Path(args.runtime_entry).resolve()
    list_path = Path(args.m1_unified_json_list).resolve()
    pose_json = Path(args.pose_json).resolve()
    normal_bg_video = Path(args.normal_bg_video).resolve()

    if not runtime_entry.exists():
        raise FileNotFoundError(f"missing runtime_entry: {runtime_entry}")
    if not list_path.exists():
        raise FileNotFoundError(f"missing m1_unified_json_list: {list_path}")
    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")
    if not normal_bg_video.exists():
        raise FileNotFoundError(f"missing normal_bg_video: {normal_bg_video}")

    input_jsons = [Path(p).resolve() for p in _load_json_list_file(list_path)]

    for i, m1_json in enumerate(input_jsons):
        if not m1_json.exists():
            raise FileNotFoundError(f"missing m1_unified_json[{i}]: {m1_json}")

        session_id = f"{args.session_id_prefix}_{i:03d}"
        utt_id = f"{args.utt_id_prefix}_{i:03d}"

        cmd = [
            sys.executable,
            str(runtime_entry),
            "--session_id", session_id,
            "--utt_id", utt_id,
            "--m1_unified_json", str(m1_json),
            "--pose_json", str(pose_json),
            "--normal_bg_video", str(normal_bg_video),
        ]

        if args.skip_bridge:
            cmd.append("--skip_bridge")
        if args.resume:
            cmd.append("--resume")
        if args.no_clean:
            cmd.append("--no_clean")
        if args.no_verify:
            cmd.append("--no_verify")

        print(f"[LOOP] utterance_index={i}")
        print(f"[LOOP] session_id={session_id}")
        print(f"[LOOP] utt_id={utt_id}")
        print(f"[LOOP] m1_unified_json={m1_json}")

        _run(cmd, cwd=runtime_entry.parent.parent)

    print("[run_runtime_session_loop][OK]")
    print("  count:", len(input_jsons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())