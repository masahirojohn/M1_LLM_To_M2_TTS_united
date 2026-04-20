#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd: List[str], *, cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
    except subprocess.CalledProcessError as e:
        print("[ERROR] child command failed")
        print("[ERROR] returncode:", e.returncode)
        print("[ERROR] cmd:", " ".join(e.cmd))
        raise


def _stable_json_files(inbox_dir: Path) -> List[Path]:
    # .part は未完成扱いで無視
    files = [
        p for p in sorted(inbox_dir.glob("*.json"))
        if p.is_file() and not p.name.endswith(".part.json")
    ]
    return files

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Watch inbox dir and run runtime session for each arriving unified json"
    )
    ap.add_argument("--inbox_dir", required=True)
    ap.add_argument("--done_dir", default=None)
    ap.add_argument("--error_dir", default=None)

    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--normal_bg_video", required=True)

    ap.add_argument(
        "--runtime_entry",
        default="/workspaces/M1_LLM_To_M2_TTS_united/scripts/run_runtime_session.py",
    )
    ap.add_argument("--session_id_prefix", default="sess_runtime_watch")
    ap.add_argument("--utt_id_prefix", default="utt_runtime_watch")

    ap.add_argument("--poll_sec", type=float, default=2.0)
    ap.add_argument("--max_items", type=int, default=0, help="0 means unlimited")
    ap.add_argument(
        "--max_runtime_sec",
        type=float,
        default=0,
        help="0 means unlimited; stop watcher after this many seconds",
    )
    ap.add_argument("--skip_bridge", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_clean", action="store_true")
    ap.add_argument("--no_verify", action="store_true")
    ap.add_argument("--dry_run_child", action="store_true")

    args = ap.parse_args()

    inbox_dir = Path(args.inbox_dir).resolve()
    done_dir = Path(args.done_dir).resolve() if args.done_dir else (inbox_dir / "done")
    error_dir = Path(args.error_dir).resolve() if args.error_dir else (inbox_dir / "error")

    pose_json = Path(args.pose_json).resolve()
    normal_bg_video = Path(args.normal_bg_video).resolve()
    runtime_entry = Path(args.runtime_entry).resolve()

    if not inbox_dir.exists():
        raise FileNotFoundError(f"missing inbox_dir: {inbox_dir}")
    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")
    if not normal_bg_video.exists():
        raise FileNotFoundError(f"missing normal_bg_video: {normal_bg_video}")
    if not runtime_entry.exists():
        raise FileNotFoundError(f"missing runtime_entry: {runtime_entry}")

    _ensure_dir(done_dir)
    _ensure_dir(error_dir)

    processed = 0
    loop_index = 0
    start_time = time.time()

    print("[WATCH] inbox_dir       :", inbox_dir)
    print("[WATCH] done_dir        :", done_dir)
    print("[WATCH] error_dir       :", error_dir)
    print("[WATCH] runtime_entry   :", runtime_entry)
    print("[WATCH] poll_sec        :", args.poll_sec)
    print("[WATCH] max_items       :", args.max_items)
    print("[WATCH] max_runtime_sec :", args.max_runtime_sec)

    while True:
        elapsed_sec = time.time() - start_time
        if args.max_runtime_sec > 0 and elapsed_sec >= args.max_runtime_sec:
            print("[WATCH] reached max_runtime_sec -> stop")
            break

        files = _stable_json_files(inbox_dir)

        if not files:
            time.sleep(args.poll_sec)
            continue

        target = files[0]
        session_id = f"{args.session_id_prefix}_{loop_index:03d}"
        utt_id = f"{args.utt_id_prefix}_{loop_index:03d}"

        cmd = [
            sys.executable,
            str(runtime_entry),
            "--session_id", session_id,
            "--utt_id", utt_id,
            "--m1_unified_json", str(target),
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
        if args.dry_run_child:
            cmd.append("--dry_run")

        print(f"[WATCH] picked={target}")
        print(f"[WATCH] session_id={session_id}")
        print(f"[WATCH] utt_id={utt_id}")

        try:
            _run(cmd, cwd=runtime_entry.parent.parent)
            dst = done_dir / target.name
            shutil.move(str(target), str(dst))
            print(f"[WATCH] moved to done: {dst}")
        except Exception:
            dst = error_dir / target.name
            shutil.move(str(target), str(dst))
            print(f"[WATCH] moved to error: {dst}")
            raise

        processed += 1
        loop_index += 1

        if args.max_items > 0 and processed >= args.max_items:
            print("[WATCH] reached max_items -> stop")
            break

    print("[run_runtime_session_watch][OK]")
    print("  processed:", processed)
    print("  done_dir :", done_dir)
    print("  error_dir:", error_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())