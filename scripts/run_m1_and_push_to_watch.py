#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import re
from pathlib import Path

DEFAULT_PUSH_SCRIPT = Path(
    "/workspaces/M1_LLM_To_M2_TTS_united/scripts/push_unified_json_to_watch_inbox.py"
)
DEFAULT_INBOX_DIR = Path(
    "/workspaces/M1_LLM_To_M2_TTS_united/out/runtime_watch_inbox"
)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _next_auto_watch_name(inbox_dir: Path) -> str:
    used: set[int] = set()
    pat = re.compile(r"^test_(\d+)\.json$")

    scan_dirs = [
        inbox_dir,
        inbox_dir / "done",
        inbox_dir / "error",
    ]

    for d in scan_dirs:
        if not d.exists():
            continue
        for p in d.glob("test_*.json"):
            m = pat.match(p.name)
            if not m:
                continue
            used.add(int(m.group(1)))

    n = 305
    while n in used:
        n += 1
    return f"test_{n:03d}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run an M1 command, then push its unified json to watcher inbox."
    )
    ap.add_argument(
        "--m1_unified_json",
        required=True,
        help="absolute path to unified json expected after M1 command finishes",
    )
    ap.add_argument(
        "--watch_name",
        default=None,
        help="optional final watcher file basename without extension; if omitted, auto-generate test_XXX",
    )
    ap.add_argument(
        "--inbox_dir",
        default=str(DEFAULT_INBOX_DIR),
        help="watcher inbox dir",
    )
    ap.add_argument(
        "--push_script",
        default=str(DEFAULT_PUSH_SCRIPT),
        help="push_unified_json_to_watch_inbox script path",
    )
    ap.add_argument(
        "--cwd",
        default=".",
        help="current working directory for m1_cmd",
    )
    ap.add_argument(
        "m1_cmd",
        nargs=argparse.REMAINDER,
        help="M1 command to run. Put it after '--'",
    )
    args = ap.parse_args()

    m1_unified_json = Path(args.m1_unified_json).resolve()
    inbox_dir = Path(args.inbox_dir).resolve()
    push_script = Path(args.push_script).resolve()
    cwd = Path(args.cwd).resolve()

    if not push_script.exists():
        raise FileNotFoundError(f"missing push_script: {push_script}")

    if not args.m1_cmd:
        raise ValueError("missing m1_cmd. Use '--' then the M1 command.")

    m1_cmd = list(args.m1_cmd)
    if m1_cmd and m1_cmd[0] == "--":
        m1_cmd = m1_cmd[1:]

    if not m1_cmd:
        raise ValueError("empty m1_cmd after '--'")

    # 1) M1 command
    _run(m1_cmd, cwd=cwd)

    # 2) verify output json exists
    if not m1_unified_json.exists():
        raise FileNotFoundError(
            f"M1 command finished but unified json not found: {m1_unified_json}"
        )

    watch_name = str(args.watch_name).strip() if args.watch_name else _next_auto_watch_name(inbox_dir)

    # 3) push to watcher inbox
    push_cmd = [
        sys.executable,
        str(push_script),
        "--src_json",
        str(m1_unified_json),
        "--inbox_dir",
        str(inbox_dir),
        "--name",
        str(watch_name),
    ]
    _run(push_cmd)

    print("[run_m1_and_push_to_watch][OK]")
    print("  m1_unified_json:", m1_unified_json)
    print("  watch_name     :", watch_name)
    print("  inbox_dir      :", inbox_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())