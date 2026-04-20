#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


DEFAULT_INBOX_DIR = Path("/workspaces/M1_LLM_To_M2_TTS_united/out/runtime_watch_inbox")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _validate_unified_json(obj: Any) -> None:
    if not isinstance(obj, dict):
        raise ValueError("input json must be an object")

    # 最低限の軽い検証だけ行う
    speech = obj.get("speech")
    if speech is not None and not isinstance(speech, dict):
        raise ValueError("input json field 'speech' must be an object if present")

    events = obj.get("events")
    if events is not None and not isinstance(events, list):
        raise ValueError("input json field 'events' must be a list if present")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Safely push m1_unified_output.json into runtime watcher inbox via .part.json -> rename"
    )
    ap.add_argument(
        "--src_json",
        required=True,
        help="absolute path to input m1_unified_output.json",
    )
    ap.add_argument(
        "--inbox_dir",
        default=str(DEFAULT_INBOX_DIR),
        help="absolute path to watcher inbox dir",
    )
    ap.add_argument(
        "--name",
        required=True,
        help="final file basename without extension, e.g. test_301",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing target json if already exists",
    )
    args = ap.parse_args()

    src_json = Path(args.src_json).resolve()
    inbox_dir = Path(args.inbox_dir).resolve()
    final_name = str(args.name).strip()

    if not src_json.exists():
        raise FileNotFoundError(f"missing src_json: {src_json}")

    if not final_name:
        raise ValueError("--name must not be empty")
    if "/" in final_name or "\\" in final_name:
        raise ValueError("--name must be a basename only")

    _ensure_dir(inbox_dir)
    _ensure_dir(inbox_dir / "done")
    _ensure_dir(inbox_dir / "error")

    tmp_path = inbox_dir / f"{final_name}.part.json"
    final_path = inbox_dir / f"{final_name}.json"

    if tmp_path.exists():
        raise FileExistsError(f"temp file already exists: {tmp_path}")

    if final_path.exists() and not args.force:
        raise FileExistsError(f"final file already exists: {final_path}")

    obj = _load_json(src_json)
    _validate_unified_json(obj)

    tmp_path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 原子的に近い move / rename
    shutil.move(str(tmp_path), str(final_path))

    print("[push_unified_json_to_watch_inbox][OK]")
    print("  src_json   :", src_json)
    print("  inbox_dir  :", inbox_dir)
    print("  final_json :", final_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())