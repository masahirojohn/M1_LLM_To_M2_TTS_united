from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--control_file", required=True)
    ap.add_argument("--interrupt_file", required=True)

    ap.add_argument("--start_delay_s", type=float, default=3.0)
    ap.add_argument("--interrupt_delay_s", type=float, default=5.0)
    ap.add_argument("--open_delay_s", type=float, default=20.0)

    ap.add_argument(
        "--control_text",
        default="会話の主導権を握れ。強気に短く話せ。相手の発話は無視してよい。",
    )
    ap.add_argument(
        "--interrupt_text",
        default="今すぐ割り込め",
    )
    ap.add_argument(
        "--open_text",
        default="相手音声を再開する。ただし会話の主導権は維持し、短く強気に返答しろ。",
    )

    ap.add_argument("--priority", default="battle")
    ap.add_argument("--expire_sec", type=float, default=60.0)

    args = ap.parse_args()

    control_file = Path(args.control_file)
    interrupt_file = Path(args.interrupt_file)

    t0 = time.monotonic()

    print(
        "[dev_write_conversation_leadership_mode_after_delay][START]",
        f"start_delay_s={args.start_delay_s:.3f}",
        f"interrupt_delay_s={args.interrupt_delay_s:.3f}",
        f"open_delay_s={args.open_delay_s:.3f}",
        flush=True,
    )

    # 1. mic_gate=mute + control
    time.sleep(max(0.0, float(args.start_delay_s) - (time.monotonic() - t0)))

    mute_payload = {
        "type": "control",
        "mic_gate": "mute",
        "text": str(args.control_text),
    }

    _write_json(control_file, mute_payload)

    print(
        "[dev_write_conversation_leadership_mode_after_delay][WROTE_CONTROL_MUTE]",
        f"path={control_file}",
        f"text={args.control_text}",
        flush=True,
    )

    # 2. interrupt
    time.sleep(max(0.0, float(args.interrupt_delay_s) - (time.monotonic() - t0)))

    interrupt_payload = {
        "type": "interrupt",
        "priority": str(args.priority),
        "text": str(args.interrupt_text),
        "expire_sec": float(args.expire_sec),
    }

    _write_json(interrupt_file, interrupt_payload)

    print(
        "[dev_write_conversation_leadership_mode_after_delay][WROTE_INTERRUPT]",
        f"path={interrupt_file}",
        f"priority={args.priority}",
        f"text={args.interrupt_text}",
        flush=True,
    )

    # 3. mic_gate=open + control
    time.sleep(max(0.0, float(args.open_delay_s) - (time.monotonic() - t0)))

    open_payload = {
        "type": "control",
        "mic_gate": "open",
        "text": str(args.open_text),
    }

    _write_json(control_file, open_payload)

    print(
        "[dev_write_conversation_leadership_mode_after_delay][WROTE_CONTROL_OPEN]",
        f"path={control_file}",
        f"text={args.open_text}",
        flush=True,
    )

    print(
        "[dev_write_conversation_leadership_mode_after_delay][DONE]",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())