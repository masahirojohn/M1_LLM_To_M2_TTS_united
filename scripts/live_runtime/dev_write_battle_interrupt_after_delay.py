from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--delay_s",
        type=float,
        default=1.0,
        help="Seconds to wait before writing interrupt file.",
    )

    ap.add_argument(
        "--interrupt_file",
        required=True,
        help="battle_interrupt_live.txt path",
    )

    ap.add_argument(
        "--text",
        required=True,
        help="Interrupt text",
    )

    ap.add_argument(
        "--priority",
        default="battle",
        help="interrupt priority",
    )

    ap.add_argument(
        "--expire_sec",
        type=float,
        default=60.0,
    )

    args = ap.parse_args()

    path = Path(args.interrupt_file)

    print(
        "[dev_write_battle_interrupt_after_delay][WAIT]",
        f"delay_s={args.delay_s:.3f}",
        flush=True,
    )

    time.sleep(float(args.delay_s))

    payload = {
        "type": "interrupt",
        "priority": str(args.priority),
        "text": str(args.text),
        "expire_sec": float(args.expire_sec),
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        "[dev_write_battle_interrupt_after_delay][WROTE]",
        f"path={path}",
        f"text={args.text}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())