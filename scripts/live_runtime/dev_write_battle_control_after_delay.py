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
        help="Seconds to wait before writing control file.",
    )

    ap.add_argument(
        "--control_file",
        required=True,
        help="battle_control_live.txt path",
    )

    ap.add_argument(
        "--text",
        default="",
        help="Control text. Empty text means clear if --action is clear.",
    )

    ap.add_argument(
        "--action",
        default="set",
        choices=["set", "clear"],
        help="set: write control text / clear: clear active battle_control",
    )

    ap.add_argument(
        "--mic_gate",
        default="",
        choices=["", "mute", "open"],
        help="Optional mic gate control: mute/open.",
    )

    args = ap.parse_args()

    path = Path(args.control_file)

    print(
        "[dev_write_battle_control_after_delay][WAIT]",
        f"delay_s={args.delay_s:.3f}",
        f"action={args.action}",
        f"mic_gate={args.mic_gate}",
        flush=True,
    )

    time.sleep(float(args.delay_s))

    if args.action == "clear":
        payload = {
            "type": "control",
            "action": "clear",
        }
    else:
        payload = {
            "type": "control",
            "text": str(args.text),
        }

    mic_gate = str(args.mic_gate or "").strip().lower()
    if mic_gate in ("mute", "open"):
        payload["mic_gate"] = mic_gate

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    print(    
        "[dev_write_battle_control_after_delay][WROTE]",
        f"path={path}",
        f"action={args.action}",
        f"mic_gate={args.mic_gate}",
        f"text={args.text}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())