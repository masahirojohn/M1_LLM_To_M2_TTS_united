#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aituber_llm_tts.obs_router import OBSActionRouter, OBSRouterConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs_actions_json", required=True)
    ap.add_argument(
        "--mode",
        default="dry_run",
        choices=["dry_run", "websocket", "html_overlay_ws"],
    )
    ap.add_argument(
        "--log_jsonl",
        default="out/dev_obs_router/obs_router.log.jsonl",
    )
    ap.add_argument(
        "--websocket_url",
        default="",
        help="将来の websocket / html_overlay_ws 用",
    )
    args = ap.parse_args()

    obs_actions = json.loads(
        Path(args.obs_actions_json).read_text(encoding="utf-8")
    )

    if not isinstance(obs_actions, list):
        raise ValueError("obs_actions_json must be a JSON array")

    router = OBSActionRouter(
        OBSRouterConfig(
            mode=args.mode,
            log_jsonl_path=args.log_jsonl,
            websocket_url=(args.websocket_url or None),
        )
    )
    router.dispatch_all(obs_actions)


if __name__ == "__main__":
    main()