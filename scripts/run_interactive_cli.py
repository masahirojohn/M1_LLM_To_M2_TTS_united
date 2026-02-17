# scripts/run_interactive_cli.py

from __future__ import annotations

import argparse
from pathlib import Path

from aituber_llm_tts.config import load_config
from aituber_llm_tts.orchestrator import Orchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AItuber LLM+TTS interactive CLI runner (Phase2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    orchestrator = Orchestrator(config)
    orchestrator.run_interactive_cli()


if __name__ == "__main__":
    main()
