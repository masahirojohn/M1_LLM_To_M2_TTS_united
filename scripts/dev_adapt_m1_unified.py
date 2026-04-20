#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aituber_llm_tts.config import load_config
from aituber_llm_tts.m1_unified_adapter import (
    unified_to_legacy_m1output,
    legacy_m1output_to_dict,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    unified = json.loads(
        Path(args.input_json).read_text(encoding="utf-8")
    )

    adapted = unified_to_legacy_m1output(
        unified,
        schema_version=cfg.schema_version,
        mode=cfg.llm.mode,
    )

    out_obj = legacy_m1output_to_dict(adapted)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()