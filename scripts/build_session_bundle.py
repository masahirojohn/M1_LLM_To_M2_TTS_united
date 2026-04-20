#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_id", required=True)
    ap.add_argument("--audio_ms", type=int, required=True)
    ap.add_argument("--mouth_json", required=True)
    ap.add_argument("--expr_json", required=True)
    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--audio_path", default=None)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = {
        "schema_version": "session_runtime_v0.2",
        "session_id": args.session_id,
        "session_audio_ms": args.audio_ms,
        "mouth_timeline": args.mouth_json,
        "expression_timeline": args.expr_json,
        "pose_timeline": args.pose_json,
    }

    if args.audio_path:
        session["audio_path"] = args.audio_path

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)

    print("[build_session_bundle][OK]")
    print(f"  out_json: {out_path}")
    print(f"  session_id: {args.session_id}")
    print(f"  audio_ms: {args.audio_ms}")


if __name__ == "__main__":
    main()