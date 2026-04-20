from __future__ import annotations

import argparse
import json

from aituber_llm_tts.comment_ingest import (
    calc_comment_rate_per_min,
    load_comments,
    select_priority_comments,
)
from aituber_llm_tts.priority_router import build_router_state
from aituber_llm_tts.prompt_builder import build_prompt_payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--comments_json",
        default="examples/dummy_comments.json",
    )
    ap.add_argument(
        "--max_comments",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--session_id",
        default="sess_real_01",
    )
    ap.add_argument(
        "--utt_id",
        default="utt_0001",
    )
    ap.add_argument(
        "--source",
        default="dummy_comments_json",
    )
    args = ap.parse_args()

    comments = load_comments(args.comments_json)
    selected = select_priority_comments(comments, max_comments=args.max_comments)
    rate = calc_comment_rate_per_min(comments)

    state = build_router_state(
        selected_comments=selected,
        comment_rate_per_min=rate,
    )

    payload = build_prompt_payload(
        state=state,
        session_id=args.session_id,
        utt_id=args.utt_id,
        source=args.source,
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()