from __future__ import annotations

import argparse
import json

from aituber_llm_tts.comment_ingest import (
    calc_comment_rate_per_min,
    load_comments,
    select_priority_comments,
)
from aituber_llm_tts.priority_router import (
    build_router_state,
    router_state_to_dict,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--comments_json",
        default="examples/dummy_comments.json",
        help="Path to dummy comments json",
    )
    ap.add_argument(
        "--max_comments",
        type=int,
        default=5,
        help="Number of priority-selected comments",
    )
    args = ap.parse_args()

    comments = load_comments(args.comments_json)
    selected = select_priority_comments(comments, max_comments=args.max_comments)
    rate = calc_comment_rate_per_min(comments)

    state = build_router_state(
        selected_comments=selected,
        comment_rate_per_min=rate,
    )

    print(json.dumps(router_state_to_dict(state), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()