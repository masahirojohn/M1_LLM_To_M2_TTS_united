from __future__ import annotations

from pathlib import Path
from typing import List

from aituber_llm_tts.comment_ingest import NormalizedComment, load_comments_from_any_json
from .base import CommentSource, ensure_path


class DummyJsonCommentSource(CommentSource):
    def __init__(self, path: str | Path) -> None:
        self.path = ensure_path(path)

    def load_comments(self) -> List[NormalizedComment]:
        return load_comments_from_any_json(self.path)