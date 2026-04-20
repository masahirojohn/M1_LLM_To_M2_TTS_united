from __future__ import annotations

from pathlib import Path
from typing import List, Protocol

from aituber_llm_tts.comment_ingest import NormalizedComment


class CommentSource(Protocol):
    def load_comments(self) -> List[NormalizedComment]:
        ...


def ensure_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)