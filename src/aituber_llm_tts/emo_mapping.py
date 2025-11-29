# src/aituber_llm_tts/emo_mapping.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml


@dataclass
class EmoTtsMapper:
    mapping: Dict[str, str]

    @classmethod
    def from_yaml(cls, path: Path | None) -> "EmoTtsMapper":
        if path is None or not path.exists():
            # 最小フォールバック
            return cls(mapping={})
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(mapping={str(k): str(v) for k, v in data.items()})

    def get_style_prompt(self, emo_id: str) -> str:
        if emo_id in self.mapping:
            return self.mapping[emo_id]
        # フォールバック（ neutral な女性声 ）
        return "A friendly female voice"
