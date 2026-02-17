# src/aituber_llm_tts/llm_client.py

from __future__ import annotations

from dataclasses import dataclass, asdict
import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import AppConfig


# ========= 共通出力型 =========


@dataclass
class M1Output:
    schema_version: str
    session_id: str
    utt_id: str
    mode: str
    text: str
    emo_id: str
    emo_group: Optional[int]
    emo_level: Optional[int]
    tts_style_prompt: Optional[str]
    next_wait_ms: int
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ========= emo_id ユーティリティ =========


def parse_emo_id(emo_id: str) -> tuple[Optional[int], Optional[int]]:
    try:
        g, l = emo_id.split("_", 1)
        return int(g), int(l)
    except Exception:
        return None, None


def sanitize_emo_id(emo_id: Optional[str]) -> str:
    """
    不正 / None の場合は "0_0" にフォールバック。
    """
    if not isinstance(emo_id, str):
        return "0_0"
    emo_id = emo_id.strip()
    if emo_id == "":
        return "0_0"
    # ざっくり形式チェック (数値_数値 ならOK)
    g, l = parse_emo_id(emo_id)
    if g is None or l is None:
        return "0_0"
    return f"{g}_{l}"


# ========= Dummy LLM Client =========


class DummyLLMClient:
    """
    これまで使っていたダミー版。
    OpenAI 版と同じインタフェース（next_utterance）を持つ。
    """

    def __init__(self, config: AppConfig, emo_ids: Optional[List[str]] = None) -> None:
        self._config = config
        self._rng = random.Random(config.session.seed)
        self._utt_counter = 0

        if emo_ids is None:
            emo_ids = ["1_0", "1_1", "2_1", "3_1", "7_1", "9_1", "0_0"]
        self._emo_cycle = itertools.cycle(emo_ids)

        self._text_candidates = [
            "今日はいい天気だね！",
            "見て見て、このリボンかわいいでしょ？",
            "ちょっと緊張してきたかも……。",
            "ねむくなってきちゃった……。",
            "びっくりした！今のコメント面白いね！",
        ]

    def _next_utt_id(self) -> str:
        self._utt_counter += 1
        return f"utt_{self._utt_counter:06d}"

    def next_utterance(
        self,
        session_id: str,
        utt_id: Optional[str] = None,
        mode: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        viewer_comment: Optional[str] = None,
        episode_theme: Optional[str] = None,
    ) -> M1Output:
        # 既存コードとの互換のため、viewer_comment / episode_theme は
        # あってもなくても動く形で扱う
        if utt_id is None:
            utt_id = self._next_utt_id()
        if mode is None:
            mode = self._config.llm.mode

        text = self._rng.choice(self._text_candidates)
        emo_id = next(self._emo_cycle)
        emo_group, emo_level = parse_emo_id(emo_id)
        next_wait_ms = self._rng.randint(1000, 3000)

        meta: Dict[str, Any] = {
            "source": "dummy_llm",
            "viewer_comment_used": bool(viewer_comment),
            "viewer_comment_text": viewer_comment,
            "episode_theme": episode_theme
            or getattr(self._config.session, "episode_theme", None),
        }

        return M1Output(
            schema_version=self._config.schema_version,
            session_id=session_id,
            utt_id=utt_id,
            mode=mode,
            text=text,
            emo_id=emo_id,
            emo_group=emo_group,
            emo_level=emo_level,
            tts_style_prompt=None,
            next_wait_ms=next_wait_ms,
            meta=meta,
        )


# ========= OpenAI ベース Real LLM Client =========


class RealLLMClient:
    """
    OpenAI Chat Completions API を使って M1Output を生成する本番用クライアント。
    """

    def __init__(self, config: AppConfig, client: Optional[OpenAI] = None) -> None:
        self._config = config
        self._client = client or OpenAI()
        self._system_prompt = self._load_system_prompt(config.llm.system_prompt_path)

    @staticmethod
    def _load_system_prompt(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"System prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

    def next_utterance(
        self,
        session_id: str,
        utt_id: Optional[str] = None,
        mode: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        viewer_comment: Optional[str] = None,
        episode_theme: Optional[str] = None,
    ) -> M1Output:
        """
        viewer_comment / episode_theme を含めた payload を LLM に渡す。

        - viewer_comment: 視聴者コメント文字列（なければ None）
        - episode_theme: 引数で渡されなければ config.session.episode_theme を利用
        """
        if mode is None:
            mode = self._config.llm.mode
        if utt_id is None:
            # orchestrator 側で決めるのが基本だが、念のためフォールバック
            utt_id = "utt_auto"

        if episode_theme is None:
            # config 側に episode_theme があればそれを優先
            episode_theme = getattr(self._config.session, "episode_theme", None)

        payload: Dict[str, Any] = {
            "session_id": session_id,
            "utt_id": utt_id,
            "turn_id": 1,  # 必要なら orchestrator で管理
            "mode": mode,
            "history": history or [],
            "viewer_comment": viewer_comment,
            "episode_theme": episode_theme,
            "internal_state": {},
        }

        resp = self._client.chat.completions.create(
            model=self._config.llm.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            response_format={"type": "json_object"},
            temperature=self._config.llm.temperature,
        )

        content = resp.choices[0].message.content
        data = json.loads(content)

        # 必須フィールド抽出＋フォールバック
        text = data.get("text", "").strip()
        emo_id_raw = data.get("emo_id")
        emo_id = sanitize_emo_id(emo_id_raw)
        emo_group = data.get("emo_group")
        emo_level = data.get("emo_level")
        if emo_group is None or emo_level is None:
            g, l = parse_emo_id(emo_id)
            emo_group = g
            emo_level = l

        next_wait_ms = int(data.get("next_wait_ms", 2000))

        return M1Output(
            schema_version=data.get(
                "schema_version", self._config.schema_version
            ),
            session_id=session_id,  # 入力を優先
            utt_id=utt_id,
            mode=data.get("mode", mode),
            text=text,
            emo_id=emo_id,
            emo_group=emo_group,
            emo_level=emo_level,
            tts_style_prompt=data.get("tts_style_prompt"),
            next_wait_ms=next_wait_ms,
            meta=data.get("meta") or {},
        )


# ========= factory =========


def create_llm_client(config: AppConfig):
    provider = config.llm.provider.lower()
    if provider == "openai":
        return RealLLMClient(config)
    else:
        return DummyLLMClient(config)
