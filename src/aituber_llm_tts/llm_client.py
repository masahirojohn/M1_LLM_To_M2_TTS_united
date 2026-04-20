from __future__ import annotations

from dataclasses import dataclass, asdict
import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import AppConfig
from .m1_output_parser import parse_m1_output_text
from .prompt_builder import (
    build_input_summary_dict,
    build_priority_inputs_dict,
    build_prompt_payload,
)
from .priority_router import RouterState


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
    g, l = parse_emo_id(emo_id)
    if g is None or l is None:
        return "0_0"
    return f"{g}_{l}"


def emotion_hint_to_emo_id(emotion_hint: str) -> str:
    """
    unified output の emotion_hint から既存 emo_id へ寄せる仮マッピング。
    後で調整可能な最小版。
    """
    mapping = {
        "normal": "1_0",
        "happy": "1_1",
        "excited": "2_1",
        "sleepy": "9_1",
        "surprised": "3_1",
        "sad": "7_1",
    }
    return mapping.get(str(emotion_hint or "normal"), "1_0")


# ========= Dummy LLM Client =========


class DummyLLMClient:
    """
    これまで使っていたダミー版。
    OpenAI 版と同じインタフェース（next_utterance）を持つ。
    さらに unified output 向け stub 実行メソッドも持つ。
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
        """
        既存互換の旧 M1Output 返却。
        """
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

    def _build_dummy_raw_unified_output_text(self, state: RouterState) -> str:
        """
        LLM未接続段階の仮出力。
        priority_inputs / sleep_risk を見て speech / events / obs_actions を返す。
        返すのは「生JSON文字列」。
        """
        p = state.priority_inputs

        text = "こんにちは！来てくれてありがとう。"
        emotion_hint = "normal"
        events: List[Dict[str, Any]] = []
        obs_actions: List[Dict[str, Any]] = []

        if p.high_value_gift_detected:
            text = "ギフトありがとう。すごく嬉しいよ。"
            emotion_hint = "happy"

        if state.sleep_risk:
            text = "ちょっと眠いけど、まだ頑張るよ。"
            emotion_hint = "sleepy"
            obs_actions.append(
                {
                    "action": "screen_effect",
                    "params": {
                        "effect_id": "darken_overlay",
                        "duration_ms": 3000,
                    },
                }
            )
            obs_actions.append(
                {
                    "action": "set_energy_level",
                    "params": {
                        "value": state.energy_level,
                        "max": 10,
                    },
                }
            )

        if p.admin_comment_detected:
            admin_comment = ""
            for c in state.selected_comments:
                if c.priority == "admin":
                    admin_comment = c.comment.strip()
                    break

            if "配信終了" in admin_comment:
                text = "そろそろ眠くて限界かも……今日はここまでにするね。"
                emotion_hint = "sleepy"
                events.append(
                    {
                        "type": "event",
                        "event_id": "evt_end_001",
                        "t_ms": 120000,
                        "duration_ms": 0,
                        "event_mode": "end_stream",
                        "trigger_source": "admin_comment",
                    }
                )
                obs_actions.append(
                    {
                        "action": "play_sound",
                        "params": {
                            "sound_id": "sleep_end",
                        },
                    }
                )
            elif "event1発動" in admin_comment:
                text = "ギフトありがとう。ちょっと眠いけど、まだ頑張るよ。"
                emotion_hint = "sleepy"
                events.append(
                    {
                        "type": "event",
                        "event_id": "evt_001",
                        "t_ms": 8000,
                        "duration_ms": 1600,
                        "bg_video": "in/05_0111.mp4",
                        "pose_json": "in/pose_timeline_pnp_fixed_05015.json",
                        "event_mode": "bg_only",
                        "trigger_source": "admin_comment",
                    }
                )

        raw_obj = {
            "schema": "m1_unified_output.v0.1",
            "speech": {
                "text": text,
                "style": "normal",
                "emotion_hint": emotion_hint,
            },
            "events": events,
            "obs_actions": obs_actions,
        }
        return json.dumps(raw_obj, ensure_ascii=False)

    def generate_unified_output_stub(
        self,
        *,
        state: RouterState,
        session_id: str,
        utt_id: Optional[str] = None,
        source: str = "dummy_comments_json",
    ) -> Dict[str, Any]:
        """
        unified output の stub 実行。
        prompt_builder で input_summary / priority_inputs を作り、
        ダミーの生JSON文字列を parser に通して最終 unified output を返す。
        """
        if utt_id is None:
            utt_id = self._next_utt_id()

        input_summary = build_input_summary_dict(state=state, source=source)
        priority_inputs = build_priority_inputs_dict(state=state)

        # 将来の実LLM接続と揃えるため prompt payload も内部生成
        _ = build_prompt_payload(
            state=state,
            session_id=session_id,
            utt_id=utt_id,
            source=source,
            system_prompt_path=str(self._config.llm.system_prompt_path),
        )

        raw_text = self._build_dummy_raw_unified_output_text(state=state)

        parsed = parse_m1_output_text(
            raw_text,
            session_id=session_id,
            utt_id=utt_id,
            input_summary=input_summary,
            priority_inputs=priority_inputs,
        )
        return parsed


# ========= OpenAI ベース Real LLM Client =========


class RealLLMClient:
    """
    OpenAI Chat Completions API を使って M1Output を生成する本番用クライアント。
    さらに unified output 生成メソッドも持つ。
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
        旧 M1Output 返却。
        viewer_comment / episode_theme を含めた payload を LLM に渡す。
        """
        if mode is None:
            mode = self._config.llm.mode
        if utt_id is None:
            utt_id = "utt_auto"

        if episode_theme is None:
            episode_theme = getattr(self._config.session, "episode_theme", None)

        payload: Dict[str, Any] = {
            "session_id": session_id,
            "utt_id": utt_id,
            "turn_id": 1,
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
            schema_version=data.get("schema_version", self._config.schema_version),
            session_id=session_id,
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

    def generate_unified_output(
        self,
        *,
        state: RouterState,
        session_id: str,
        utt_id: Optional[str] = None,
        source: str = "dummy_comments_json",
    ) -> Dict[str, Any]:
        """
        prompt_builder で組んだ system/user prompt を使い、
        LLM に unified output を生成させ、parser で正規化して返す。
        """
        if utt_id is None:
            utt_id = "utt_auto"

        prompt_payload = build_prompt_payload(
            state=state,
            session_id=session_id,
            utt_id=utt_id,
            source=source,
            system_prompt_path=str(self._config.llm.system_prompt_path),
        )

        input_summary = build_input_summary_dict(state=state, source=source)
        priority_inputs = build_priority_inputs_dict(state=state)

        resp = self._client.chat.completions.create(
            model=self._config.llm.model,
            messages=[
                {"role": "system", "content": prompt_payload["system_prompt"]},
                {"role": "user", "content": prompt_payload["user_prompt"]},
            ],
            response_format={"type": "json_object"},
            temperature=self._config.llm.temperature,
        )

        content = resp.choices[0].message.content

        parsed = parse_m1_output_text(
            content,
            session_id=session_id,
            utt_id=utt_id,
            input_summary=input_summary,
            priority_inputs=priority_inputs,
        )
        return parsed


# ========= factory =========


def create_llm_client(config: AppConfig):
    provider = config.llm.provider.lower()
    if provider == "openai":
        return RealLLMClient(config)
    else:
        return DummyLLMClient(config)