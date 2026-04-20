from __future__ import annotations

from typing import Any, Dict

from .llm_client import (
    M1Output,
    emotion_hint_to_emo_id,
    parse_emo_id,
)


def unified_to_legacy_m1output(
    unified: Dict[str, Any],
    *,
    schema_version: str,
    mode: str,
) -> M1Output:
    speech = unified.get("speech", {}) or {}

    text = str(speech.get("text", "")).strip()

    emotion_hint = str(speech.get("emotion_hint", "normal"))
    emo_id = emotion_hint_to_emo_id(emotion_hint)
    emo_group, emo_level = parse_emo_id(emo_id)

    session_id = str(unified.get("session_id", "sess_real_01"))
    utt_id = str(unified.get("utt_id", "utt_0001"))

    meta = {
        "source": "m1_unified_adapter",
        "input_summary": unified.get("input_summary", {}),
        "priority_inputs": unified.get("priority_inputs", {}),
        "events": unified.get("events", []),
        "obs_actions": unified.get("obs_actions", []),
        "unified_schema": unified.get("schema"),
    }

    return M1Output(
        schema_version=schema_version,
        session_id=session_id,
        utt_id=utt_id,
        mode=mode,
        text=text,
        emo_id=emo_id,
        emo_group=emo_group,
        emo_level=emo_level,
        tts_style_prompt=None,
        next_wait_ms=1500,
        meta=meta,
    )


def legacy_m1output_to_dict(obj: M1Output) -> Dict[str, Any]:
    """
    dev_adapt_m1_unified.py 用。
    dataclass を JSON 出力可能 dict に変換する。
    """
    return {
        "schema_version": obj.schema_version,
        "session_id": obj.session_id,
        "utt_id": obj.utt_id,
        "mode": obj.mode,
        "text": obj.text,
        "emo_id": obj.emo_id,
        "emo_group": obj.emo_group,
        "emo_level": obj.emo_level,
        "tts_style_prompt": obj.tts_style_prompt,
        "next_wait_ms": obj.next_wait_ms,
        "meta": obj.meta,
    }