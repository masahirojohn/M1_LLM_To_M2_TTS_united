from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List


M1_SCHEMA_NAME = "m1_unified_output.v0.1"


def build_empty_m1_output(
    session_id: str,
    utt_id: str,
    input_summary: Dict[str, Any],
    priority_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema": M1_SCHEMA_NAME,
        "session_id": session_id,
        "utt_id": utt_id,
        "input_summary": deepcopy(input_summary),
        "priority_inputs": deepcopy(priority_inputs),
        "speech": {
            "text": "",
            "style": "normal",
            "emotion_hint": "normal",
        },
        "events": [],
        "obs_actions": [],
    }


def _normalize_speech(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {
            "text": "",
            "style": "normal",
            "emotion_hint": "normal",
        }

    return {
        "text": str(raw.get("text", "")),
        "style": str(raw.get("style", "normal")),
        "emotion_hint": str(raw.get("emotion_hint", "normal")),
    }


def _normalize_event(raw: Any) -> Dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    event_type = str(raw.get("type", "event"))
    event_id = str(raw.get("event_id", "evt_unknown"))
    t_ms = int(raw.get("t_ms", 0))
    duration_ms = int(raw.get("duration_ms", 0))
    event_mode = str(raw.get("event_mode", ""))

    if event_mode not in ("bg_only", "end_stream"):
        return None

    out = {
        "type": event_type,
        "event_id": event_id,
        "t_ms": t_ms,
        "duration_ms": duration_ms,
        "event_mode": event_mode,
        "trigger_source": str(raw.get("trigger_source", "llm")),
    }

    if event_mode == "bg_only":
        out["bg_video"] = str(raw.get("bg_video", ""))
        out["pose_json"] = str(raw.get("pose_json", ""))

    return out


def _normalize_obs_action(raw: Any) -> Dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    action = str(raw.get("action", ""))
    params = raw.get("params", {})

    if not isinstance(params, dict):
        params = {}

    allowed_actions = {
        "set_background",
        "play_sound",
        "set_energy_level",
        "show_barrage",
        "set_timer_state",
        "screen_effect",
    }

    if action not in allowed_actions:
        return None

    return {
        "action": action,
        "params": params,
    }


def normalize_m1_output_dict(
    raw_obj: Dict[str, Any],
    *,
    session_id: str,
    utt_id: str,
    input_summary: Dict[str, Any],
    priority_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    out = build_empty_m1_output(
        session_id=session_id,
        utt_id=utt_id,
        input_summary=input_summary,
        priority_inputs=priority_inputs,
    )

    out["speech"] = _normalize_speech(raw_obj.get("speech", {}))

    raw_events = raw_obj.get("events", [])
    if isinstance(raw_events, list):
        events: List[Dict[str, Any]] = []
        for item in raw_events:
            ev = _normalize_event(item)
            if ev is not None:
                events.append(ev)
        out["events"] = events

    raw_obs_actions = raw_obj.get("obs_actions", [])
    if isinstance(raw_obs_actions, list):
        obs_actions: List[Dict[str, Any]] = []
        for item in raw_obs_actions:
            act = _normalize_obs_action(item)
            if act is not None:
                obs_actions.append(act)
        out["obs_actions"] = obs_actions

    return out


def parse_m1_output_text(
    raw_text: str,
    *,
    session_id: str,
    utt_id: str,
    input_summary: Dict[str, Any],
    priority_inputs: Dict[str, Any],
) -> Dict[str, Any]:
    fallback = build_empty_m1_output(
        session_id=session_id,
        utt_id=utt_id,
        input_summary=input_summary,
        priority_inputs=priority_inputs,
    )

    try:
        raw_obj = json.loads(raw_text)
    except Exception:
        return fallback

    if not isinstance(raw_obj, dict):
        return fallback

    schema = str(raw_obj.get("schema", ""))
    if schema and schema != M1_SCHEMA_NAME:
        return fallback

    return normalize_m1_output_dict(
        raw_obj,
        session_id=session_id,
        utt_id=utt_id,
        input_summary=input_summary,
        priority_inputs=priority_inputs,
    )