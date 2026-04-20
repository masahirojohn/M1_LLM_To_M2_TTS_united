from __future__ import annotations

import argparse
import json
from pathlib import Path

from aituber_llm_tts.comment_sources.dummy_json import DummyJsonCommentSource
from aituber_llm_tts.comment_sources.onecomme_json import OneCommeJsonCommentSource
from aituber_llm_tts.config import load_config
from aituber_llm_tts.llm_client import create_llm_client
from aituber_llm_tts.prompt_builder import build_prompt_payload

from aituber_llm_tts.comment_router import CommentRouter
from aituber_llm_tts.comment_store_jsonl import RuntimeStores
from aituber_llm_tts.runtime_guard import RuntimeGuardConfig


def _ensure_speech_emo_id(parsed: dict, state=None) -> dict:
    speech = parsed.get("speech")
    if not isinstance(speech, dict):
        return parsed

    emo_id = speech.get("emo_id")
    if isinstance(emo_id, str) and emo_id.strip():
        return parsed

    sleepy_emo_id = getattr(state, "sleepy_emo_id", None)
    if isinstance(sleepy_emo_id, str) and sleepy_emo_id.strip():
        speech["emo_id"] = sleepy_emo_id.strip()
        return parsed

    emotion_hint = str(speech.get("emotion_hint", "normal")).strip()

    emotion_to_emo_id = {
        "normal": "1_0",
        "happy": "1_1",
        "excited": "2_1",
        "sleepy": "9_1",
        "very_sleepy": "9_2",
        "sad": "7_1",
        "angry": "5_0",
        "surprised": "3_1",
    }

    speech["emo_id"] = emotion_to_emo_id.get(emotion_hint, "1_0")
    return parsed


def _has_end_stream_event(parsed: dict) -> bool:
    events = parsed.get("events", [])
    if not isinstance(events, list):
        return False
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("event_mode") == "end_stream":
            return True
    return False


def _ensure_end_stream_from_runtime_state(
    parsed: dict,
    state=None,
    *,
    auto_end_stream_after_ms: int,
) -> dict:
    if state is None:
        return parsed

    sleepy_mode = bool(getattr(state, "sleepy_mode", False))
    sleepy_duration_ms = int(getattr(state, "sleepy_duration_ms", 0) or 0)

    if not sleepy_mode:
        return parsed
    if sleepy_duration_ms < int(auto_end_stream_after_ms):
        return parsed
    if _has_end_stream_event(parsed):
        return parsed

    events = parsed.get("events")
    if not isinstance(events, list):
        events = []
        parsed["events"] = events

    events.append(
        {
            "type": "event",
            "event_id": "evt_end_auto_001",
            "t_ms": int(auto_end_stream_after_ms),
            "duration_ms": 0,
            "event_mode": "end_stream",
            "trigger_source": "sleepy_timeout",
        }
    )
    return parsed


# ===== 追加（差分3）ここから =====
def _ensure_end_stream_from_admin_priority_inputs(parsed: dict, state=None) -> dict:
    if not isinstance(parsed, dict):
        return parsed
    if state is None:
        return parsed
    if _has_end_stream_event(parsed):
        return parsed

    p = getattr(state, "priority_inputs", None)
    if p is None:
        return parsed

    if not bool(getattr(p, "admin_comment_detected", False)):
        return parsed

    events = parsed.get("events")
    if not isinstance(events, list):
        events = []
        parsed["events"] = events

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
    return parsed
# ===== 追加ここまで =====


def _ensure_end_obs_actions_for_end_stream(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        return parsed

    events = parsed.get("events", [])
    if not isinstance(events, list):
        return parsed

    has_end_stream = any(
        isinstance(ev, dict) and ev.get("event_mode") == "end_stream"
        for ev in events
    )
    if not has_end_stream:
        return parsed

    obs_actions = parsed.get("obs_actions")
    if not isinstance(obs_actions, list):
        obs_actions = []
        parsed["obs_actions"] = obs_actions

    def _has_action(action_name: str) -> bool:
        return any(
            isinstance(a, dict) and a.get("action") == action_name
            for a in obs_actions
        )

    if not _has_action("screen_effect"):
        obs_actions.append(
            {
                "action": "screen_effect",
                "params": {
                    "effect_id": "darken_overlay",
                    "duration_ms": 3000,
                },
            }
        )

    if not _has_action("audio_control"):
        obs_actions.append(
            {
                "action": "audio_control",
                "params": {
                    "target": "bgm",
                    "operation": "fade_out",
                    "duration_ms": 3000,
                },
            }
        )

    if not _has_action("scene_control"):
        obs_actions.append(
            {
                "action": "scene_control",
                "params": {
                    "operation": "switch",
                    "scene": "Ending",
                },
            }
        )

    return parsed


def _attach_runtime_state_to_prompt_payload(
    prompt_payload: dict,
    state=None,
    *,
    auto_end_stream_after_ms: int,
) -> dict:
    if not isinstance(prompt_payload, dict):
        return prompt_payload

    prompt_payload["runtime_state"] = {
        "sleepy_mode": bool(getattr(state, "sleepy_mode", False)),
        "sleepy_emo_id": getattr(state, "sleepy_emo_id", None),
        "comment_rate_ratio_pct": getattr(state, "comment_rate_ratio_pct", None),
        "last_comment_ms": getattr(state, "last_comment_ms", None),
        "last_gift_ms": getattr(state, "last_gift_ms", None),
        "sleepy_start_ms": getattr(state, "sleepy_start_ms", None),
        "sleepy_duration_ms": getattr(state, "sleepy_duration_ms", 0),
        "auto_end_stream_after_ms": int(auto_end_stream_after_ms),
    }
    return prompt_payload


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/default.yaml",
        help="App config path",
    )
    ap.add_argument(
        "--comment_source",
        default="dummy_json",
        choices=["dummy_json", "onecomme_json"],
        help="コメント入力ソース",
    )
    ap.add_argument(
        "--comments_json",
        default="examples/dummy_comments.json",
        help="comments json path",
    )
    ap.add_argument(
        "--max_comments",
        type=int,
        default=5,
        help="最大選択コメント数",
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
        "--out_json",
        default="out/dev_m1_stub/m1_unified_output.json",
        help="Generated unified output path",
    )
    ap.add_argument(
        "--dump_prompt_json",
        default="out/dev_m1_stub/prompt_payload.json",
        help="Prompt payload dump path",
    )
    ap.add_argument(
        "--force_sleepy_duration_ms",
        type=int,
        default=None,
        help="dev only: override runtime sleepy_duration_ms before auto end_stream check",
    )

    # ===== 追加（差分1）=====
    ap.add_argument(
        "--force_admin_detected",
        action="store_true",
        help="dev only: force admin_comment_detected=true for STEP-B test",
    )
    # ===== 追加ここまで =====

    args = ap.parse_args()

    cfg = load_config(args.config)
    auto_end_stream_after_ms = int(cfg.runtime.auto_end_stream_after_ms)

    if args.comment_source == "onecomme_json":
        comments = OneCommeJsonCommentSource(args.comments_json).load_comments()
    else:
        comments = DummyJsonCommentSource(args.comments_json).load_comments()

    stores = RuntimeStores(cfg.paths.out_root)
    router = CommentRouter(
        runtime_cfg=RuntimeGuardConfig(
            queue_max=100,
            llm_max_comments=args.max_comments,
            normal_ttl_ms=30000,
        ),
        stores=stores,
        baseline_comment_rate_per_min=cfg.runtime.baseline_comment_rate_per_min,
        sleepy_threshold_pct_9_1=cfg.runtime.sleepy_threshold_pct_9_1,
        sleepy_threshold_pct_9_2=cfg.runtime.sleepy_threshold_pct_9_2,
    )

    router.ingest_comments(comments, source=args.comment_source)
    state = router.build_state(comments)

    # ===== 追加（差分2）=====
    if args.force_admin_detected:
        if hasattr(state, "priority_inputs") and state.priority_inputs is not None:
            setattr(state.priority_inputs, "admin_comment_detected", True)
            if not getattr(state.priority_inputs, "admin_user_id", None):
                setattr(state.priority_inputs, "admin_user_id", "DEV_ADMIN")
            if getattr(state.priority_inputs, "trigger_reason", None) in (None, "", "sleep_risk"):
                setattr(state.priority_inputs, "trigger_reason", "admin_comment")
    # ===== 追加ここまで =====

    if args.force_sleepy_duration_ms is not None:
        forced = max(0, int(args.force_sleepy_duration_ms))
        setattr(state, "sleepy_duration_ms", forced)
        if forced >= auto_end_stream_after_ms:
            if getattr(state, "sleepy_start_ms", None) is None and getattr(state, "last_comment_ms", None) is not None:
                setattr(state, "sleepy_start_ms", int(getattr(state, "last_comment_ms")) - forced)

    router.log_llm_payload(
        session_id=args.session_id,
        utt_id=args.utt_id,
        state=state,
        source=args.comment_source,
    )

    source_name = args.comment_source

    prompt_payload = build_prompt_payload(
        state=state,
        session_id=args.session_id,
        utt_id=args.utt_id,
        source=source_name,
        system_prompt_path=str(cfg.llm.system_prompt_path),
    )
    prompt_payload = _attach_runtime_state_to_prompt_payload(
        prompt_payload,
        state=state,
        auto_end_stream_after_ms=auto_end_stream_after_ms,
    )

    llm = create_llm_client(cfg)

    if hasattr(llm, "generate_unified_output"):
        parsed = llm.generate_unified_output(
            state=state,
            session_id=args.session_id,
            utt_id=args.utt_id,
            source=source_name,
        )
    elif hasattr(llm, "generate_unified_output_stub"):
        parsed = llm.generate_unified_output_stub(
            state=state,
            session_id=args.session_id,
            utt_id=args.utt_id,
            source=source_name,
        )
    else:
        raise TypeError(
            f"Unsupported llm client: {type(llm).__name__} "
            "(missing generate_unified_output / generate_unified_output_stub)"
        )

    parsed = _ensure_speech_emo_id(parsed, state=state)

    # ===== 修正（差分4）=====
    parsed = _ensure_end_stream_from_runtime_state(
        parsed,
        state=state,
        auto_end_stream_after_ms=auto_end_stream_after_ms,
    )
    parsed = _ensure_end_stream_from_admin_priority_inputs(parsed, state=state)
    parsed = _ensure_end_obs_actions_for_end_stream(parsed)
    # ===== 修正ここまで =====

    router.log_events(
        session_id=args.session_id,
        utt_id=args.utt_id,
        events=parsed.get("events", []),
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dump_prompt_json = Path(args.dump_prompt_json)
    dump_prompt_json.parent.mkdir(parents=True, exist_ok=True)
    dump_prompt_json.write_text(
        json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[OK] llm provider:", cfg.llm.provider)
    print("[OK] llm model   :", cfg.llm.model)
    print("[OK] wrote prompt payload:", dump_prompt_json.as_posix())
    print("[OK] wrote unified output:", out_json.as_posix())
    print()
    print("=== speech ===")
    print(json.dumps(parsed["speech"], ensure_ascii=False, indent=2))
    print()
    print("=== events ===")
    print(json.dumps(parsed["events"], ensure_ascii=False, indent=2))
    print()
    print("=== obs_actions ===")
    print(json.dumps(parsed["obs_actions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
