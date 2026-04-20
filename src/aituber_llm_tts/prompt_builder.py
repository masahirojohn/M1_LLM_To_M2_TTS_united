from __future__ import annotations

from typing import Dict
from pathlib import Path

from aituber_llm_tts.comment_ingest import NormalizedComment
from aituber_llm_tts.priority_router import RouterState


SYSTEM_PROMPT_TEMPLATE = """あなたはライブ配信中のAI猫キャラクターです。
以下の入力状況を見て、視聴者に自然に返答してください。

絶対条件:
- 返答は短く自然な日本語にする
- キャラクター性を保つ
- 管理者コメントがある場合は最優先で反映する
- コメント率が低く sleep_risk=true の場合は、少し眠そうな雰囲気を出してよい
- 高額ギフトがある場合は、お礼を優先してよい
- 必要に応じて event / OBS action を出してよい
- 出力は JSON のみ
- JSON schema に必ず従う
- input_summary と priority_inputs は与えられた内容をそのまま反映する
"""


def load_system_prompt(system_prompt_path: str | None = None) -> str:
    if system_prompt_path:
        p = Path(system_prompt_path)
        if not p.exists():
            raise FileNotFoundError(f"system prompt file not found: {p}")
        return p.read_text(encoding="utf-8")
    return SYSTEM_PROMPT_TEMPLATE


def _comment_to_line(c: NormalizedComment) -> str:
    return (
        f"[priority={c.priority}] "
        f"service={c.service} "
        f"name={c.name} "
        f"user_id={c.user_id} "
        f"comment={c.comment} "
        f"timestamp={c.timestamp}"
    )


def build_input_summary_dict(
    state: RouterState,
    source: str = "dummy_comments_json",
) -> Dict:
    return {
        "source": source,
        "selected_comments": [
            {
                "service": c.service,
                "name": c.name,
                "user_id": c.user_id,
                "comment": c.comment,
                "timestamp": c.timestamp,
                "priority": c.priority,
            }
            for c in state.selected_comments
        ],
        "comment_rate_per_min": state.comment_rate_per_min,
        "energy_level": state.energy_level,
        "sleep_risk": state.sleep_risk,
    }


def build_priority_inputs_dict(state: RouterState) -> Dict:
    p = state.priority_inputs
    return {
        "admin_comment_detected": p.admin_comment_detected,
        "admin_user_id": p.admin_user_id,
        "gift_detected": p.gift_detected,
        "high_value_gift_detected": p.high_value_gift_detected,
        "trigger_reason": p.trigger_reason,
    }


def build_user_prompt(state: RouterState) -> str:
    comment_lines = "\n".join(_comment_to_line(c) for c in state.selected_comments)

    return f"""以下は現在の配信入力です。

[input_summary]
comment_rate_per_min={state.comment_rate_per_min}
energy_level={state.energy_level}
sleep_risk={state.sleep_risk}

[selected_comments]
{comment_lines}

[priority_inputs]
admin_comment_detected={state.priority_inputs.admin_comment_detected}
admin_user_id={state.priority_inputs.admin_user_id}
gift_detected={state.priority_inputs.gift_detected}
high_value_gift_detected={state.priority_inputs.high_value_gift_detected}
trigger_reason={state.priority_inputs.trigger_reason}

出力仕様:
以下の JSON schema に厳密に従って、JSONのみを返してください。

{{
  "schema": "m1_unified_output.v0.1",
  "session_id": "sess_real_01",
  "utt_id": "utt_0001",
  "input_summary": {{
    "source": "dummy_comments_json",
    "selected_comments": [],
    "comment_rate_per_min": 0,
    "energy_level": 0,
    "sleep_risk": false
  }},
  "priority_inputs": {{
    "admin_comment_detected": false,
    "admin_user_id": null,
    "gift_detected": false,
    "high_value_gift_detected": false,
    "trigger_reason": "normal_comment"
  }},
  "speech": {{
    "text": "",
    "style": "normal",
    "emotion_hint": "normal"
  }},
  "events": [],
  "obs_actions": []
}}

必須ルール:
- input_summary は与えられた内容をそのまま反映すること
- priority_inputs は与えられた内容をそのまま反映すること
- 不明なフィールドを勝手に追加しないこと
- events / obs_actions が不要な場合は空配列にすること

イベント生成ルール:
- 管理者コメントに "配信終了" が含まれる場合、events に end_stream の event を必ず1件入れること
- その end_stream event は以下の内容にすること
  {{
    "type": "event",
    "event_id": "evt_end_001",
    "t_ms": 120000,
    "duration_ms": 0,
    "event_mode": "end_stream",
    "trigger_source": "admin_comment"
  }}
- 上記の t_ms=120000 は仮値としてそのまま出力してよい。後段で補正される
- 管理者コメントに "event1発動" が含まれる場合、events に bg_only の event を必ず1件入れること
- その bg_only event は以下の内容にすること
  {{
    "type": "event",
    "event_id": "evt_001",
    "t_ms": 8000,
    "duration_ms": 1600,
    "bg_video": "in/05_0111.mp4",
    "pose_json": "in/pose_timeline_pnp_fixed_05015.json",
    "event_mode": "bg_only",
    "trigger_source": "admin_comment"
  }}

OBS action ルール:
- sleep_risk=true の場合、obs_actions に以下の screen_effect を入れてよい
  {{
    "action": "screen_effect",
    "params": {{
      "effect_id": "darken_overlay",
      "duration_ms": 3000
    }}
  }}
- 管理者コメントに "配信終了" が含まれる場合、obs_actions に以下をこの順で入れること
  1.
  {{
    "action": "screen_effect",
    "params": {{
      "effect_id": "darken_overlay",
      "duration_ms": 3000
    }}
  }}
  2.
  {{
    "action": "audio_control",
    "params": {{
      "target": "bgm",
      "operation": "fade_out",
      "duration_ms": 3000
    }}
  }}
  3.
  {{
    "action": "scene_control",
    "params": {{
      "operation": "switch",
      "scene": "Ending"
    }}
  }}
- 高額ギフトがある場合、speech で感謝を自然に含めてよい

speech ルール:
- speech.text は短く自然な日本語にすること
- 管理者コメントが "配信終了" の場合は、配信終了の意図が明確に伝わる文にすること
- speech.style は "normal" を基本とすること
- emotion_hint は normal / happy / excited / sleepy / surprised / sad のいずれかを使うこと

最重要:
- 管理者コメントが "配信終了" なのに events を空配列にしてはいけない
- 管理者コメントが "配信終了" なのに obs_actions を空配列にしてはいけない
- 管理者コメントが "event1発動" なのに events を空配列にしてはいけない
"""


def build_prompt_payload(
    state: RouterState,
    session_id: str = "sess_real_01",
    utt_id: str = "utt_0001",
    source: str = "dummy_comments_json",
    system_prompt_path: str | None = None,
) -> Dict:
    return {
        "system_prompt": load_system_prompt(system_prompt_path),
        "user_prompt": build_user_prompt(state),
        "input_summary": build_input_summary_dict(state=state, source=source),
        "priority_inputs": build_priority_inputs_dict(state=state),
        "session_id": session_id,
        "utt_id": utt_id,
        "system_prompt_path": system_prompt_path,
    }


# 終了演出 schema（STEP-B 固定）
# 管理者コメントに "配信終了" が含まれる場合:
# 1) events に end_stream を 1件入れる
# 2) obs_actions に以下を順番どおり入れる
#    - darken_overlay
#    - bgm fade_out
#    - scene switch -> Ending