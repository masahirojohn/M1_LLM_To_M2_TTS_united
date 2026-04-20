from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from aituber_llm_tts.comment_ingest import NormalizedComment


@dataclass
class PriorityInputs:
    admin_comment_detected: bool
    admin_user_id: Optional[str]
    gift_detected: bool
    high_value_gift_detected: bool
    trigger_reason: str


@dataclass
class RouterState:
    comment_rate_per_min: int
    energy_level: int
    sleep_risk: bool
    sleepy_mode: bool
    sleepy_emo_id: Optional[str]
    comment_rate_ratio_pct: Optional[float]
    selected_comments: List[NormalizedComment]
    priority_inputs: PriorityInputs


def calc_energy_level(
    comment_rate_per_min: int,
    max_energy: int = 10,
) -> int:
    if comment_rate_per_min <= 0:
        return 0
    if comment_rate_per_min >= max_energy:
        return max_energy
    return comment_rate_per_min


def calc_sleep_risk(
    energy_level: int,
    sleepy_threshold: int = 3,
) -> bool:
    return energy_level <= sleepy_threshold


def calc_sleepy_emo_id(
    *,
    comment_rate_ratio_pct: Optional[float],
    energy_level: int,
    sleepy_threshold_pct_9_1: float = 20.0,
    sleepy_threshold_pct_9_2: float = 10.0,
) -> Optional[str]:
    """
    - comment_rate_ratio_pct がある場合:
      < sleepy_threshold_pct_9_2 -> 9_2
      < sleepy_threshold_pct_9_1 -> 9_1
    - ratio が無い場合:
      後方互換 fallback として energy_level<=3 なら 9_1
    """
    if comment_rate_ratio_pct is not None:
        if comment_rate_ratio_pct < sleepy_threshold_pct_9_2:
            return "9_2"
        if comment_rate_ratio_pct < sleepy_threshold_pct_9_1:
            return "9_1"
        return None

    if calc_sleep_risk(energy_level=energy_level):
        return "9_1"

    return None


def build_priority_inputs(selected_comments: List[NormalizedComment]) -> PriorityInputs:
    admin_comments = [c for c in selected_comments if c.priority == "admin"]
    high_value_gifts = [c for c in selected_comments if c.priority == "high_value_gift"]
    gifts = [c for c in selected_comments if c.priority in ("gift", "high_value_gift")]

    admin_comment_detected = len(admin_comments) > 0
    high_value_gift_detected = len(high_value_gifts) > 0
    gift_detected = len(gifts) > 0
    admin_user_id = admin_comments[0].user_id if admin_comments else None

    trigger_reason = "normal_comment"

    if admin_comment_detected:
        admin_text = admin_comments[0].comment.strip()
        if "配信終了" in admin_text:
            trigger_reason = "end_stream_request"
        elif "event" in admin_text:
            trigger_reason = "admin_event_command"
        else:
            trigger_reason = "admin_comment"
    elif high_value_gift_detected:
        trigger_reason = "high_value_gift"
    elif gift_detected:
        trigger_reason = "gift_priority"

    return PriorityInputs(
        admin_comment_detected=admin_comment_detected,
        admin_user_id=admin_user_id,
        gift_detected=gift_detected,
        high_value_gift_detected=high_value_gift_detected,
        trigger_reason=trigger_reason,
    )


def build_router_state(
    selected_comments: List[NormalizedComment],
    comment_rate_per_min: int,
    comment_rate_ratio_pct: Optional[float] = None,
    sleepy_threshold_pct_9_1: float = 20.0,
    sleepy_threshold_pct_9_2: float = 10.0,
) -> RouterState:
    energy_level = calc_energy_level(comment_rate_per_min=comment_rate_per_min)
    sleepy_emo_id = calc_sleepy_emo_id(
        comment_rate_ratio_pct=comment_rate_ratio_pct,
        energy_level=energy_level,
        sleepy_threshold_pct_9_1=sleepy_threshold_pct_9_1,
        sleepy_threshold_pct_9_2=sleepy_threshold_pct_9_2,
    )
    sleepy_mode = sleepy_emo_id is not None
    sleep_risk = sleepy_mode or calc_sleep_risk(energy_level=energy_level)

    priority_inputs = build_priority_inputs(selected_comments)

    if priority_inputs.trigger_reason == "normal_comment" and sleepy_mode:
        priority_inputs = PriorityInputs(
            admin_comment_detected=priority_inputs.admin_comment_detected,
            admin_user_id=priority_inputs.admin_user_id,
            gift_detected=priority_inputs.gift_detected,
            high_value_gift_detected=priority_inputs.high_value_gift_detected,
            trigger_reason="sleep_risk",
        )

    return RouterState(
        comment_rate_per_min=comment_rate_per_min,
        energy_level=energy_level,
        sleep_risk=sleep_risk,
        sleepy_mode=sleepy_mode,
        sleepy_emo_id=sleepy_emo_id,
        comment_rate_ratio_pct=comment_rate_ratio_pct,
        selected_comments=selected_comments,
        priority_inputs=priority_inputs,
    )


def router_state_to_dict(state: RouterState) -> Dict[str, Any]:
    return {
        "comment_rate_per_min": state.comment_rate_per_min,
        "energy_level": state.energy_level,
        "sleep_risk": state.sleep_risk,
        "sleepy_mode": state.sleepy_mode,
        "sleepy_emo_id": state.sleepy_emo_id,
        "comment_rate_ratio_pct": state.comment_rate_ratio_pct,
        "selected_comments": [asdict(c) for c in state.selected_comments],
        "priority_inputs": asdict(state.priority_inputs),
    }