from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(r"C:\dev\M1_LLM_To_M2_TTS_united")
DEFAULT_CONTROL_FILE = DEFAULT_ROOT / "in" / "battle_control_live.txt"
DEFAULT_INTERRUPT_FILE = DEFAULT_ROOT / "in" / "battle_interrupt_live.txt"
DEFAULT_EVENT_FILE = DEFAULT_ROOT / "in" / "event_runtime_live.txt"


DEFAULT_LEADERSHIP_CONTROL_TEXT = (
    "会話の主導権を握れ。強気に短く話せ。相手の発話は無視してよい。"
)

DEFAULT_LEADERSHIP_INTERRUPT_TEXT = "今すぐ割り込め"

DEFAULT_ZOOM_LEADERSHIP_CONTROL_TEXT = (
    "会話の主導権を握れ。強気に短く返せ。ただし相手音声は止めるな。"
)

DEFAULT_LEADERSHIP_OPEN_TEXT = (
    "相手音声を再開する。ただし会話の主導権は維持し、短く強気に返答しろ。"
)

DEFAULT_NORMAL_CONVERSATION_TEXT = (
    "通常会話へ戻れ。相手の話を聞いて自然に短く返答しろ。"
)


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def read_text_file(path: Path | str) -> str:
    path = Path(path)
    if not path.exists():
        return "(not found)"
    return path.read_text(encoding="utf-8")


def write_interrupt(
    *,
    interrupt_file: Path | str = DEFAULT_INTERRUPT_FILE,
    text: str = DEFAULT_LEADERSHIP_INTERRUPT_TEXT,
    priority: str = "battle",
    expire_sec: float = 60.0,
) -> dict[str, Any]:
    payload = {
        "type": "interrupt",
        "priority": str(priority),
        "text": str(text),
        "expire_sec": float(expire_sec),
    }

    write_json(interrupt_file, payload)

    return payload


def write_event(
    *,
    event_file: Path | str = DEFAULT_EVENT_FILE,
    event_id: str = "evt_001",
) -> dict[str, Any]:
    event_id = str(event_id or "").strip()

    if not event_id:
        raise ValueError("event_id is empty")

    payload = {
        "type": "event",
        "event_id": event_id,
    }

    write_json(event_file, payload)

    return payload


def write_control(
    *,
    control_file: Path | str = DEFAULT_CONTROL_FILE,
    text: str = "",
    mic_gate: str = "",
    action: str = "set",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "control",
    }

    action = str(action or "set").strip().lower()

    if action == "clear":
        payload["action"] = "clear"
    else:
        payload["text"] = str(text)

    mic_gate = str(mic_gate or "").strip().lower()

    if mic_gate in ("mute", "open"):
        payload["mic_gate"] = mic_gate

    write_json(control_file, payload)

    return payload


def write_mic_gate(
    *,
    control_file: Path | str = DEFAULT_CONTROL_FILE,
    mic_gate: str,
    text: str | None = None,
) -> dict[str, Any]:
    mic_gate = str(mic_gate or "").strip().lower()

    if mic_gate not in ("mute", "open"):
        raise ValueError(f"invalid mic_gate: {mic_gate}")

    if text is None:
        if mic_gate == "mute":
            text = "相手音声を停止する。"
        else:
            text = "相手音声を再開する。"

    return write_control(
        control_file=control_file,
        text=str(text),
        mic_gate=mic_gate,
    )


def clear_control(
    *,
    control_file: Path | str = DEFAULT_CONTROL_FILE,
) -> dict[str, Any]:
    return write_control(
        control_file=control_file,
        action="clear",
    )


def write_leadership_start(
    *,
    control_file: Path | str = DEFAULT_CONTROL_FILE,
    interrupt_file: Path | str = DEFAULT_INTERRUPT_FILE,
    control_text: str = DEFAULT_LEADERSHIP_CONTROL_TEXT,
    interrupt_text: str = DEFAULT_LEADERSHIP_INTERRUPT_TEXT,
    priority: str = "battle",
    expire_sec: float = 60.0,
) -> dict[str, Any]:
    control_payload = write_control(
        control_file=control_file,
        text=control_text,
        mic_gate="mute",
    )

    interrupt_payload = write_interrupt(
        interrupt_file=interrupt_file,
        text=interrupt_text,
        priority=priority,
        expire_sec=expire_sec,
    )

    return {
        "control": control_payload,
        "interrupt": interrupt_payload,
    }


def write_zoom_leadership_start(
    *,
    control_file: Path | str = DEFAULT_CONTROL_FILE,
    control_text: str = DEFAULT_ZOOM_LEADERSHIP_CONTROL_TEXT,
) -> dict[str, Any]:
    return write_control(
        control_file=control_file,
        text=control_text,
    )


def write_leadership_open(
    *,
    control_file: Path | str = DEFAULT_CONTROL_FILE,
    text: str = DEFAULT_LEADERSHIP_OPEN_TEXT,
) -> dict[str, Any]:
    return write_control(
        control_file=control_file,
        text=text,
        mic_gate="open",
    )


def write_normal_conversation(
    *,
    control_file: Path | str = DEFAULT_CONTROL_FILE,
    text: str = DEFAULT_NORMAL_CONVERSATION_TEXT,
) -> dict[str, Any]:
    return write_control(
        control_file=control_file,
        text=text,
        mic_gate="open",
    )


def read_status(
    *,
    control_file: Path | str = DEFAULT_CONTROL_FILE,
    interrupt_file: Path | str = DEFAULT_INTERRUPT_FILE,
    event_file: Path | str = DEFAULT_EVENT_FILE,
) -> dict[str, str]:
    return {
        "control": read_text_file(control_file),
        "interrupt": read_text_file(interrupt_file),
        "event": read_text_file(event_file),
    }
