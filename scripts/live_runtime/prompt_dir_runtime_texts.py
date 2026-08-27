from __future__ import annotations

from pathlib import Path


INTERRUPT_PROMPT_FILE = "interrupt_prompt.txt"
CONTROL_WRAP_FILE = "control_wrap.txt"
INTERRUPT_RESERVE_WRAP_FILE = "interrupt_reserve_wrap.txt"
LEADERSHIP_CONTROL_FILE = "leadership_control.txt"
LEADERSHIP_INTERRUPT_FILE = "leadership_interrupt.txt"
LEADERSHIP_OPEN_FILE = "leadership_open.txt"
ZOOM_LEADERSHIP_CONTROL_FILE = "zoom_leadership_control.txt"
NORMAL_CONVERSATION_FILE = "normal_conversation.txt"

FALLBACK_INTERRUPT_PROMPT = (
    "【管理者割り込み指示】"
    "相手の発話終了を待たず、今すぐ短く被せて話してください。"
    "人気ライバーのように、テンポよく、1文で返してください。"
    "指示内容: {text}"
)
FALLBACK_CONTROL_WRAP = "【管理者制御】{text}"
FALLBACK_INTERRUPT_RESERVE_WRAP = "【管理者割り込み予約】{text}"
FALLBACK_LEADERSHIP_CONTROL = (
    "会話の主導権を握れ。強気に短く話せ。相手の発話は無視してよい。"
)
FALLBACK_LEADERSHIP_INTERRUPT = "今すぐ割り込め"
FALLBACK_ZOOM_LEADERSHIP_CONTROL = (
    "会話の主導権を握れ。強気に短く返せ。ただし相手音声は止めるな。"
)
FALLBACK_LEADERSHIP_OPEN = (
    "相手音声を再開する。ただし会話の主導権は維持し、短く強気に返答しろ。"
)
FALLBACK_NORMAL_CONVERSATION = (
    "通常会話へ戻れ。相手の話を聞いて自然に短く返答しろ。"
)


def read_prompt_dir_text(prompt_dir: Path | str | None, name: str) -> str | None:
    if prompt_dir is None:
        return None
    path = Path(prompt_dir) / name
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8-sig").strip()


def resolve_prompt_dir_text(
    prompt_dir: Path | str | None,
    name: str,
    fallback: str,
) -> tuple[str, str]:
    loaded = read_prompt_dir_text(prompt_dir, name)
    if loaded is None:
        return fallback, "fallback"
    source = str((Path(prompt_dir) / name).resolve())
    return loaded, source


def format_named_template(template: str, raw_text: str) -> str:
    raw = str(raw_text).strip()
    if "{text}" in template:
        return template.replace("{text}", raw)
    if template:
        return f"{template}{raw}"
    return raw


def format_battle_interrupt_prompt(
    raw_text: str,
    prompt_dir: Path | str | None = None,
) -> tuple[str, str]:
    template, source = resolve_prompt_dir_text(
        prompt_dir,
        INTERRUPT_PROMPT_FILE,
        FALLBACK_INTERRUPT_PROMPT,
    )
    return format_named_template(template, raw_text), source


def format_battle_control_prompt(
    raw_text: str,
    prompt_dir: Path | str | None = None,
) -> tuple[str, str]:
    template, source = resolve_prompt_dir_text(
        prompt_dir,
        CONTROL_WRAP_FILE,
        FALLBACK_CONTROL_WRAP,
    )
    return format_named_template(template, raw_text), source


def format_battle_interrupt_reserve_prompt(
    raw_text: str,
    prompt_dir: Path | str | None = None,
) -> tuple[str, str]:
    loaded = read_prompt_dir_text(prompt_dir, INTERRUPT_RESERVE_WRAP_FILE)
    if loaded is None:
        template, source = resolve_prompt_dir_text(
            prompt_dir,
            INTERRUPT_PROMPT_FILE,
            FALLBACK_INTERRUPT_RESERVE_WRAP,
        )
        if source == "fallback":
            template = FALLBACK_INTERRUPT_RESERVE_WRAP
    else:
        template = loaded
        source = str((Path(prompt_dir) / INTERRUPT_RESERVE_WRAP_FILE).resolve())
    return format_named_template(template, raw_text), source


def resolve_leadership_control_text(
    prompt_dir: Path | str | None = None,
) -> str:
    text, _source = resolve_prompt_dir_text(
        prompt_dir,
        LEADERSHIP_CONTROL_FILE,
        FALLBACK_LEADERSHIP_CONTROL,
    )
    return text


def resolve_leadership_interrupt_text(
    prompt_dir: Path | str | None = None,
) -> str:
    text, _source = resolve_prompt_dir_text(
        prompt_dir,
        LEADERSHIP_INTERRUPT_FILE,
        FALLBACK_LEADERSHIP_INTERRUPT,
    )
    return text


def resolve_leadership_open_text(
    prompt_dir: Path | str | None = None,
) -> str:
    text, _source = resolve_prompt_dir_text(
        prompt_dir,
        LEADERSHIP_OPEN_FILE,
        FALLBACK_LEADERSHIP_OPEN,
    )
    return text


def resolve_zoom_leadership_control_text(
    prompt_dir: Path | str | None = None,
) -> str:
    text, _source = resolve_prompt_dir_text(
        prompt_dir,
        ZOOM_LEADERSHIP_CONTROL_FILE,
        FALLBACK_ZOOM_LEADERSHIP_CONTROL,
    )
    return text


def resolve_normal_conversation_text(
    prompt_dir: Path | str | None = None,
) -> str:
    text, _source = resolve_prompt_dir_text(
        prompt_dir,
        NORMAL_CONVERSATION_FILE,
        FALLBACK_NORMAL_CONVERSATION,
    )
    return text
