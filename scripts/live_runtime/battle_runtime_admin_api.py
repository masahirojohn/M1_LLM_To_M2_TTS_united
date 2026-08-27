from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from scripts.live_runtime.prompt_dir_runtime_texts import (
        FALLBACK_LEADERSHIP_CONTROL,
        FALLBACK_LEADERSHIP_INTERRUPT,
        FALLBACK_LEADERSHIP_OPEN,
        FALLBACK_NORMAL_CONVERSATION,
        FALLBACK_ZOOM_LEADERSHIP_CONTROL,
        resolve_leadership_control_text,
        resolve_leadership_interrupt_text,
        resolve_leadership_open_text,
        resolve_normal_conversation_text,
        resolve_zoom_leadership_control_text,
    )
except ImportError:
    from prompt_dir_runtime_texts import (
        FALLBACK_LEADERSHIP_CONTROL,
        FALLBACK_LEADERSHIP_INTERRUPT,
        FALLBACK_LEADERSHIP_OPEN,
        FALLBACK_NORMAL_CONVERSATION,
        FALLBACK_ZOOM_LEADERSHIP_CONTROL,
        resolve_leadership_control_text,
        resolve_leadership_interrupt_text,
        resolve_leadership_open_text,
        resolve_normal_conversation_text,
        resolve_zoom_leadership_control_text,
    )
from scripts.live_runtime.obs_runtime_control import (
    DEFAULT_ATEFURI_LIVE_FILE,
    DEFAULT_CONFIG_FILE as DEFAULT_OBS_CONFIG_FILE,
    DEFAULT_LOCAL_CONFIG_FILE as DEFAULT_OBS_LOCAL_CONFIG_FILE,
    is_atefuri_enabled,
    list_catalog_items,
    load_obs_config,
    parse_ws_settings,
    probe_connection,
    set_atefuri_zoom,
    set_background_image,
    set_bgm_media,
    set_smith_effect,
    write_atefuri_live_enabled,
)


DEFAULT_ROOT = Path(r"C:\dev\M1_LLM_To_M2_TTS_united")
DEFAULT_CONTROL_FILE = DEFAULT_ROOT / "in" / "battle_control_live.txt"
DEFAULT_INTERRUPT_FILE = DEFAULT_ROOT / "in" / "battle_interrupt_live.txt"
DEFAULT_EVENT_FILE = DEFAULT_ROOT / "in" / "event_runtime_live.txt"
DEFAULT_VAD_PROFILE_FILE = DEFAULT_ROOT / "in" / "vad_profile_live.txt"

VAD_PROFILE_ALLOWED_SILENCE_MS = frozenset({250, 350})


DEFAULT_LEADERSHIP_CONTROL_TEXT = FALLBACK_LEADERSHIP_CONTROL
DEFAULT_LEADERSHIP_INTERRUPT_TEXT = FALLBACK_LEADERSHIP_INTERRUPT
DEFAULT_ZOOM_LEADERSHIP_CONTROL_TEXT = FALLBACK_ZOOM_LEADERSHIP_CONTROL
DEFAULT_LEADERSHIP_OPEN_TEXT = FALLBACK_LEADERSHIP_OPEN
DEFAULT_NORMAL_CONVERSATION_TEXT = FALLBACK_NORMAL_CONVERSATION


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
    text: str | None = None,
    priority: str = "battle",
    expire_sec: float = 60.0,
    prompt_dir: Path | str | None = None,
) -> dict[str, Any]:
    if text is None:
        text = resolve_leadership_interrupt_text(prompt_dir)
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


def write_vad_profile(
    *,
    vad_profile_file: Path | str = DEFAULT_VAD_PROFILE_FILE,
    silence_ms: int,
) -> dict[str, Any]:
    """Write mic_vad_silence_ms profile for R2 file-watch (plain 250|350 one line)."""
    try:
        value = int(silence_ms)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid vad_profile silence_ms: {silence_ms!r}") from e

    if value not in VAD_PROFILE_ALLOWED_SILENCE_MS:
        raise ValueError(
            f"vad_profile silence_ms must be one of "
            f"{sorted(VAD_PROFILE_ALLOWED_SILENCE_MS)}, got {value}"
        )

    path = Path(vad_profile_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    # R2 watch: plain integer one line (same as existing in/vad_profile_live.txt)
    path.write_text(f"{value}\n", encoding="utf-8")

    return {
        "type": "vad_profile",
        "mic_vad_silence_ms": value,
        "path": str(path),
    }


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
    control_text: str | None = None,
    interrupt_text: str | None = None,
    priority: str = "battle",
    expire_sec: float = 60.0,
    prompt_dir: Path | str | None = None,
) -> dict[str, Any]:
    if control_text is None:
        control_text = resolve_leadership_control_text(prompt_dir)
    if interrupt_text is None:
        interrupt_text = resolve_leadership_interrupt_text(prompt_dir)
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
        prompt_dir=prompt_dir,
    )

    return {
        "control": control_payload,
        "interrupt": interrupt_payload,
    }


def write_zoom_leadership_start(
    *,
    control_file: Path | str = DEFAULT_CONTROL_FILE,
    control_text: str | None = None,
    prompt_dir: Path | str | None = None,
) -> dict[str, Any]:
    if control_text is None:
        control_text = resolve_zoom_leadership_control_text(prompt_dir)
    return write_control(
        control_file=control_file,
        text=control_text,
    )


def write_leadership_open(
    *,
    control_file: Path | str = DEFAULT_CONTROL_FILE,
    text: str | None = None,
    prompt_dir: Path | str | None = None,
) -> dict[str, Any]:
    if text is None:
        text = resolve_leadership_open_text(prompt_dir)
    return write_control(
        control_file=control_file,
        text=text,
        mic_gate="open",
    )


def write_normal_conversation(
    *,
    control_file: Path | str = DEFAULT_CONTROL_FILE,
    text: str | None = None,
    prompt_dir: Path | str | None = None,
) -> dict[str, Any]:
    if text is None:
        text = resolve_normal_conversation_text(prompt_dir)
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
    vad_profile_file: Path | str = DEFAULT_VAD_PROFILE_FILE,
) -> dict[str, str]:
    return {
        "control": read_text_file(control_file),
        "interrupt": read_text_file(interrupt_file),
        "event": read_text_file(event_file),
        "vad_profile": read_text_file(vad_profile_file),
    }


def read_obs_admin_state(
    *,
    obs_config_file: Path | str = DEFAULT_OBS_CONFIG_FILE,
    obs_local_config_file: Path | str | None = DEFAULT_OBS_LOCAL_CONFIG_FILE,
) -> dict[str, Any]:
    """Load OBS catalog + connection probe for admin UI (never raises)."""
    try:
        cfg = load_obs_config(obs_config_file, obs_local_config_file)
        backgrounds = list_catalog_items(cfg, "backgrounds")
        bgm = list_catalog_items(cfg, "bgm")
        sources = cfg.get("sources") or {}
        ws = cfg.get("websocket") or {}
        settings = parse_ws_settings(cfg)
        atefuri = {
            "enabled": is_atefuri_enabled(
                config_file=obs_config_file,
                local_config_file=obs_local_config_file,
            ),
            "enabled_default": settings.atefuri_enabled_default,
            "normal_source": settings.atefuri_normal,
            "zoom_source": settings.atefuri_zoom,
            "scene": settings.atefuri_scene,
            "live_file": str(DEFAULT_ATEFURI_LIVE_FILE),
        }
        smith = {
            "clone_sources": list(settings.smith_clone_sources),
            "audio_source": settings.smith_audio_source,
            "filter_name": settings.smith_filter_name,
            "scene": settings.smith_scene,
        }
        config_error = ""
    except Exception as e:
        backgrounds = []
        bgm = []
        sources = {}
        ws = {}
        atefuri = {
            "enabled": True,
            "enabled_default": True,
            "normal_source": "",
            "zoom_source": "",
            "scene": "",
            "live_file": str(DEFAULT_ATEFURI_LIVE_FILE),
        }
        smith = {
            "clone_sources": [],
            "audio_source": "",
            "filter_name": "",
            "scene": "",
        }
        config_error = f"{type(e).__name__}: {e}"

    probe = probe_connection(
        config_file=obs_config_file,
        local_config_file=obs_local_config_file,
    )
    return {
        "backgrounds": backgrounds,
        "bgm": bgm,
        "sources": sources,
        "websocket": {
            "host": ws.get("host", ""),
            "port": ws.get("port", ""),
            # never expose password to UI status
            "password_set": bool(str(ws.get("password") or "")),
        },
        "probe": probe,
        "config_error": config_error,
        "config_file": str(obs_config_file),
        "atefuri": atefuri,
        "smith": smith,
    }


def write_obs_background(
    *,
    item_id: str,
    obs_config_file: Path | str = DEFAULT_OBS_CONFIG_FILE,
    obs_local_config_file: Path | str | None = DEFAULT_OBS_LOCAL_CONFIG_FILE,
) -> dict[str, Any]:
    return set_background_image(
        item_id=item_id,
        config_file=obs_config_file,
        local_config_file=obs_local_config_file,
    )


def write_obs_bgm(
    *,
    item_id: str,
    obs_config_file: Path | str = DEFAULT_OBS_CONFIG_FILE,
    obs_local_config_file: Path | str | None = DEFAULT_OBS_LOCAL_CONFIG_FILE,
) -> dict[str, Any]:
    return set_bgm_media(
        item_id=item_id,
        config_file=obs_config_file,
        local_config_file=obs_local_config_file,
    )


def write_atefuri_enabled(
    *,
    enabled: bool,
    live_file: Path | str = DEFAULT_ATEFURI_LIVE_FILE,
    restore_when_off: bool = True,
    obs_config_file: Path | str = DEFAULT_OBS_CONFIG_FILE,
    obs_local_config_file: Path | str | None = DEFAULT_OBS_LOCAL_CONFIG_FILE,
) -> dict[str, Any]:
    """Admin toggle. OFF also restores normal source visibility (best-effort)."""
    live = write_atefuri_live_enabled(bool(enabled), live_file=live_file)
    restore: dict[str, Any] | None = None
    if restore_when_off and not bool(enabled):
        restore = set_atefuri_zoom(
            zoomed=False,
            config_file=obs_config_file,
            local_config_file=obs_local_config_file,
        )
    return {"live": live, "restore": restore}


def write_atefuri_zoom(
    *,
    zoomed: bool,
    obs_config_file: Path | str = DEFAULT_OBS_CONFIG_FILE,
    obs_local_config_file: Path | str | None = DEFAULT_OBS_LOCAL_CONFIG_FILE,
) -> dict[str, Any]:
    return set_atefuri_zoom(
        zoomed=zoomed,
        config_file=obs_config_file,
        local_config_file=obs_local_config_file,
    )


def write_smith_effect(
    *,
    active: bool,
    obs_config_file: Path | str = DEFAULT_OBS_CONFIG_FILE,
    obs_local_config_file: Path | str | None = DEFAULT_OBS_LOCAL_CONFIG_FILE,
) -> dict[str, Any]:
    """Admin smith start/reset. Visibility + filter only. Never raises."""
    return set_smith_effect(
        active=active,
        config_file=obs_config_file,
        local_config_file=obs_local_config_file,
    )
