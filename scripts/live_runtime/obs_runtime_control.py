"""OBS WebSocket control for Phase O1 (admin), O2 (atefuri), O3 (smith).

O1 background/BGM: admin panel only.
O2 atefuri: visibility show/hide of pre-placed sources. May be called from
session_loop as fire-and-forget (never block the Live loop). No Python scale.
O3 smith: admin explicit button. Pre-placed clone visibility + audio filter
enable. All-at-once (no stagger sleep). No transform. Not first_audio.

Does not touch VirtualCam / pose / AI PCM. Connection failures are returned
as Result dicts; callers must not crash the Live pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


DEFAULT_ROOT = Path(r"C:\dev\M1_LLM_To_M2_TTS_united")
DEFAULT_CONFIG_FILE = DEFAULT_ROOT / "in" / "obs_control_config.json"
DEFAULT_LOCAL_CONFIG_FILE = DEFAULT_ROOT / "in" / "obs_control_config.local.json"
DEFAULT_ATEFURI_LIVE_FILE = DEFAULT_ROOT / "in" / "obs_atefuri_live.json"

# Media restart action (obs-websocket v5)
_MEDIA_RESTART = "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_RESTART"

_atefuri_obs_lock = threading.Lock()
_smith_obs_lock = threading.Lock()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_obs_config(
    config_file: Path | str = DEFAULT_CONFIG_FILE,
    local_config_file: Path | str | None = DEFAULT_LOCAL_CONFIG_FILE,
) -> dict[str, Any]:
    path = Path(config_file)
    if not path.exists():
        raise FileNotFoundError(f"obs config not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("obs config root must be an object")

    if local_config_file is not None:
        local_path = Path(local_config_file)
        if local_path.exists():
            local = json.loads(local_path.read_text(encoding="utf-8"))
            if isinstance(local, dict):
                # Accept either nested {"websocket":{...}} or flat
                # {"host","port","password"} (common local-file mistake).
                local = _normalize_local_overlay(local)
                data = _deep_merge(data, local)

    # Env overrides (never hardcode secrets in source)
    ws = data.setdefault("websocket", {})
    if not isinstance(ws, dict):
        raise ValueError("obs config.websocket must be an object")

    env_host = os.environ.get("OBS_WS_HOST", "").strip()
    env_port = os.environ.get("OBS_WS_PORT", "").strip()
    env_password = os.environ.get("OBS_WS_PASSWORD")

    if env_host:
        ws["host"] = env_host
    if env_port:
        ws["port"] = int(env_port)
    if env_password is not None:
        ws["password"] = env_password

    return data


def _normalize_local_overlay(local: dict[str, Any]) -> dict[str, Any]:
    """Map flat host/port/password into websocket.* before merge."""
    flat_keys = ("host", "port", "password", "timeout_sec")
    if any(k in local for k in flat_keys) and "websocket" not in local:
        ws = {k: local[k] for k in flat_keys if k in local}
        rest = {k: v for k, v in local.items() if k not in flat_keys}
        out = dict(rest)
        out["websocket"] = ws
        return out
    return local


def resolve_media_path(path_str: str, *, root: Path = DEFAULT_ROOT) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


@dataclass(frozen=True)
class ObsWsSettings:
    host: str
    port: int
    password: str
    timeout_sec: float
    background_source: str
    bgm_source: str
    atefuri_normal: str = ""
    atefuri_zoom: str = ""
    atefuri_scene: str = ""
    atefuri_enabled_default: bool = True
    smith_clone_sources: tuple[str, ...] = ()
    smith_audio_source: str = ""
    smith_filter_name: str = ""
    smith_scene: str = ""


def parse_ws_settings(cfg: dict[str, Any]) -> ObsWsSettings:
    ws = cfg.get("websocket") or {}
    sources = cfg.get("sources") or {}
    atefuri = cfg.get("atefuri") or {}
    smith = cfg.get("smith") or {}
    if not isinstance(ws, dict) or not isinstance(sources, dict):
        raise ValueError("obs config websocket/sources must be objects")
    if not isinstance(atefuri, dict):
        atefuri = {}
    if not isinstance(smith, dict):
        smith = {}

    host = str(ws.get("host") or "localhost").strip() or "localhost"
    port = int(ws.get("port") or 4455)
    password = str(ws.get("password") or "")
    timeout_sec = float(ws.get("timeout_sec") or 3.0)
    bg_src = str(sources.get("background_image") or "").strip()
    bgm_src = str(sources.get("bgm_media") or "").strip()
    if not bg_src or not bgm_src:
        raise ValueError(
            "obs config sources.background_image and sources.bgm_media are required"
        )

    enabled_default = True
    if "enabled" in atefuri:
        enabled_default = bool(atefuri.get("enabled"))

    return ObsWsSettings(
        host=host,
        port=port,
        password=password,
        timeout_sec=timeout_sec,
        background_source=bg_src,
        bgm_source=bgm_src,
        atefuri_normal=str(sources.get("atefuri_normal") or "").strip(),
        atefuri_zoom=str(sources.get("atefuri_zoom") or "").strip(),
        atefuri_scene=str(atefuri.get("scene") or "").strip(),
        atefuri_enabled_default=enabled_default,
        smith_clone_sources=_parse_smith_clone_sources(smith.get("clone_sources")),
        smith_audio_source=str(smith.get("audio_source") or "").strip(),
        smith_filter_name=str(smith.get("filter_name") or "").strip(),
        smith_scene=str(smith.get("scene") or "").strip(),
    )


def _parse_smith_clone_sources(raw: Any) -> tuple[str, ...]:
    names: list[str] = []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return ()
    for item in raw:
        name = str(item or "").strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def list_catalog_items(cfg: dict[str, Any], kind: str) -> list[dict[str, str]]:
    """kind: 'backgrounds' | 'bgm'."""
    items = cfg.get(kind) or []
    if not isinstance(items, list):
        return []
    out: list[dict[str, str]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("id") or "").strip()
        label = str(raw.get("label") or item_id).strip()
        path = str(raw.get("path") or "").strip()
        if not item_id or not path:
            continue
        out.append({"id": item_id, "label": label, "path": path})
    return out


@contextmanager
def _quiet_obsws_logs() -> Iterator[None]:
    """obsws-python logs full tracebacks on refuse; keep admin UI quiet."""
    names = ("obsws_python", "obsws_python.baseclient", "ObsClient")
    prev: list[tuple[logging.Logger, int]] = []
    for name in names:
        log = logging.getLogger(name)
        prev.append((log, log.level))
        log.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        for log, level in prev:
            log.setLevel(level)


def _connect(settings: ObsWsSettings) -> Any:
    try:
        import obsws_python as obs
    except ImportError as e:
        raise RuntimeError(
            "obsws-python is not installed (pip install obsws-python)"
        ) from e

    with _quiet_obsws_logs():
        return obs.ReqClient(
            host=settings.host,
            port=settings.port,
            password=settings.password,
            timeout=settings.timeout_sec,
        )


def probe_connection(
    *,
    config_file: Path | str = DEFAULT_CONFIG_FILE,
    local_config_file: Path | str | None = DEFAULT_LOCAL_CONFIG_FILE,
) -> dict[str, Any]:
    """Try OBS WebSocket; never raises — returns ok/error for UI."""
    host = ""
    port = 0
    bg_src = ""
    bgm_src = ""
    ate_n = ""
    ate_z = ""
    smith_clones: tuple[str, ...] = ()
    smith_audio = ""
    smith_filter = ""
    try:
        cfg = load_obs_config(config_file, local_config_file)
        settings = parse_ws_settings(cfg)
        host = settings.host
        port = settings.port
        bg_src = settings.background_source
        bgm_src = settings.bgm_source
        ate_n = settings.atefuri_normal
        ate_z = settings.atefuri_zoom
        smith_clones = settings.smith_clone_sources
        smith_audio = settings.smith_audio_source
        smith_filter = settings.smith_filter_name
        client = _connect(settings)
        try:
            version = client.get_version()
            obs_ver = getattr(version, "obs_version", None) or getattr(
                version, "obsVersion", "?"
            )
        finally:
            try:
                client.base_client.ws.close()
            except Exception:
                pass
        return {
            "ok": True,
            "host": host,
            "port": port,
            "background_source": bg_src,
            "bgm_source": bgm_src,
            "atefuri_normal": ate_n,
            "atefuri_zoom": ate_z,
            "smith_clone_sources": list(smith_clones),
            "smith_audio_source": smith_audio,
            "smith_filter_name": smith_filter,
            "obs_version": str(obs_ver),
            "error": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "host": host,
            "port": port,
            "background_source": bg_src,
            "bgm_source": bgm_src,
            "atefuri_normal": ate_n,
            "atefuri_zoom": ate_z,
            "smith_clone_sources": list(smith_clones),
            "smith_audio_source": smith_audio,
            "smith_filter_name": smith_filter,
            "obs_version": "",
            "error": f"{type(e).__name__}: {e}",
        }


def set_background_image(
    *,
    item_id: str,
    config_file: Path | str = DEFAULT_CONFIG_FILE,
    local_config_file: Path | str | None = DEFAULT_LOCAL_CONFIG_FILE,
    root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    """Switch OBS image source file. Safe on failure (returns ok=False)."""
    item_id = str(item_id or "").strip()
    try:
        cfg = load_obs_config(config_file, local_config_file)
        settings = parse_ws_settings(cfg)
        items = {x["id"]: x for x in list_catalog_items(cfg, "backgrounds")}
        if item_id not in items:
            raise ValueError(f"unknown background id: {item_id!r}")
        media_path = resolve_media_path(items[item_id]["path"], root=root)
        if not media_path.is_file():
            raise FileNotFoundError(f"background file missing: {media_path}")

        client = _connect(settings)
        try:
            client.set_input_settings(
                settings.background_source,
                {"file": str(media_path)},
                True,
            )
        finally:
            try:
                client.base_client.ws.close()
            except Exception:
                pass

        return {
            "ok": True,
            "type": "obs_background",
            "id": item_id,
            "source": settings.background_source,
            "path": str(media_path),
            "error": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "type": "obs_background",
            "id": item_id,
            "source": "",
            "path": "",
            "error": f"{type(e).__name__}: {e}",
        }


def set_bgm_media(
    *,
    item_id: str,
    config_file: Path | str = DEFAULT_CONFIG_FILE,
    local_config_file: Path | str | None = DEFAULT_LOCAL_CONFIG_FILE,
    root: Path = DEFAULT_ROOT,
    restart: bool = True,
) -> dict[str, Any]:
    """Switch OBS media source local_file (BGM). Does not touch AI PCM."""
    item_id = str(item_id or "").strip()
    try:
        cfg = load_obs_config(config_file, local_config_file)
        settings = parse_ws_settings(cfg)
        items = {x["id"]: x for x in list_catalog_items(cfg, "bgm")}
        if item_id not in items:
            raise ValueError(f"unknown bgm id: {item_id!r}")
        media_path = resolve_media_path(items[item_id]["path"], root=root)
        if not media_path.is_file():
            raise FileNotFoundError(f"bgm file missing: {media_path}")

        client = _connect(settings)
        try:
            client.set_input_settings(
                settings.bgm_source,
                {
                    "is_local_file": True,
                    "local_file": str(media_path),
                    "looping": True,
                },
                True,
            )
            if restart:
                try:
                    client.trigger_media_input_action(
                        settings.bgm_source,
                        _MEDIA_RESTART,
                    )
                except Exception:
                    # Some source kinds ignore media actions; file switch still applied.
                    pass
        finally:
            try:
                client.base_client.ws.close()
            except Exception:
                pass

        return {
            "ok": True,
            "type": "obs_bgm",
            "id": item_id,
            "source": settings.bgm_source,
            "path": str(media_path),
            "error": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "type": "obs_bgm",
            "id": item_id,
            "source": "",
            "path": "",
            "error": f"{type(e).__name__}: {e}",
        }


def _obs_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if obj is None:
            break
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return default


def _close_client(client: Any) -> None:
    try:
        client.base_client.ws.close()
    except Exception:
        pass


def _read_atefuri_live(
    live_file: Path | str | None = DEFAULT_ATEFURI_LIVE_FILE,
) -> dict[str, Any] | None:
    if live_file is None:
        return None
    path = Path(live_file)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    return raw


def is_atefuri_enabled(
    *,
    config_file: Path | str = DEFAULT_CONFIG_FILE,
    local_config_file: Path | str | None = DEFAULT_LOCAL_CONFIG_FILE,
    live_file: Path | str | None = DEFAULT_ATEFURI_LIVE_FILE,
) -> bool:
    """Live overlay wins when present; else config atefuri.enabled; else True."""
    try:
        live = _read_atefuri_live(live_file)
        if live is not None and "enabled" in live:
            return bool(live.get("enabled"))
    except Exception:
        pass
    try:
        cfg = load_obs_config(config_file, local_config_file)
        settings = parse_ws_settings(cfg)
        return bool(settings.atefuri_enabled_default)
    except Exception:
        return True


def write_atefuri_live_enabled(
    enabled: bool,
    *,
    live_file: Path | str = DEFAULT_ATEFURI_LIVE_FILE,
) -> dict[str, Any]:
    """Persist admin ON/OFF overlay. Never writes the committed config."""
    path = Path(live_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": bool(enabled)}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "type": "atefuri_live",
        "enabled": bool(enabled),
        "path": str(path),
    }


def set_atefuri_zoom(
    *,
    zoomed: bool,
    config_file: Path | str = DEFAULT_CONFIG_FILE,
    local_config_file: Path | str | None = DEFAULT_LOCAL_CONFIG_FILE,
) -> dict[str, Any]:
    """Show/hide pre-placed normal vs zoom sources. No transform/scale. Never raises."""
    zoomed = bool(zoomed)
    scene = ""
    normal = ""
    zoom = ""
    with _atefuri_obs_lock:
        try:
            cfg = load_obs_config(config_file, local_config_file)
            settings = parse_ws_settings(cfg)
            normal = settings.atefuri_normal
            zoom = settings.atefuri_zoom
            if not normal or not zoom:
                raise ValueError(
                    "obs config sources.atefuri_normal and sources.atefuri_zoom "
                    "are required for atefuri"
                )

            client = _connect(settings)
            try:
                scene = settings.atefuri_scene
                if not scene:
                    cur = client.get_current_program_scene()
                    scene = str(
                        _obs_attr(
                            cur,
                            "current_program_scene_name",
                            "currentProgramSceneName",
                            default="",
                        )
                        or ""
                    ).strip()
                if not scene:
                    raise RuntimeError("OBS current program scene is empty")

                normal_id = int(
                    _obs_attr(
                        client.get_scene_item_id(scene, normal),
                        "scene_item_id",
                        "sceneItemId",
                    )
                )
                zoom_id = int(
                    _obs_attr(
                        client.get_scene_item_id(scene, zoom),
                        "scene_item_id",
                        "sceneItemId",
                    )
                )
                client.set_scene_item_enabled(scene, normal_id, not zoomed)
                client.set_scene_item_enabled(scene, zoom_id, zoomed)
            finally:
                _close_client(client)

            return {
                "ok": True,
                "type": "atefuri",
                "zoomed": zoomed,
                "scene": scene,
                "normal_source": normal,
                "zoom_source": zoom,
                "error": "",
            }
        except Exception as e:
            return {
                "ok": False,
                "type": "atefuri",
                "zoomed": zoomed,
                "scene": scene,
                "normal_source": normal,
                "zoom_source": zoom,
                "error": f"{type(e).__name__}: {e}",
            }


def _resolve_program_scene(client: Any, configured: str) -> str:
    scene = str(configured or "").strip()
    if scene:
        return scene
    cur = client.get_current_program_scene()
    return str(
        _obs_attr(
            cur,
            "current_program_scene_name",
            "currentProgramSceneName",
            default="",
        )
        or ""
    ).strip()


def _smith_result(
    *,
    ok: bool,
    active: bool,
    scene: str,
    clones: list[str],
    applied: list[str],
    audio: str,
    filt: str,
    filter_ok: bool,
    error: str,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "type": "smith",
        "active": bool(active),
        "scene": scene,
        "clone_sources": list(clones),
        "applied": list(applied),
        "audio_source": audio,
        "filter_name": filt,
        "filter_ok": bool(filter_ok),
        "error": error,
    }


def set_smith_effect(
    *,
    active: bool,
    config_file: Path | str = DEFAULT_CONFIG_FILE,
    local_config_file: Path | str | None = DEFAULT_LOCAL_CONFIG_FILE,
) -> dict[str, Any]:
    """Show/hide pre-placed clones + toggle AI audio filter. No transform.

    All-at-once (O3 minimum). Never raises. Must not target the BGM source.
    """
    active = bool(active)
    scene = ""
    clones: list[str] = []
    applied: list[str] = []
    audio = ""
    filt = ""
    filter_ok = False
    errors: list[str] = []
    with _smith_obs_lock:
        try:
            cfg = load_obs_config(config_file, local_config_file)
            settings = parse_ws_settings(cfg)
            clones = list(settings.smith_clone_sources)
            audio = settings.smith_audio_source
            filt = settings.smith_filter_name
            if not clones:
                raise ValueError("obs config smith.clone_sources is empty")
            if not audio or not filt:
                raise ValueError(
                    "obs config smith.audio_source and smith.filter_name "
                    "are required"
                )
            if audio == settings.bgm_source:
                raise ValueError(
                    "smith.audio_source must not be the BGM media source "
                    f"({settings.bgm_source})"
                )

            client = _connect(settings)
            try:
                scene = _resolve_program_scene(client, settings.smith_scene)
                if not scene:
                    raise RuntimeError("OBS current program scene is empty")

                for name in clones:
                    try:
                        item_id = int(
                            _obs_attr(
                                client.get_scene_item_id(scene, name),
                                "scene_item_id",
                                "sceneItemId",
                            )
                        )
                        client.set_scene_item_enabled(scene, item_id, active)
                        applied.append(name)
                    except Exception as e:
                        errors.append(f"{name}: {type(e).__name__}: {e}")

                try:
                    client.set_source_filter_enabled(audio, filt, active)
                    filter_ok = True
                except Exception as e:
                    errors.append(
                        f"filter {audio}/{filt}: {type(e).__name__}: {e}"
                    )
            finally:
                _close_client(client)

            return _smith_result(
                ok=not errors,
                active=active,
                scene=scene,
                clones=clones,
                applied=applied,
                audio=audio,
                filt=filt,
                filter_ok=filter_ok,
                error="; ".join(errors),
            )
        except Exception as e:
            return _smith_result(
                ok=False,
                active=active,
                scene=scene,
                clones=clones,
                applied=applied,
                audio=audio,
                filt=filt,
                filter_ok=filter_ok,
                error=f"{type(e).__name__}: {e}",
            )
