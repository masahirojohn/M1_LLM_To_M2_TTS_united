#!/usr/bin/env python3
"""Phase O2: atefuri config overlay + optional OBS visibility switch selfcheck."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "live_runtime"))

from obs_runtime_control import (  # noqa: E402
    is_atefuri_enabled,
    load_obs_config,
    parse_ws_settings,
    probe_connection,
    set_atefuri_zoom,
    write_atefuri_live_enabled,
)


def _fail(msg: str) -> int:
    print(f"[phase_o2][FAIL] {msg}", flush=True)
    return 1


def main() -> int:
    cfg = load_obs_config()
    settings = parse_ws_settings(cfg)
    print(
        "[phase_o2][config]",
        f"bg={settings.background_source}",
        f"bgm={settings.bgm_source}",
        f"normal={settings.atefuri_normal}",
        f"zoom={settings.atefuri_zoom}",
        f"enabled_default={int(settings.atefuri_enabled_default)}",
        flush=True,
    )
    if settings.background_source != "OBS_BG_Still" or settings.bgm_source != "OBS_BGM":
        return _fail("O1 background/bgm source names changed")
    if not settings.atefuri_normal or not settings.atefuri_zoom:
        return _fail("atefuri source names missing")
    if not settings.atefuri_enabled_default:
        return _fail("atefuri.enabled default must be true")

    with tempfile.TemporaryDirectory() as td:
        live = Path(td) / "obs_atefuri_live.json"
        if not is_atefuri_enabled(live_file=live):
            return _fail("missing live file must follow config default ON")
        write_atefuri_live_enabled(False, live_file=live)
        if is_atefuri_enabled(live_file=live):
            return _fail("live overlay OFF did not win")
        write_atefuri_live_enabled(True, live_file=live)
        if not is_atefuri_enabled(live_file=live):
            return _fail("live overlay ON did not win")
        print("[phase_o2][overlay] ok", flush=True)

    probe = probe_connection()
    print(
        "[phase_o2][probe]",
        f"ok={int(bool(probe.get('ok')))}",
        f"err={probe.get('error') or ''}",
        f"bg={probe.get('background_source')}",
        f"bgm={probe.get('bgm_source')}",
        flush=True,
    )
    if not probe.get("ok"):
        disconnected = set_atefuri_zoom(zoomed=True)
        if disconnected.get("ok"):
            return _fail("atefuri zoom must not report ok when OBS is down")
        print(
            "[phase_o2][disconnected_noop]",
            disconnected.get("error") or "ok=0",
            flush=True,
        )
        print(
            "[phase_o2][SKIP] OBS not connected; overlay+config Pass. "
            "Live continues without atefuri switch.",
            flush=True,
        )
        print("[phase_o2][OK] config_overlay", flush=True)
        return 0

    zoom_in = set_atefuri_zoom(zoomed=True)
    print("[phase_o2][zoom_in]", json.dumps(zoom_in, ensure_ascii=False), flush=True)
    zoom_out = set_atefuri_zoom(zoomed=False)
    print("[phase_o2][zoom_out]", json.dumps(zoom_out, ensure_ascii=False), flush=True)
    if not zoom_in.get("ok"):
        print(
            "[phase_o2][WARN] OBS connected but sources missing or not in current scene. "
            f"{zoom_in.get('error')}",
            flush=True,
        )
        print("[phase_o2][OK] config_overlay_obs_connected_no_sources", flush=True)
        return 0
    if not zoom_out.get("ok"):
        return _fail(f"zoom restore failed: {zoom_out.get('error')}")
    print("[phase_o2][OK] visibility_switch", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
