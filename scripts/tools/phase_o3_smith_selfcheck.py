#!/usr/bin/env python3
"""Phase O3: smith config + optional OBS clone/filter selfcheck.

Does not touch session_loop. O1/O2 source names must remain unchanged.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "live_runtime"))

from obs_runtime_control import (  # noqa: E402
    load_obs_config,
    parse_ws_settings,
    probe_connection,
    set_smith_effect,
)


def _fail(msg: str) -> int:
    print(f"[phase_o3][FAIL] {msg}", flush=True)
    return 1


def main() -> int:
    cfg = load_obs_config()
    settings = parse_ws_settings(cfg)
    print(
        "[phase_o3][config]",
        f"bg={settings.background_source}",
        f"bgm={settings.bgm_source}",
        f"normal={settings.atefuri_normal}",
        f"zoom={settings.atefuri_zoom}",
        f"clones={list(settings.smith_clone_sources)}",
        f"audio={settings.smith_audio_source}",
        f"filter={settings.smith_filter_name}",
        flush=True,
    )
    if settings.background_source != "OBS_BG_Still" or settings.bgm_source != "OBS_BGM":
        return _fail("O1 background/bgm source names changed")
    if settings.atefuri_normal != "OBS_AI_Normal" or settings.atefuri_zoom != "OBS_AI_Zoom":
        return _fail("O2 atefuri source names changed")
    if not settings.smith_clone_sources:
        return _fail("smith.clone_sources missing")
    if not settings.smith_audio_source or not settings.smith_filter_name:
        return _fail("smith.audio_source / filter_name missing")
    if settings.smith_audio_source == settings.bgm_source:
        return _fail("smith.audio_source must not be BGM")
    if settings.smith_filter_name != "Smith_Effect":
        return _fail("smith.filter_name default should be Smith_Effect")

    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "obs_bad.json"
        bad.write_text(
            json.dumps(
                {
                    "websocket": {
                        "host": "127.0.0.1",
                        "port": 1,
                        "password": "",
                        "timeout_sec": 0.2,
                    },
                    "sources": {
                        "background_image": "OBS_BG_Still",
                        "bgm_media": "OBS_BGM",
                    },
                    "smith": {
                        "clone_sources": ["OBS_Smith_1"],
                        "audio_source": "OBS_BGM",
                        "filter_name": "Smith_Effect",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        bgm_reject = set_smith_effect(
            active=True,
            config_file=bad,
            local_config_file=None,
        )
        if bgm_reject.get("ok") or "BGM" not in str(bgm_reject.get("error") or ""):
            return _fail(f"BGM audio_source must be rejected: {bgm_reject}")
        print("[phase_o3][bgm_reject] ok", flush=True)

    session_loop = ROOT / "scripts" / "live_runtime" / "run_mic_input_obs_realtime_session_loop.py"
    sl_text = session_loop.read_text(encoding="utf-8")
    if "set_smith_effect" in sl_text or "smith.clone" in sl_text:
        return _fail("session_loop must not call smith")

    probe = probe_connection()
    print(
        "[phase_o3][probe]",
        f"ok={int(bool(probe.get('ok')))}",
        f"err={probe.get('error') or ''}",
        f"bg={probe.get('background_source')}",
        f"bgm={probe.get('bgm_source')}",
        f"ate_n={probe.get('atefuri_normal')}",
        f"ate_z={probe.get('atefuri_zoom')}",
        flush=True,
    )
    if not probe.get("ok"):
        disconnected = set_smith_effect(active=True)
        if disconnected.get("ok"):
            return _fail("smith start must not report ok when OBS is down")
        print(
            "[phase_o3][disconnected_noop]",
            disconnected.get("error") or "ok=0",
            flush=True,
        )
        print(
            "[phase_o3][SKIP] OBS not connected; config Pass. "
            "Live continues without smith switch.",
            flush=True,
        )
        print("[phase_o3][OK] config_overlay", flush=True)
        return 0

    start = set_smith_effect(active=True)
    print("[phase_o3][start]", json.dumps(start, ensure_ascii=False), flush=True)
    reset = set_smith_effect(active=False)
    print("[phase_o3][reset]", json.dumps(reset, ensure_ascii=False), flush=True)

    if not start.get("ok"):
        print(
            "[phase_o3][WARN] OBS connected but clones/filter missing. "
            f"{start.get('error')}",
            flush=True,
        )
        print("[phase_o3][OK] config_overlay_obs_connected_no_sources", flush=True)
        return 0
    if not reset.get("ok"):
        return _fail(f"smith reset failed: {reset.get('error')}")
    if not start.get("filter_ok") or not reset.get("filter_ok"):
        return _fail("filter enable/disable did not succeed")
    print("[phase_o3][OK] clone_visibility_and_filter", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
