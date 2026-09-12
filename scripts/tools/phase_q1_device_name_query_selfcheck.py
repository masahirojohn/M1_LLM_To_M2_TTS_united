#!/usr/bin/env python3
"""Phase Q1: name-query helper + session_loop bind. No Live. No file bake."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LIVE = _ROOT / "scripts" / "live_runtime"
if str(_LIVE) not in sys.path:
    sys.path.insert(0, str(_LIVE))

from audio_device_name_query import (  # noqa: E402
    AudioRouteResolveError,
    MmeDevice,
    apply_audio_route_profile_to_args,
    bind_audio_route_or_refuse,
    list_mme_devices,
    resolve_audio_route_profile,
)


def _dev(
    index: int,
    name: str,
    *,
    inp: int = 0,
    out: int = 0,
    ha: str = "MME",
) -> MmeDevice:
    return MmeDevice(
        index=index,
        name=name,
        max_input_channels=inp,
        max_output_channels=out,
        hostapi_name=ha,
    )


_BASE = [
    _dev(1, "マイク (2- USB PnP Audio Device)", inp=1),
    _dev(3, "マイク (WebCamera)", inp=1),
    _dev(8, "CABLE Output (VB-Audio Virtual Cable)", inp=2),
    _dev(9, "Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)", inp=8),
    _dev(15, "CABLE In 16ch (VB-Audio Virtual Cable)", out=16),
    _dev(17, "スピーカー (2- USB PnP Audio Device)", out=2),
    _dev(19, "Voicemeeter Input (VB-Audio Voicemeeter VAIO)", out=8),
    _dev(23, "CABLE Input (VB-Audio Virtual Cable)", out=16),
]


def _expect_ok(profile: str, devices: list[MmeDevice], out_i: int, in_i: int) -> None:
    resolved = resolve_audio_route_profile(profile, devices=devices)
    assert resolved.output.index == out_i, (profile, resolved)
    assert resolved.input.index == in_i, (profile, resolved)
    print(
        f"[selfcheck][ok] profile={profile} "
        f"out={resolved.output.index}:{resolved.output.name} "
        f"in={resolved.input.index}:{resolved.input.name}",
        flush=True,
    )


def _expect_refuse(profile: str, devices: list[MmeDevice], needle: str) -> None:
    try:
        resolve_audio_route_profile(profile, devices=devices)
    except AudioRouteResolveError as exc:
        text = exc.format_refuse()
        assert needle in exc.reason or needle in text, (exc.reason, text, needle)
        print(text, flush=True)
        print(f"[selfcheck][refuse-ok] profile={profile} needle={needle!r}", flush=True)
        return
    raise AssertionError(f"expected refuse profile={profile} needle={needle}")


def _check_exclusions() -> None:
    resolved = resolve_audio_route_profile("zoom", devices=_BASE)
    assert "CABLE In 16ch" not in resolved.output.name
    assert "CABLE Output" not in resolved.input.name
    usb = resolve_audio_route_profile("local_usb", devices=_BASE)
    assert "WebCamera" not in usb.input.name
    print("[selfcheck][ok] exclusions CABLE In 16ch / CABLE Output / WebCamera", flush=True)


def _check_session_loop_bind() -> None:
    src = (_LIVE / "run_mic_input_obs_realtime_session_loop.py").read_text(
        encoding="utf-8"
    )
    bind_at = src.find("bind_audio_route_or_refuse(args)")
    run_at = src.find("return asyncio.run(_run(args))")
    assert bind_at > 0 and run_at > bind_at, "bind must run before asyncio.run(_run)"
    assert "time.sleep" not in src[bind_at:run_at]
    print(
        "[selfcheck][ok] session_loop bind before asyncio.run(_run); no sleep",
        flush=True,
    )

    args = argparse.Namespace(
        audio_route_profile="zoom",
        audio_device="15",
        mic_input_device=None,
        ai_audio_output_device=None,
    )
    rc = bind_audio_route_or_refuse(
        args,
        argv=["prog", "--audio_route_profile", "zoom"],
        devices=_BASE,
    )
    assert rc == 0, rc
    assert args.audio_device == 23, args.audio_device
    assert args.mic_input_device == 9, args.mic_input_device
    if args.ai_audio_output_device is None:
        args.ai_audio_output_device = args.audio_device
    print(
        "[battle_audio_routing] "
        f"mic_input_device={args.mic_input_device} "
        f"ai_audio_output_device={args.ai_audio_output_device}",
        flush=True,
    )

    fail_args = argparse.Namespace(
        audio_route_profile="zoom",
        audio_device="15",
        mic_input_device=None,
        ai_audio_output_device=None,
    )
    no_b1 = [d for d in _BASE if "Voicemeeter Out B1" not in d.name]
    rc = bind_audio_route_or_refuse(
        fail_args,
        argv=["prog", "--audio_route_profile", "zoom"],
        devices=no_b1,
    )
    assert rc == 2, rc
    print("[selfcheck][ok] session_loop bind refuse rc=2 (B1 missing)", flush=True)


def _check_no_bake() -> None:
    helper = (_LIVE / "audio_device_name_query.py").read_text(encoding="utf-8")
    banned = (
        "write_text",
        "write_bytes",
        "json.dump",
        "to_json",
        "open(",
    )
    for token in banned:
        assert token not in helper, token
    admin = (_LIVE / "admin_control_panel.py").read_text(encoding="utf-8")
    assert "st.number_input" not in admin
    assert "asyncio.run" not in admin
    assert "バトル開始" not in admin
    print(
        "[selfcheck][ok] helper writes no files; admin has no index edit / Live",
        flush=True,
    )


def _check_live_query() -> None:
    listing = list_mme_devices()
    print(f"[selfcheck][live] mme_n={len(listing)}", flush=True)
    for profile in ("zoom", "local_usb"):
        resolved = resolve_audio_route_profile(profile)
        args = argparse.Namespace(
            audio_route_profile=profile,
            audio_device="15",
            mic_input_device=None,
            ai_audio_output_device=None,
        )
        apply_audio_route_profile_to_args(
            args,
            argv=["prog", "--audio_route_profile", profile],
            devices=listing,
        )
        print(
            f"[selfcheck][live-bind] profile={profile} "
            f"audio_device={args.audio_device} "
            f"mic_input_device={args.mic_input_device} "
            f"out={resolved.output.name!r} in={resolved.input.name!r}",
            flush=True,
        )


def main() -> int:
    _expect_ok("zoom", _BASE, 23, 9)
    _expect_ok("local_usb", _BASE, 17, 1)
    _check_exclusions()
    _expect_refuse("zoom", [], "0件")
    _expect_refuse(
        "zoom",
        [d for d in _BASE if "Voicemeeter Out B1" not in d.name],
        "Voicemeeter Out B1 無し",
    )
    _expect_refuse(
        "zoom",
        _BASE
        + [_dev(40, "CABLE Input (VB-Audio Virtual Cable) copy", out=16)],
        "曖昧",
    )
    _expect_refuse(
        "local_usb",
        [d for d in _BASE if "USB PnP" not in d.name],
        "0件",
    )
    _check_session_loop_bind()
    _check_no_bake()
    _check_live_query()
    print("[selfcheck] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
