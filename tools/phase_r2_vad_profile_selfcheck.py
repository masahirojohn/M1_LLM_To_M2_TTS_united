#!/usr/bin/env python3
"""Phase R2: offline selfcheck for vad_profile parse / resolve / allowlist / watch."""
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "live_runtime"))

from run_mic_input_obs_realtime_session_loop import (  # noqa: E402
    _parse_vad_profile_file_text,
    _read_vad_profile_file,
    _resolve_mic_vad_silence_ms,
    _start_vad_profile_file_thread,
    _VAD_PROFILE_ALLOWED_SILENCE_MS,
    _VAD_PROFILE_DEFAULT_SILENCE_MS,
)


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


async def _watch_smoke(td: Path) -> None:
    p = td / "vad_profile_live.txt"
    p.write_text("350\n", encoding="utf-8")
    ref: dict[str, int] = {"value": 350}
    loop = asyncio.get_running_loop()
    _start_vad_profile_file_thread(
        path=p,
        silence_ms_ref=ref,
        loop=loop,
        poll_s=0.05,
    )
    await asyncio.sleep(0.15)
    p.write_text("600\n", encoding="utf-8")
    await asyncio.sleep(0.25)
    _expect(ref["value"] == 350, f"reject keep got {ref['value']}")
    p.write_text("250\n", encoding="utf-8")
    deadline = time.time() + 2.0
    while time.time() < deadline and int(ref["value"]) != 250:
        await asyncio.sleep(0.05)
    _expect(ref["value"] == 250, f"watch set 250 got {ref['value']}")


def main() -> int:
    _expect(_VAD_PROFILE_ALLOWED_SILENCE_MS == frozenset({250, 350}), "allowlist")
    _expect(_VAD_PROFILE_DEFAULT_SILENCE_MS == 350, "default")

    _expect(_parse_vad_profile_file_text("350") == 350, "plain 350")
    _expect(_parse_vad_profile_file_text("250") == 250, "plain 250")
    _expect(_parse_vad_profile_file_text("600") is None, "reject 600")
    _expect(_parse_vad_profile_file_text("200") is None, "reject 200")
    _expect(_parse_vad_profile_file_text("") is None, "empty")
    _expect(
        _parse_vad_profile_file_text('{"mic_vad_silence_ms": 250}') == 250,
        "json 250",
    )
    _expect(
        _parse_vad_profile_file_text('{"silence_ms": 600}') is None,
        "json reject 600",
    )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        p = td_path / "vad_profile_live.txt"
        p.write_text("250\n", encoding="utf-8")
        _expect(_read_vad_profile_file(p) == 250, "read file 250")

        ms, src = _resolve_mic_vad_silence_ms(cli_value=350, profile_path=p)
        _expect((ms, src) == (350, "cli"), f"cli wins got {(ms, src)}")

        ms, src = _resolve_mic_vad_silence_ms(cli_value=None, profile_path=p)
        _expect((ms, src) == (250, "file"), f"file wins got {(ms, src)}")

        p.write_text("999\n", encoding="utf-8")
        ms, src = _resolve_mic_vad_silence_ms(cli_value=None, profile_path=p)
        _expect((ms, src) == (350, "default"), f"invalid file → default got {(ms, src)}")

        ms, src = _resolve_mic_vad_silence_ms(cli_value=None, profile_path=None)
        _expect((ms, src) == (350, "default"), f"no file → default got {(ms, src)}")

        asyncio.run(_watch_smoke(td_path))

    print("[phase_r2_vad_profile_selfcheck][OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
