#!/usr/bin/env python3
"""Phase G1: admin 「今すぐ割り込め」PLAYING gate. Fixtures only. No Live."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LIVE = _ROOT / "scripts" / "live_runtime"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.live_runtime.battle_runtime_admin_api import (  # noqa: E402
    write_interrupt,
    write_interrupt_unless_playing,
    write_leadership_open,
    write_leadership_start,
    write_zoom_leadership_start,
)


def _state_file(td: Path, state: str | None, *, raw: str | None = None) -> Path:
    td.mkdir(parents=True, exist_ok=True)
    path = td / "playback_state.json"
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return path
    path.write_text(
        json.dumps({"ok": True, "state": state, "played_samples": 1}),
        encoding="utf-8",
    )
    return path


def _expect_discarded(
    label: str,
    interrupt: Path,
    state_path: Path | None,
    *,
    sessions_root: Path | None = None,
) -> None:
    sentinel = "SENTINEL_UNCHANGED"
    interrupt.write_text(sentinel, encoding="utf-8")
    result = write_interrupt_unless_playing(
        interrupt_file=interrupt,
        playback_state_file=state_path,
        text="今すぐ割り込め",
        priority="battle",
        expire_sec=60.0,
        sessions_root=sessions_root,
    )
    assert result["discarded"] is True, (label, result)
    assert result["state"] == "PLAYING", (label, result)
    assert interrupt.read_text(encoding="utf-8") == sentinel, label
    print(f"[selfcheck][ok] discard {label} file_unchanged", flush=True)


def _expect_written(
    label: str,
    interrupt: Path,
    state_path: Path | None,
    *,
    sessions_root: Path | None = None,
) -> None:
    if interrupt.exists():
        interrupt.write_text("OLD", encoding="utf-8")
    result = write_interrupt_unless_playing(
        interrupt_file=interrupt,
        playback_state_file=state_path,
        text="今すぐ割り込め",
        priority="battle",
        expire_sec=60.0,
        sessions_root=sessions_root,
    )
    assert result["discarded"] is False, (label, result)
    payload = json.loads(interrupt.read_text(encoding="utf-8"))
    assert payload["type"] == "interrupt", (label, payload)
    assert payload["text"] == "今すぐ割り込め", (label, payload)
    assert payload["priority"] == "battle", (label, payload)
    assert float(payload["expire_sec"]) == 60.0, (label, payload)
    print(
        f"[selfcheck][ok] write {label} state={result.get('state')}",
        flush=True,
    )


def _check_keep_buttons() -> None:
    admin = (_LIVE / "admin_control_panel.py").read_text(encoding="utf-8")
    assert 'st.button("今すぐ割り込め"' in admin
    assert "write_interrupt_unless_playing(" in admin
    assert 'st.button("主導権奪取"' in admin
    assert "write_leadership_start(" in admin
    assert 'st.button("Zoom主導権"' in admin
    assert "write_zoom_leadership_start(" in admin
    assert 'st.button("主導権解除 OPEN"' in admin
    assert "write_leadership_open(" in admin
    assert "asyncio.run" not in admin
    assert "st.number_input" not in admin
    assert "audio_route_profile" in admin
    assert "_maybe_user_voice_talkover_cut_in" not in admin
    print("[selfcheck][ok] buttons/talkover/Q1 keep in admin panel", flush=True)

    loop = (_LIVE / "run_mic_input_obs_realtime_session_loop.py").read_text(
        encoding="utf-8"
    )
    assert "def _maybe_user_voice_talkover_cut_in" in loop
    assert "battle_talkover_cut_in_on_interrupt" in loop
    assert "bind_audio_route_or_refuse" in loop
    assert "--audio_route_profile" in loop
    print("[selfcheck][ok] session_loop talkover/Q1 untouched markers", flush=True)


def _check_leadership_not_gated(td: Path) -> None:
    playing = _state_file(td, "PLAYING")
    interrupt = td / "leadership_interrupt.txt"
    control = td / "leadership_control.txt"
    interrupt.write_text("SENTINEL_LEADERSHIP", encoding="utf-8")
    write_leadership_start(
        control_file=control,
        interrupt_file=interrupt,
        control_text="主導権",
        interrupt_text="今すぐ割り込め",
        priority="battle",
        expire_sec=60.0,
    )
    payload = json.loads(interrupt.read_text(encoding="utf-8"))
    assert payload["text"] == "今すぐ割り込め", payload
    write_zoom_leadership_start(control_file=control, control_text="Zoom主導権")
    write_leadership_open(control_file=control, text="OPEN")
    print("[selfcheck][ok] leadership/zoom still write (not G1-gated)", flush=True)


def _check_write_interrupt_defaults() -> None:
    src = (_LIVE / "battle_runtime_admin_api.py").read_text(encoding="utf-8")
    assert 'priority: str = "battle"' in src
    assert "expire_sec: float = 60.0" in src
    assert "def write_interrupt(" in src
    assert "def write_interrupt_unless_playing(" in src
    print("[selfcheck][ok] write_interrupt signature kept", flush=True)


def main() -> int:
    _check_keep_buttons()
    _check_write_interrupt_defaults()
    with tempfile.TemporaryDirectory() as raw:
        td = Path(raw)
        empty_sessions = td / "empty_sessions"
        empty_sessions.mkdir()
        interrupt = td / "battle_interrupt_live.txt"
        _expect_discarded("PLAYING", interrupt, _state_file(td / "playing", "PLAYING"))
        absent = td / "absent_interrupt.txt"
        assert not absent.exists()
        result = write_interrupt_unless_playing(
            interrupt_file=absent,
            playback_state_file=_state_file(td / "playing2", "PLAYING"),
            text="今すぐ割り込め",
            priority="battle",
            expire_sec=60.0,
        )
        assert result["discarded"] is True, result
        assert not absent.exists()
        print("[selfcheck][ok] discard PLAYING created_no_interrupt_file", flush=True)

        for name in ("IDLE", "BUFFERING", "REBUFFERING", "UNKNOWN"):
            _expect_written(
                name,
                td / f"int_{name}.txt",
                _state_file(td / name.lower(), name),
            )
        _expect_written("MISSING_path", td / "int_missing.txt", td / "no_such.json")
        _expect_written(
            "MISSING_none",
            td / "int_none.txt",
            None,
            sessions_root=empty_sessions,
        )
        _expect_written(
            "UNREADABLE",
            td / "int_bad.txt",
            _state_file(td / "bad", None, raw="{not-json"),
        )
        empty = td / "empty" / "playback_state.json"
        empty.parent.mkdir(parents=True, exist_ok=True)
        empty.write_text("", encoding="utf-8")
        _expect_written("EMPTY", td / "int_empty.txt", empty)

        sessions = td / "sessions"
        sess = sessions / "sess_fixture" / "sync"
        sess.mkdir(parents=True)
        (sess / "playback_state.json").write_text(
            json.dumps({"state": "IDLE"}),
            encoding="utf-8",
        )
        _expect_written(
            "AUTO_LATEST_IDLE",
            td / "int_latest.txt",
            None,
            sessions_root=sessions,
        )
        playing_sessions = td / "playing_sessions"
        pdir = playing_sessions / "sess_playing" / "sync"
        pdir.mkdir(parents=True)
        (pdir / "playback_state.json").write_text(
            json.dumps({"state": "PLAYING"}),
            encoding="utf-8",
        )
        _expect_discarded(
            "AUTO_LATEST_PLAYING",
            td / "int_latest_playing.txt",
            None,
            sessions_root=playing_sessions,
        )

        _check_leadership_not_gated(td / "lead")
        # write_interrupt itself is ungated (leadership path).
        other = td / "direct_interrupt.txt"
        write_interrupt(
            interrupt_file=other,
            text="今すぐ割り込め",
            priority="battle",
            expire_sec=60.0,
        )
        assert json.loads(other.read_text(encoding="utf-8"))["priority"] == "battle"
        print("[selfcheck][ok] write_interrupt still writes", flush=True)
    print("[selfcheck][ok] phase_g1 fixtures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
