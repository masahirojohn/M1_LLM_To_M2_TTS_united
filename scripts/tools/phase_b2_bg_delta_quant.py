#!/usr/bin/env python3
"""Phase B2: quantify Δ(bg_est, audio_ms) from virtualcam logs / clock model.

Observation-only helper. Does not change sync behavior.
Δ = bg_elapsed_ms - audio_elapsed_ms relative to first PLAYING (or turn) anchor.
Legacy logs use sent≈bg_read (each send path does one cap.read).
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

RE_SYNC = re.compile(
    r"\[sync\]\[virtualcam\](?!\[)"
    r".*?audio_ms=(?P<audio_ms>-?\d+)"
    r".*?frame_offset=(?P<frame_offset>-?\d+)"
    r".*?state=(?P<state>\S+)"
)
RE_IDLE = re.compile(
    r"\[sync\]\[virtualcam\]\[IDLE_BG_ADVANCE\]"
    r".*?state=(?P<state>\S+)"
    r".*?audio_ms=(?P<audio_ms>-?\d+)"
    r".*?sent=(?P<sent>\d+)"
)
RE_B2 = re.compile(
    r"\[sync\]\[virtualcam\]\[B2_OBS\]"
    r".*?state=(?P<state>\S+)"
    r".*?audio_ms=(?P<audio_ms>-?\d+)"
    r".*?bg_read_n=(?P<bg_read_n>\d+)"
    r".*?delta_ms=(?P<delta_ms>-?\d+)"
    r".*?frame_offset=(?P<frame_offset>-?\d+)"
    r".*?sent=(?P<sent>\d+)"
)
RE_SENT = re.compile(r"\[virtualcam_persistent\] sent=(?P<sent>\d+)")


@dataclass
class Sample:
    kind: str
    state: str
    audio_ms: int
    bg_n: int
    frame_offset: int
    delta_ms: int | None = None


def _parse_legacy(text: str, fps: float = 25.0) -> list[Sample]:
    period = 1000.0 / float(max(1.0, fps))
    samples: list[Sample] = []
    last_sent = 0
    for line in text.splitlines():
        m_sent = RE_SENT.search(line)
        if m_sent:
            last_sent = int(m_sent.group("sent"))
        m_b2 = RE_B2.search(line)
        if m_b2:
            samples.append(
                Sample(
                    kind="b2",
                    state=str(m_b2.group("state")),
                    audio_ms=int(m_b2.group("audio_ms")),
                    bg_n=int(m_b2.group("bg_read_n")),
                    frame_offset=int(m_b2.group("frame_offset")),
                    delta_ms=int(m_b2.group("delta_ms")),
                )
            )
            continue
        m_idle = RE_IDLE.search(line)
        if m_idle:
            samples.append(
                Sample(
                    kind="idle",
                    state=str(m_idle.group("state")),
                    audio_ms=int(m_idle.group("audio_ms")),
                    bg_n=int(m_idle.group("sent")),
                    frame_offset=-1,
                )
            )
            continue
        m_sync = RE_SYNC.search(line)
        if m_sync:
            samples.append(
                Sample(
                    kind="sync",
                    state=str(m_sync.group("state")),
                    audio_ms=int(m_sync.group("audio_ms")),
                    bg_n=int(last_sent),
                    frame_offset=int(m_sync.group("frame_offset")),
                )
            )
    # Fill delta for legacy samples from first PLAYING anchor / turn offset.
    anchor_bg: int | None = None
    anchor_audio: int | None = None
    anchor_off: int | None = None
    out: list[Sample] = []
    for s in samples:
        if s.delta_ms is not None:
            out.append(s)
            continue
        if s.state == "PLAYING":
            if anchor_bg is None or (
                s.frame_offset >= 0
                and anchor_off is not None
                and s.frame_offset > int(anchor_off)
            ):
                anchor_bg = int(s.bg_n)
                anchor_audio = int(s.audio_ms)
                anchor_off = int(s.frame_offset) if s.frame_offset >= 0 else anchor_off
        if anchor_bg is None or anchor_audio is None:
            s.delta_ms = 0
        else:
            bg_elapsed = int(round((int(s.bg_n) - int(anchor_bg)) * period))
            audio_elapsed = int(s.audio_ms) - int(anchor_audio)
            s.delta_ms = int(bg_elapsed - audio_elapsed)
        out.append(s)
    return out


def _summarize(samples: list[Sample]) -> dict:
    playing = [s for s in samples if s.state == "PLAYING" and s.delta_ms is not None]
    reb = [s for s in samples if s.state == "REBUFFERING" and s.delta_ms is not None]
    idleish = [
        s
        for s in samples
        if s.state not in ("PLAYING",) and s.delta_ms is not None
    ]
    # Turn boundaries: frame_offset jumps on PLAYING samples.
    turns: list[tuple[int, int]] = []
    prev_off: int | None = None
    for s in samples:
        if s.state != "PLAYING" or s.frame_offset < 0:
            continue
        if prev_off is not None and s.frame_offset > prev_off:
            turns.append((prev_off, s.frame_offset))
        prev_off = s.frame_offset

    def _stats(xs: list[Sample]) -> tuple[int, int, int, int]:
        if not xs:
            return (0, 0, 0, 0)
        ds = [int(s.delta_ms or 0) for s in xs]
        return (min(ds), max(ds), int(round(sum(ds) / len(ds))), ds[-1] - ds[0])

    # REBUFFERING episode growth: consecutive idle/reb with same frozen audio_ms.
    reb_episodes: list[int] = []
    i = 0
    while i < len(samples):
        s = samples[i]
        if s.state != "REBUFFERING":
            i += 1
            continue
        a0 = s.audio_ms
        bg0 = s.bg_n
        j = i
        while j + 1 < len(samples) and samples[j + 1].state != "PLAYING":
            if samples[j + 1].audio_ms == a0:
                j += 1
            else:
                break
        bg1 = samples[j].bg_n
        # Only count when we saw multiple markers or can bound by next PLAYING.
        growth = max(0, int(bg1) - int(bg0)) * 40
        if j > i:
            reb_episodes.append(growth)
        elif i + 1 < len(samples) and samples[i + 1].state == "PLAYING":
            # Single IDLE marker: lower-bound unknown; use 0..40ms opaque → skip
            pass
        i = j + 1

    # End-of-turn idle: last frozen audio streak.
    end_growth = 0
    if samples:
        k = len(samples) - 1
        while k > 0 and samples[k].state == "PLAYING":
            k -= 1
        if k > 0 and samples[k].state != "PLAYING":
            a_end = samples[k].audio_ms
            t = k
            while t > 0 and samples[t - 1].audio_ms == a_end and samples[t - 1].state != "PLAYING":
                t -= 1
            end_growth = max(0, samples[k].bg_n - samples[t].bg_n) * 40

    return {
        "n": len(samples),
        "playing_minmax_mean_drift": _stats(playing),
        "reb_minmax_mean_drift": _stats(reb),
        "nonplaying_minmax_mean_drift": _stats(idleish),
        "turn_boundaries": len(turns),
        "reb_episode_growths_ms": reb_episodes,
        "end_idle_growth_ms": end_growth,
        "final_delta_ms": int(samples[-1].delta_ms or 0) if samples else 0,
        "max_abs_delta_ms": max((abs(int(s.delta_ms or 0)) for s in samples), default=0),
    }


def _clock_model_selfcheck(fps: float = 25.0) -> dict:
    """Deterministic: PLAYING co-advance; REBUFFERING/IDLE advance BG only."""
    period = 1000.0 / fps
    bg = 0
    audio = 0
    # 2s PLAYING
    play_d0 = play_d1 = 0
    for i in range(50):
        bg += 1
        audio += int(period)
        d = int(round(bg * period - audio))
        if i == 0:
            play_d0 = d
        play_d1 = d
    d_before_reb = play_d1
    # 400ms REBUFFERING (audio frozen)
    for _ in range(10):
        bg += 1
    d_after_reb = int(round(bg * period - audio))
    # 1s PLAYING resume (Δ stays elevated; co-advance again)
    for _ in range(25):
        bg += 1
        audio += int(period)
    d_after_resume = int(round(bg * period - audio))
    # turn gap 3s IDLE (audio frozen; BG keeps advancing; no BG reset)
    for _ in range(75):
        bg += 1
    d_after_idle = int(round(bg * period - audio))
    # new turn: audio_ms restarts (base_played_samples); BG capture not reset
    # carry = prior wall lead over previous audio timeline
    carry = d_after_idle
    audio2 = 0
    bg_at_turn = bg
    for _ in range(25):
        bg += 1
        audio2 += int(period)
    # Within-turn Δ after re-anchor would be ~0; unresected absolute lead remains carry
    within_turn_d = int(round((bg - bg_at_turn) * period - audio2))
    return {
        "playing_delta_stable": play_d0 == play_d1 == 0,
        "reb_delta_growth_ms": d_after_reb - d_before_reb,
        "resume_delta_held_ms": d_after_resume,
        "idle_gap_growth_ms": d_after_idle - d_after_resume,
        "within_turn_reanchor_delta_ms": within_turn_d,
        "carry_if_no_bg_reset_ms": carry,
        "expected_reb_ms": 400,
        "expected_idle_ms": 3000,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase B2 BG↔audio_ms Δ quantify")
    ap.add_argument("log", nargs="?", default=None, help="session log path")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    model = _clock_model_selfcheck(fps=float(args.fps))
    print("[B2][clock_model]", model)

    if not args.log:
        ok = (
            model["playing_delta_stable"]
            and model["reb_delta_growth_ms"] == model["expected_reb_ms"]
            and model["idle_gap_growth_ms"] == model["expected_idle_ms"]
            and model["within_turn_reanchor_delta_ms"] == 0
            and model["resume_delta_held_ms"] == model["expected_reb_ms"]
            and model["carry_if_no_bg_reset_ms"]
            == model["expected_reb_ms"] + model["expected_idle_ms"]
        )
        print("[B2][selfcheck]", "PASS" if ok else "FAIL")
        return 0 if ok else 2

    path = Path(args.log)
    text = path.read_text(encoding="utf-8", errors="replace")
    samples = _parse_legacy(text, fps=float(args.fps))
    summary = _summarize(samples)
    print(f"[B2][log] path={path.name} samples={summary['n']}")
    pmin, pmax, pmean, pdrift = summary["playing_minmax_mean_drift"]
    print(
        f"[B2][PLAYING] delta_ms min={pmin} max={pmax} mean={pmean} "
        f"end-start={pdrift}"
    )
    rmin, rmax, rmean, rdrift = summary["reb_minmax_mean_drift"]
    print(
        f"[B2][REBUFFERING] delta_ms min={rmin} max={rmax} mean={rmean} "
        f"end-start={rdrift}"
    )
    print(
        f"[B2][reb_episodes] growths_ms={summary['reb_episode_growths_ms']} "
        f"end_idle_growth_ms={summary['end_idle_growth_ms']}"
    )
    print(
        f"[B2][turns] boundaries={summary['turn_boundaries']} "
        f"final_delta_ms={summary['final_delta_ms']} "
        f"max_abs_delta_ms={summary['max_abs_delta_ms']}"
    )
    # Compact table for parent summary.
    print("[B2][table]")
    print("interval\tdelta note")
    print(
        f"steady PLAYING\tmin/max/mean={pmin}/{pmax}/{pmean} "
        "(legacy sent proxy coarse in PLAYING; prefer B2_OBS / clock model)"
    )
    eg = summary["reb_episode_growths_ms"]
    if eg:
        print(
            f"REBUFFERING episodes\tgrowth_ms={eg} "
            f"sum={sum(eg)} (BG advances, audio_ms frozen)"
        )
    else:
        print(
            "REBUFFERING episodes\tshort REB often 1 marker "
            f"(logged delta span end-start={rdrift})"
        )
    print(
        f"turn/end idle\tend_idle_growth_ms={summary['end_idle_growth_ms']} "
        f"turn_boundaries={summary['turn_boundaries']} "
        "(BG VideoCapture not reset; IDLE_BG_ADVANCE keeps wall pace)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
