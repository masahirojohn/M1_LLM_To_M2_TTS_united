#!/usr/bin/env python3
"""Phase 16: mid-band supply / wait / REB probe for N-scale A/B (helper only)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[int(round((len(s) - 1) * p))]


def analyze(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"file={path.name}")

    m = re.search(r"start m0 tcp worker pool n=(\d+)", text)
    print(f"pool_n={m.group(1) if m else '?'}")
    rss = re.findall(r"\[m0_pool\]\[rss_after_spawn\] n=(\d+) sum_mb=([0-9.]+)", text)
    if rss:
        print(f"rss_after_spawn_n={rss[0][0]} sum_mb={rss[0][1]}")
    rss_later = re.findall(r"\[m0_pool\]\[rss[^\]]*\][^\n]*sum_mb=([0-9.]+)", text)
    if rss_later:
        print(f"rss_sum_mb_samples={','.join(rss_later[:8])}")

    reb_lines = len(re.findall(r"\[REBUFFERING\]", text))
    reb_events: list[tuple[float, float, int]] = []
    prev_rc = 0
    for line in text.splitlines():
        if "__AUDIO_PLAYER_RESPONSE__" not in line:
            continue
        try:
            j = json.loads(line.split("__AUDIO_PLAYER_RESPONSE__", 1)[1].strip())
        except Exception:
            continue
        rc = int(j.get("rebuffer_count") or 0)
        if rc > prev_rc:
            reb_events.append(
                (
                    float(j.get("player_local_ms") or 0.0),
                    float(j.get("pending_ms") or 0.0),
                    rc,
                )
            )
            prev_rc = rc
    mid_reb = sum(1 for pl, _pend, _rc in reb_events if 2500.0 <= pl <= 20000.0)
    late_reb = sum(1 for pl, _pend, _rc in reb_events if pl > 20000.0)
    print(f"REBUFFERING_lines={reb_lines}")
    print(f"REBUFFERING_events={len(reb_events)} mid_2p5_20s={mid_reb} late_gt20s={late_reb}")
    if reb_events:
        print(
            f"REBUFFERING_player_local_first={reb_events[0][0]:.0f} "
            f"last={reb_events[-1][0]:.0f}"
        )

    pending_all: list[float] = []
    pending_mid: list[float] = []
    lt240 = 0
    lt240_mid = 0
    for line in text.splitlines():
        if "__AUDIO_PLAYER_RESPONSE__" not in line:
            continue
        try:
            j = json.loads(line.split("__AUDIO_PLAYER_RESPONSE__", 1)[1].strip())
        except Exception:
            continue
        if str(j.get("cmd") or "") != "play":
            continue
        pend = float(j.get("pending_ms") or 0)
        pl = float(j.get("player_local_ms") or 0)
        pending_all.append(pend)
        if pend < 240.0:
            lt240 += 1
        if 2500.0 <= pl <= 12000.0:
            pending_mid.append(pend)
            if pend < 240.0:
                lt240_mid += 1

    print(
        f"pending_p50={_pct(pending_all, 0.5):.1f} lt240={lt240}/{len(pending_all)} "
        f"mid_pending_p50={_pct(pending_mid, 0.5):.1f} mid_lt240={lt240_mid}/{len(pending_mid)}"
    )

    rows = []
    for line in text.splitlines():
        if "[sync][pipeline_chunk] " not in line or "pipeline_seq=" not in line:
            continue

        def g(k: str) -> float | None:
            m2 = re.search(rf"{k}=([0-9.]+)", line)
            return float(m2.group(1)) if m2 else None

        rows.append(
            {
                "chunk_idx": g("chunk_idx"),
                "m0_ms": g("m0_ms"),
                "m0_wait_mouth_ms": g("m0_wait_mouth_ms"),
                "m0_lock_ms": g("m0_lock_ms"),
                "m0_png_wait_ms": g("m0_png_wait_ms"),
            }
        )
    n = len(rows)
    lo = int(n * 0.35) if n else 0
    hi = int(n * 0.75) if n else 0
    mid = rows[lo:hi] if n else []

    def stats(label: str, key: str, src: list) -> None:
        vals = [float(r[key]) for r in src if r.get(key) is not None]
        if not vals:
            print(f"{label}.{key}: n=0")
            return
        print(
            f"{label}.{key}: n={len(vals)} p50={_pct(vals, 0.5):.1f} "
            f"p90={_pct(vals, 0.9):.1f} max={max(vals):.1f}"
        )

    print(f"pipeline_chunks={n} mid_slice={lo}:{hi}")
    for key in ("m0_wait_mouth_ms", "m0_lock_ms", "m0_png_wait_ms", "m0_ms"):
        stats("all", key, rows)
        stats("mid", key, mid)

    for k in (
        "AUDIO_BEFORE_M0",
        r"\[sync\]\[virtualcam\]\[SSOT_WAIT\]",
        r"\[sync\]\[virtualcam\]\[SSOT_CATCHUP\]",
        r"\[sync\]\[virtualcam\]\[IDLE_BG",
        "clear_queue_sent",
        "reason=talkover_cut_in",
        "fast_inmemory=False",
        r"run_mic_input_obs_realtime_session_loop\]\[OK\]",
    ):
        print(f"count[{k}]={len(re.findall(k, text))}")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: phase16_n_scale_probe.py <log>")
        return 2
    analyze(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
