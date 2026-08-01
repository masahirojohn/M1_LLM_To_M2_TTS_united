#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_log(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    if len(raw) > 4 and raw[1:2] == b"\x00" and raw[3:4] == b"\x00":
        return raw.decode("utf-16-le")
    return raw.decode("utf-8", errors="replace")


def analyze_multi(sid: str) -> None:
    text = read_log(ROOT / "logs" / f"{sid}.log")
    print(f"==== {sid} chars={len(text)}")
    keys = [
        "[REBUFFERING]",
        "AUDIO_BEFORE_M0",
        "ENQUEUE_BLOCKED",
        "clear_queue_sent",
        "no_clear_queue",
        "SSOT_WAIT",
        "SSOT_CATCHUP",
        "IDLE_BG",
        "hang_used=True",
        "png_verified=False",
        "mouth_hold_extend",
        "[phase24][rss]",
        "[m0_pool][turn_ready]",
        "automatic_activity_detection.disabled=True",
        "skip_response_trigger=True",
        "skip_response_trigger=False",
    ]
    for k in keys:
        print(f"  {k}: {text.count(k)}")

    parts = re.split(r"\[m0_pool\]\[turn_ready\] turn=(\d+)", text)
    reb_by: dict[int, int] = {}
    catch_by: dict[int, int] = {}
    hold_by: dict[int, int] = {}
    ab_by: dict[int, int] = {}
    wait_by: dict[int, tuple[float, float]] = {}
    i = 1
    while i + 1 < len(parts):
        t = int(parts[i])
        body = parts[i + 1]
        reb_by[t] = body.count("[REBUFFERING]")
        catch_by[t] = body.count("SSOT_CATCHUP")
        hold_by[t] = body.count("mouth_hold_extend")
        ab_by[t] = body.count("AUDIO_BEFORE_M0")
        waits = [float(x) for x in re.findall(r"m0_wait_mouth_ms=([0-9.]+)", body)]
        wait_by[t] = (
            st.median(waits) if waits else 0.0,
            max(waits) if waits else 0.0,
        )
        i += 2

    print(f"  turns={len(reb_by)}")
    if reb_by:
        print(
            "  REB sum/p50/max",
            sum(reb_by.values()),
            st.median(reb_by.values()),
            max(reb_by.values()),
            "worst",
            sorted(reb_by.items(), key=lambda kv: -kv[1])[:6],
        )
        print(
            "  CATCHUP sum/p50/max",
            sum(catch_by.values()),
            st.median(catch_by.values()),
            max(catch_by.values()),
            "worst",
            sorted(catch_by.items(), key=lambda kv: -kv[1])[:6],
        )
        print(
            "  hold sum",
            sum(hold_by.values()),
            "AB sum",
            sum(ab_by.values()),
            "AB turns",
            [t for t, v in ab_by.items() if v],
        )
        early = [reb_by[t] for t in range(1, 15) if t in reb_by]
        mid = [reb_by[t] for t in range(15, 24) if t in reb_by]
        late = [reb_by[t] for t in range(24, 29) if t in reb_by]
        print(
            "  REB mean early1-14/mid15-23/late24-28",
            round(st.mean(early), 1) if early else None,
            round(st.mean(mid), 1) if mid else None,
            round(st.mean(late), 1) if late else None,
        )

    root = ROOT / "out" / "obs_realtime_session_loop" / sid
    suspect = []
    print("  --- mouth last3s ---")
    for td in sorted(root.glob("turn_*")):
        mouth = td / "01_audio_stream_bridge/stream_mouth/mouth.json"
        if not mouth.exists():
            continue
        frames = json.loads(mouth.read_text(encoding="utf-8")).get("frames") or []
        if not frames:
            continue
        last = float(frames[-1].get("t_ms") or 0)
        lo = max(0.0, last - 3000.0)
        late_ids = [
            int(f.get("mouth_id") or 0)
            for f in frames
            if float(f.get("t_ms") or 0) >= lo
        ]
        early_ids = [
            int(f.get("mouth_id") or 0)
            for f in frames
            if float(f.get("t_ms") or 0) < last * 0.5
        ]

        def ch(ids: list[int]) -> int:
            return sum(1 for a, b in zip(ids, ids[1:]) if a != b)

        def nzp(ids: list[int]) -> float:
            return 100.0 * sum(1 for i in ids if i) / len(ids) if ids else 0.0

        fr = ch(late_ids) <= 2 and nzp(late_ids) < 25 and ch(early_ids) >= 8 and last >= 7000
        tno = int(td.name.split("_")[1])
        if fr:
            suspect.append(td.name)
        if fr or tno in (1, 5, 10, 15, 20, 24, 25, 26, 27, 28):
            print(
                f"  {td.name} last={last:.0f} early_ch={ch(early_ids)} "
                f"early_nz={nzp(early_ids):.0f}% late3_ch={ch(late_ids)} "
                f"late3_nz={nzp(late_ids):.0f}% REB={reb_by.get(tno)} "
                f"CATCH={catch_by.get(tno)} wait_max={wait_by.get(tno, (0, 0))[1]:.0f} "
                f"suspect={fr}"
            )
    print("  suspect_freeze_turns", len(suspect), suspect)

    rss_path = ROOT / "logs" / f"{sid}_rss.csv"
    if rss_path.exists():
        rows = list(csv.DictReader(rss_path.open(encoding="utf-8-sig")))
        sl = [float(r["rss_mb"]) for r in rows if r["name"] == "session_loop"]
        m0 = [
            float(r["rss_mb"])
            for r in rows
            if r["name"] == "m0_worker" and float(r["rss_mb"]) >= 20
        ]
        if sl:
            print(
                "  RSS parent first/p50/last/max",
                sl[0],
                st.median(sl),
                sl[-1],
                max(sl),
            )
        if m0:
            print(
                "  RSS m0active first/p50/last/max",
                m0[0],
                st.median(m0),
                m0[-1],
                max(m0),
                "n",
                len(m0),
            )


def analyze_long(sid: str) -> None:
    text = read_log(ROOT / "logs" / f"{sid}.log")
    print(f"==== {sid}")
    for k in [
        "[REBUFFERING]",
        "AUDIO_BEFORE_M0",
        "ENQUEUE_BLOCKED",
        "clear_queue_sent",
        "SSOT_CATCHUP",
        "mouth_hold_extend",
        "automatic_activity_detection.disabled=True",
        "skip_response_trigger=False",
        "skip_response_trigger=True",
    ]:
        print(f"  {k}: {text.count(k)}")
    rss_path = ROOT / "logs" / f"{sid}_rss.csv"
    if rss_path.exists():
        rows = list(csv.DictReader(rss_path.open(encoding="utf-8-sig")))
        by = defaultdict(list)
        for r in rows:
            by[r["name"]].append(float(r["rss_mb"]))
        for k, xs in sorted(by.items()):
            print(
                f"  RSS {k}: n={len(xs)} min={min(xs):.1f} max={max(xs):.1f} "
                f"last={xs[-1]:.1f}"
            )


def main() -> int:
    args = sys.argv[1:] or [
        "sess_phase10_step2_subj_20260801_165233",
        "sess_phase11_subj_20260801_165649",
    ]
    for sid in args:
        if "phase11" in sid or "multi" in sid:
            analyze_multi(sid)
        else:
            analyze_long(sid)
            # still print mouth selfgate summary
            root = ROOT / "out" / "obs_realtime_session_loop" / sid / "turn_001"
            mouth = root / "01_audio_stream_bridge/stream_mouth/mouth.json"
            if mouth.exists():
                frames = json.loads(mouth.read_text(encoding="utf-8")).get("frames") or []
                print(f"  mouth frames={len(frames)} last_t={frames[-1].get('t_ms') if frames else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
