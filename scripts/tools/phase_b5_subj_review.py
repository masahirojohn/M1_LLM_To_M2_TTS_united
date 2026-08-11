#!/usr/bin/env python3
"""Review B5/B5hf pose_base vs B3 RELOCK gaps from session logs.

Emits per-turn |B5_POSE_BASE − nearest RELOCK bg/ideal| table.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def _read_log(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if b"\x00" in raw[:200]:
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8", errors="replace")


def analyze(path: Path) -> None:
    text = _read_log(path)
    print("=" * 60)
    print(path.name)
    turns = re.findall(r"\[session_loop\] turn=(\d+)", text)
    print("turns", turns)

    events: list[tuple] = []
    for m in re.finditer(
        r"\[sync\]\[B5_POSE_BASE\]\s+(.*)",
        text,
    ):
        line = m.group(1)
        fo_m = re.search(r"frame_offset=(\d+)", line)
        mode_m = re.search(r"mode=(\S+)", line)
        cur_m = re.search(r"bg_cursor=(\S+)", line)
        base_m = re.search(r"pose_base_frame=(\d+|pending)", line)
        fo = int(fo_m.group(1)) if fo_m else -1
        mode = mode_m.group(1) if mode_m else "?"
        cur = cur_m.group(1) if cur_m else "?"
        if base_m and base_m.group(1) != "pending":
            base_v: int | None = int(base_m.group(1))
        else:
            base_v = None
        events.append(("base", m.start(), base_v, fo, mode, cur))
    for m in re.finditer(
        r"\[sync\]\[virtualcam\]\[B3_BG_RELOCK\] reason=(\S+) audio_ms=(\d+) "
        r"bg_frame=(\d+) frame_offset=(\d+)",
        text,
    ):
        events.append(
            (
                "relock",
                m.start(),
                m.group(1),
                int(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
            )
        )
    events.sort(key=lambda x: x[1])

    last_base: int | None = None
    last_mode = "?"
    gaps_enter: list[float] = []
    # Per frozen absolute base: gap to next RELOCK (prefer turn, else enter_playing).
    pending_base: tuple[int, int, str] | None = None  # base, fo, cur
    rows: list[tuple] = []

    print("timeline (BASE / RELOCK):")
    for e in events:
        if e[0] == "base":
            base_v, fo, mode, cur = e[2], int(e[3]), str(e[4]), str(e[5])
            if base_v is None:
                print(
                    f"  BASE pending fo={fo} mode={mode} cursor={cur}"
                )
                continue
            last_base = int(base_v)
            last_mode = mode
            print(
                f"  BASE pose_base={last_base} fo={fo} mode={mode} cursor={cur}"
            )
            # Legacy B5 logs omit mode=; treat numeric freeze as absolute.
            if mode in ("absolute", "?"):
                pending_base = (int(last_base), int(fo), cur)
            continue
        reason, a, bg, fo = e[2], int(e[3]), int(e[4]), int(e[5])
        ideal = float(bg) - float(a) / 40.0
        if last_base is None:
            print(
                f"  RELOCK reason={reason:14} a={a:5d} bg={bg:4d} fo={fo:4d} "
                f"ideal_base~={ideal:.1f} (no BASE yet)"
            )
            continue
        pose_at = float(last_base) + float(a) / 40.0
        gap = pose_at - float(bg)
        mark = ""
        if reason == "enter_playing":
            gaps_enter.append(gap)
            mark = "  << enter"
        print(
            f"  RELOCK reason={reason:14} a={a:5d} bg={bg:4d} fo={fo:4d} "
            f"ideal_base~={ideal:.1f} pose_at~={pose_at:.1f} gap={gap:+.1f}"
            f" mode={last_mode}{mark}"
        )
        if pending_base is not None and int(pending_base[1]) == int(fo):
            b0, bfo, bcur = pending_base
            # |POSE_BASE - RELOCK ideal_base| and |POSE_BASE - bg| at a≈0 prefer.
            gap_ideal = float(b0) - float(ideal)
            gap_bg = float(b0) - float(bg)
            rows.append(
                (bfo, b0, reason, a, bg, ideal, gap_ideal, gap_bg, bcur)
            )
            if reason in ("turn", "enter_playing"):
                pending_base = None

    print()
    print("|turn_fo|pose_base|relock|a_ms|bg|ideal|base-ideal|base-bg|cursor|")
    print("|---:|---:|---|---:|---:|---:|---:|---:|---|")
    if not rows:
        print("| (no absolute BASE-RELOCK pairs) |")
    for bfo, b0, reason, a, bg, ideal, gap_ideal, gap_bg, bcur in rows:
        print(
            f"|{bfo}|{b0}|{reason}|{a}|{bg}|{ideal:.1f}|"
            f"{gap_ideal:+.1f}|{gap_bg:+.1f}|{bcur}|"
        )

    if gaps_enter:
        abs_g = [abs(g) for g in gaps_enter]
        print(
            f"enter_playing gaps: n={len(gaps_enter)} "
            f"absmean={sum(abs_g)/len(abs_g):.1f} maxabs={max(abs_g):.1f}"
        )
    if rows:
        abs_ideal = [abs(r[6]) for r in rows]
        print(
            f"POSE_BASE-ideal gaps: n={len(rows)} "
            f"absmean={sum(abs_ideal)/len(abs_ideal):.1f} "
            f"maxabs={max(abs_ideal):.1f}"
        )


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        paths = [
            Path("logs/sess_phase11_subj_20260810_232331.log"),
            Path("logs/sess_phase11_subj_20260810_232453.log"),
        ]
    for p in paths:
        analyze(p.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
