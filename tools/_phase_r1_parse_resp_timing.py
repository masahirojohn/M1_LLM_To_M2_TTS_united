#!/usr/bin/env python3
"""Phase R1: parse [resp_timing] markers and emit interval table."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


_RE = re.compile(
    r"\[resp_timing\]\s+(speech_end|activity_end|first_audio)\s+perf_ms=([0-9.]+)"
    r"(?:\s+reason=(\S+))?"
)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\xff\xfe" or (len(raw) > 2 and raw[1] == 0 and raw[3] == 0):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse_intervals(text: str) -> list[dict[str, float | str]]:
    events: list[tuple[str, float, str]] = []
    for m in _RE.finditer(text):
        events.append((m.group(1), float(m.group(2)), m.group(3) or ""))

    out: list[dict[str, float | str]] = []
    last_speech_end: float | None = None
    pending_end: float | None = None
    pending_reason = ""
    turn_i = 0

    for kind, perf_ms, reason in events:
        if kind == "speech_end":
            last_speech_end = perf_ms
        elif kind == "activity_end":
            pending_end = perf_ms
            pending_reason = reason or "activity_end"
            if pending_reason != "vad_silence":
                # A/B validity: only silence-ended turns measure silence_wait
                last_speech_end = None
                pending_end = None
                pending_reason = ""
        elif kind == "first_audio":
            if last_speech_end is None or pending_end is None:
                continue
            turn_i += 1
            silence_wait = pending_end - last_speech_end
            end_to_first = perf_ms - pending_end
            total = perf_ms - last_speech_end
            out.append(
                {
                    "turn": float(turn_i),
                    "speech_end": last_speech_end,
                    "activity_end": pending_end,
                    "first_audio": perf_ms,
                    "silence_wait_ms": silence_wait,
                    "end_to_first_pcm_ms": end_to_first,
                    "total_ms": total,
                    "marker": pending_reason,
                }
            )
            last_speech_end = None
            pending_end = None
            pending_reason = ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    path = Path(args.log)
    rows = parse_intervals(_read_text(path))
    label = args.label or path.name
    print(f"LABEL={label}")
    print(
        "turn\tsilence_wait_ms\tend_to_first_pcm_ms\ttotal_ms"
        "\tspeech_end\tactivity_end\tfirst_audio"
    )
    if not rows:
        print("(no complete speech_end→activity_end→first_audio triples)")
        return 1
    for r in rows:
        print(
            f"{int(r['turn'])}\t{r['silence_wait_ms']:.1f}\t"
            f"{r['end_to_first_pcm_ms']:.1f}\t{r['total_ms']:.1f}\t"
            f"{r['speech_end']:.3f}\t{r['activity_end']:.3f}\t{r['first_audio']:.3f}"
        )
    sw = [float(r["silence_wait_ms"]) for r in rows]
    ef = [float(r["end_to_first_pcm_ms"]) for r in rows]
    tot = [float(r["total_ms"]) for r in rows]
    n = len(rows)

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    print(
        f"AVG\t{avg(sw):.1f}\t{avg(ef):.1f}\t{avg(tot):.1f}\tn={n}"
    )
    if avg(tot) > 0:
        print(
            f"SHARE\tsilence={100.0 * avg(sw) / avg(tot):.1f}%"
            f"\tend_to_first={100.0 * avg(ef) / avg(tot):.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
