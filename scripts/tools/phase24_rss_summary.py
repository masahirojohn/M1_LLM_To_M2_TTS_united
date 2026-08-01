#!/usr/bin/env python3
"""Summarize Phase24 RSS CSV (per-pid + per-name)."""
from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "")
    if not path.exists():
        print(f"MISSING {path}")
        return 1
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    by_name: dict[str, list[float]] = defaultdict(list)
    by_pid: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        name = r["name"]
        pid = r["pid"]
        mb = float(r["rss_mb"])
        el = float(r.get("elapsed_s") or 0.0)
        by_name[name].append(mb)
        by_pid[(name, pid)].append((el, mb))

    print(f"==== {path.name} rows={len(rows)}")
    print("-- per name (all samples; may mix warm/cold pids) --")
    for k, ys in sorted(by_name.items()):
        n = len(ys)
        early = ys[: max(1, n // 5)]
        late = ys[-max(1, n // 5) :]
        print(
            f"  {k}: n={n} min={min(ys):.1f} p50={st.median(ys):.1f} max={max(ys):.1f} "
            f"early_mean={st.mean(early):.1f} late_mean={st.mean(late):.1f} "
            f"d={st.mean(late) - st.mean(early):.1f}"
        )

    print("-- per pid (active only, max>=20MB) --")
    for (name, pid), xs in sorted(by_pid.items(), key=lambda kv: -max(v for _, v in kv[1])):
        ys = [v for _, v in xs]
        if max(ys) < 20:
            continue
        n = len(ys)
        early = ys[: max(1, n // 5)]
        late = ys[-max(1, n // 5) :]
        step = max(1, n // 6)
        series = " -> ".join(f"{ys[i]:.0f}" for i in range(0, n, step))
        print(
            f"  {name} pid={pid}: n={n} min={min(ys):.1f} max={max(ys):.1f} "
            f"early={st.mean(early):.1f} late={st.mean(late):.1f} "
            f"d={st.mean(late) - st.mean(early):.1f} series={series} -> {ys[-1]:.0f}"
        )

    # qsize peaks if present
    inflight = [int(r["pipeline_inflight"]) for r in rows if r.get("pipeline_inflight")]
    eq = [int(r["enqueue_qsize"]) for r in rows if r.get("enqueue_qsize")]
    wq = [int(r["worker_qsize"]) for r in rows if r.get("worker_qsize")]
    if inflight or eq or wq:
        print("-- qsize peaks --")
        if inflight:
            print(f"  pipeline_inflight max={max(inflight)} last={inflight[-1]}")
        if eq:
            print(f"  enqueue_qsize max={max(eq)} last={eq[-1]}")
        if wq:
            print(f"  worker_qsize max={max(wq)} last={wq[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
