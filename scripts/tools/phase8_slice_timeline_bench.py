#!/usr/bin/env python3
"""Phase 8 suspicion-B offline bench: timeline slice cost vs N.

Compares legacy O(N) full scan vs current bisect path in step1.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path


STEP1 = (
    Path(__file__).resolve().parents[1]
    / "live_runtime"
    / "run_mic_input_obs_realtime_step1.py"
)


def _load_step1():
    spec = importlib.util.spec_from_file_location("step1_bench", STEP1)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {STEP1}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_timeline(n: int) -> dict:
    return {
        "schema_version": "bench",
        "frames": [{"t_ms": i * 40, "mouth_id": i % 5} for i in range(n)],
    }


def _slice_linear(raw: dict, t0_ms: int, t1_ms: int) -> list[dict]:
    frames = list(raw.get("frames") or [])
    inside = [
        fr for fr in frames if t0_ms <= int(fr.get("t_ms", 0) or 0) < t1_ms
    ]
    prev = None
    for fr in frames:
        t = int(fr.get("t_ms", 0) or 0)
        if t < t0_ms:
            prev = fr
        else:
            break
    out: list[dict] = []
    if prev is not None:
        fr0 = dict(prev)
        fr0["t_ms"] = 0
        out.append(fr0)
    for fr in inside:
        fr2 = dict(fr)
        fr2["t_ms"] = int(fr2.get("t_ms", 0) or 0) - int(t0_ms)
        out.append(fr2)
    return out


def _bench(fn, raw, windows: list[tuple[int, int]], repeats: int = 40) -> float:
    # warmup
    for t0, t1 in windows[:3]:
        fn(raw, t0, t1)
    t0 = time.perf_counter()
    for _ in range(repeats):
        for a, b in windows:
            fn(raw, a, b)
    return (time.perf_counter() - t0) * 1000.0 / float(repeats)


def main() -> int:
    mod = _load_step1()
    print(f"step1={STEP1}")
    for n in (64, 256, 1024, 4096, 16384):
        raw = _make_timeline(n)
        # ~20 chunk windows across the timeline (120ms chunks)
        windows = [(i * 120, i * 120 + 120) for i in range(0, min(20, max(1, n // 3)))]
        linear_ms = _bench(_slice_linear, raw, windows)
        bisect_ms = _bench(mod._slice_shift_timeline, raw, windows)
        # correctness spot-check
        a, b = windows[len(windows) // 2]
        got = mod._as_frames(mod._slice_shift_timeline(raw, a, b))
        exp = _slice_linear(raw, a, b)
        ok = [(x.get("t_ms"), x.get("mouth_id")) for x in got] == [
            (x.get("t_ms"), x.get("mouth_id")) for x in exp
        ]
        print(
            f"N={n:5d} windows={len(windows):2d} "
            f"linear_ms/iter={linear_ms:.3f} bisect_ms/iter={bisect_ms:.3f} "
            f"speedup={linear_ms / max(bisect_ms, 1e-9):.1f}x ok={ok}"
        )
    print(
        "note: live Before (phase7 180003) m0_slice p90~1.1ms / frac_of_m0 << 1% "
        "-> suspicion B non-dominant; bisect closes growth risk"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
