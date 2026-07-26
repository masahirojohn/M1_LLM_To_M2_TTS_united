#!/usr/bin/env python3
"""Phase 6 log metrics summary (observability helper)."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: phase6_metrics_summary.py <utf8-log>")
        return 2
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8", errors="replace")

    def count(pat: str) -> int:
        return len(re.findall(pat, text))

    print(f"file={path}")
    print(f"turns={count(r'\[session_loop\] turn=')}")
    print(f"turn_sent={count(r'\[session_loop\]\[turn_sent\]')}")
    print(f"first_audio={count(r'turn_first_audio_detected')}")
    print(f"ACTIVITY_START={count(r'ACTIVITY_START')}")
    print(f"ACTIVITY_END={count(r'ACTIVITY_END')}")
    print(f"gen_complete_marker={count(r'marker_only no_clear_queue')}")
    print(f"clear_queue_sent={count(r'clear_queue_sent')}")
    print(f"talkover_clear={count(r'reason=talkover_cut_in')}")
    print(f"interrupt_clear_begin={count(r'interrupt_clear_begin')}")
    print(f"AUDIO_BEFORE_M0={count(r'AUDIO_BEFORE_M0')}")
    print(f"ENQUEUE_BLOCKED={count(r'ENQUEUE_BLOCKED')}")
    print(f"REBUFFERING={count(r'\[REBUFFERING\]')}")
    print(f"SSOT_WAIT={count(r'\[sync\]\[virtualcam\]\[SSOT_WAIT\]')}")
    print(f"SSOT_CATCHUP={count(r'\[sync\]\[virtualcam\]\[SSOT_CATCHUP\]')}")
    print(f"m0_breakdown={count(r'\[m0_breakdown\]')}")
    print(f"ssot_audio_ms={count(r'mode=audio_ms')}")
    print(f"fast_inmemory_False={count(r'fast_inmemory=False')}")
    print(f"fast_inmemory_True={count(r'fast_inmemory=True')}")
    print(f"fast_inmemory_ENABLED={count(r'\[fast_inmemory\]\[ENABLED\]')}")
    print(f"knn_inmemory_updated={count(r'\[knn_inmemory\]\[mouth_obj_updated\]')}")
    print(f"knn_mode_incremental={count(r'mode=incremental')}")
    print(f"knn_mode_full={count(r'mode=full')}")
    print(f"OK={count(r'run_mic_input_obs_realtime_session_loop\]\[OK\]')}")

    rows = []
    for line in text.splitlines():
        if "[sync][pipeline_chunk] " in line and "pipeline_seq=" in line:

            def g(k: str) -> float | None:
                m = re.search(rf"{k}=([0-9.]+)", line)
                return float(m.group(1)) if m else None

            rows.append(
                {
                    k: g(k)
                    for k in [
                        "chunk_idx",
                        "knn_ms",
                        "m0_ms",
                        "queue_wait_ms",
                        "total_ms",
                        "m0_wait_mouth_ms",
                        "m0_lock_ms",
                        "m0_slice_ms",
                        "m0_disk_ms",
                        "m0_req_send_ms",
                        "m0_png_wait_ms",
                        "m0_verify_ms",
                        "m0_other_ms",
                    ]
                }
            )

    print(f"pipeline_chunks={len(rows)}")

    def stats(key: str) -> None:
        vals = sorted(r[key] for r in rows if r.get(key) is not None)
        if not vals:
            return

        def pct(p: float) -> float:
            return vals[int(round((len(vals) - 1) * p))]

        print(
            f"{key}: n={len(vals)} min={vals[0]:.1f} p50={pct(0.5):.1f} "
            f"p90={pct(0.9):.1f} max={vals[-1]:.1f}"
        )

    for k in [
        "knn_ms",
        "m0_ms",
        "queue_wait_ms",
        "total_ms",
        "m0_wait_mouth_ms",
        "m0_lock_ms",
        "m0_slice_ms",
        "m0_disk_ms",
        "m0_png_wait_ms",
        "m0_other_ms",
    ]:
        stats(k)

    early = [r for r in rows if r["chunk_idx"] is not None and r["chunk_idx"] <= 5]
    late = [r for r in rows if r["chunk_idx"] is not None and r["chunk_idx"] >= 10]

    def avg(rs: list[dict], k: str) -> float:
        vs = [r[k] for r in rs if r.get(k) is not None]
        return sum(vs) / len(vs) if vs else 0.0

    print(
        f"early(<=5) avg knn={avg(early,'knn_ms'):.1f} m0={avg(early,'m0_ms'):.1f} "
        f"slice={avg(early,'m0_slice_ms'):.1f}"
    )
    print(
        f"late(>=10) avg knn={avg(late,'knn_ms'):.1f} m0={avg(late,'m0_ms'):.1f} "
        f"slice={avg(late,'m0_slice_ms'):.1f}"
    )

    und = [int(x) for x in re.findall(r"active_playback_underrun_count.: (\d+)", text)]
    reb = [int(x) for x in re.findall(r"rebuffer_count.: (\d+)", text)]
    print(f"max_underrun_count={max(und) if und else 0}")
    print(f"max_rebuffer_count={max(reb) if reb else 0}")

    # Phase 8: lock vs png_wait (Phase 9 defer evidence when lock ≈ peer png_wait).
    lock_vals = [float(r["m0_lock_ms"]) for r in rows if r.get("m0_lock_ms") is not None]
    png_vals = [float(r["m0_png_wait_ms"]) for r in rows if r.get("m0_png_wait_ms") is not None]
    slice_vals = [float(r["m0_slice_ms"]) for r in rows if r.get("m0_slice_ms") is not None]
    lock_sum = sum(lock_vals)
    png_sum = sum(png_vals)
    slice_sum = sum(slice_vals)
    m0_sum = sum(float(r["m0_ms"]) for r in rows if r.get("m0_ms") is not None)
    print(f"sum_m0_ms={m0_sum:.1f}")
    print(f"sum_m0_lock_ms={lock_sum:.1f}")
    print(f"sum_m0_png_wait_ms={png_sum:.1f}")
    print(f"sum_m0_slice_ms={slice_sum:.1f}")
    if png_sum > 1.0:
        print(f"lock_over_png_wait={lock_sum / png_sum:.2f}")
    if m0_sum > 1.0:
        print(f"lock_frac_of_m0={lock_sum / m0_sum:.2f}")
        print(f"png_wait_frac_of_m0={png_sum / m0_sum:.2f}")
        print(f"slice_frac_of_m0={slice_sum / m0_sum:.4f}")

    # VirtualCam freeze signal: longest run of identical displayed_frame on SSOT_WAIT.
    wait_disp = [
        int(x)
        for x in re.findall(
            r"\[sync\]\[virtualcam\]\[SSOT_WAIT\].*?displayed_frame=([0-9]+)",
            text,
        )
    ]
    max_stuck = 0
    run = 0
    prev = None
    for d in wait_disp:
        if prev is not None and d == prev:
            run += 1
        else:
            run = 1
        max_stuck = max(max_stuck, run)
        prev = d
    print(f"SSOT_WAIT_max_same_displayed_run={max_stuck}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
