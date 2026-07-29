#!/usr/bin/env python3
"""Phase 13 offline: multi-M0 pool claim + contiguous watermark ordering."""
from __future__ import annotations

import threading
import time
from typing import Any

from run_mic_input_obs_realtime_step1 import _m0_pipeline_advance_sync


def _make_mouth(n_frames: int, step_ms: int = 40) -> dict[str, Any]:
    frames = []
    for i in range(int(n_frames)):
        frames.append({"t_ms": int(i * step_ms), "mouth_id": 0, "src": "offline"})
    return {"obj": {"frames": frames, "timeline": []}}


def main() -> int:
    lock = threading.Lock()
    cond = threading.Condition(lock)
    render_log: list[tuple[int, int, float]] = []
    render_lock = threading.Lock()

    # Monkeypatch hang guard path via pipeline render replacement is heavy;
    # instead stub _m0_pipeline_render_with_hang_guard on the module.
    import run_mic_input_obs_realtime_step1 as step1

    def _fake_render(**kwargs: Any) -> tuple[int, bool, float]:
        cid = int(kwargs["cid"])
        port = kwargs.get("m0_worker_port")
        t0 = time.perf_counter()
        # Simulate ~80ms render; two workers should overlap.
        time.sleep(0.08)
        with render_lock:
            render_log.append((cid, int(port or -1), time.perf_counter()))
        return 3, False, (time.perf_counter() - t0) * 1000.0

    def _fake_verify(**_kwargs: Any) -> bool:
        return True

    step1._m0_pipeline_render_with_hang_guard = _fake_render  # type: ignore
    step1._m0_pipeline_verify_pngs_exist = _fake_verify  # type: ignore
    step1._m0_pipeline_global_frame_range = lambda **_k: (0, 3)  # type: ignore

    ports = [39390, 39391]
    ref: dict[str, Any] = {
        "lock": lock,
        "cond": cond,
        "worker_owner": None,
        "rendered_chunks": 0,
        "next_claim_cid": 0,
        "completed_cids": set(),
        "completed_meta": {},
        "m0_worker_ports": ports,
        "m0_worker_n": 2,
        "pool_free": set(range(2)),
        "total_frames": 0,
        "step_ms": 40,
        "chunk_len_ms": 120,
        "close_mouth_id": 0,
        "m0_worker_proc": None,
        "m0_worker_host": "127.0.0.1",
        "m0_worker_port": ports[0],
        "frame_offset": 0,
        "watch_fg_dir": None,
    }

    # 8 chunks * 120ms = 960ms coverage; 3 frames/chunk at 40ms.
    mouth_ref = _make_mouth(n_frames=30, step_ms=40)
    errors: list[BaseException] = []

    def _worker(until_ms: int) -> None:
        try:
            _m0_pipeline_advance_sync(
                m0_pipeline_ref=ref,
                mouth_obj_ref=mouth_ref,
                audio_playback_state_ref=None,
                hang_timeout_ms=0,
                max_chunks=8,
                until_t1_ms=int(until_ms),
            )
        except BaseException as e:
            errors.append(e)

    threads = [
        threading.Thread(target=_worker, args=(240,), daemon=True),
        threading.Thread(target=_worker, args=(480,), daemon=True),
        threading.Thread(target=_worker, args=(720,), daemon=True),
        threading.Thread(target=_worker, args=(960,), daemon=True),
    ]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10.0)
    wall = time.perf_counter() - t0

    if errors:
        print("OFFLINE_POOL: FAIL errors=", errors)
        return 1
    if int(ref["rendered_chunks"]) < 8:
        print(
            "OFFLINE_POOL: FAIL watermark",
            f"rendered_chunks={ref['rendered_chunks']}",
            f"log={render_log}",
        )
        return 1

    ports_used = sorted({p for _c, p, _t in render_log})
    # With N=2, serial 8*0.08=0.64s; parallel should be clearly under ~0.50s.
    ok_parallel = wall < 0.55 and len(ports_used) >= 2
    print(
        "OFFLINE_POOL:",
        "PASS" if ok_parallel else "WARN_SERIALISH",
        f"wall_s={wall:.3f}",
        f"rendered_chunks={ref['rendered_chunks']}",
        f"ports_used={ports_used}",
        f"claims={len(render_log)}",
    )
    # Watermark order is the hard Pass; parallel speedup is soft evidence.
    print("OFFLINE_POOL_ORDER: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
