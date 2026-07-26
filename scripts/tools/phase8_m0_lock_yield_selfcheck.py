#!/usr/bin/env python3
"""Phase 8 selfcheck: worker_owner + state-lock yield lets covered peers enqueue early."""
from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from unittest import mock


STEP1 = (
    Path(__file__).resolve().parents[1]
    / "live_runtime"
    / "run_mic_input_obs_realtime_step1.py"
)


def _load_step1():
    spec = importlib.util.spec_from_file_location("step1_lock_check", STEP1)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {STEP1}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    mod = _load_step1()
    lock = threading.Lock()
    cond = threading.Condition(lock)
    ref = {
        "lock": lock,
        "cond": cond,
        "worker_owner": None,
        "rendered_chunks": 0,
        "total_frames": 0,
        "chunk_len_ms": 120,
        "step_ms": 40,
        "close_mouth_id": 0,
        "frame_offset": 0,
        "watch_fg_dir": Path("."),
    }
    frames = [{"t_ms": i * 40, "mouth_id": 0} for i in range(20)]
    mouth_obj_ref = {"obj": {"frames": frames}}

    render_events: list[str] = []
    peer_saw_mid = threading.Event()
    a_kept_owner_during_peer = threading.Event()

    def fake_render(**kwargs):
        cid = int(kwargs["cid"])
        render_events.append(f"start:{cid}")
        # Peer should be able to finish while A still owns worker and renders more.
        if cid == 0:
            time.sleep(0.12)
        else:
            time.sleep(0.12)
        render_events.append(f"end:{cid}")
        return 3, False, 120.0

    def worker_a():
        with mock.patch.object(mod, "_m0_pipeline_render_with_hang_guard", side_effect=fake_render):
            with mock.patch.object(mod, "_m0_pipeline_verify_pngs_exist", return_value=True):
                mod._m0_pipeline_advance_sync(
                    m0_pipeline_ref=ref,
                    mouth_obj_ref=mouth_obj_ref,
                    audio_playback_state_ref=None,
                    max_chunks=3,
                    until_t1_ms=360,
                )

    def worker_b():
        t0 = time.perf_counter()
        with mock.patch.object(mod, "_m0_pipeline_render_with_hang_guard", side_effect=fake_render):
            with mock.patch.object(mod, "_m0_pipeline_verify_pngs_exist", return_value=True):
                time.sleep(0.03)
                mod._m0_pipeline_advance_sync(
                    m0_pipeline_ref=ref,
                    mouth_obj_ref=mouth_obj_ref,
                    audio_playback_state_ref=None,
                    max_chunks=8,
                    until_t1_ms=120,
                )
        elapsed = time.perf_counter() - t0
        # A keeps rendering 3 chunks (~0.36s). B only needs 120ms coverage (~1 chunk).
        if elapsed < 0.28:
            peer_saw_mid.set()
        # A should still own worker when B finishes early.
        if ref.get("worker_owner") is not None:
            a_kept_owner_during_peer.set()
        render_events.append(f"peer_done_s={elapsed:.3f}")

    ta = threading.Thread(target=worker_a)
    tb = threading.Thread(target=worker_b)
    ta.start()
    tb.start()
    ta.join(timeout=5)
    tb.join(timeout=5)

    ok = (
        peer_saw_mid.is_set()
        and a_kept_owner_during_peer.is_set()
        and ref["rendered_chunks"] >= 3
        and ref.get("worker_owner") is None
    )
    print(f"events={render_events}")
    print(f"rendered_chunks={ref['rendered_chunks']} owner={ref['worker_owner']}")
    print(
        f"peer_early_exit={peer_saw_mid.is_set()} "
        f"a_kept_owner={a_kept_owner_during_peer.is_set()} ok={ok}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
