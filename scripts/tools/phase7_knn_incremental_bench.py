#!/usr/bin/env python3
"""Offline Before/After microbench for Phase7 incremental KNN (standalone)."""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

KNN = Path(r"C:\dev\M3_Live_API_1_united\tools\knn_from_formant_raw_to_mouth_timeline.py")
GT = str(Path(r"C:\dev\M3_Live_API_1_united\data\knn_db") / "*.f1f2.json")


def _load_knn_mod():
    spec = importlib.util.spec_from_file_location("knn_bench_mod", KNN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {KNN}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_raw(n: int) -> dict:
    frames = []
    for i in range(n):
        frames.append(
            {
                "t_ms": i * 40,
                "vad_active": 1 if (i % 7) else 0,
                "f1_hz": 400.0 + (i % 20) * 15.0,
                "f2_hz": 1200.0 + (i % 30) * 25.0,
            }
        )
    return {"frames": frames, "step_ms": 40}


def mouth_frames_from_raw(mod, frames_in, db_z, z, k=5, fallback_id_active=2, min_conf_ratio=1.0):
    out = []
    for fr in frames_in:
        t_ms = fr.get("t_ms")
        if t_ms is None:
            continue
        t_ms = int(t_ms)
        vad = int(fr.get("vad_active") or 0)
        if vad == 0:
            out.append({"t_ms": t_ms, "mouth_id": 0})
            continue
        f1 = fr.get("f1_hz")
        f2 = fr.get("f2_hz")
        if not (
            isinstance(f1, (int, float))
            and isinstance(f2, (int, float))
            and f1 == f1
            and f2 == f2
        ):
            out.append({"t_ms": t_ms, "mouth_id": int(fallback_id_active) or 2})
            continue
        qz = mod._z_point(float(f1), float(f2), z)
        pred, top, top2 = mod._predict_knn(qz, db_z, int(k))
        ratio = (top / top2) if top2 > 0 else 999.0
        if ratio < float(min_conf_ratio):
            pred = int(fallback_id_active) if int(fallback_id_active) != 0 else 2
        out.append({"t_ms": t_ms, "mouth_id": int(pred)})
    return out


def run_incremental(mod, raw_obj, db_z, z, step_ms, mouth_obj_ref):
    t0 = time.perf_counter()
    raw_frames = list(raw_obj.get("frames") or [])
    prev_n = int(mouth_obj_ref.get("knn_raw_frames_done", 0) or 0)
    if len(raw_frames) < prev_n:
        prev_n = 0
        mouth_obj_ref["obj"] = None
    if len(raw_frames) == prev_n:
        existing = mouth_obj_ref.get("obj")
        if isinstance(existing, dict):
            total_n = len(existing.get("frames") or [])
            return existing, 0.0, 0, total_n
        prev_n = 0
    delta = raw_frames[prev_n:]
    delta_out = mouth_frames_from_raw(mod, delta, db_z, z)
    prev_obj = mouth_obj_ref.get("obj")
    if isinstance(prev_obj, dict) and prev_n > 0:
        prev_mouth = list(prev_obj.get("frames") or [])
    else:
        prev_mouth = []
    merged = prev_mouth + delta_out
    out_obj = {"audio": "", "step_ms": int(step_ms), "frames": merged}
    mouth_obj_ref["knn_raw_frames_done"] = len(raw_frames)
    mouth_obj_ref["obj"] = out_obj
    return out_obj, time.perf_counter() - t0, len(delta), len(merged)


def main() -> int:
    mod = _load_knn_mod()
    chunk_sizes = list(range(4, 81, 4))

    full = []
    for n in chunk_sizes:
        raw = make_raw(n)
        t0 = time.perf_counter()
        out = mod.run_knn_from_raw_obj(
            raw_obj=raw,
            gt_glob=GT,
            step_ms=40,
            k=5,
            fallback_id_active=2,
            min_conf_ratio=1.0,
        )
        full.append((n, (time.perf_counter() - t0) * 1000.0, len(out.get("frames") or [])))

    import glob

    gt_paths = sorted(glob.glob(GT))
    db = mod._load_knn_db(gt_paths)
    z = mod._compute_z(db)
    db_z = [(mod._z_point(f1, f2, z), vid) for (f1, f2, vid) in db]

    ref: dict = {"obj": None, "knn_raw_frames_done": 0}
    inc = []
    for n in chunk_sizes:
        raw = make_raw(n)
        _out, sec, delta, frames_n = run_incremental(mod, raw, db_z, z, 40, ref)
        inc.append((n, sec * 1000.0, delta, frames_n))

    def avg(vals):
        return sum(vals) / len(vals) if vals else 0.0

    full_early = avg([t for n, t, _ in full if n <= 20])
    full_late = avg([t for n, t, _ in full if n >= 60])
    inc_early = avg([t for n, t, _, _ in inc if n <= 20])
    inc_late = avg([t for n, t, _, _ in inc if n >= 60])

    print(
        f"FULL(reload+all): early(n<=20)={full_early:.1f}ms "
        f"late(n>=60)={full_late:.1f}ms ratio={full_late / max(full_early, 1e-9):.2f}"
    )
    print(
        f"INCR(delta+cache): early(n<=20)={inc_early:.1f}ms "
        f"late(n>=60)={inc_late:.1f}ms ratio={inc_late / max(inc_early, 1e-9):.2f}"
    )
    print("sample FULL:", [(n, round(t, 2)) for n, t, _ in full[:3] + full[-3:]])
    print("sample INCR:", [(n, round(t, 2), d) for n, t, d, _ in inc[:3] + inc[-3:]])

    full_last = mod.run_knn_from_raw_obj(
        raw_obj=make_raw(chunk_sizes[-1]),
        gt_glob=GT,
        step_ms=40,
        k=5,
        fallback_id_active=2,
        min_conf_ratio=1.0,
    )
    inc_ids = [fr.get("mouth_id") for fr in (ref["obj"] or {}).get("frames") or []]
    full_ids = [fr.get("mouth_id") for fr in full_last.get("frames") or []]
    if inc_ids != full_ids:
        print("MISMATCH", len(inc_ids), len(full_ids))
        for i, (a, b) in enumerate(zip(inc_ids, full_ids)):
            if a != b:
                print("first_diff", i, a, b)
                break
        return 1
    print("OK mouth_id sequence match n=", len(inc_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
