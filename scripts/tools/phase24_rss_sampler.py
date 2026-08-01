#!/usr/bin/env python3
"""Phase24: sample parent session_loop + M0 worker RSS (+ optional qsize from log tail)."""
from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def _classify(cmdline: str) -> str | None:
    s = (cmdline or "").lower().replace("\\", "/")
    if "m0_persistent_worker" in s or "m0_worker" in s:
        return "m0_worker"
    if "run_mic_input_obs_realtime_session_loop" in s:
        return "session_loop"
    if "dev_audio_chunk_player" in s:
        return "audio_player"
    if "run_virtualcam_persistent" in s or "virtualcam" in s:
        return "virtualcam"
    return None


def _qsize_from_log(log_path: Path | None) -> dict[str, str]:
    out = {
        "pipeline_inflight": "",
        "enqueue_qsize": "",
        "worker_qsize": "",
    }
    if log_path is None or not log_path.exists():
        return out
    try:
        # Read last ~64KB only.
        data = log_path.read_bytes()
        text = data[-65536:].decode("utf-8", errors="replace")
    except Exception:
        return out
    m = re.findall(r"pipeline_inflight=(\d+)", text)
    if m:
        out["pipeline_inflight"] = m[-1]
    m = re.findall(r"enqueue_queue=(\d+)", text)
    if m:
        out["enqueue_qsize"] = m[-1]
    m = re.findall(r"worker_queue=(\d+)", text)
    if m:
        out["worker_qsize"] = m[-1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval_s", type=float, default=2.0)
    ap.add_argument("--duration_s", type=float, default=240.0)
    ap.add_argument("--session_log", default="")
    args = ap.parse_args()

    if psutil is None:
        raise SystemExit("psutil required for phase24_rss_sampler")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.session_log) if args.session_log else None

    fields = [
        "ts",
        "elapsed_s",
        "pid",
        "name",
        "rss_mb",
        "pipeline_inflight",
        "enqueue_qsize",
        "worker_qsize",
    ]
    t0 = time.time()
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        f.flush()
        while (time.time() - t0) < float(args.duration_s):
            now = datetime.now(timezone.utc).astimezone().isoformat()
            elapsed = time.time() - t0
            q = _qsize_from_log(log_path)
            rows = []
            for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
                try:
                    cmd = " ".join(p.info.get("cmdline") or [])
                    kind = _classify(cmd)
                    if kind is None:
                        continue
                    rss = float(p.info["memory_info"].rss) / (1024.0 * 1024.0)
                    rows.append(
                        {
                            "ts": now,
                            "elapsed_s": f"{elapsed:.1f}",
                            "pid": str(p.info["pid"]),
                            "name": kind,
                            "rss_mb": f"{rss:.1f}",
                            "pipeline_inflight": q["pipeline_inflight"],
                            "enqueue_qsize": q["enqueue_qsize"],
                            "worker_qsize": q["worker_qsize"],
                        }
                    )
                except (psutil.Error, TypeError, KeyError):
                    continue
            for r in rows:
                w.writerow(r)
            f.flush()
            time.sleep(max(0.2, float(args.interval_s)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
