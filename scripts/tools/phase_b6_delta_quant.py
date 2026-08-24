#!/usr/bin/env python3
"""Phase B6: summarize overlay-tick Δ = display_bg_idx − pose_idx (obs only)."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def _as_frames(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [f for f in raw if isinstance(f, dict)]
    if isinstance(raw, dict):
        for k in ("frames", "timeline", "poses"):
            v = raw.get(k)
            if isinstance(v, list):
                return [f for f in v if isinstance(f, dict)]
    return []


def _ty(fr: dict[str, Any]) -> float | None:
    for k in ("ty", "t_y", "y"):
        if k in fr and fr[k] is not None:
            return float(fr[k])
    tr = fr.get("transform") or fr.get("pose") or {}
    if isinstance(tr, dict):
        for k in ("ty", "t_y", "y"):
            if k in tr and tr[k] is not None:
                return float(tr[k])
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _median(xs: list[int]) -> float:
    if not xs:
        return float("nan")
    return float(statistics.median(xs))


def _sign_label(xs: list[int]) -> str:
    if not xs:
        return "n/a"
    pos = sum(1 for x in xs if x > 0)
    neg = sum(1 for x in xs if x < 0)
    zer = sum(1 for x in xs if x == 0)
    if pos == 0 and neg == 0:
        return "Δ=0"
    if pos >= neg:
        return "BG進み"
    return "pose進み"


def _zone_rows(
    rows: list[dict[str, Any]],
    *,
    dty: list[float],
    thr: float,
    edge_n: int,
) -> dict[str, list[int]]:
    n = len(rows)
    states = [str(r.get("state") or "") for r in rows]
    fos = [int(r.get("frame_offset", 0) or 0) for r in rows]
    relocks = [str(r.get("relock") or "") for r in rows]
    turn_mark = [False] * n
    idle_edge = [False] * n
    for i in range(n):
        if relocks[i] == "turn":
            for j in range(max(0, i - edge_n), min(n, i + edge_n + 1)):
                turn_mark[j] = True
        if i > 0 and fos[i] > fos[i - 1]:
            for j in range(max(0, i - edge_n), min(n, i + edge_n + 1)):
                turn_mark[j] = True
        if i > 0 and (states[i] == "PLAYING") != (states[i - 1] == "PLAYING"):
            for j in range(max(0, i - edge_n), min(n, i + edge_n + 1)):
                idle_edge[j] = True
        if states[i] != "PLAYING":
            idle_edge[i] = True

    buckets: dict[str, list[int]] = {
        "定常PLAYING": [],
        "idle境": [],
        "ターン境": [],
        "高速上下": [],
    }
    for i, r in enumerate(rows):
        d = r.get("delta")
        if d is None:
            continue
        try:
            d_i = int(d)
        except Exception:
            continue
        bg = int(r.get("display_bg_idx", -1) or -1)
        hi = False
        if bg >= 0 and dty:
            ad = abs(dty[int(bg) % len(dty)])
            hi = ad >= float(thr)
        if turn_mark[i]:
            buckets["ターン境"].append(d_i)
        elif idle_edge[i]:
            buckets["idle境"].append(d_i)
        elif hi:
            buckets["高速上下"].append(d_i)
        elif states[i] == "PLAYING":
            buckets["定常PLAYING"].append(d_i)
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta_jsonl", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument(
        "--pose_json",
        default=r"C:\dev\M0_session_renderer_final_1\timelines\pose\pose_timeline_final_with1_4.json",
    )
    ap.add_argument("--edge_n", type=int, default=8)
    args = ap.parse_args()

    rows = _load_jsonl(Path(args.delta_jsonl))
    usable = [r for r in rows if r.get("delta") is not None]
    pose_frames = _as_frames(json.loads(Path(args.pose_json).read_text(encoding="utf-8")))
    tys = [_ty(f) for f in pose_frames]
    dty = [0.0] * len(tys)
    for i in range(1, len(tys)):
        a = tys[i]
        b = tys[i - 1]
        if a is None or b is None:
            continue
        dty[i] = float(a) - float(b)
    ad = sorted(abs(x) for x in dty[1:]) if len(dty) > 1 else [0.0]
    thr = ad[int(0.9 * (len(ad) - 1))] if ad else 3.655

    buckets = _zone_rows(usable, dty=dty, thr=float(thr), edge_n=int(args.edge_n))
    print(f"lang={args.lang} ticks={len(rows)} usable={len(usable)} dty_p90={thr:.3f}")
    print("言語 | 区間 | n | Δ中央 | Δ最大 | 符号")
    for zone in ("定常PLAYING", "idle境", "ターン境", "高速上下"):
        xs = buckets[zone]
        if not xs:
            print(f"{args.lang} | {zone} | 0 | n/a | n/a | n/a")
            continue
        print(
            f"{args.lang} | {zone} | {len(xs)} | {_median(xs):.1f} | {max(abs(x) for x in xs)} "
            f"(raw_max={max(xs)} raw_min={min(xs)}) | {_sign_label(xs)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
