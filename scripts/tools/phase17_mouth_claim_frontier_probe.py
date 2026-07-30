#!/usr/bin/env python3
"""Phase 17: mouth frontier vs claim until_ms mid-band probe (helper only).

Reconstructs mouth last t_ms from [knn][mouth_obj_updated] frames=N
(approx last_t = (N-1)*step_ms) and compares to enqueue_timeline_end_ms
(= claim/coverage need) alongside wait_mouth / knn_ms.

Does not change runtime behavior. Optional: parse new
[sync][pipeline_chunk][mouth_claim_frontier] lines if present.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[int(round((len(s) - 1) * p))]


def _g(line: str, k: str) -> float | None:
    m = re.search(rf"{k}=([0-9.\-]+)", line)
    return float(m.group(1)) if m else None


def analyze(path: Path, step_ms: int = 40) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"file={path.name}")

    mouth_upd: list[tuple[int, int]] = []
    rows: list[dict] = []
    frontier_logs: list[dict] = []

    for i, line in enumerate(text.splitlines()):
        if "[knn][mouth_obj_updated]" in line:
            m = re.search(r"frames=(\d+)", line)
            if m:
                mouth_upd.append((i, int(m.group(1))))
        if "[sync][pipeline_chunk][mouth_claim_frontier]" in line:
            frontier_logs.append(
                {
                    "line": i,
                    "until_ms": _g(line, "until_ms") or _g(line, "need_t1_ms"),
                    "mouth_last_t_ms": _g(line, "mouth_last_t_ms"),
                    "mouth_cov_ms": _g(line, "mouth_cov_ms"),
                    "gap_ms": _g(line, "gap_ms"),
                    "next_claim_t1_ms": _g(line, "next_claim_t1_ms"),
                    "rendered_end_ms": _g(line, "rendered_end_ms"),
                    "mouth_ready": _g(line, "mouth_ready"),
                }
            )
        if "[sync][pipeline_chunk] " in line and "pipeline_seq=" in line:
            rows.append(
                {
                    "line": i,
                    "chunk_idx": _g(line, "chunk_idx"),
                    "knn_ms": _g(line, "knn_ms"),
                    "wait": _g(line, "m0_wait_mouth_ms"),
                    "lock": _g(line, "m0_lock_ms"),
                    "png": _g(line, "m0_png_wait_ms"),
                    "m0": _g(line, "m0_ms"),
                    "mouth_n": _g(line, "mouth_frames_n"),
                    "until": _g(line, "enqueue_timeline_end_ms"),
                }
            )

    # mouth_frames_n on pipeline_chunk is captured at knn_done (before wait_mouth),
    # so it approximates mouth frontier at wait start — not end-of-wait contamination.
    # mouth_obj_updated nearest before the summary line is end-biased; keep as secondary.
    j = 0
    for r in rows:
        while j + 1 < len(mouth_upd) and mouth_upd[j + 1][0] <= r["line"]:
            j += 1
        mf_end = mouth_upd[j][1] if mouth_upd and mouth_upd[j][0] <= r["line"] else None
        mn = r["mouth_n"]
        if mn is not None and mn > 0:
            mouth_last = float((mn - 1) * step_ms)
            mouth_cov = float(mn * step_ms)  # last_t + step_ms
        else:
            mouth_last = 0.0
            mouth_cov = 0.0
        until = float(r["until"] or 0.0)
        gap = until - mouth_cov
        # Claims are chunk_len_ms (120) aligned; last needed claim has
        # t0 = floor((until-1)/120)*120, t1 = t0+120 (when until>0).
        chunk_len = 120.0
        if until > 0:
            last_claim_t0 = float(int((until - 1) // chunk_len) * chunk_len)
            claim_t1 = last_claim_t0 + chunk_len
        else:
            claim_t1 = 0.0
        claim_gap = claim_t1 - mouth_cov
        r["mouth_upd_n_end"] = mf_end
        r["mouth_last"] = mouth_last
        r["mouth_cov"] = mouth_cov
        r["gap"] = gap
        r["claim_t1"] = claim_t1
        r["claim_gap"] = claim_gap
        r["behind"] = gap > 0.0
        r["ahead"] = gap <= 0.0
        r["claim_behind"] = claim_gap > 0.0
        r["claim_ahead"] = claim_gap <= 0.0
        if mf_end is not None and mf_end > 0:
            end_cov = float(mf_end * step_ms)
            r["end_gap"] = until - end_cov
            r["end_claim_gap"] = claim_t1 - end_cov
        else:
            r["end_gap"] = None
            r["end_claim_gap"] = None

    n = len(rows)
    lo = int(n * 0.35) if n else 0
    hi = int(n * 0.75) if n else 0
    mid = rows[lo:hi] if n else []
    print(f"pipeline_chunks={n} mid_slice={lo}:{hi} mouth_upd_events={len(mouth_upd)}")
    if mouth_upd:
        print(
            f"mouth_upd first_n={mouth_upd[0][1]} last_n={mouth_upd[-1][1]} "
            f"last_t~={(mouth_upd[-1][1] - 1) * step_ms}"
        )
    print(f"mouth_claim_frontier_logs={len(frontier_logs)}")

    for label, src in (("all", rows), ("mid", mid)):
        waits = [float(r["wait"]) for r in src if r["wait"] is not None]
        knns = [float(r["knn_ms"]) for r in src if r["knn_ms"] is not None]
        gaps = [float(r["gap"]) for r in src if r["gap"] is not None]
        cgaps = [float(r["claim_gap"]) for r in src if r["claim_gap"] is not None]
        high = [r for r in src if r["wait"] is not None and float(r["wait"]) >= 200.0]
        hw_behind = sum(1 for r in high if r["behind"])
        hw_ahead = sum(1 for r in high if r["ahead"])
        hw_cbehind = sum(1 for r in high if r["claim_behind"])
        hw_cahead = sum(1 for r in high if r["claim_ahead"])
        behind = sum(1 for r in src if r["behind"])
        ahead = sum(1 for r in src if r["ahead"])
        cbehind = sum(1 for r in src if r["claim_behind"])
        cahead = sum(1 for r in src if r["claim_ahead"])
        print(
            f"{label}.wait: n={len(waits)} p50={_pct(waits, 0.5):.1f} "
            f"p90={_pct(waits, 0.9):.1f} max={max(waits) if waits else 0:.1f}"
        )
        print(
            f"{label}.knn: n={len(knns)} p50={_pct(knns, 0.5):.1f} "
            f"p90={_pct(knns, 0.9):.1f} max={max(knns) if knns else 0:.1f}"
        )
        print(
            f"{label}.gap(until-mouth_cov): n={len(gaps)} p50={_pct(gaps, 0.5):.1f} "
            f"p90={_pct(gaps, 0.9):.1f} max={max(gaps) if gaps else 0:.1f} "
            f"behind={behind} ahead={ahead}"
        )
        print(
            f"{label}.claim_gap(claim_t1-mouth_cov): n={len(cgaps)} "
            f"p50={_pct(cgaps, 0.5):.1f} p90={_pct(cgaps, 0.9):.1f} "
            f"max={max(cgaps) if cgaps else 0:.1f} "
            f"behind={cbehind} ahead={cahead}"
        )
        print(
            f"{label}.high_wait>=200: n={len(high)} until_behind={hw_behind} "
            f"until_ahead={hw_ahead} claim_behind={hw_cbehind} claim_ahead={hw_cahead}"
        )
        if high:
            step = max(1, len(high) // 6)
            print(f"{label}.high_wait samples:")
            for r in high[::step][:8]:
                print(
                    f"  c={int(r['chunk_idx'] or -1)} until={r['until']:.0f} "
                    f"claim_t1={r['claim_t1']:.0f} mouth_last={r['mouth_last']:.0f} "
                    f"cov={r['mouth_cov']:.0f} gap={r['gap']:.0f} "
                    f"cgap={r['claim_gap']:.0f} wait={r['wait']:.0f} knn={r['knn_ms']}"
                )

    print(
        "mid series (sampled): chunk until claim_t1 mouth_last@knn cov "
        "gap cgap wait knn lock"
    )
    step = max(1, len(mid) // 12) if mid else 1
    for r in mid[::step][:14]:
        print(
            f"  c={int(r['chunk_idx'] or -1)} u={r['until']:.0f} "
            f"ct1={r['claim_t1']:.0f} ml={r['mouth_last']:.0f} "
            f"cov={r['mouth_cov']:.0f} gap={r['gap']:.0f} "
            f"cgap={r['claim_gap']:.0f} w={r['wait']:.0f} "
            f"knn={r['knn_ms']:.1f} lock={r['lock']:.0f}"
        )

    # Lead of mouth vs player in mid: pair nearest player_local after chunk line.
    player_pts: list[tuple[int, float]] = []
    for i, line in enumerate(text.splitlines()):
        if "__AUDIO_PLAYER_RESPONSE__" not in line:
            continue
        try:
            import json

            jobj = json.loads(line.split("__AUDIO_PLAYER_RESPONSE__", 1)[1].strip())
        except Exception:
            continue
        if str(jobj.get("cmd") or "") != "play":
            continue
        player_pts.append((i, float(jobj.get("player_local_ms") or 0.0)))
    if mid and player_pts:
        leads = []
        pk = 0
        for r in mid:
            while pk + 1 < len(player_pts) and player_pts[pk + 1][0] <= r["line"]:
                pk += 1
            if player_pts and player_pts[pk][0] <= r["line"]:
                leads.append(r["mouth_cov"] - player_pts[pk][1])
        if leads:
            print(
                f"mid.mouth_cov_minus_player_local: n={len(leads)} "
                f"p50={_pct(leads, 0.5):.1f} p10={_pct(leads, 0.1):.1f} "
                f"min={min(leads):.1f} neg={sum(1 for x in leads if x < 0)}"
            )

    # Verdict: prefer claim_t1 gap (true mouth_ready gate) over until gap.
    mid_high = [r for r in mid if r["wait"] is not None and float(r["wait"]) >= 200.0]
    if mid_high:
        b = sum(1 for r in mid_high if r["claim_behind"])
        a = sum(1 for r in mid_high if r["claim_ahead"])
        print(f"verdict_hint mid_high_wait claim_behind={b} claim_ahead={a}")
        if b >= max(1, int(0.7 * len(mid_high))):
            print("verdict_hint=A_mouth_slow")
        elif a >= max(1, int(0.7 * len(mid_high))):
            print("verdict_hint=B_mouth_ready_but_waiting")
        else:
            print("verdict_hint=mixed")
    else:
        mid_behind = sum(1 for r in mid if r["claim_behind"])
        mid_ahead = sum(1 for r in mid if r["claim_ahead"])
        print(f"verdict_hint mid_claim_gap behind={mid_behind} ahead={mid_ahead}")
        if mid_behind > mid_ahead * 2:
            print("verdict_hint=A_mouth_slow")
        elif mid_ahead > mid_behind * 2:
            print("verdict_hint=B_mouth_ready_but_waiting")
        else:
            print("verdict_hint=mixed")

    if frontier_logs:
        gaps_f = [float(x["gap_ms"]) for x in frontier_logs if x["gap_ms"] is not None]
        print(
            f"frontier_log.gap: n={len(gaps_f)} p50={_pct(gaps_f, 0.5):.1f} "
            f"p90={_pct(gaps_f, 0.9):.1f} behind={sum(1 for g in gaps_f if g > 0)} "
            f"ahead={sum(1 for g in gaps_f if g <= 0)}"
        )


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: phase17_mouth_claim_frontier_probe.py <log> [log...]")
        return 2
    for arg in sys.argv[1:]:
        print("=" * 60)
        analyze(Path(arg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
