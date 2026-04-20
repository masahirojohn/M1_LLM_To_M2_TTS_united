#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# helpers
# ============================================================

def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd: List[str], *, cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _read_audio_ms(audio_meta_json: Path) -> int:
    obj = _load_json(audio_meta_json)
    audio_ms = obj.get("audio_ms")
    if audio_ms is None:
        raise ValueError(f"audio_meta.json missing audio_ms: {audio_meta_json}")
    return int(audio_ms)


def _read_step_ms(audio_meta_json: Path, default_step_ms: int = 40) -> int:
    obj = _load_json(audio_meta_json)
    return int(obj.get("step_ms", default_step_ms))


def _read_events_from_m1_unified(m1_unified_json: Path) -> List[Dict[str, Any]]:
    obj = _load_json(m1_unified_json)
    events = obj.get("events", [])
    if events is None:
        return []
    if not isinstance(events, list):
        raise ValueError(f"m1 unified events must be list: {m1_unified_json}")
    return [ev for ev in events if isinstance(ev, dict)]


# ============================================================
# main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sleepy full E2E orchestrator: M1 -> expr -> mouth clamp -> session -> manifest -> plan -> bridge"
    )

    # ---- repo roots
    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m0_repo_root", default="/workspaces/M0_session_renderer_final_1")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    # ---- required inputs
    ap.add_argument(
        "--m1_unified_json",
        default="/workspaces/M1_LLM_To_M2_TTS_united/out/dev_m1_stub/m1_unified_output_9_2_long_end.json",
    )
    ap.add_argument(
        "--audio_meta_json",
        default="/workspaces/M0_session_renderer_final_1/in/e2e_sleepy_9_2_long/audio_meta.json",
    )
    ap.add_argument(
        "--mouth_json",
        default="/workspaces/M0_session_renderer_final_1/in/e2e_sleepy_9_2_long/mouth.json",
    )
    ap.add_argument(
        "--pose_json",
        default="/workspaces/M0_session_renderer_final_1/in/e2e_sleepy_9_2_long/pose.json",
    )

    # ---- session / bg
    ap.add_argument("--session_id", default="sess_sleepy_9_2_long")
    ap.add_argument("--audio_path", default="/workspaces/M0_session_renderer_final_1/in/e2e_sleepy_9_2_long/audio.wav")
    ap.add_argument("--normal_bg_video", default="/workspaces/M3.5_final/in/with1.mp4")

    # ---- config knobs
    ap.add_argument("--chunk_len_ms", type=int, default=400)
    ap.add_argument("--step_ms", type=int, default=None, help="default: read from audio_meta.json, fallback=40")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--m0_base_config", default="configs/smoke_pose_improved.yaml")
    ap.add_argument("--blink_duration_ms", type=int, default=None, help="default: step_ms")
    ap.add_argument("--event_id", default="sleepy_auto_end")
    ap.add_argument("--trigger_source", default="auto_sleepy_timeout")

    # ---- output roots
    ap.add_argument("--m1_out_dir", default=None, help="default: <m1_repo>/out/dev_sleepy_full_e2e/<session_id>")
    ap.add_argument("--m0_in_dir", default=None, help="default: <m0_repo>/in/<session_id>")
    ap.add_argument("--session_json", default=None, help="default: <m0_repo>/sessions/<session_id>.session.json")
    ap.add_argument("--manifest_json", default=None, help="default: <m0_repo>/out/manifests/<session_id>.chunk400.json")
    ap.add_argument("--plan_json", default=None, help="default: <m35_repo>/out/plans/<session_id>.plan.json")
    ap.add_argument("--bridge_out_root", default=None, help="default: <m35_repo>/out/runtime/<session_id>")

    # ---- controls
    ap.add_argument("--skip_add_end_stream", action="store_true")
    ap.add_argument("--skip_run_chunk", action="store_true", help="skip M0 tools/run_chunk.py")
    ap.add_argument("--skip_bridge", action="store_true", help="stop after plan.json")
    ap.add_argument("--plan_only", action="store_true", help="same as --skip_run_chunk --skip_bridge")
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    if args.plan_only:
        args.skip_run_chunk = True
        args.skip_bridge = True

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    m1_unified_json = Path(args.m1_unified_json).resolve()
    audio_meta_json = Path(args.audio_meta_json).resolve()
    mouth_json = Path(args.mouth_json).resolve()
    pose_json = Path(args.pose_json).resolve()
    audio_path = Path(args.audio_path).resolve()
    normal_bg_video = Path(args.normal_bg_video).resolve()

    for p in [m1_unified_json, audio_meta_json, mouth_json, pose_json]:
        if not p.exists():
            raise FileNotFoundError(f"missing required input: {p}")

    audio_ms = _read_audio_ms(audio_meta_json)
    step_ms = int(args.step_ms) if args.step_ms is not None else _read_step_ms(audio_meta_json, 40)
    blink_duration_ms = int(args.blink_duration_ms) if args.blink_duration_ms is not None else step_ms

    m1_out_dir = Path(args.m1_out_dir).resolve() if args.m1_out_dir else (m1_repo / "out" / "dev_sleepy_full_e2e" / args.session_id)
    m0_in_dir = Path(args.m0_in_dir).resolve() if args.m0_in_dir else (m0_repo / "in" / args.session_id)
    session_json = Path(args.session_json).resolve() if args.session_json else (m0_repo / "sessions" / f"{args.session_id}.session.json")
    manifest_json = Path(args.manifest_json).resolve() if args.manifest_json else (m0_repo / "out" / "manifests" / f"{args.session_id}.chunk{args.chunk_len_ms}.json")
    plan_json = Path(args.plan_json).resolve() if args.plan_json else (m35_repo / "out" / "plans" / f"{args.session_id}.plan.json")
    bridge_out_root = Path(args.bridge_out_root).resolve() if args.bridge_out_root else (m35_repo / "out" / "runtime" / args.session_id)

    _ensure_dir(m1_out_dir)
    _ensure_dir(m0_in_dir)
    _ensure_dir(session_json.parent)
    _ensure_dir(manifest_json.parent)
    _ensure_dir(plan_json.parent)
    _ensure_dir(bridge_out_root)

    # ---- derived outputs
    m1_end_json = m1_out_dir / f"{args.session_id}.m1_unified.end.json"
    expression_chunks_json = m1_out_dir / f"{args.session_id}.expression_chunks.v1.json"
    expr_json = m0_in_dir / "expr.json"
    mouth_clamped_json = m0_in_dir / "mouth.clamped.json"
    run_chunk_log = m0_repo / "out" / f"{args.session_id}.run_chunk.log.json"

    print("[INFO] session_id        :", args.session_id)
    print("[INFO] audio_ms          :", audio_ms)
    print("[INFO] step_ms           :", step_ms)
    print("[INFO] chunk_len_ms      :", args.chunk_len_ms)
    print("[INFO] m1_unified_json   :", m1_unified_json)
    print("[INFO] audio_meta_json   :", audio_meta_json)
    print("[INFO] mouth_json        :", mouth_json)
    print("[INFO] pose_json         :", pose_json)
    print("[INFO] normal_bg_video   :", normal_bg_video)
    print("[INFO] m1_out_dir        :", m1_out_dir)
    print("[INFO] m0_in_dir         :", m0_in_dir)
    print("[INFO] session_json      :", session_json)
    print("[INFO] manifest_json     :", manifest_json)
    print("[INFO] plan_json         :", plan_json)
    print("[INFO] bridge_out_root   :", bridge_out_root)

    if args.dry_run:
        print("[DRY_RUN] stop before execution")
        return 0

    # --------------------------------------------------------
    # 1) add_auto_end_stream
    # --------------------------------------------------------
    if args.skip_add_end_stream:
        m1_end_json = m1_unified_json
        print("[SKIP] add_auto_end_stream -> use input as-is")
    else:
        _run(
            [
                sys.executable,
                str((m1_repo / "scripts" / "add_auto_end_stream.py").resolve()),
                "--in_json", str(m1_unified_json),
                "--out_json", str(m1_end_json),
                "--audio_ms", str(audio_ms),
                "--step_ms", str(step_ms),
                "--event_id", str(args.event_id),
                "--trigger_source", str(args.trigger_source),
            ],
            cwd=m1_repo,
        )

    # --------------------------------------------------------
    # 2) build_expression_chunks_from_m1
    # --------------------------------------------------------
    _run(
        [
            sys.executable,
            str((m1_repo / "scripts" / "build_expression_chunks_from_m1.py").resolve()),
            "--m1_unified_json", str(m1_end_json),
            "--audio_meta_json", str(audio_meta_json),
            "--out", str(expression_chunks_json),
            "--session_id", str(args.session_id),
            "--chunk_start_ms", "0",
            "--step_ms", str(step_ms),
            "--emo_map_yaml", str((m1_repo / "configs" / "emo_id_to_expression.yaml").resolve()),
        ],
        cwd=m1_repo,
    )

    # --------------------------------------------------------
    # 3) build_session_expression_timeline_from_chunks --auto_blink
    # --------------------------------------------------------
    _run(
        [
            sys.executable,
            str((m3_repo / "scripts" / "build_session_expression_timeline_from_chunks.py").resolve()),
            "--in_chunks", str(expression_chunks_json),
            "--out", str(expr_json),
            "--session_id", str(args.session_id),
            "--auto_blink",
            "--blink_duration_ms", str(blink_duration_ms),
        ],
        cwd=m3_repo,
    )

    # --------------------------------------------------------
    # 4) build_clamped_mouth
    # --------------------------------------------------------
    _run(
        [
            sys.executable,
            str((m3_repo / "scripts" / "build_clamped_mouth.py").resolve()),
            "--mouth_json", str(mouth_json),
            "--out_json", str(mouth_clamped_json),
            "--audio_ms", str(audio_ms),
            "--step_ms", str(step_ms),
        ],
        cwd=m3_repo,
    )

    # --------------------------------------------------------
    # 5) build_session_bundle
    # --------------------------------------------------------
    _run(
        [
            sys.executable,
            str((m1_repo / "scripts" / "build_session_bundle.py").resolve()),
            "--session_id", str(args.session_id),
            "--audio_ms", str(audio_ms),
            "--mouth_json", str(mouth_clamped_json),
            "--expr_json", str(expr_json),
            "--pose_json", str(pose_json),
            "--audio_path", str(audio_path),
            "--out_json", str(session_json),
        ],
        cwd=m1_repo,
    )

    # --------------------------------------------------------
    # 6) build_chunk_manifest
    # --------------------------------------------------------
    _run(
        [
            sys.executable,
            str((m0_repo / "tools" / "build_chunk_manifest.py").resolve()),
            "--session_audio_ms", str(audio_ms),
            "--session_id", str(args.session_id),
            "--chunk_len_ms", str(args.chunk_len_ms),
            "--step_ms", str(step_ms),
            "--fps", str(args.fps),
            "--out", str(manifest_json),
        ],
        cwd=m0_repo,
    )

    # --------------------------------------------------------
    # 7) run_chunk
    # --------------------------------------------------------
    if not args.skip_run_chunk:
        _run(
            [
                sys.executable,
                str((m0_repo / "tools" / "run_chunk.py").resolve()),
                "--session_json", str(session_json),
                "--manifest", str(manifest_json),
                "--chunk_len_ms", str(args.chunk_len_ms),
                "--clean",
                "--verify",
                "--log", str(run_chunk_log),
            ],
            cwd=m0_repo,
        )
    else:
        print("[SKIP] run_chunk")

    # --------------------------------------------------------
    # 8) plan generation
    # --------------------------------------------------------
    # use uploaded / repo implementation by import path
    sys.path.insert(0, str(m35_repo))
    from m3_5.event_to_plan import build_plan_from_chunks_and_events, summarize_plan  # type: ignore
    from m3_5.run_chunk_bridge import run_from_manifest_and_events  # type: ignore

    manifest_obj = _load_json(manifest_json)
    chunks = manifest_obj.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"manifest chunks must be list: {manifest_json}")

    events = _read_events_from_m1_unified(m1_end_json)
    plan = build_plan_from_chunks_and_events(
        chunks,
        events,
        normal_pose_json=str(pose_json),
        normal_bg_playlist_json=None,
        normal_bg_video=str(normal_bg_video),
        normal_pose_by_bg_video=None,
    )
    _dump_json(plan_json, plan)

    print("[PLAN] chunks:", len(plan))
    if plan:
        print("[PLAN] first :", summarize_plan(plan)[0])
        print("[PLAN] last  :", summarize_plan(plan)[-1])

    # --------------------------------------------------------
    # 9) bridge
    # --------------------------------------------------------
    if not args.skip_bridge:
        run_from_manifest_and_events(
            manifest_json=str(manifest_json),
            events=events,
            session_id=str(args.session_id),
            normal_pose_json=str(pose_json),
            normal_bg_playlist_json=None,
            normal_bg_video=str(normal_bg_video),
            normal_pose_by_bg_video=None,
            normal_mouth_json=str(mouth_clamped_json),
            normal_expr_json=str(expr_json),
            out_root=str(bridge_out_root),
            m0_repo_root=str(m0_repo),
            m3_5_repo_root=str(m35_repo),
            m0_base_config=str(args.m0_base_config),
            cli_cfg=None,
            use_fg_index=True,
            fg_mode="follow",
            debug_print_plan=True,
            debug_plan_range=None,
        )
    else:
        print("[SKIP] bridge")

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------
    print("[dev_run_sleepy_full_e2e][OK]")
    print("  audio_ms             :", audio_ms)
    print("  step_ms              :", step_ms)
    print("  m1_end_json          :", m1_end_json)
    print("  expression_chunks    :", expression_chunks_json)
    print("  expr_json            :", expr_json)
    print("  mouth_clamped_json   :", mouth_clamped_json)
    print("  session_json         :", session_json)
    print("  manifest_json        :", manifest_json)
    print("  plan_json            :", plan_json)
    print("  bridge_out_root      :", bridge_out_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())