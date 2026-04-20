#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], *, cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_audio_ms(audio_meta_json: Path) -> int:
    obj = _load_json(audio_meta_json)
    return int(obj["audio_ms"])


def _read_events_from_unified(m1_unified_json: Path) -> List[Dict[str, Any]]:
    obj = _load_json(m1_unified_json)
    events = obj.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"events must be list: {m1_unified_json}")
    return [ev for ev in events if isinstance(ev, dict)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PCM push full E2E: M1/M2 PCM -> M3 raw -> mouth -> expr -> session -> M0 -> M3.5 bridge"
    )

    # repo roots
    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m0_repo_root", default="/workspaces/M0_session_renderer_final_1")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    # main ids
    ap.add_argument("--session_id", default="sess_sleepy_pcm_9_2_long")
    ap.add_argument("--utt_id", default="utt_sleepy_9_2_long")

    # inputs
    ap.add_argument(
        "--m1_unified_json",
        default="/workspaces/M1_LLM_To_M2_TTS_united/out/dev_m1_stub/m1_unified_output_9_2_long_end.json",
    )
    ap.add_argument(
        "--pose_json",
        default="/workspaces/M0_session_renderer_final_1/in/e2e_sleepy_9_2_long/pose.json",
    )
    ap.add_argument(
        "--audio_path",
        default="/workspaces/M0_session_renderer_final_1/in/e2e_sleepy_9_2_long/audio.wav",
    )
    ap.add_argument(
        "--normal_bg_video",
        default="/workspaces/M3.5_final/in/with1.mp4",
    )

    # phase-E first half (PCM push)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--analysis_sr", type=int, default=16000)
    ap.add_argument("--vad_energy_thr", type=float, default=0.0004)
    ap.add_argument("--max_buffer_ms", type=int, default=3000)
    ap.add_argument("--sleep_consumer_ms", type=int, default=0)

    # downstream knobs
    ap.add_argument("--chunk_len_ms", type=int, default=400)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--blink_duration_ms", type=int, default=40)
    ap.add_argument("--m0_base_config", default="configs/smoke_pose_improved.yaml")
    ap.add_argument("--knn_gt_glob", default="data/knn_db/*.f1f2.json")

    # output dirs
    ap.add_argument(
        "--pcm_out_dir",
        default=None,
        help="default: <m1_repo>/out/phase5_pcm_smoke/<session_id>",
    )
    ap.add_argument(
        "--m0_in_dir",
        default=None,
        help="default: <m0_repo>/in/<session_id>",
    )
    ap.add_argument(
        "--session_json",
        default=None,
        help="default: <m0_repo>/sessions/<session_id>.session.json",
    )
    ap.add_argument(
        "--manifest_json",
        default=None,
        help="default: <m0_repo>/out/manifests/<session_id>.chunk400.json",
    )
    ap.add_argument(
        "--plan_json",
        default=None,
        help="default: <m35_repo>/out/plans/<session_id>.plan.json",
    )
    ap.add_argument(
        "--bridge_out_root",
        default=None,
        help="default: <m35_repo>/out/runtime/<session_id>",
    )
    ap.add_argument(
        "--run_chunk_log",
        default=None,
        help="default: <m0_repo>/out/<session_id>.run_chunk.log.json",
    )

    # controls
    ap.add_argument("--plan_only", action="store_true")
    ap.add_argument("--skip_bridge", action="store_true")
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    if args.plan_only:
        args.skip_bridge = True

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    m1_unified_json = Path(args.m1_unified_json).resolve()
    pose_json = Path(args.pose_json).resolve()
    audio_path = Path(args.audio_path).resolve()
    normal_bg_video = Path(args.normal_bg_video).resolve()

    if not m1_unified_json.exists():
        raise FileNotFoundError(f"missing m1_unified_json: {m1_unified_json}")
    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")

    pcm_out_dir = Path(args.pcm_out_dir).resolve() if args.pcm_out_dir else (
        m1_repo / "out" / "phase5_pcm_smoke" / args.session_id
    )
    m0_in_dir = Path(args.m0_in_dir).resolve() if args.m0_in_dir else (
        m0_repo / "in" / args.session_id
    )
    session_json = Path(args.session_json).resolve() if args.session_json else (
        m0_repo / "sessions" / f"{args.session_id}.session.json"
    )
    manifest_json = Path(args.manifest_json).resolve() if args.manifest_json else (
        m0_repo / "out" / "manifests" / f"{args.session_id}.chunk{args.chunk_len_ms}.json"
    )
    plan_json = Path(args.plan_json).resolve() if args.plan_json else (
        m35_repo / "out" / "plans" / f"{args.session_id}.plan.json"
    )
    bridge_out_root = Path(args.bridge_out_root).resolve() if args.bridge_out_root else (
        m35_repo / "out" / "runtime" / args.session_id
    )
    run_chunk_log = Path(args.run_chunk_log).resolve() if args.run_chunk_log else (
        m0_repo / "out" / f"{args.session_id}.run_chunk.log.json"
    )

    _ensure_dir(pcm_out_dir)
    _ensure_dir(m0_in_dir)
    _ensure_dir(session_json.parent)
    _ensure_dir(manifest_json.parent)
    _ensure_dir(plan_json.parent)
    _ensure_dir(bridge_out_root)

    # outputs
    raw_json = pcm_out_dir / "mouth_timeline.formant.raw.json"
    audio_meta_json = pcm_out_dir / "audio_meta.json"
    expression_chunks_json = pcm_out_dir / "expression_chunks.v1.json"

    mouth_json = m0_in_dir / "mouth.json"
    mouth_clamped_json = m0_in_dir / "mouth.clamped.json"
    expr_json = m0_in_dir / "expr.json"

    print("[INFO] session_id      :", args.session_id)
    print("[INFO] utt_id          :", args.utt_id)
    print("[INFO] pcm_out_dir     :", pcm_out_dir)
    print("[INFO] m0_in_dir       :", m0_in_dir)
    print("[INFO] session_json    :", session_json)
    print("[INFO] manifest_json   :", manifest_json)
    print("[INFO] plan_json       :", plan_json)
    print("[INFO] bridge_out_root :", bridge_out_root)

    if args.dry_run:
        print("[DRY_RUN] stop before execution")
        return 0

    # --------------------------------------------------------
    # 1) M1/M2 PCM -> M3 raw
    # --------------------------------------------------------
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = (
        f"{m1_repo / 'src'}:{m3_repo / 'src'}:" + env.get("PYTHONPATH", "")
    )

    cmd_pcm = [
        sys.executable,
        str((m1_repo / "scripts" / "dev_f1_tts_pcm_to_m3_raw.py").resolve()),
        "--config", args.config,
        "--unified_json", str(m1_unified_json),
        "--out_dir", str(pcm_out_dir),
        "--session_id", args.session_id,
        "--utt_id", args.utt_id,
        "--step_ms", str(args.step_ms),
        "--analysis_sr", str(args.analysis_sr),
        "--vad_energy_thr", str(args.vad_energy_thr),
        "--max_buffer_ms", str(args.max_buffer_ms),
        "--sleep_consumer_ms", str(args.sleep_consumer_ms),
    ]
    print("[RUN]", " ".join(cmd_pcm))
    subprocess.run(cmd_pcm, check=True, cwd=str(m1_repo), env=env)

    if not raw_json.exists():
        raise FileNotFoundError(f"missing raw_json: {raw_json}")
    if not audio_meta_json.exists():
        raise FileNotFoundError(f"missing audio_meta_json: {audio_meta_json}")

    audio_ms = _read_audio_ms(audio_meta_json)
    print("[INFO] audio_ms (SSOT):", audio_ms)

    # --------------------------------------------------------
    # 2) raw -> mouth.json
    # --------------------------------------------------------
    _run(
        [
            sys.executable,
            str((m3_repo / "tools" / "knn_from_formant_raw_to_mouth_timeline.py").resolve()),
            "--raw", str(raw_json),
            "--gt_glob", args.knn_gt_glob,
            "--out", str(mouth_json),
        ],
        cwd=m3_repo,
    )

    # --------------------------------------------------------
    # 3) clamp mouth
    # --------------------------------------------------------
    _run(
        [
            sys.executable,
            str((m3_repo / "scripts" / "build_clamped_mouth.py").resolve()),
            "--mouth_json", str(mouth_json),
            "--out_json", str(mouth_clamped_json),
            "--audio_ms", str(audio_ms),
            "--step_ms", str(args.step_ms),
        ],
        cwd=m3_repo,
    )

    # --------------------------------------------------------
    # 4) unified -> expression_chunks
    # --------------------------------------------------------
    _run(
        [
            sys.executable,
            str((m1_repo / "scripts" / "build_expression_chunks_from_m1.py").resolve()),
            "--m1_unified_json", str(m1_unified_json),
            "--audio_meta_json", str(audio_meta_json),
            "--out", str(expression_chunks_json),
            "--session_id", str(args.session_id),
            "--chunk_start_ms", "0",
            "--step_ms", str(args.step_ms),
            "--emo_map_yaml", str((m1_repo / "configs" / "emo_id_to_expression.yaml").resolve()),
        ],
        cwd=m1_repo,
    )

    # --------------------------------------------------------
    # 5) expression_chunks -> expr.json
    # --------------------------------------------------------
    _run(
        [
            sys.executable,
            str((m3_repo / "scripts" / "build_session_expression_timeline_from_chunks.py").resolve()),
            "--in_chunks", str(expression_chunks_json),
            "--out", str(expr_json),
            "--session_id", str(args.session_id),
            "--auto_blink",
            "--blink_duration_ms", str(args.blink_duration_ms),
        ],
        cwd=m3_repo,
    )

    # --------------------------------------------------------
    # 6) session bundle
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
    # 7) manifest
    # --------------------------------------------------------
    _run(
        [
            sys.executable,
            str((m0_repo / "tools" / "build_chunk_manifest.py").resolve()),
            "--session_audio_ms", str(audio_ms),
            "--session_id", str(args.session_id),
            "--chunk_len_ms", str(args.chunk_len_ms),
            "--step_ms", str(args.step_ms),
            "--fps", str(args.fps),
            "--out", str(manifest_json),
        ],
        cwd=m0_repo,
    )

    # --------------------------------------------------------
    # 8) run_chunk
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 9) plan + bridge
    # --------------------------------------------------------
    sys.path.insert(0, str(m35_repo))
    from m3_5.event_to_plan import build_plan_from_chunks_and_events  # type: ignore
    from m3_5.run_chunk_bridge import run_from_manifest_and_events  # type: ignore

    manifest_obj = _load_json(manifest_json)
    events = _read_events_from_unified(m1_unified_json)

    plan = build_plan_from_chunks_and_events(
        manifest_obj["chunks"],
        events,
        normal_pose_json=str(pose_json),
        normal_bg_playlist_json=None,
        normal_bg_video=str(normal_bg_video),
        normal_pose_by_bg_video=None,
    )
    _dump_json(plan_json, plan)

    print("[OK] wrote plan_json:", plan_json)
    print("[OK] plan_chunks:", len(plan))
    print("[OK] last_chunk:", plan[-1])

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

    print("[dev_run_sleepy_pcm_full_e2e][OK]")
    print("  audio_ms           :", audio_ms)
    print("  raw_json           :", raw_json)
    print("  mouth_json         :", mouth_json)
    print("  mouth_clamped_json :", mouth_clamped_json)
    print("  expr_json          :", expr_json)
    print("  session_json       :", session_json)
    print("  manifest_json      :", manifest_json)
    print("  plan_json          :", plan_json)
    print("  bridge_out_root    :", bridge_out_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())