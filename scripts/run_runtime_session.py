#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_STEP_MS = 40
DEFAULT_CHUNK_LEN_MS = 400
DEFAULT_FPS = 25
DEFAULT_EVENT_ID = "sleepy_auto_end"
DEFAULT_TRIGGER_SOURCE = "auto_sleepy_timeout"


def _run(cmd: List[str], *, cwd: Path | None = None, env: Dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


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


def _has_end_stream_event(events: List[Dict[str, Any]]) -> bool:
    for ev in events:
        if str(ev.get("event_mode", "")) == "end_stream":
            return True
    return False


def _build_runtime_env(m1_repo: Path, m3_repo: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{m1_repo / 'src'}:{m3_repo / 'src'}:" + env.get("PYTHONPATH", "")
    return env


def run_m2_and_pcm_push(
    *,
    python_exe: str,
    m1_repo: Path,
    m3_repo: Path,
    m1_unified_json: Path,
    pcm_out_dir: Path,
    session_id: str,
    utt_id: str,
    config: str,
    step_ms: int,
    analysis_sr: int,
    vad_energy_thr: float,
    max_buffer_ms: int,
    sleep_consumer_ms: int,
) -> tuple[Path, Path]:
    env = _build_runtime_env(m1_repo, m3_repo)

    _run(
        [
            python_exe,
            str((m1_repo / "scripts" / "dev_f1_tts_pcm_to_m3_raw.py").resolve()),
            "--config", config,
            "--unified_json", str(m1_unified_json),
            "--out_dir", str(pcm_out_dir),
            "--session_id", session_id,
            "--utt_id", utt_id,
            "--step_ms", str(step_ms),
            "--analysis_sr", str(analysis_sr),
            "--vad_energy_thr", str(vad_energy_thr),
            "--max_buffer_ms", str(max_buffer_ms),
            "--sleep_consumer_ms", str(sleep_consumer_ms),
        ],
        cwd=m1_repo,
        env=env,
    )

    raw_json = pcm_out_dir / "mouth_timeline.formant.raw.json"
    audio_meta_json = pcm_out_dir / "audio_meta.json"
    if not raw_json.exists():
        raise FileNotFoundError(f"missing raw_json: {raw_json}")
    if not audio_meta_json.exists():
        raise FileNotFoundError(f"missing audio_meta_json: {audio_meta_json}")
    return raw_json, audio_meta_json


def build_mouth_from_raw(
    *,
    python_exe: str,
    m3_repo: Path,
    raw_json: Path,
    knn_gt_glob: str,
    mouth_json: Path,
) -> None:
    _run(
        [
            python_exe,
            str((m3_repo / "tools" / "knn_from_formant_raw_to_mouth_timeline.py").resolve()),
            "--raw", str(raw_json),
            "--gt_glob", knn_gt_glob,
            "--out", str(mouth_json),
        ],
        cwd=m3_repo,
    )


def build_clamped_mouth_runtime(
    *,
    python_exe: str,
    m3_repo: Path,
    mouth_json: Path,
    mouth_clamped_json: Path,
    audio_ms: int,
    step_ms: int,
) -> None:
    _run(
        [
            python_exe,
            str((m3_repo / "scripts" / "build_clamped_mouth.py").resolve()),
            "--mouth_json", str(mouth_json),
            "--out_json", str(mouth_clamped_json),
            "--audio_ms", str(audio_ms),
            "--step_ms", str(step_ms),
        ],
        cwd=m3_repo,
    )


def build_expr_from_unified(
    *,
    python_exe: str,
    m1_repo: Path,
    m3_repo: Path,
    m1_unified_json: Path,
    audio_meta_json: Path,
    expression_chunks_json: Path,
    expr_json: Path,
    session_id: str,
    step_ms: int,
    blink_duration_ms: int,
) -> None:
    _run(
        [
            python_exe,
            str((m1_repo / "scripts" / "build_expression_chunks_from_m1.py").resolve()),
            "--m1_unified_json", str(m1_unified_json),
            "--audio_meta_json", str(audio_meta_json),
            "--out", str(expression_chunks_json),
            "--session_id", str(session_id),
            "--chunk_start_ms", "0",
            "--step_ms", str(step_ms),
            "--emo_map_yaml", str((m1_repo / "configs" / "emo_id_to_expression.yaml").resolve()),
        ],
        cwd=m1_repo,
    )

    _run(
        [
            python_exe,
            str((m3_repo / "scripts" / "build_session_expression_timeline_from_chunks.py").resolve()),
            "--in_chunks", str(expression_chunks_json),
            "--out", str(expr_json),
            "--session_id", str(session_id),
            "--auto_blink",
            "--blink_duration_ms", str(blink_duration_ms),
        ],
        cwd=m3_repo,
    )


def build_session_bundle_runtime(
    *,
    python_exe: str,
    m1_repo: Path,
    session_id: str,
    audio_ms: int,
    mouth_clamped_json: Path,
    expr_json: Path,
    pose_json: Path,
    audio_path: Path | None,
    session_json: Path,
) -> None:
    cmd = [
        python_exe,
        str((m1_repo / "scripts" / "build_session_bundle.py").resolve()),
        "--session_id", str(session_id),
        "--audio_ms", str(audio_ms),
        "--mouth_json", str(mouth_clamped_json),
        "--expr_json", str(expr_json),
        "--pose_json", str(pose_json),
        "--out_json", str(session_json),
    ]

    if audio_path is not None:
        cmd += ["--audio_path", str(audio_path)]

    _run(cmd, cwd=m1_repo)


def build_manifest_runtime(
    *,
    python_exe: str,
    m0_repo: Path,
    session_id: str,
    audio_ms: int,
    chunk_len_ms: int,
    step_ms: int,
    fps: int,
    manifest_json: Path,
) -> None:
    _run(
        [
            python_exe,
            str((m0_repo / "tools" / "build_chunk_manifest.py").resolve()),
            "--session_audio_ms", str(audio_ms),
            "--session_id", str(session_id),
            "--chunk_len_ms", str(chunk_len_ms),
            "--step_ms", str(step_ms),
            "--fps", str(fps),
            "--out", str(manifest_json),
        ],
        cwd=m0_repo,
    )


def run_chunk_runtime(
    *,
    python_exe: str,
    m0_repo: Path,
    session_json: Path,
    manifest_json: Path,
    chunk_len_ms: int,
    run_chunk_log: Path,
    base_config: str,
    clean: bool,
    verify: bool,
    resume: bool,
) -> None:
    cmd = [
        python_exe,
        str((m0_repo / "tools" / "run_chunk.py").resolve()),
        "--session_json", str(session_json),
        "--manifest", str(manifest_json),
        "--chunk_len_ms", str(chunk_len_ms),
        "--base_config", str(base_config),
        "--log", str(run_chunk_log),
    ]
    if clean:
        cmd.append("--clean")
    if verify:
        cmd.append("--verify")
    if resume:
        cmd.append("--resume")

    _run(cmd, cwd=m0_repo)


def plan_and_bridge_runtime(
    *,
    m35_repo: Path,
    manifest_json: Path,
    m1_unified_json: Path,
    session_id: str,
    pose_json: Path,
    normal_bg_video: Path,
    mouth_clamped_json: Path,
    expr_json: Path,
    plan_json: Path,
    bridge_out_root: Path,
    m0_repo: Path,
    m0_base_config: str,
    skip_bridge: bool,
) -> Dict[str, Any]:
    sys.path.insert(0, str(m35_repo))
    from m3_5.event_to_plan import build_plan_from_chunks_and_events  # type: ignore
    from m3_5.run_chunk_bridge import run_from_manifest_and_events  # type: ignore

    manifest_obj = _load_json(manifest_json)
    chunks = manifest_obj.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"manifest chunks must be list: {manifest_json}")

    events = _read_events_from_unified(m1_unified_json)

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

    if skip_bridge:
        print("[SKIP] bridge")
        return {
            "plan_chunks": len(plan),
            "end_stream_present": any(ch.get("mode") == "end" for ch in plan),
            "end_stream_chunk_id": next(
                (i for i, ch in enumerate(plan) if ch.get("mode") == "end"),
                None,
            ),
        }

    run_from_manifest_and_events(
        manifest_json=str(manifest_json),
        events=events,
        session_id=str(session_id),
        normal_pose_json=str(pose_json),
        normal_bg_playlist_json=None,
        normal_bg_video=str(normal_bg_video),
        normal_pose_by_bg_video=None,
        normal_mouth_json=str(mouth_clamped_json),
        normal_expr_json=str(expr_json),
        out_root=str(bridge_out_root),
        m0_repo_root=str(m0_repo),
        m3_5_repo_root=str(m35_repo),
        m0_base_config=str(m0_base_config),
        cli_cfg=None,
        use_fg_index=True,
        fg_mode="follow",
        debug_print_plan=True,
        debug_plan_range=None,
    )

    return {
        "plan_chunks": len(plan),
        "end_stream_present": any(ch.get("mode") == "end" for ch in plan),
        "end_stream_chunk_id": next(
            (i for i, ch in enumerate(plan) if ch.get("mode") == "end"),
            None,
        ),
    }


def run_one_utterance(
    *,
    python_exe: str,
    args: argparse.Namespace,
    m1_repo: Path,
    m3_repo: Path,
    m0_repo: Path,
    m35_repo: Path,
    m1_unified_json: Path,
    pose_json: Path,
    audio_path: Path | None,
    normal_bg_video: Path,
    pcm_out_dir: Path,
    m0_in_dir: Path,
    session_json: Path,
    manifest_json: Path,
    plan_json: Path,
    bridge_out_root: Path,
    run_chunk_log: Path,
) -> Dict[str, Any]:
    raw_json = pcm_out_dir / "mouth_timeline.formant.raw.json"
    audio_meta_json = pcm_out_dir / "audio_meta.json"
    expression_chunks_json = pcm_out_dir / "expression_chunks.v1.json"

    m1_end_json = pcm_out_dir / f"{args.session_id}.m1_unified.end.json"
    mouth_json = m0_in_dir / "mouth.json"
    mouth_clamped_json = m0_in_dir / "mouth.clamped.json"
    expr_json = m0_in_dir / "expr.json"

    runtime_unified_json = m1_unified_json

    raw_json, audio_meta_json = run_m2_and_pcm_push(
        python_exe=python_exe,
        m1_repo=m1_repo,
        m3_repo=m3_repo,
        m1_unified_json=m1_unified_json,
        pcm_out_dir=pcm_out_dir,
        session_id=args.session_id,
        utt_id=args.utt_id,
        config=args.config,
        step_ms=args.step_ms,
        analysis_sr=args.analysis_sr,
        vad_energy_thr=args.vad_energy_thr,
        max_buffer_ms=args.max_buffer_ms,
        sleep_consumer_ms=args.sleep_consumer_ms,
    )

    audio_ms = _read_audio_ms(audio_meta_json)
    print("[INFO] audio_ms (SSOT):", audio_ms)

    if not args.skip_add_end_stream:
        _run(
            [
                python_exe,
                str((m1_repo / "scripts" / "add_auto_end_stream.py").resolve()),
                "--in_json", str(m1_unified_json),
                "--out_json", str(m1_end_json),
                "--audio_ms", str(audio_ms),
                "--step_ms", str(args.step_ms),
                "--event_id", str(args.event_id),
                "--trigger_source", str(args.trigger_source),
            ],
            cwd=m1_repo,
        )
        runtime_unified_json = m1_end_json

    events = _read_events_from_unified(runtime_unified_json)
    print("[INFO] end_stream_present:", _has_end_stream_event(events))

    build_mouth_from_raw(
        python_exe=python_exe,
        m3_repo=m3_repo,
        raw_json=raw_json,
        knn_gt_glob=args.knn_gt_glob,
        mouth_json=mouth_json,
    )

    build_clamped_mouth_runtime(
        python_exe=python_exe,
        m3_repo=m3_repo,
        mouth_json=mouth_json,
        mouth_clamped_json=mouth_clamped_json,
        audio_ms=audio_ms,
        step_ms=args.step_ms,
    )

    build_expr_from_unified(
        python_exe=python_exe,
        m1_repo=m1_repo,
        m3_repo=m3_repo,
        m1_unified_json=runtime_unified_json,
        audio_meta_json=audio_meta_json,
        expression_chunks_json=expression_chunks_json,
        expr_json=expr_json,
        session_id=args.session_id,
        step_ms=args.step_ms,
        blink_duration_ms=args.blink_duration_ms,
    )

    build_session_bundle_runtime(
        python_exe=python_exe,
        m1_repo=m1_repo,
        session_id=args.session_id,
        audio_ms=audio_ms,
        mouth_clamped_json=mouth_clamped_json,
        expr_json=expr_json,
        pose_json=pose_json,
        audio_path=audio_path,
        session_json=session_json,
    )

    build_manifest_runtime(
        python_exe=python_exe,
        m0_repo=m0_repo,
        session_id=args.session_id,
        audio_ms=audio_ms,
        chunk_len_ms=args.chunk_len_ms,
        step_ms=args.step_ms,
        fps=args.fps,
        manifest_json=manifest_json,
    )

    run_chunk_runtime(
        python_exe=python_exe,
        m0_repo=m0_repo,
        session_json=session_json,
        manifest_json=manifest_json,
        chunk_len_ms=args.chunk_len_ms,
        run_chunk_log=run_chunk_log,
        base_config=args.m0_base_config,
        clean=not args.no_clean,
        verify=not args.no_verify,
        resume=args.resume,
    )

    bridge_result = plan_and_bridge_runtime(
        m35_repo=m35_repo,
        manifest_json=manifest_json,
        m1_unified_json=runtime_unified_json,
        session_id=args.session_id,
        pose_json=pose_json,
        normal_bg_video=normal_bg_video,
        mouth_clamped_json=mouth_clamped_json,
        expr_json=expr_json,
        plan_json=plan_json,
        bridge_out_root=bridge_out_root,
        m0_repo=m0_repo,
        m0_base_config=args.m0_base_config,
        skip_bridge=args.skip_bridge,
    )

    return {
        "audio_ms": audio_ms,
        "plan_chunks": bridge_result["plan_chunks"],
        "end_stream_present": bridge_result["end_stream_present"],
        "end_stream_chunk_id": bridge_result["end_stream_chunk_id"],
        "raw_json": str(raw_json),
        "mouth_json": str(mouth_json),
        "mouth_clamped_json": str(mouth_clamped_json),
        "expr_json": str(expr_json),
        "session_json": str(session_json),
        "manifest_json": str(manifest_json),
        "plan_json": str(plan_json),
        "bridge_out_root": str(bridge_out_root),
    }

def _load_json_list_file(path: Path) -> List[str]:
    obj = _load_json(path)
    if not isinstance(obj, list):
        raise ValueError(f"json list file must be list: {path}")
    out = []
    for i, v in enumerate(obj):
        if not isinstance(v, str):
            raise ValueError(f"json list file item[{i}] must be str: {path}")
        out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="STEP-F minimal runtime entry skeleton based on current E2E scripts"
    )

    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m0_repo_root", default="/workspaces/M0_session_renderer_final_1")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    ap.add_argument("--session_id", default="sess_sleepy_pcm_9_2_long")
    ap.add_argument("--utt_id", default="utt_sleepy_9_2_long")

    ap.add_argument("--m1_unified_json", required=True)
    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--audio_path", default=None)
    ap.add_argument("--normal_bg_video", required=True)

    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--step_ms", type=int, default=DEFAULT_STEP_MS)
    ap.add_argument("--chunk_len_ms", type=int, default=DEFAULT_CHUNK_LEN_MS)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--analysis_sr", type=int, default=16000)
    ap.add_argument("--vad_energy_thr", type=float, default=0.0004)
    ap.add_argument("--max_buffer_ms", type=int, default=3000)
    ap.add_argument("--sleep_consumer_ms", type=int, default=0)
    ap.add_argument("--blink_duration_ms", type=int, default=DEFAULT_STEP_MS)
    ap.add_argument("--m0_base_config", default="configs/smoke_pose_improved.yaml")
    ap.add_argument("--knn_gt_glob", default="data/knn_db/*.f1f2.json")
    ap.add_argument("--event_id", default=DEFAULT_EVENT_ID)
    ap.add_argument("--trigger_source", default=DEFAULT_TRIGGER_SOURCE)

    ap.add_argument("--pcm_out_dir", default=None)
    ap.add_argument("--m0_in_dir", default=None)
    ap.add_argument("--session_json", default=None)
    ap.add_argument("--manifest_json", default=None)
    ap.add_argument("--plan_json", default=None)
    ap.add_argument("--bridge_out_root", default=None)
    ap.add_argument("--run_chunk_log", default=None)

    ap.add_argument("--skip_add_end_stream", action="store_true")
    ap.add_argument("--skip_bridge", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_clean", action="store_true")
    ap.add_argument("--no_verify", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--repeat_n", type=int, default=1)
    ap.add_argument("--m1_unified_json_list", default=None)

    args = ap.parse_args()

    if args.step_ms != 40:
        raise ValueError("step_ms must be 40")
    if args.chunk_len_ms != 400:
        raise ValueError("chunk_len_ms must be 400")

    python_exe = sys.executable

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    m1_unified_json = Path(args.m1_unified_json).resolve()
    pose_json = Path(args.pose_json).resolve()
    audio_path = Path(args.audio_path).resolve() if args.audio_path else None
    normal_bg_video = Path(args.normal_bg_video).resolve()

    if not m1_unified_json.exists():
        raise FileNotFoundError(f"missing m1_unified_json: {m1_unified_json}")
    if not pose_json.exists():
        raise FileNotFoundError(f"missing pose_json: {pose_json}")

    pcm_out_dir = Path(args.pcm_out_dir).resolve() if args.pcm_out_dir else (
        m1_repo / "out" / "runtime_session" / args.session_id / "pcm"
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

    print("[INFO] session_id      :", args.session_id)
    print("[INFO] utt_id          :", args.utt_id)
    print("[INFO] m1_unified_json :", m1_unified_json)
    print("[INFO] pcm_out_dir     :", pcm_out_dir)
    print("[INFO] m0_in_dir       :", m0_in_dir)
    print("[INFO] session_json    :", session_json)
    print("[INFO] manifest_json   :", manifest_json)
    print("[INFO] plan_json       :", plan_json)
    print("[INFO] bridge_out_root :", bridge_out_root)

    if args.dry_run:
        print("[DRY_RUN] stop before execution")
        return 0

    runtime_summary_json = pcm_out_dir / "runtime_summary.json"

    if args.m1_unified_json_list:
        m1_unified_json_list_path = Path(args.m1_unified_json_list).resolve()
        input_jsons = [Path(p).resolve() for p in _load_json_list_file(m1_unified_json_list_path)]
    else:
        input_jsons = [m1_unified_json for _ in range(args.repeat_n)]

    all_results = []

    for i, loop_m1_unified_json in enumerate(input_jsons):
        print(f"[LOOP] utterance_index={i}")
        print(f"[LOOP] m1_unified_json={loop_m1_unified_json}")

        if not loop_m1_unified_json.exists():
            raise FileNotFoundError(f"missing loop m1_unified_json: {loop_m1_unified_json}")

        loop_pcm_out_dir = pcm_out_dir.parent / f"pcm_{i:03d}"
        loop_m0_in_dir = m0_in_dir.parent / f"{m0_in_dir.name}_{i:03d}"
        loop_session_json = session_json.parent / f"{session_json.stem}_{i:03d}{session_json.suffix}"
        loop_manifest_json = manifest_json.parent / f"{manifest_json.stem}_{i:03d}{manifest_json.suffix}"
        loop_plan_json = plan_json.parent / f"{plan_json.stem}_{i:03d}{plan_json.suffix}"
        loop_bridge_out_root = bridge_out_root.parent / f"{bridge_out_root.name}_{i:03d}"
        loop_run_chunk_log = run_chunk_log.parent / f"{run_chunk_log.stem}_{i:03d}{run_chunk_log.suffix}"

        _ensure_dir(loop_pcm_out_dir)
        _ensure_dir(loop_m0_in_dir)
        _ensure_dir(loop_bridge_out_root)

        result = run_one_utterance(
            python_exe=python_exe,
            args=args,
            m1_repo=m1_repo,
            m3_repo=m3_repo,
            m0_repo=m0_repo,
            m35_repo=m35_repo,
            m1_unified_json=loop_m1_unified_json,
            pose_json=pose_json,
            audio_path=audio_path,
            normal_bg_video=normal_bg_video,
            pcm_out_dir=loop_pcm_out_dir,
            m0_in_dir=loop_m0_in_dir,
            session_json=loop_session_json,
            manifest_json=loop_manifest_json,
            plan_json=loop_plan_json,
            bridge_out_root=loop_bridge_out_root,
            run_chunk_log=loop_run_chunk_log,
        )

        result["utterance_index"] = i
        result["m1_unified_json"] = str(loop_m1_unified_json)
        all_results.append(result)

    summary = {
        "session_id": args.session_id,
        "utt_id": args.utt_id,
        "repeat_n": args.repeat_n,
        "m1_unified_json_list": args.m1_unified_json_list,
        "results": all_results,
    }
    _dump_json(runtime_summary_json, summary)
    print("  runtime_summary_json:", runtime_summary_json)

    result = all_results[-1]

    print("[run_runtime_session][OK]")
    print("  audio_ms           :", result["audio_ms"])
    print("  raw_json           :", result["raw_json"])
    print("  mouth_json         :", result["mouth_json"])
    print("  mouth_clamped_json :", result["mouth_clamped_json"])
    print("  expr_json          :", result["expr_json"])
    print("  session_json       :", result["session_json"])
    print("  manifest_json      :", result["manifest_json"])
    print("  plan_json          :", result["plan_json"])
    print("  bridge_out_root    :", result["bridge_out_root"])
    print("  plan_chunks        :", result["plan_chunks"])
    print("  end_stream_present :", result["end_stream_present"])
    print("  end_stream_chunk_id:", result["end_stream_chunk_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())