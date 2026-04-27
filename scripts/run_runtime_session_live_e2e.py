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


def _run(cmd: List[str], *, cwd: Path | None = None, env: Dict[str, str] | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)


def _spawn(cmd: List[str], *, cwd: Path | None = None, env: Dict[str, str] | None = None) -> subprocess.Popen:
    print("[SPAWN]", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_runtime_env(m1_repo: Path, m3_repo: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{m1_repo / 'src'}:{m3_repo / 'src'}:" + env.get("PYTHONPATH", "")
    return env


def _read_audio_meta(audio_meta_json: Path) -> Dict[str, Any]:
    obj = _load_json(audio_meta_json)
    if not isinstance(obj, dict):
        raise ValueError(f"audio_meta must be object: {audio_meta_json}")
    return obj


def _read_audio_ms(audio_meta_json: Path) -> int:
    obj = _read_audio_meta(audio_meta_json)
    return int(obj["audio_ms"])


def _read_expr_events(expr_json: Path) -> List[Dict[str, Any]]:
    obj = _load_json(expr_json)
    if not isinstance(obj, list):
        raise ValueError(f"expression json must be list: {expr_json}")
    return [x for x in obj if isinstance(x, dict)]


def spawn_live_mouth(
    *,
    python_exe: str,
    m1_repo: Path,
    m3_repo: Path,
    out_dir: Path,
    session_id: str,
    model: str,
    prompt: str,
    dev_stop_s: float,
    target_audio_ms: int,
    api_version: str,
) -> tuple[subprocess.Popen, Dict[str, Path]]:
    env = _build_runtime_env(m1_repo, m3_repo)

    cmd = [
        python_exe,
        str((m1_repo / "scripts" / "dev_live_to_m3_raw.py").resolve()),
        "--session_id", str(session_id),
        "--out_dir", str(out_dir),
        "--model", str(model),
        "--prompt", str(prompt),
        "--dev_stop_s", str(dev_stop_s),
        "--target_audio_ms", str(target_audio_ms),
        "--api_version", str(api_version),
    ]
    proc = _spawn(cmd, cwd=m1_repo, env=env)

    paths = {
        "streamer_json": out_dir / "mouth_timeline.live.json",
        "raw_json": out_dir / "mouth_timeline.formant.raw.json",
        "audio_meta_json": out_dir / "audio_meta.json",
        "tool_calls_json": out_dir / "tool_calls.live.json",
        "text_chunks_json": out_dir / "text_chunks.live.json",
        "events_jsonl": out_dir / "events.live.jsonl",
        "profile_json": out_dir / "profile.live.json",
    }
    return proc, paths


def spawn_live_expression(
    *,
    python_exe: str,
    m1_repo: Path,
    m3_repo: Path,
    out_dir: Path,
    session_id: str,
    model: str,
    prompt: str,
    dev_stop_s: float,
    stop_after_first_tool_s: float,
    api_version: str,
) -> tuple[subprocess.Popen, Dict[str, Path]]:
    env = _build_runtime_env(m1_repo, m3_repo)

    cmd = [
        python_exe,
        str((m1_repo / "scripts" / "dev_live_to_expression_json.py").resolve()),
        "--session_id", str(session_id),
        "--out_dir", str(out_dir),
        "--model", str(model),
        "--prompt", str(prompt),
        "--dev_stop_s", str(dev_stop_s),
        "--stop_after_first_tool_s", str(stop_after_first_tool_s),
        "--api_version", str(api_version),
    ]
    proc = _spawn(cmd, cwd=m1_repo, env=env)

    paths = {
        "expr_json": out_dir / "expression_timeline.live.json",
        "tool_calls_json": out_dir / "tool_calls.live.json",
        "text_chunks_json": out_dir / "text_chunks.live.json",
        "events_jsonl": out_dir / "events.live.jsonl",
        "profile_json": out_dir / "profile.live.json",
    }
    return proc, paths


def _assert_paths_exist(paths: Dict[str, Path], *, label: str) -> None:
    missing = [f"{k}: {v}" for k, v in paths.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"missing {label} outputs:\n" + "\n".join(missing))


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


def build_expr_from_live_events(
    *,
    live_expr_json: Path,
    session_id: str,
    expr_json: Path,
    step_ms: int,
) -> None:
    """
    live expression event list -> M0 が読む session_expression_timeline_v0.1 形式へ包装。
    auto_blink はまだ入れない。まずは主感情イベントで通す。
    """
    events = _read_expr_events(live_expr_json)

    out_obj = {
        "schema_version": "session_expression_timeline_v0.1",
        "session_id": session_id,
        "step_ms": int(step_ms),
        "timeline": events,
        "meta": {
            "source": str(live_expr_json),
            "source_type": "live_expression_events",
            "events_n": len(events),
            "auto_blink": False,
        },
    }
    _dump_json(expr_json, out_obj)


def write_fallback_expr_normal(
    *,
    session_id: str,
    expr_json: Path,
    step_ms: int,
) -> None:
    _dump_json(
        expr_json,
        {
            "schema_version": "session_expression_timeline_v0.1",
            "session_id": str(session_id),
            "step_ms": int(step_ms),
            "timeline": [
                {"t_ms": 0, "expression": "normal", "source": "init"}
            ],
            "meta": {
                "source_type": "fallback_normal_only",
                "events_n": 1,
                "auto_blink": False,
            },
        },
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


def plan_and_bridge_runtime_no_events(
    *,
    m35_repo: Path,
    manifest_json: Path,
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

    events: List[Dict[str, Any]] = []

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
            "end_stream_present": False,
            "end_stream_chunk_id": None,
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
        "end_stream_present": False,
        "end_stream_chunk_id": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Minimal live runtime E2E: Live mouth + Live expression -> M0/M3.5"
    )

    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m0_repo_root", default="/workspaces/M0_session_renderer_final_1")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    ap.add_argument("--session_id", default="sess_live_runtime_001")
    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--normal_bg_video", required=True)
    ap.add_argument("--audio_path", default=None)

    ap.add_argument("--step_ms", type=int, default=DEFAULT_STEP_MS)
    ap.add_argument("--chunk_len_ms", type=int, default=DEFAULT_CHUNK_LEN_MS)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--blink_duration_ms", type=int, default=DEFAULT_STEP_MS)
    ap.add_argument("--m0_base_config", default="configs/smoke_pose_improved.yaml")
    ap.add_argument("--knn_gt_glob", default="data/knn_db/*.f1f2.json")

    ap.add_argument("--mouth_model", default="gemini-2.5-flash-native-audio-preview-12-2025")
    ap.add_argument("--mouth_prompt", default="元気よく自己紹介して！短めで。")
    ap.add_argument("--mouth_dev_stop_s", type=float, default=8.0)
    ap.add_argument("--target_audio_ms", type=int, default=0)

    ap.add_argument("--expr_model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--expr_prompt", default="短く自己紹介して。最初に必ず set_emotion を呼び、感情は smile にしてください。")
    ap.add_argument("--expr_dev_stop_s", type=float, default=12.0)
    ap.add_argument("--stop_after_first_tool_s", type=float, default=3.0)

    ap.add_argument("--api_version", default="v1alpha")

    ap.add_argument("--live_root", default=None)
    ap.add_argument("--m0_in_dir", default=None)
    ap.add_argument("--session_json", default=None)
    ap.add_argument("--manifest_json", default=None)
    ap.add_argument("--plan_json", default=None)
    ap.add_argument("--bridge_out_root", default=None)
    ap.add_argument("--run_chunk_log", default=None)

    ap.add_argument("--skip_expression", action="store_true")
    ap.add_argument("--skip_bridge", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_clean", action="store_true")
    ap.add_argument("--no_verify", action="store_true")

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

    pose_json = Path(args.pose_json).resolve()
    normal_bg_video = Path(args.normal_bg_video).resolve()
    audio_path = Path(args.audio_path).resolve() if args.audio_path else None

    live_root = Path(args.live_root).resolve() if args.live_root else (m1_repo / "out" / "live_runtime_e2e" / args.session_id)
    mouth_live_dir = live_root / "mouth_live"
    expr_live_dir = live_root / "expr_live"
    m0_in_dir = Path(args.m0_in_dir).resolve() if args.m0_in_dir else (live_root / "m0_in")
    session_json = Path(args.session_json).resolve() if args.session_json else (live_root / "sessions" / f"{args.session_id}.session.json")
    manifest_json = Path(args.manifest_json).resolve() if args.manifest_json else (live_root / "manifests" / f"{args.session_id}.chunk400.json")
    plan_json = Path(args.plan_json).resolve() if args.plan_json else (live_root / "plans" / f"{args.session_id}.plan.json")
    bridge_out_root = Path(args.bridge_out_root).resolve() if args.bridge_out_root else (live_root / "bridge_out")
    run_chunk_log = Path(args.run_chunk_log).resolve() if args.run_chunk_log else (live_root / "logs" / f"{args.session_id}.run_chunk.log.json")

    _ensure_dir(live_root)
    _ensure_dir(mouth_live_dir)
    _ensure_dir(expr_live_dir)
    _ensure_dir(m0_in_dir)
    _ensure_dir(session_json.parent)
    _ensure_dir(manifest_json.parent)
    _ensure_dir(plan_json.parent)
    _ensure_dir(bridge_out_root)
    _ensure_dir(run_chunk_log.parent)

    # ------------------------------------------------------------
    # 1) Live mouth / expression を並列起動
    # ------------------------------------------------------------
    mouth_proc, mouth_paths = spawn_live_mouth(
        python_exe=python_exe,
        m1_repo=m1_repo,
        m3_repo=m3_repo,
        out_dir=mouth_live_dir,
        session_id=str(args.session_id),
        model=str(args.mouth_model),
        prompt=str(args.mouth_prompt),
        dev_stop_s=float(args.mouth_dev_stop_s),
        target_audio_ms=int(args.target_audio_ms),
        api_version=str(args.api_version),
    )

    expr_proc = None
    expr_paths: Dict[str, Path] | None = None
    if not args.skip_expression:
        expr_proc, expr_paths = spawn_live_expression(
            python_exe=python_exe,
            m1_repo=m1_repo,
            m3_repo=m3_repo,
            out_dir=expr_live_dir,
            session_id=str(args.session_id),
            model=str(args.expr_model),
            prompt=str(args.expr_prompt),
            dev_stop_s=float(args.expr_dev_stop_s),
            stop_after_first_tool_s=float(args.stop_after_first_tool_s),
            api_version=str(args.api_version),
        )

    mouth_rc = mouth_proc.wait()
    if mouth_rc != 0:
        raise RuntimeError(f"mouth process failed rc={mouth_rc}")

    if expr_proc is not None:
        expr_rc = expr_proc.wait()
        if expr_rc != 0:
            raise RuntimeError(f"expression process failed rc={expr_rc}")

    _assert_paths_exist(mouth_paths, label="mouth")
    if expr_paths is not None:
        _assert_paths_exist(expr_paths, label="expression")

    # ------------------------------------------------------------
    # 2) audio_ms を SSOT として後段生成
    # ------------------------------------------------------------
    audio_ms = _read_audio_ms(mouth_paths["audio_meta_json"])
    print("[INFO] audio_ms (SSOT):", audio_ms)

    if audio_ms <= 0:
        raise RuntimeError(
            f"Live mouth produced invalid audio_ms={audio_ms}. "
            f"Check mouth profile: {mouth_paths['profile_json']}"
        )

    raw_json = mouth_paths["raw_json"]
    mouth_json = m0_in_dir / "mouth.json"
    mouth_clamped_json = m0_in_dir / "mouth.clamped.json"
    expr_json = m0_in_dir / "expr.json"

    build_mouth_from_raw(
        python_exe=python_exe,
        m3_repo=m3_repo,
        raw_json=raw_json,
        knn_gt_glob=str(args.knn_gt_glob),
        mouth_json=mouth_json,
    )

    build_clamped_mouth_runtime(
        python_exe=python_exe,
        m3_repo=m3_repo,
        mouth_json=mouth_json,
        mouth_clamped_json=mouth_clamped_json,
        audio_ms=audio_ms,
        step_ms=int(args.step_ms),
    )

    if expr_paths is not None:
        build_expr_from_live_events(
            live_expr_json=expr_paths["expr_json"],
            session_id=str(args.session_id),
            expr_json=expr_json,
            step_ms=int(args.step_ms),
        )
    else:
        write_fallback_expr_normal(
            session_id=str(args.session_id),
            expr_json=expr_json,
            step_ms=int(args.step_ms),
        )

    build_session_bundle_runtime(
        python_exe=python_exe,
        m1_repo=m1_repo,
        session_id=str(args.session_id),
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
        session_id=str(args.session_id),
        audio_ms=audio_ms,
        chunk_len_ms=int(args.chunk_len_ms),
        step_ms=int(args.step_ms),
        fps=int(args.fps),
        manifest_json=manifest_json,
    )

    #run_chunk_runtime(
        #python_exe=python_exe,
        #m0_repo=m0_repo,
        #session_json=session_json,
        #manifest_json=manifest_json,
        #chunk_len_ms=int(args.chunk_len_ms),
        #run_chunk_log=run_chunk_log,
        #base_config=str(args.m0_base_config),
        #clean=not bool(args.no_clean),
        #verify=not bool(args.no_verify),
        #resume=bool(args.resume),
    #)

    bridge_result = plan_and_bridge_runtime_no_events(
        m35_repo=m35_repo,
        manifest_json=manifest_json,
        session_id=str(args.session_id),
        pose_json=pose_json,
        normal_bg_video=normal_bg_video,
        mouth_clamped_json=mouth_clamped_json,
        expr_json=expr_json,
        plan_json=plan_json,
        bridge_out_root=bridge_out_root,
        m0_repo=m0_repo,
        m0_base_config=str(args.m0_base_config),
        skip_bridge=bool(args.skip_bridge),
    )

    summary = {
        "format": "runtime_session_live_e2e_summary.v0.1",
        "session_id": str(args.session_id),
        "audio_ms": int(audio_ms),
        "mouth_raw_json": str(raw_json),
        "mouth_json": str(mouth_json),
        "mouth_clamped_json": str(mouth_clamped_json),
        "expr_json": str(expr_json),
        "session_json": str(session_json),
        "manifest_json": str(manifest_json),
        "plan_json": str(plan_json),
        "bridge_out_root": str(bridge_out_root),
        "plan_chunks": bridge_result["plan_chunks"],
        "end_stream_present": bridge_result["end_stream_present"],
        "end_stream_chunk_id": bridge_result["end_stream_chunk_id"],
    }
    summary_json = live_root / "runtime_session_live_e2e_summary.json"
    _dump_json(summary_json, summary)

    print("[run_runtime_session_live_e2e][OK]")
    print("  audio_ms         :", audio_ms)
    print("  mouth_raw_json   :", raw_json)
    print("  mouth_json       :", mouth_json)
    print("  mouth_clamped    :", mouth_clamped_json)
    print("  expr_json        :", expr_json)
    print("  session_json     :", session_json)
    print("  manifest_json    :", manifest_json)
    print("  plan_json        :", plan_json)
    print("  bridge_out_root  :", bridge_out_root)
    print("  plan_chunks      :", bridge_result["plan_chunks"])
    print("  summary_json     :", summary_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())