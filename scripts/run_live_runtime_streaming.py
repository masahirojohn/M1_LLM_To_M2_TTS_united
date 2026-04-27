#!/usr/bin/env python3
from __future__ import annotations

"""
Pseudo streaming v0.

目的:
- 既存合格済み Live E2E 経路を壊さず、
  400ms chunk 単位の streaming 風実行入口を作る。
- v0 では Live API から target_audio_ms 分を先に取得し、
  その後 manifest の chunk を順に M3.5 bridge へ流す。
- 将来は mouth worker / expression worker 常時化に差し替える。
"""

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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_env(m1_repo: Path, m3_repo: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{m1_repo / 'src'}:{m3_repo / 'src'}:" + env.get("PYTHONPATH", "")
    return env


def _read_audio_ms(audio_meta_json: Path) -> int:
    obj = _load_json(audio_meta_json)
    return int(obj["audio_ms"])


def _run_live_e2e_fast(
    *,
    python_exe: str,
    m1_repo: Path,
    m3_repo: Path,
    m0_repo: Path,
    m35_repo: Path,
    session_id: str,
    pose_json: Path,
    normal_bg_video: Path,
    live_root: Path,
    mouth_model: str,
    mouth_prompt: str,
    mouth_dev_stop_s: float,
    target_audio_ms: int,
    expr_model: str,
    expr_prompt: str,
    expr_dev_stop_s: float,
    stop_after_first_tool_s: float,
    api_version: str,
    m0_base_config: str,
    knn_gt_glob: str,
) -> Path:
    """
    既存合格済み run_runtime_session_live_e2e.py を --skip_bridge で使い、
    mouth/expr/session/manifest/plan まで作る。
    """
    cmd = [
        python_exe,
        str((m1_repo / "scripts" / "run_runtime_session_live_e2e.py").resolve()),
        "--m1_repo_root", str(m1_repo),
        "--m3_repo_root", str(m3_repo),
        "--m0_repo_root", str(m0_repo),
        "--m35_repo_root", str(m35_repo),
        "--session_id", session_id,
        "--pose_json", str(pose_json),
        "--normal_bg_video", str(normal_bg_video),
        "--mouth_model", mouth_model,
        "--mouth_prompt", mouth_prompt,
        "--mouth_dev_stop_s", str(mouth_dev_stop_s),
        "--target_audio_ms", str(target_audio_ms),
        "--expr_model", expr_model,
        "--expr_prompt", expr_prompt,
        "--expr_dev_stop_s", str(expr_dev_stop_s),
        "--stop_after_first_tool_s", str(stop_after_first_tool_s),
        "--api_version", api_version,
        "--m0_base_config", m0_base_config,
        "--knn_gt_glob", knn_gt_glob,
        "--live_root", str(live_root),
        "--skip_bridge",
    ]

    _run(cmd, cwd=m1_repo, env=_build_env(m1_repo, m3_repo))

    summary_json = live_root / "runtime_session_live_e2e_summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"missing summary_json: {summary_json}")

    return summary_json


def _build_plan_from_manifest(
    *,
    m35_repo: Path,
    manifest_json: Path,
    pose_json: Path,
    normal_bg_video: Path,
    plan_json: Path,
) -> list[Dict[str, Any]]:
    sys.path.insert(0, str(m35_repo))
    from m3_5.event_to_plan import build_plan_from_chunks_and_events  # type: ignore

    manifest_obj = _load_json(manifest_json)
    chunks = manifest_obj.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"manifest chunks must be list: {manifest_json}")

    events: list[Dict[str, Any]] = []

    plan = build_plan_from_chunks_and_events(
        chunks,
        events,
        normal_pose_json=str(pose_json),
        normal_bg_playlist_json=None,
        normal_bg_video=str(normal_bg_video),
        normal_pose_by_bg_video=None,
    )
    _dump_json(plan_json, plan)
    return plan


def _run_plan_chunk_by_chunk(
    *,
    m35_repo: Path,
    m0_repo: Path,
    session_id: str,
    plan: list[Dict[str, Any]],
    mouth_clamped_json: Path,
    expr_json: Path,
    normal_bg_video: Path,
    out_root: Path,
    m0_base_config: str,
    max_chunks: int | None,
) -> int:
    """
    疑似streaming:
    plan を全件一括 run_all_chunks せず、1 chunk ずつ bridge.run_chunk へ流す。
    """
    sys.path.insert(0, str(m35_repo))
    from m3_5.run_chunk_bridge import run_chunk  # type: ignore

    ran = 0
    for i, chunk in enumerate(plan):
        if max_chunks is not None and ran >= max_chunks:
            print(f"[streaming_v0] max_chunks reached: {max_chunks}")
            break

        mode = chunk.get("mode")
        if mode == "end":
            print(
                f"[streaming_v0] end_stream detected at chunk {i} "
                f"event_id={chunk.get('event_id')} -> stopping"
            )
            break

        print(
            f"[streaming_v0] chunk={i:06d} "
            f"t=[{chunk['chunk_start_ms']},{chunk['chunk_end_ms']}) "
            f"mode={mode}"
        )

        run_chunk(
            chunk_id=i,
            chunk_start_ms=int(chunk["chunk_start_ms"]),
            chunk_end_ms=int(chunk["chunk_end_ms"]),
            pose_json=str(chunk["pose_json"]),
            mode=str(chunk["mode"]),
            bg_playlist_json=chunk.get("bg_playlist_json"),
            bg_video=chunk.get("bg_video"),
            bg_start_ms=chunk.get("bg_start_ms"),
            session_id=str(session_id),
            normal_mouth_json=str(mouth_clamped_json),
            normal_expr_json=str(expr_json),
            out_root=str(out_root),
            m0_repo_root=str(m0_repo),
            m3_5_repo_root=str(m35_repo),
            m0_base_config=str(m0_base_config),
            normal_bg_video=str(normal_bg_video),
            cli_cfg=None,
            use_fg_index=True,
            fg_mode="follow",
        )
        ran += 1

    return ran


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pseudo streaming v0: Live API -> prepared timelines -> chunk-by-chunk bridge"
    )

    ap.add_argument("--m1_repo_root", default="/workspaces/M1_LLM_To_M2_TTS_united")
    ap.add_argument("--m3_repo_root", default="/workspaces/M3_Live_API_1_united")
    ap.add_argument("--m0_repo_root", default="/workspaces/M0_session_renderer_final_1")
    ap.add_argument("--m35_repo_root", default="/workspaces/M3.5_final")

    ap.add_argument("--session_id", default="sess_live_streaming_v0_001")
    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--normal_bg_video", required=True)

    ap.add_argument("--step_ms", type=int, default=DEFAULT_STEP_MS)
    ap.add_argument("--chunk_len_ms", type=int, default=DEFAULT_CHUNK_LEN_MS)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)

    ap.add_argument("--mouth_model", default="gemini-2.5-flash-native-audio-preview-12-2025")
    ap.add_argument("--mouth_prompt", default="元気よく自己紹介して！短めで。")
    ap.add_argument("--mouth_dev_stop_s", type=float, default=20.0)
    ap.add_argument("--target_audio_ms", type=int, default=3000)

    ap.add_argument("--expr_model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--expr_prompt", default="短く自己紹介して。最初に必ず set_emotion を呼び、感情は smile にしてください。")
    ap.add_argument("--expr_dev_stop_s", type=float, default=12.0)
    ap.add_argument("--stop_after_first_tool_s", type=float, default=3.0)

    ap.add_argument("--api_version", default="v1alpha")
    ap.add_argument("--m0_base_config", default="configs/smoke_pose_improved.yaml")
    ap.add_argument("--knn_gt_glob", default="data/knn_db/*.f1f2.json")

    ap.add_argument("--live_root", default=None)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--max_chunks", type=int, default=None)

    args = ap.parse_args()

    if int(args.step_ms) != 40:
        raise ValueError("step_ms must be 40")
    if int(args.chunk_len_ms) != 400:
        raise ValueError("chunk_len_ms must be 400")
    if int(args.fps) != 25:
        raise ValueError("fps must be 25")

    python_exe = sys.executable

    m1_repo = Path(args.m1_repo_root).resolve()
    m3_repo = Path(args.m3_repo_root).resolve()
    m0_repo = Path(args.m0_repo_root).resolve()
    m35_repo = Path(args.m35_repo_root).resolve()

    pose_json = Path(args.pose_json).resolve()
    normal_bg_video = Path(args.normal_bg_video).resolve()

    live_root = (
        Path(args.live_root).resolve()
        if args.live_root
        else (m1_repo / "out" / "live_runtime_streaming" / args.session_id)
    )
    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else (live_root / "bridge_streaming_out")
    )

    _ensure_dir(live_root)
    _ensure_dir(out_root)

    print("[streaming_v0] prepare timelines via live_e2e --skip_bridge")

    summary_json = _run_live_e2e_fast(
        python_exe=python_exe,
        m1_repo=m1_repo,
        m3_repo=m3_repo,
        m0_repo=m0_repo,
        m35_repo=m35_repo,
        session_id=str(args.session_id),
        pose_json=pose_json,
        normal_bg_video=normal_bg_video,
        live_root=live_root,
        mouth_model=str(args.mouth_model),
        mouth_prompt=str(args.mouth_prompt),
        mouth_dev_stop_s=float(args.mouth_dev_stop_s),
        target_audio_ms=int(args.target_audio_ms),
        expr_model=str(args.expr_model),
        expr_prompt=str(args.expr_prompt),
        expr_dev_stop_s=float(args.expr_dev_stop_s),
        stop_after_first_tool_s=float(args.stop_after_first_tool_s),
        api_version=str(args.api_version),
        m0_base_config=str(args.m0_base_config),
        knn_gt_glob=str(args.knn_gt_glob),
    )

    summary = _load_json(summary_json)

    audio_ms = int(summary["audio_ms"])
    if audio_ms <= 0:
        raise RuntimeError(f"invalid audio_ms={audio_ms}: {summary_json}")

    manifest_json = Path(summary["manifest_json"]).resolve()
    mouth_clamped_json = Path(summary["mouth_clamped_json"]).resolve()
    expr_json = Path(summary["expr_json"]).resolve()
    plan_json = live_root / "plans" / f"{args.session_id}.streaming_v0.plan.json"

    print("[streaming_v0] audio_ms:", audio_ms)
    print("[streaming_v0] manifest_json:", manifest_json)
    print("[streaming_v0] mouth_clamped_json:", mouth_clamped_json)
    print("[streaming_v0] expr_json:", expr_json)

    plan = _build_plan_from_manifest(
        m35_repo=m35_repo,
        manifest_json=manifest_json,
        pose_json=pose_json,
        normal_bg_video=normal_bg_video,
        plan_json=plan_json,
    )

    print("[streaming_v0] plan_chunks:", len(plan))
    print("[streaming_v0] run chunk-by-chunk bridge")

    ran_chunks = _run_plan_chunk_by_chunk(
        m35_repo=m35_repo,
        m0_repo=m0_repo,
        session_id=str(args.session_id),
        plan=plan,
        mouth_clamped_json=mouth_clamped_json,
        expr_json=expr_json,
        normal_bg_video=normal_bg_video,
        out_root=out_root,
        m0_base_config=str(args.m0_base_config),
        max_chunks=args.max_chunks,
    )

    streaming_summary = {
        "format": "live_runtime_streaming_v0_summary.v0.1",
        "session_id": str(args.session_id),
        "audio_ms": int(audio_ms),
        "target_audio_ms": int(args.target_audio_ms),
        "plan_json": str(plan_json),
        "plan_chunks": int(len(plan)),
        "ran_chunks": int(ran_chunks),
        "summary_from_live_e2e": str(summary_json),
        "manifest_json": str(manifest_json),
        "mouth_clamped_json": str(mouth_clamped_json),
        "expr_json": str(expr_json),
        "out_root": str(out_root),
    }
    streaming_summary_json = live_root / "live_runtime_streaming_v0_summary.json"
    _dump_json(streaming_summary_json, streaming_summary)

    print("[run_live_runtime_streaming][OK]")
    print("  audio_ms              :", audio_ms)
    print("  target_audio_ms       :", int(args.target_audio_ms))
    print("  plan_chunks           :", len(plan))
    print("  ran_chunks            :", ran_chunks)
    print("  plan_json             :", plan_json)
    print("  out_root              :", out_root)
    print("  streaming_summary_json:", streaming_summary_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())