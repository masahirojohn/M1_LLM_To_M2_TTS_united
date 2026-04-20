#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print()
    print("[RUN]", " ".join(cmd))
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(p.stdout, end="")
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run M1 stub -> expression_chunks -> expr.json in one shot."
    )
    ap.add_argument(
        "--m1_repo_root",
        default=".",
        help="M1 repo root (default: current dir)",
    )
    ap.add_argument(
        "--m3_repo_root",
        default="/workspaces/M3_Live_API_1_united",
        help="M3 repo root",
    )

    ap.add_argument(
        "--config",
        default="configs/default.yaml",
        help="M1 config path (repo-relative from m1_repo_root)",
    )
    ap.add_argument(
        "--comment_source",
        default="dummy_json",
        choices=["dummy_json", "onecomme_json"],
        help="comment source",
    )
    ap.add_argument(
        "--comments_json",
        default="examples/dummy_comments.json",
        help="comments json path (repo-relative from m1_repo_root)",
    )
    ap.add_argument(
        "--session_id",
        default="sess_real_01",
    )
    ap.add_argument(
        "--utt_id",
        default="utt_0001",
    )

    ap.add_argument(
        "--out_unified_json",
        default="out/dev_m1_stub/m1_unified_output.json",
        help="M1 unified output path (repo-relative from m1_repo_root)",
    )
    ap.add_argument(
        "--dump_prompt_json",
        default="out/dev_m1_stub/prompt_payload.json",
        help="prompt payload path (repo-relative from m1_repo_root)",
    )
    ap.add_argument(
        "--audio_meta_json",
        default="out/dev_expr/audio_meta_long.json",
        help="audio_meta path (repo-relative from m1_repo_root)",
    )
    ap.add_argument(
        "--emo_map_yaml",
        default="configs/emo_id_to_expression.yaml",
        help="emo_id -> expression map yaml (repo-relative from m1_repo_root)",
    )
    ap.add_argument(
        "--out_chunks_json",
        default="out/dev_expr/expression_chunks.long.v1.json",
        help="expression_chunks output path (repo-relative from m1_repo_root)",
    )
    ap.add_argument(
        "--out_expr_json",
        default="out/dev_expr/expr.long.json",
        help="expr output path (repo-relative from m1_repo_root)",
    )

    ap.add_argument(
        "--auto_blink",
        action="store_true",
        help="enable auto blink when building expr.json",
    )
    ap.add_argument(
        "--max_comments",
        type=int,
        default=5,
        help="max comments for dev_run_m1_stub.py",
    )
    args = ap.parse_args()

    m1_repo_root = Path(args.m1_repo_root).resolve()
    m3_repo_root = Path(args.m3_repo_root).resolve()

    if not m1_repo_root.exists():
        raise SystemExit(f"[ERROR] m1_repo_root not found: {m1_repo_root}")
    if not m3_repo_root.exists():
        raise SystemExit(f"[ERROR] m3_repo_root not found: {m3_repo_root}")

    dev_run_m1_stub = m1_repo_root / "scripts" / "dev_run_m1_stub.py"
    build_expr_chunks = m1_repo_root / "scripts" / "build_expression_chunks_from_m1.py"
    build_session_expr = m3_repo_root / "scripts" / "build_session_expression_timeline_from_chunks.py"

    for p in [dev_run_m1_stub, build_expr_chunks, build_session_expr]:
        if not p.exists():
            raise SystemExit(f"[ERROR] required script not found: {p}")

    out_unified_abs = (m1_repo_root / args.out_unified_json).resolve()
    out_chunks_abs = (m1_repo_root / args.out_chunks_json).resolve()
    out_expr_abs = (m1_repo_root / args.out_expr_json).resolve()

    run_cmd(
        [
            sys.executable,
            "scripts/dev_run_m1_stub.py",
            "--config",
            args.config,
            "--comment_source",
            args.comment_source,
            "--comments_json",
            args.comments_json,
            "--max_comments",
            str(args.max_comments),
            "--session_id",
            args.session_id,
            "--utt_id",
            args.utt_id,
            "--out_json",
            args.out_unified_json,
            "--dump_prompt_json",
            args.dump_prompt_json,
        ],
        cwd=m1_repo_root,
    )

    if not out_unified_abs.exists():
        raise SystemExit(f"[ERROR] unified output not created: {out_unified_abs}")

    run_cmd(
        [
            sys.executable,
            "scripts/build_expression_chunks_from_m1.py",
            "--m1_unified_json",
            args.out_unified_json,
            "--audio_meta_json",
            args.audio_meta_json,
            "--out",
            args.out_chunks_json,
            "--session_id",
            args.session_id,
            "--emo_map_yaml",
            args.emo_map_yaml,
        ],
        cwd=m1_repo_root,
    )

    if not out_chunks_abs.exists():
        raise SystemExit(f"[ERROR] expression_chunks not created: {out_chunks_abs}")

    cmd = [
        sys.executable,
        "scripts/build_session_expression_timeline_from_chunks.py",
        "--in_chunks",
        str(out_chunks_abs),
        "--out",
        str(out_expr_abs),
        "--session_id",
        args.session_id,
    ]
    if args.auto_blink:
        cmd.append("--auto_blink")

    run_cmd(cmd, cwd=m3_repo_root)

    if not out_expr_abs.exists():
        raise SystemExit(f"[ERROR] expr output not created: {out_expr_abs}")

    print()
    print("[dev_run_m1_to_expr][OK]")
    print(" unified_json :", out_unified_abs.as_posix())
    print(" chunks_json  :", out_chunks_abs.as_posix())
    print(" expr_json    :", out_expr_abs.as_posix())
    print(" auto_blink   :", bool(args.auto_blink))


if __name__ == "__main__":
    main()
