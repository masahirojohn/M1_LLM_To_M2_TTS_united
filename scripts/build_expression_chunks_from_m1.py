#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_emo_id_to_expression_map(path: Path) -> Dict[str, str]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML が必要です。未導入なら `pip install pyyaml` を実行してください。"
        ) from e

    if not path.exists():
        raise FileNotFoundError(f"emo_id_to_expression yaml not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"emo_id_to_expression yaml root must be dict: {path}")

    out: Dict[str, str] = {}
    for k, v in raw.items():
        out[str(k)] = str(v)
    return out


def ceil_to_grid(ms: int, step_ms: int) -> int:
    return int(math.ceil(int(ms) / int(step_ms)) * int(step_ms))


def resolve_emo_id(speech: Dict[str, Any]) -> str:
    """
    新仕様:
    - speech.emo_id を最優先
    旧仕様 fallback:
    - speech.emotion_hint から仮変換
    """
    emo_id = speech.get("emo_id")
    if isinstance(emo_id, str) and emo_id.strip():
        return emo_id.strip()

    emotion_hint = str(speech.get("emotion_hint", "normal")).strip()

    hint_to_emo = {
        "normal": "1_0",
        "happy": "1_1",
        "excited": "2_1",
        "sleepy": "9_1",
        "very_sleepy": "9_2",
        "surprised": "3_1",
        "sad": "7_1",
        "angry": "5_0",
    }

    return hint_to_emo.get(emotion_hint, "1_0")


def emo_id_to_expression(emo_id: str, mapping: Dict[str, str]) -> str:
    return mapping.get(str(emo_id), "normal")


def build_chunk_events_from_m1(
    *,
    m1_unified: Dict[str, Any],
    audio_ms: int,
    step_ms: int,
    emo_map: Dict[str, str],
    default_source: str = "m1_emo",
) -> List[Dict[str, Any]]:
    speech = m1_unified.get("speech", {}) or {}

    emo_id = resolve_emo_id(speech)
    emotion_hint = str(speech.get("emotion_hint", "normal")).strip()
    expression = emo_id_to_expression(emo_id, emo_map)

    events: List[Dict[str, Any]] = [
        {
            "t_ms": 0,
            "expression": expression,
            "source": default_source,
            "emo_id": emo_id,
            "emotion_hint": emotion_hint,
        }
    ]

    return events


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build expression_chunks.v1.json from M1 unified output + audio_meta."
    )
    ap.add_argument("--m1_unified_json", required=True, help="path to m1_unified_output.json")
    ap.add_argument("--audio_meta_json", required=True, help="path to audio_meta.json")
    ap.add_argument("--out", required=True, help="output expression_chunks.v1.json")
    ap.add_argument("--session_id", default="", help="override session_id if needed")
    ap.add_argument("--chunk_start_ms", type=int, default=0, help="absolute chunk start ms")
    ap.add_argument("--step_ms", type=int, default=40, help="frame step ms")
    ap.add_argument(
        "--emo_map_yaml",
        default="configs/emo_id_to_expression.yaml",
        help="emo_id -> expression yaml path",
    )
    args = ap.parse_args()

    m1_path = Path(args.m1_unified_json)
    audio_meta_path = Path(args.audio_meta_json)
    out_path = Path(args.out)
    emo_map_path = Path(args.emo_map_yaml)

    m1_unified = load_json(m1_path)
    audio_meta = load_json(audio_meta_path)
    emo_map = load_emo_id_to_expression_map(emo_map_path)

    step_ms = int(audio_meta.get("step_ms", args.step_ms))
    audio_ms = int(audio_meta["audio_ms"])
    chunk_start_ms = int(args.chunk_start_ms)

    session_id = args.session_id or str(
        m1_unified.get("session_id")
        or audio_meta.get("session_id")
        or "sess_real_01"
    )

    chunk_end_ms = chunk_start_ms + ceil_to_grid(audio_ms, step_ms)

    events = build_chunk_events_from_m1(
        m1_unified=m1_unified,
        audio_ms=audio_ms,
        step_ms=step_ms,
        emo_map=emo_map,
    )

    out_obj = {
        "schema": "expression_chunks.v1",
        "session_id": session_id,
        "step_ms": step_ms,
        "chunks": [
            {
                "chunk_id": 0,
                "chunk_start_ms": chunk_start_ms,
                "chunk_end_ms": chunk_end_ms,
                "audio_ms": audio_ms,
                "events": events,
                "meta": {
                    "source_m1_unified_json": m1_path.as_posix(),
                    "source_audio_meta_json": audio_meta_path.as_posix(),
                    "source_emo_map_yaml": emo_map_path.as_posix(),
                },
            }
        ],
        "meta": {
            "source": "build_expression_chunks_from_m1.py",
            "chunks_n": 1,
        },
    }

    save_json(out_path, out_obj)

    print("[build_expression_chunks_from_m1][OK]")
    print(" m1_unified :", m1_path.as_posix())
    print(" audio_meta :", audio_meta_path.as_posix())
    print(" emo_map    :", emo_map_path.as_posix())
    print(" out        :", out_path.as_posix())
    print(" session_id :", session_id)
    print(" step_ms    :", step_ms)
    print(" audio_ms   :", audio_ms)
    print(" chunk_end_ms:", chunk_end_ms)
    print(" emo_id     :", events[0]["emo_id"])
    print(" expression :", events[0]["expression"])


if __name__ == "__main__":
    main()