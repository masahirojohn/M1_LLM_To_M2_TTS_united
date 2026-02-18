#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from aituber_llm_tts.config import load_config
from aituber_llm_tts.tts_client import RealTTSClient  # synthesize_pcm_bytes を追加済み前提

# M3側（PYTHONPATHでM3/srcを通すこと）
from m3p.live.mouth_streamer_oc import MouthStreamerOC, MouthOCConfig


def project_streamer_to_raw(streamer_json_path: Path) -> dict:
    data = json.loads(streamer_json_path.read_text(encoding="utf-8"))
    frames_in = data.get("frames", [])
    frames_out = []
    for fr in frames_in:
        frames_out.append(
            {
                "t_ms": fr.get("t_ms"),
                "vad_active": int(fr.get("vad_active", 0) or 0),
                "f1_hz": fr.get("f1_hz", None),
                "f2_hz": fr.get("f2_hz", None),
                "src": "mouth_streamer_oc",
            }
        )
    return {
        "version": "m3p.mouth.timeline.v1",
        "step_ms": int(data.get("step_ms", 40)),
        "frames": frames_out,
        "meta": data.get("meta", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--session_id", default="sess_tts_stream_test")
    ap.add_argument("--utt_id", default="utt_000001")

    # SSOT
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--analysis_sr", type=int, default=16000)
    ap.add_argument("--vad_energy_thr", type=float, default=0.0004)

    # outputs
    ap.add_argument("--out_streamer_json", default="streamer.json")
    ap.add_argument("--out_raw_json", default="mouth_timeline.formant.raw.json")
    ap.add_argument("--out_audio_meta_json", default="audio_meta.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tts = RealTTSClient(cfg)

    # M1Outputは必須ではない（duck typing）。textだけ使うので簡易でOK。
    m1_like = SimpleNamespace(
        session_id=args.session_id,
        utt_id=args.utt_id,
        text=args.text,
    )

    pcm_bytes, sample_rate, audio_ms = tts.synthesize_pcm_bytes(m1_like)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    streamer_json = out_dir / args.out_streamer_json
    raw_json = out_dir / args.out_raw_json
    audio_meta_json = out_dir / args.out_audio_meta_json

    # M3 streamer init（1回のみ、再生成禁止）
    m3_cfg = MouthOCConfig(
        step_ms=int(args.step_ms),
        analysis_sr=int(args.analysis_sr),
        input_sr_default=int(sample_rate),
        vad_energy_thr=float(args.vad_energy_thr),
    )
    ms = MouthStreamerOC(out_json=str(streamer_json), session_id=str(args.session_id), cfg=m3_cfg)

    # PCM bytes -> 40ms刻みでpush（wav無し）
    bytes_per_sample = 2  # PCM16 mono
    chunk_samples = int(sample_rate * int(args.step_ms) / 1000)
    chunk_bytes = chunk_samples * bytes_per_sample

    # 2byte境界は必ず守る
    pcm_bytes = pcm_bytes[: (len(pcm_bytes) // 2) * 2]

    for off in range(0, len(pcm_bytes), chunk_bytes):
        b = pcm_bytes[off : off + chunk_bytes]
        if not b:
            break
        ms.push_pcm16_mono(b, input_sr=int(sample_rate))

    ms.finalize()

    # M2 SSOT: audio_ms を別JSONに書き出す
    audio_meta = {
        "format": "audio_meta.v1",
        "session_id": str(args.session_id),
        "utt_id": str(args.utt_id),
        "step_ms": int(args.step_ms),
        "sample_rate": int(sample_rate),
        "audio_ms": int(audio_ms),
    }
    audio_meta_json.write_text(json.dumps(audio_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_obj = project_streamer_to_raw(streamer_json)
    raw_json.write_text(json.dumps(raw_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] sample_rate =", sample_rate)
    print("[OK] audio_ms (M2 SSOT) =", audio_ms)
    print("[OK] wrote:", audio_meta_json.as_posix())
    print("[OK] wrote:", raw_json.as_posix())


if __name__ == "__main__":
    main()
