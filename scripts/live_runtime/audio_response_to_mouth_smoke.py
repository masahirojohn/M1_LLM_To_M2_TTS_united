#!/usr/bin/env python3
from __future__ import annotations

"""
audio_response_to_mouth_smoke.py

目的:
- audio_stream_bridge.py が保存した audio_response.pcm を
  M3 MouthStreamerOC に流す単体スモーク。
- Live API音声入力 → Gemini音声応答 → M3 mouth解析 の接続確認。

入力:
- audio_response.pcm
- sample_rate: Gemini Live API audio output は通常 24000Hz PCM16 mono

出力:
- mouth_streamer.json
- mouth_timeline.formant.raw.json
- audio_meta.json
- audio_response_to_mouth_smoke.summary.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from m3p.live.mouth_streamer_oc import MouthStreamerOC, MouthOCConfig


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_bytes(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"missing input pcm: {path}")
    data = path.read_bytes()
    if not data:
        raise RuntimeError(f"empty input pcm: {path}")
    if len(data) % 2 != 0:
        raise RuntimeError(f"PCM16 bytes must be even length: {path} bytes={len(data)}")
    return data


def _chunk_bytes(data: bytes, *, sample_rate: int, chunk_ms: int) -> list[bytes]:
    bytes_per_sample = 2
    chunk_size = int(round(sample_rate * chunk_ms / 1000.0)) * bytes_per_sample
    chunk_size = max(2, chunk_size)
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def _project_streamer_to_raw(streamer_json_path: Path) -> dict[str, Any]:
    data = json.loads(streamer_json_path.read_text(encoding="utf-8"))
    frames_in = data.get("frames", []) if isinstance(data, dict) else []

    frames_out: list[dict[str, Any]] = []
    for fr in frames_in:
        if not isinstance(fr, dict):
            continue
        frames_out.append(
            {
                "t_ms": fr.get("t_ms"),
                "vad_active": int(fr.get("vad_active", 0) or 0),
                "f1_hz": fr.get("f1_hz"),
                "f2_hz": fr.get("f2_hz"),
                "src": "audio_response_to_mouth_smoke",
            }
        )

    return {
        "version": "m3p.mouth.timeline.v1",
        "step_ms": int(data.get("step_ms", 40)),
        "frames": frames_out,
        "meta": data.get("meta", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="audio_response.pcm -> MouthStreamerOC smoke"
    )

    ap.add_argument("--session_id", default="sess_audio_response_to_mouth_001")
    ap.add_argument("--input_pcm", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--input_sr", type=int, default=24000)
    ap.add_argument("--step_ms", type=int, default=40)
    ap.add_argument("--chunk_ms", type=int, default=40)

    ap.add_argument("--analysis_sr", type=int, default=16000)
    ap.add_argument("--window_ms", type=int, default=240)

    ap.add_argument("--rms_thr", type=float, default=0.015)
    ap.add_argument("--vad_energy_thr", type=float, default=0.0004)
    ap.add_argument("--vad_min_speech_ms", type=int, default=80)
    ap.add_argument("--vad_min_silence_ms", type=int, default=120)

    ap.add_argument("--open_id", type=int, default=1)
    ap.add_argument("--close_id", type=int, default=0)

    ap.add_argument("--flush_every_frames", type=int, default=25)
    ap.add_argument("--max_buffer_s", type=float, default=10.0)

    ap.add_argument("--vowel_mode", choices=["formant", "simple"], default="formant")
    ap.add_argument("--formant_window_ms", type=int, default=200)
    ap.add_argument("--formant_max_hz", type=int, default=5500)

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_pcm = Path(args.input_pcm).resolve()
    pcm = _load_bytes(input_pcm)

    streamer_json = out_dir / "mouth_streamer.json"
    raw_json = out_dir / "mouth_timeline.formant.raw.json"
    audio_meta_json = out_dir / "audio_meta.json"
    summary_json = out_dir / "audio_response_to_mouth_smoke.summary.json"

    streamer = MouthStreamerOC(
        out_json=str(streamer_json),
        session_id=str(args.session_id),
        cfg=MouthOCConfig(
            step_ms=int(args.step_ms),
            window_ms=int(args.window_ms),
            analysis_sr=int(args.analysis_sr),
            input_sr_default=int(args.input_sr),
            rms_thr=float(args.rms_thr),
            vad_energy_thr=float(args.vad_energy_thr),
            vad_min_speech_ms=int(args.vad_min_speech_ms),
            vad_min_silence_ms=int(args.vad_min_silence_ms),
            open_id=int(args.open_id),
            close_id=int(args.close_id),
            flush_every_frames=int(args.flush_every_frames),
            max_buffer_s=float(args.max_buffer_s),
            vowel_mode=str(args.vowel_mode),
            formant_window_ms=int(args.formant_window_ms),
            formant_max_hz=int(args.formant_max_hz),
        ),
    )

    chunks = _chunk_bytes(
        pcm,
        sample_rate=int(args.input_sr),
        chunk_ms=int(args.chunk_ms),
    )

    total_bytes = 0
    for ch in chunks:
        if not ch:
            continue
        streamer.push_pcm16_mono(ch, input_sr=int(args.input_sr))
        total_bytes += len(ch)

    streamer.finalize()

    raw_obj = _project_streamer_to_raw(streamer_json)
    _write_json(raw_json, raw_obj)

    total_samples = total_bytes // 2
    audio_ms = int(round(total_samples * 1000.0 / int(args.input_sr)))
    target_frames = int(math.ceil(audio_ms / int(args.step_ms)))

    frames = raw_obj.get("frames", [])
    voiced_frames = [
        fr for fr in frames
        if int(fr.get("vad_active", 0) or 0) == 1
    ]

    audio_meta = {
        "schema": "audio_meta.live.v0.1",
        "session_id": str(args.session_id),
        "sample_rate": int(args.input_sr),
        "channels": 1,
        "sample_width_bytes": 2,
        "audio_bytes": int(total_bytes),
        "audio_samples": int(total_samples),
        "audio_ms": int(audio_ms),
        "step_ms": int(args.step_ms),
        "source": str(input_pcm),
    }
    _write_json(audio_meta_json, audio_meta)

    summary = {
        "format": "audio_response_to_mouth_smoke.summary.v0",
        "session_id": str(args.session_id),
        "input_pcm": str(input_pcm),
        "input_sr": int(args.input_sr),
        "audio_bytes": int(total_bytes),
        "audio_ms": int(audio_ms),
        "step_ms": int(args.step_ms),
        "target_frames": int(target_frames),
        "frames_n": int(len(frames)),
        "voiced_frames_n": int(len(voiced_frames)),
        "outputs": {
            "streamer_json": str(streamer_json),
            "raw_json": str(raw_json),
            "audio_meta_json": str(audio_meta_json),
        },
    }
    _write_json(summary_json, summary)

    print("[audio_response_to_mouth_smoke][OK]")
    print(f"  input_pcm      : {input_pcm}")
    print(f"  audio_ms       : {audio_ms}")
    print(f"  target_frames  : {target_frames}")
    print(f"  frames_n       : {len(frames)}")
    print(f"  voiced_frames_n: {len(voiced_frames)}")
    print(f"  streamer_json  : {streamer_json}")
    print(f"  raw_json       : {raw_json}")
    print(f"  audio_meta_json: {audio_meta_json}")
    print(f"  summary_json   : {summary_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())